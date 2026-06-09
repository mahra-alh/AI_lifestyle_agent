"""
Full recommendation pipeline: FAISS retrieval --> LightGBM reranking.

Entry point for the agent tool. Handles:
  - Hot path: FAISS retrieval --> LightGBM reranking using user profile
  - Cold-start fallback: new users with no profile get popularity-based ranking
  - Inference feature logging: writes feature vectors to Firestore so
    the feedback loop can retrain on real labels
  - Model loading from the registry: always serves the active model version

Usage (from recommendation_tool.py):
    from ml.recommendation_service import get_recommendations
    result = get_recommendations(request)
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from ml.faiss.semantic_search import get_retriever
from ml.feedback_join import InferenceFeatureLog, write_inference_log
from ml.lightgbm.feature_builder import (
    build_lightgbm_feature_frame,
    calculate_rule_rank_score,
)
from ml.schemas.feature_contract import assert_booster_matches
from ml.schemas.recommendation_contracts import RecommendationRequest, RecommendationResult

# ---------------------------------------------------------------------------
# Model loader — lazy singleton, nothing touches Firebase at import time
# ---------------------------------------------------------------------------

_model: Any = None
_active_version: str = "rule_fallback"

# near the top, after other imports
_venue_pool: Optional[pd.DataFrame] = None

def _get_venue_pool() -> pd.DataFrame:
    global _venue_pool
    if _venue_pool is None:
        import os
        pool_path = os.getenv("VENUE_POOL_PATH", "ai_agent/data/unified_venue_pool.csv")
        _venue_pool = pd.read_csv(pool_path)
    return _venue_pool

def _get_registry():
    """Instantiate ModelRegistry only when first needed."""
    from ml.registry.model_registry import ModelRegistry
    return ModelRegistry()


def _load_active_model() -> tuple[Any, str]:
    """
    Load the active model from the registry.

    Falls back to rule-based scoring if no model is registered or the
    artifact file is missing.
    """
    global _model, _active_version

    try:
        meta = _get_registry().get_active()
        if meta is None:
            return None, "rule_fallback"

        artifact_path = meta["artifact_path"]
        if not Path(artifact_path).exists():
            return None, "rule_fallback"

        model = joblib.load(artifact_path)
        assert_booster_matches(model)
        _model = model
        _active_version = meta["version"]
        return _model, _active_version

    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning(
            "Could not load active model (%s). Falling back to rule-based scoring.", exc
        )
        return None, "rule_fallback"


def get_model() -> tuple[Any, str]:
    """Return the cached model and its version string."""
    if _model is None:
        return _load_active_model()
    return _model, _active_version


# ---------------------------------------------------------------------------
# Cold-start fallback
# ---------------------------------------------------------------------------

def _cold_start_rank(
    candidates: pd.DataFrame,
    user_area: Optional[str] = None,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Rank candidates for a new user with no profile.

    Strategy (in priority order):
      1. popularity_score (descending) if the column exists in the venue pool
      2. faiss_score (descending) — semantic match to the query
      3. meal_cost_for_one (ascending) — cheaper is better for unknown budget

    Returns the top_n rows with a synthetic `model_score` column.
    """
    df = candidates.copy()

    sort_cols: list[str] = []
    ascending: list[bool] = []

    if "popularity_score" in df.columns:
        sort_cols.append("popularity_score")
        ascending.append(False)

    sort_cols.append("faiss_score")
    ascending.append(False)

    if "meal_cost_for_one" in df.columns:
        sort_cols.append("meal_cost_for_one")
        ascending.append(True)

    df = df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    n = len(df)
    df["model_score"] = [(n - i) / n for i in range(n)]

    return df.head(top_n)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def get_recommendations(
    request: RecommendationRequest,
    top_n: int = 5,
    faiss_k: int = 50,
    log_features: bool = True,
) -> RecommendationResult:
    """
    Run the full recommendation pipeline for one user request.

    Pipeline:
      1. FAISS retrieval — semantic top-k candidates from the venue pool
      2. Feature building — numeric feature matrix per candidate
      3. LightGBM reranking — or rule-based fallback if no model is loaded
      4. Cold-start fallback — if the user has no profile, skip reranking
         and use popularity + FAISS score instead
      5. Inference log — write feature vectors to Firestore for retraining
         (skipped when log_features=False, e.g. during tests)

    Args:
        request:      Structured recommendation request.
        top_n:        Number of final recommendations to return.
        faiss_k:      Number of candidates to retrieve from FAISS before reranking.
        log_features: Write InferenceFeatureLogs to Firestore (set False in tests).

    Returns:
        RecommendationResult with ranked venues and metadata.
    """
    recommendation_id = f"rec_{uuid.uuid4().hex[:12]}"

    retriever = get_retriever()
    model, model_version = get_model()
    is_cold_start = not request.profile or not _profile_is_warm(request.profile)

    # Step 1 — FAISS retrieval
    candidates_df = retriever.search(request.user_query, k=faiss_k)

    if candidates_df.empty:
        return RecommendationResult(
            recommendation_id=recommendation_id,
            recommendations=[],
            model_version=model_version,
            is_cold_start=is_cold_start,
            message="No matching venues found for this query.",
        )

    # Step 2 — Cold-start path (no profile → popularity fallback)
    if is_cold_start:
        ranked = _cold_start_rank(
            candidates_df,
            user_area=getattr(request.profile, "home_area", None),
            top_n=top_n,
        )
        recommendations = _build_recommendation_list(
            ranked_df=ranked,
            recommendation_id=recommendation_id,
            model_version="cold_start",
        )
        return RecommendationResult(
            recommendation_id=recommendation_id,
            recommendations=recommendations,
            model_version="cold_start",
            is_cold_start=True,
            message=(
                "New user — showing popular options near you. "
                "Your recommendations will personalise as you use the app."
            ),
        )

    # Step 3 — Feature building
    candidate_series = [candidates_df.iloc[i] for i in range(len(candidates_df))]
    feature_frame = build_lightgbm_feature_frame(
        candidates=candidate_series,
        request=request,
    )

    # Step 4 — Reranking (LightGBM or rule-based fallback)
    if model is not None:
        scores = model.predict(feature_frame)
    else:
        scores = calculate_rule_rank_score(feature_frame).values

    candidates_df = candidates_df.iloc[:len(feature_frame)].copy().reset_index(drop=True)
    candidates_df["model_score"] = scores
    ranked = (
        candidates_df
        .sort_values("model_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    # Step 5 — Log feature vectors for retraining (skipped in tests)
    if log_features:
        _log_inference_features(
            ranked_df=ranked,
            feature_frame=feature_frame,
            candidates_df=candidates_df,
            recommendation_id=recommendation_id,
            user_id=request.user_id,
            model_version=model_version,
            session_id=request.session_id,
        )

    recommendations = _build_recommendation_list(
        ranked_df=ranked,
        recommendation_id=recommendation_id,
        model_version=model_version,
    )

    return RecommendationResult(
        recommendation_id=recommendation_id,
        recommendations=recommendations,
        model_version=model_version,
        is_cold_start=False,
        message=f"Found {len(recommendations)} recommendations.",
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _profile_is_warm(profile: Any) -> bool:
    """
    Return True if the profile has enough data for personalised ranking.

    A profile is considered warm when it has at least a home area and a
    budget level — the two most influential features in the ranker.
    """
    if profile is None:
        return False
    home_area = getattr(profile, "home_area", None) or (
        profile.get("home_area") if isinstance(profile, dict) else None
    )
    budget = getattr(profile, "budget_level", None) or (
        profile.get("budget_level") if isinstance(profile, dict) else None
    )
    return bool(home_area) and bool(budget)


def _log_inference_features(
    ranked_df: pd.DataFrame,
    feature_frame: pd.DataFrame,
    candidates_df: pd.DataFrame,
    recommendation_id: str,
    user_id: str,
    model_version: str,
    session_id: Optional[str],
) -> None:
    """Write one InferenceFeatureLog row per ranked candidate to Firestore."""
    for rank_pos, row in ranked_df.iterrows():
        original_pos = candidates_df.index[candidates_df.index == row.name]
        if len(original_pos) == 0 or original_pos[0] >= len(feature_frame):
            continue

        feature_row = feature_frame.iloc[original_pos[0]].to_dict()
        venue_id = str(row.get("venue_id", row.get("name", f"venue_{rank_pos}")))

        log = InferenceFeatureLog.from_row(
            recommendation_id=recommendation_id,
            user_id=user_id,
            venue_id=venue_id,
            model_version=model_version,
            rank=int(rank_pos),
            model_score=float(row["model_score"]),
            feature_row=feature_row,
            session_id=session_id,
        )
        try:
            write_inference_log(log)
        except Exception:
            pass  # Never let logging failures break the serving path.


def _build_recommendation_list(
    ranked_df: pd.DataFrame,
    recommendation_id: str,
    model_version: str,
) -> List[Dict[str, Any]]:
    """Convert ranked rows into the agent-facing recommendation list."""
    
    # Join venue pool to resolve names and metadata
    pool = _get_venue_pool()
    ranked_df = ranked_df.merge(pool, on="venue_id", how="left")

    results: List[Dict[str, Any]] = []
    for rank_pos, row in enumerate(ranked_df.itertuples(index=False)):
        rec: Dict[str, Any] = {
            "rank": rank_pos + 1,
            "recommendation_id": recommendation_id,
            "model_version": model_version,
            "venue_id": str(getattr(row, "venue_id", f"venue_{rank_pos}")),
            "name": getattr(row, "name", None) or getattr(row, "venue_name", "Unknown venue"),
            "model_score": round(float(getattr(row, "model_score", 0.0)), 4),
            "faiss_score": round(float(getattr(row, "faiss_score", 0.0)), 4),
        }
        for col in [
            "area", "location_area", "category", "primary_category",
            "meal_cost_for_one", "budget_level", "description",
            "has_outdoor_seating", "serves_alcohol", "has_shisha",
            "latitude", "longitude",
        ]:
            val = getattr(row, col, None)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                rec[col] = val

        results.append(rec)

    return results

    for rank_pos, row in enumerate(ranked_df.itertuples(index=False)):
        rec: Dict[str, Any] = {
            "rank": rank_pos + 1,
            "recommendation_id": recommendation_id,
            "model_version": model_version,
            "venue_id": str(getattr(row, "venue_id", f"venue_{rank_pos}")),
            "name": getattr(row, "name", "Unknown venue"),
            "model_score": round(float(getattr(row, "model_score", 0.0)), 4),
            "faiss_score": round(float(getattr(row, "faiss_score", 0.0)), 4),
        }
        for col in [
            "area", "location_area", "category", "primary_category",
            "meal_cost_for_one", "budget_level", "description",
            "has_outdoor_seating", "serves_alcohol", "has_shisha",
            "latitude", "longitude",
        ]:
            val = getattr(row, col, None)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                rec[col] = val

        results.append(rec)

    return results