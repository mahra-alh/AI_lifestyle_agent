"""
ML recommender service for the Lifestyle AI Agent.

Purpose:
- Receive a validated RecommendationRequest.
- Build a semantic query from the user profile + prompt.
- Retrieve candidate activities using FAISS.
- Apply simple hard filters such as budget and weather.
- Return a validated RecommendationResponse.

Later:
- Add LightGBM ranking after FAISS candidate retrieval.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.schemas.recommendation_contracts import (
    RecommendationRequest,
    RecommendationResponse,
    RecommendationItem,
)

try:
    import faiss
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

class LifestyleRecommender:
    """
    Main ML recommender service.

    This class is the bridge between:
    - AI Agent
    - FAISS semantic retrieval
    - LightGBM ranking
    """

    def __init__(
        self,
        artifacts_dir: str | Path | None = None,
        embedding_model_name: str = "all-MiniLM-L6-v2",
    ):
        self.ml_dir = Path(__file__).resolve().parent

        # Main artifact location we want to use going forward.
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else self.ml_dir / "artifacts"

        # Compatibility location because the current FAISS code may still use ml/models.
        self.models_dir = self.ml_dir / "models"

        self.embedding_model_name = embedding_model_name

        self.index_path = self._find_existing_file(
            [
                self.artifacts_dir / "faiss_index.bin",
                self.models_dir / "faiss_index.bin",
            ]
        )

        self.lookup_path = self._find_existing_file(
            [
                self.artifacts_dir / "faiss_lookup.csv",
                self.models_dir / "faiss_lookup.csv",
            ]
        )

        self.lightgbm_model_path = self._find_existing_file(
            [
                self.artifacts_dir / "lightgbm_model.txt",
                self.models_dir / "lightgbm_model.txt",
            ]
        )

        self.embedding_model = None
        self.faiss_index = None
        self.lookup_df = None
        self.lightgbm_model = None

        self._load_artifacts()

    def _find_existing_file(self, candidate_paths: list[Path]) -> Path | None:
        """
        Return the first existing file from a list of possible paths.
        """
        for path in candidate_paths:
            if path.exists():
                return path
        return None

    def _load_artifacts(self) -> None:
        """
        Load FAISS, lookup CSV, embedding model, and optional LightGBM model.
        This function does not crash if artifacts are missing.
        """

        if self.index_path and self.lookup_path and faiss is not None and SentenceTransformer is not None:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            self.faiss_index = faiss.read_index(str(self.index_path))
            self.lookup_df = pd.read_csv(self.lookup_path)

        if self.lightgbm_model_path and lgb is not None:
            self.lightgbm_model = lgb.Booster(model_file=str(self.lightgbm_model_path))

    def recommend(self, request_data: RecommendationRequest | dict[str, Any]) -> RecommendationResponse:
        """
        Main public method.

        Input:
            RecommendationRequest or dictionary

        Output:
            RecommendationResponse
        """

        request = (
            request_data
            if isinstance(request_data, RecommendationRequest)
            else RecommendationRequest.model_validate(request_data)
        )

        if not self._faiss_ready():
            return RecommendationResponse(
                status="no_results",
                source="fallback",
                request_id=request.request_id,
                model_version="faiss_not_loaded",
                items=[],
                message=(
                    "FAISS artifacts are not loaded yet. "
                    "Expected faiss_index.bin and faiss_lookup.csv in ml/artifacts or ml/models."
                ),
                fallback_reason="missing_faiss_artifacts_or_dependencies",
            )

        query_text = self._build_query_text(request)
        raw_candidates = self._search_faiss(query_text, search_k=max(request.top_k * 5, 20))

        if not raw_candidates:
            return RecommendationResponse(
                status="no_results",
                source="ml",
                request_id=request.request_id,
                model_version=self._model_version(),
                items=[],
                message="FAISS search returned no candidates.",
            )

        filtered_candidates = self._apply_hard_filters(raw_candidates, request)

        if not filtered_candidates:
            return RecommendationResponse(
                status="no_results",
                source="fallback",
                request_id=request.request_id,
                model_version=self._model_version(),
                items=[],
                message="Candidates were found, but all were removed by hard filters.",
                fallback_reason="all_candidates_filtered_out",
            )

        ranked_candidates = self._rank_candidates(filtered_candidates, request)
        recommendation_items = self._format_items(ranked_candidates[: request.top_k], request)

        return RecommendationResponse(
            status="success",
            source="ml",
            request_id=request.request_id,
            model_version=self._model_version(),
            items=recommendation_items,
            message="Recommendations generated successfully.",
        )

    def _faiss_ready(self) -> bool:
        """
        Check if FAISS retrieval is ready.
        """
        return (
            self.embedding_model is not None
            and self.faiss_index is not None
            and self.lookup_df is not None
        )

    def _model_version(self) -> str:
        """
        Return a simple model version string for logs and debugging.
        """
        faiss_status = "faiss_loaded" if self._faiss_ready() else "faiss_missing"
        lightgbm_status = "lightgbm_loaded" if self.lightgbm_model is not None else "lightgbm_missing"
        return f"{faiss_status}_{lightgbm_status}"

    def _build_query_text(self, request: RecommendationRequest) -> str:
        """
        Convert user profile + prompt into one text query for FAISS.
        """

        profile = request.profile
        time_context = request.time_context
        weather_context = request.weather_context

        parts = [
            f"User request: {request.user_query}",
            f"Budget: {profile.budget_level}",
            f"Preferred areas: {', '.join(profile.preferred_areas) if profile.preferred_areas else 'unknown'}",
            f"Activity preferences: {', '.join(profile.activity_preferences) if profile.activity_preferences else 'unknown'}",
            f"Disliked activities: {', '.join(profile.disliked_activities) if profile.disliked_activities else 'none'}",
            f"Transport mode: {profile.transport_mode}",
            f"Requested date: {time_context.requested_date}",
            f"Requested start time: {time_context.requested_start_time}",
            f"Weather condition: {weather_context.condition or 'unknown'}",
            f"Outdoor suitable: {weather_context.outdoor_suitable}",
        ]

        return " | ".join(parts)

    def _search_faiss(self, query_text: str, search_k: int) -> list[pd.Series]:
        """
        Search FAISS and return matching rows from the lookup DataFrame.
        """

        query_embedding = self.embedding_model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        scores, indices = self.faiss_index.search(query_embedding, search_k)

        results = []

        for score, index_position in zip(scores[0], indices[0]):
            if index_position < 0:
                continue

            row = self.lookup_df.iloc[int(index_position)].copy()
            row["_similarity_score"] = float(score)
            results.append(row)

        return results

    def _apply_hard_filters(
        self,
        candidates: list[pd.Series],
        request: RecommendationRequest,
    ) -> list[pd.Series]:
        """
        Apply basic hard filters before ranking.

        Important:
        - Hard filters remove impossible or clearly unsuitable options.
        - Soft preferences should be handled by ranking, not filtering.
        """

        filtered = []

        for row in candidates:
            if self._fails_budget_filter(row, request):
                continue

            if self._fails_weather_filter(row, request):
                continue

            if self._fails_disliked_activity_filter(row, request):
                continue

            filtered.append(row)

        return filtered

    def _fails_budget_filter(self, row: pd.Series, request: RecommendationRequest) -> bool:
        """
        Remove clearly expensive activities for low/free budget users.
        """

        user_budget = request.profile.budget_level
        item_budget = self._get_value(
            row,
            ["estimated_budget_level", "budget_level", "budget", "price_level", "cost_level"],
            default="unknown",
        )

        item_budget = str(item_budget).lower().strip()

        allowed_by_user_budget = {
            "free": {"free", "unknown"},
            "low": {"free", "low", "unknown"},
            "medium": {"free", "low", "medium", "unknown"},
            "high": {"free", "low", "medium", "high", "unknown"},
            "unknown": {"free", "low", "medium", "high", "unknown"},
        }

        return item_budget not in allowed_by_user_budget.get(user_budget, {"unknown"})

    def _fails_weather_filter(self, row: pd.Series, request: RecommendationRequest) -> bool:
        """
        If weather is not outdoor-suitable, remove outdoor-only activities.
        """

        outdoor_suitable = request.weather_context.outdoor_suitable

        if outdoor_suitable is not False:
            return False

        indoor_outdoor = self._get_value(
            row,
            ["indoor_outdoor", "environment", "activity_environment"],
            default="unknown",
        )

        indoor_outdoor = str(indoor_outdoor).lower().strip()

        return indoor_outdoor == "outdoor"

    def _fails_disliked_activity_filter(self, row: pd.Series, request: RecommendationRequest) -> bool:
        """
        Remove activities that clearly match disliked words.
        """

        disliked = [x.lower().strip() for x in request.profile.disliked_activities if x.strip()]

        if not disliked:
            return False

        searchable_text = " ".join(
            [
                str(self._get_value(row, ["name", "title", "activity_name"], default="")),
                str(self._get_value(row, ["category", "primary_category", "item_type"], default="")),
                str(self._get_value(row, ["faiss_text", "description", "text"], default="")),
            ]
        ).lower()

        return any(word in searchable_text for word in disliked)

    def _rank_candidates(
        self,
        candidates: list[pd.Series],
        request: RecommendationRequest,
    ) -> list[pd.Series]:
        """
        Rank candidates.

        Current behavior:
        - Uses FAISS similarity score.
        - LightGBM will be added after the trained model artifact and feature list are ready.
        """

        return sorted(
            candidates,
            key=lambda row: float(row.get("_similarity_score", 0.0)),
            reverse=True,
        )

    def _format_items(
        self,
        ranked_rows: list[pd.Series],
        request: RecommendationRequest,
    ) -> list[RecommendationItem]:
        """
        Convert ranked DataFrame rows into RecommendationItem objects.
        """

        items = []

        for row in ranked_rows:
            raw_score = float(row.get("_similarity_score", 0.0))
            score = self._normalize_score(raw_score)

            name = self._get_value(row, ["name", "title", "activity_name", "venue_name"], default="Unknown activity")
            category = self._get_value(row, ["category", "primary_category", "item_type"], default="unknown")
            area = self._get_value(row, ["area", "location_area", "district", "neighborhood"], default=None)

            item = RecommendationItem(
                activity_id=str(
                    self._get_value(
                        row,
                        ["activity_id", "place_id", "id", "event_id", "venue_id"],
                        default=f"candidate_{len(items) + 1}",
                    )
                ),
                name=str(name),
                category=str(category),
                area=str(area) if area is not None else None,
                latitude=self._to_float_or_none(
                    self._get_value(row, ["latitude", "lat"], default=None)
                ),
                longitude=self._to_float_or_none(
                    self._get_value(row, ["longitude", "lon", "lng"], default=None)
                ),
                estimated_budget_level=str(
                    self._get_value(
                        row,
                        ["estimated_budget_level", "budget_level", "budget", "price_level", "cost_level"],
                        default="unknown",
                    )
                ).lower(),
                indoor_outdoor=str(
                    self._get_value(
                        row,
                        ["indoor_outdoor", "environment", "activity_environment"],
                        default="unknown",
                    )
                ).lower(),
                score=score,
                reason=self._build_reason(row, request),
                constraints_matched=self._matched_constraints(row, request),
                risk_flags=self._risk_flags(row, request),
                metadata={
                    "raw_similarity_score": raw_score,
                },
            )

            items.append(item)

        return items

    def _build_reason(self, row: pd.Series, request: RecommendationRequest) -> str:
        """
        Create a short reason for the AI Agent to explain the recommendation.
        """

        matched = self._matched_constraints(row, request)

        if matched:
            return f"Recommended because it matches: {', '.join(matched)}."

        return "Recommended based on semantic similarity to the user's request and profile."

    def _matched_constraints(self, row: pd.Series, request: RecommendationRequest) -> list[str]:
        """
        Identify which user constraints this candidate seems to match.
        """

        matched = []

        item_budget = str(
            self._get_value(
                row,
                ["estimated_budget_level", "budget_level", "budget", "price_level", "cost_level"],
                default="unknown",
            )
        ).lower()

        if item_budget == request.profile.budget_level or item_budget == "unknown":
            matched.append("budget")

        item_area = str(
            self._get_value(row, ["area", "location_area", "district", "neighborhood"], default="")
        ).lower()

        preferred_areas = [area.lower() for area in request.profile.preferred_areas]

        if item_area and any(area in item_area or item_area in area for area in preferred_areas):
            matched.append("location")

        if request.calendar_context.is_free:
            matched.append("calendar")

        if request.weather_context.outdoor_suitable is not False:
            matched.append("weather")

        return matched

    def _risk_flags(self, row: pd.Series, request: RecommendationRequest) -> list[str]:
        """
        Add warnings that the agent/frontend can display or use.
        """

        risks = []

        if request.weather_context.outdoor_suitable is False:
            indoor_outdoor = str(
                self._get_value(
                    row,
                    ["indoor_outdoor", "environment", "activity_environment"],
                    default="unknown",
                )
            ).lower()

            if indoor_outdoor in {"outdoor", "unknown"}:
                risks.append("weather_uncertain")

        if not request.calendar_context.is_free:
            risks.append("calendar_conflict")

        return risks

    def _get_value(self, row: pd.Series, possible_columns: list[str], default: Any = None) -> Any:
        """
        Safely read a value from a row using possible column names.
        """

        for column in possible_columns:
            if column in row.index and pd.notna(row[column]):
                return row[column]

        return default

    def _to_float_or_none(self, value: Any) -> float | None:
        """
        Convert a value to float if possible.
        """

        if value is None or pd.isna(value):
            return None

        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _normalize_score(self, score: float) -> float:
        """
        Convert similarity score to a safe 0–1 range for the response contract.
        """

        if score < 0:
            score = (score + 1) / 2

        return max(0.0, min(1.0, score))