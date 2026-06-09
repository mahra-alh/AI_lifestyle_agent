"""
Closes the learning loop: turns real feedback signals into labeled
training rows that can retrain the LightGBM ranker.

Two responsibilities:
  1. InferenceFeatureLog — written at serve time so features are
     paired with outcomes at retraining.
  2. build_training_frame() — joins feature logs + feedback logs
     into a model-ready DataFrame.

Firestore collections read:
  activity_preference_logs    — implicit signals (clicked/booked/declined)
  recommendation_feedback     — explicit signals (liked/disliked/neutral + rating)
  inference_feature_logs      — feature vectors logged at serve time

Firestore collections written:
  inference_feature_logs      — one doc per scored candidate per request

Usage (serve time — inside recommendation_service.py):
    from ml.feedback_join import InferenceFeatureLog, write_inference_log
    write_inference_log(InferenceFeatureLog.from_row(...))

Usage (retraining):
    from ml.feedback_join import load_training_frame
    train_df = load_training_frame(drop_leaked=True)
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

import firebase_admin
from firebase_admin import credentials, firestore

from ml.schemas.feature_contract import FEATURE_COLUMNS, training_feature_columns

_CRED_PATH = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH", "config/serviceAccountKey.json")

_COL_FEATURE_LOGS  = "inference_feature_logs"
_COL_ACTIVITY_LOGS = "activity_preference_logs"
_COL_FEEDBACK_LOGS = "recommendation_feedback"


def _get_db():
    """Return the Firestore client, initializing Firebase on first call."""
    if not firebase_admin._apps:
        cred = credentials.Certificate(_CRED_PATH)
        firebase_admin.initialize_app(cred)
    return firestore.client()


# ---------------------------------------------------------------------------
# Feature set hash
# ---------------------------------------------------------------------------

def feature_set_hash() -> str:
    """Stable 12-char hash of the current feature contract column order."""
    payload = json.dumps(FEATURE_COLUMNS, sort_keys=False).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


# ---------------------------------------------------------------------------
# InferenceFeatureLog — written at serve time
# ---------------------------------------------------------------------------

@dataclass
class InferenceFeatureLog:
    """
    One row per scored candidate.

    Written at serve time so every recommendation can be traced back to
    the exact feature values that produced it. The join key
    (recommendation_id, venue_id) links to outcome events.
    """
    recommendation_id: str
    user_id: str
    venue_id: str
    model_version: str
    rank: int
    model_score: float
    features: dict[str, float]
    feature_set_hash: str = field(default_factory=feature_set_hash)
    session_id: Optional[str] = None
    scored_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @classmethod
    def from_row(
        cls,
        recommendation_id: str,
        user_id: str,
        venue_id: str,
        model_version: str,
        rank: int,
        model_score: float,
        feature_row: dict[str, Any],
        session_id: Optional[str] = None,
    ) -> "InferenceFeatureLog":
        """Build from a raw feature dict, keeping only contract columns."""
        features = {
            col: float(feature_row.get(col, 0) or 0)
            for col in FEATURE_COLUMNS
        }
        return cls(
            recommendation_id=recommendation_id,
            user_id=user_id,
            venue_id=venue_id,
            model_version=model_version,
            rank=rank,
            model_score=model_score,
            features=features,
            session_id=session_id,
        )

    def to_storage(self) -> dict[str, Any]:
        return asdict(self)


def write_inference_log(log: InferenceFeatureLog) -> None:
    """Write one InferenceFeatureLog entry to Firestore."""
    doc_id = f"{log.recommendation_id}_{log.venue_id}"
    _get_db().collection(_COL_FEATURE_LOGS).document(doc_id).set(log.to_storage())


# ---------------------------------------------------------------------------
# Label derivation
# ---------------------------------------------------------------------------

IMPLICIT_LABEL: dict[str, float] = {
    "booked":   1.0,
    "clicked":  0.5,
    "declined": 0.0,
}
EXPLICIT_LABEL: dict[str, float] = {
    "liked":    1.0,
    "neutral":  0.4,
    "disliked": 0.0,
}
IMPRESSION_ONLY_LABEL: float | None = 0.1


def derive_label(
    signal: Any = None,
    feedback: Any = None,
    rating: Any = None,
) -> float | None:
    """
    Combine implicit + explicit signals into one 0–1 relevance label.
    Explicit feedback always outranks implicit behavior.
    Returns None when there is no usable signal.
    """
    if feedback is not None and not _is_na(feedback):
        base = EXPLICIT_LABEL.get(str(feedback).strip().lower())
        if base is not None:
            if rating is not None and not _is_na(rating):
                return round(0.5 * base + 0.5 * (float(rating) - 1.0) / 4.0, 4)
            return base
    if rating is not None and not _is_na(rating):
        return max(0.0, min(1.0, (float(rating) - 1.0) / 4.0))
    if signal is not None and not _is_na(signal):
        return IMPLICIT_LABEL.get(str(signal).strip().lower())
    return None


def _is_na(value: Any) -> bool:
    if isinstance(value, float):
        import math
        return math.isnan(value)
    return value is None


# ---------------------------------------------------------------------------
# Firestore loaders
# ---------------------------------------------------------------------------

def _load_collection(collection: str) -> pd.DataFrame:
    """Load all documents from a Firestore collection into a DataFrame."""
    docs = _get_db().collection(collection).stream()
    rows = [doc.to_dict() for doc in docs]
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def load_feature_logs() -> pd.DataFrame:
    return _load_collection(_COL_FEATURE_LOGS)


def load_activity_logs() -> pd.DataFrame:
    return _load_collection(_COL_ACTIVITY_LOGS)


def load_feedback_logs() -> pd.DataFrame:
    return _load_collection(_COL_FEEDBACK_LOGS)


# ---------------------------------------------------------------------------
# Training frame builder
# ---------------------------------------------------------------------------

def load_training_frame(*, drop_leaked: bool = True) -> pd.DataFrame:
    """
    Load all logs from Firestore and return a model-ready training frame.
    """
    feature_log  = load_feature_logs()
    activity_log = load_activity_logs()
    feedback_log = load_feedback_logs()

    if feature_log.empty:
        raise RuntimeError(
            "No inference feature logs found in Firestore. "
            "The model must serve at least some requests before retraining "
            "on real labels is possible."
        )

    return build_training_frame(
        feature_log=feature_log,
        activity_log=activity_log if not activity_log.empty else None,
        feedback_log=feedback_log if not feedback_log.empty else None,
        drop_leaked=drop_leaked,
    )


def build_training_frame(
    feature_log: pd.DataFrame,
    activity_log: pd.DataFrame | None = None,
    feedback_log: pd.DataFrame | None = None,
    *,
    drop_leaked: bool = True,
) -> pd.DataFrame:
    """
    Join feature logs with outcome signals to produce a labeled training frame.
    """
    fl = feature_log.copy()
    fl["venue_id"]          = fl["venue_id"].astype(str)
    fl["recommendation_id"] = fl["recommendation_id"].astype(str)

    implicit = pd.DataFrame(columns=["recommendation_id", "venue_id", "performance_signal"])
    if activity_log is not None and len(activity_log):
        a = activity_log.rename(columns={"activity_id": "venue_id"}).copy()
        a["venue_id"]          = a["venue_id"].astype(str)
        a["recommendation_id"] = a["recommendation_id"].astype(str)
        a["_w"] = a["performance_signal"].map(IMPLICIT_LABEL).fillna(0.0)
        a = a.sort_values("_w").drop_duplicates(["recommendation_id", "venue_id"], keep="last")
        implicit = a[["recommendation_id", "venue_id", "performance_signal"]]

    explicit = pd.DataFrame(columns=["recommendation_id", "venue_id", "feedback", "rating"])
    if feedback_log is not None and len(feedback_log):
        e = feedback_log.rename(columns={"activity_id": "venue_id"}).copy()
        e["venue_id"]          = e["venue_id"].astype(str)
        e["recommendation_id"] = e["recommendation_id"].astype(str)
        if "created_at" in e.columns:
            e = e.sort_values("created_at")
        e = e.drop_duplicates(["recommendation_id", "venue_id"], keep="last")
        cols = ["recommendation_id", "venue_id", "feedback"]
        if "rating" in e.columns:
            cols.append("rating")
        explicit = e[cols]

    df = fl.merge(implicit, on=["recommendation_id", "venue_id"], how="left")
    df = df.merge(explicit, on=["recommendation_id", "venue_id"], how="left")

    rating_col = df["rating"] if "rating" in df.columns else pd.Series([None] * len(df))
    df["label"] = [
        derive_label(sig, fb, rt)
        for sig, fb, rt in zip(
            df.get("performance_signal"),
            df.get("feedback"),
            rating_col,
        )
    ]
    df["label_source"] = df.apply(
        lambda r: "explicit"
        if pd.notna(r.get("feedback")) or pd.notna(r.get("rating"))
        else ("implicit" if pd.notna(r.get("performance_signal")) else "impression"),
        axis=1,
    )

    if IMPRESSION_ONLY_LABEL is None:
        df = df[df["label"].notna()]
    else:
        df["label"] = df["label"].fillna(IMPRESSION_ONLY_LABEL)

    feats = pd.json_normalize(df["features"]).reindex(
        columns=FEATURE_COLUMNS, fill_value=0.0
    )
    feats.index = df.index
    keep_cols = training_feature_columns(drop_leaked=drop_leaked)
    meta = df[[
        "user_id", "venue_id", "recommendation_id",
        "model_version", "label", "label_source",
    ]].reset_index(drop=True)

    return pd.concat(
        [feats[keep_cols].reset_index(drop=True), meta],
        axis=1,
    )