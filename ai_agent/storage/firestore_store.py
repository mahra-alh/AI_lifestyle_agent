"""
Firestore storage adapter.

Single place in the Python codebase that talks to Firebase Firestore.

Collections:
  user_profiles            — one document per user (keyed by normalized email)
  activity_preference_logs — implicit signals (clicked/booked/declined)
  recommendation_feedback  — explicit signals (liked/disliked/neutral + rating)
  inference_feature_logs   — feature vectors written at serve time for retraining
"""
from __future__ import annotations

import os
from typing import Any, Dict

import firebase_admin
from firebase_admin import credentials, firestore


_CRED_PATH = os.getenv(
    "FIREBASE_SERVICE_ACCOUNT_PATH",
    "config/serviceAccountKey.json",
)

if not firebase_admin._apps:
    cred = credentials.Certificate(_CRED_PATH)
    firebase_admin.initialize_app(cred)

_db = firestore.client()

# Collection names
_COL_PROFILES        = "user_profiles"
_COL_ACTIVITY_LOGS   = "activity_preference_logs"
_COL_FEEDBACK_LOGS   = "recommendation_feedback"
_COL_INFERENCE_LOGS  = "inference_feature_logs"

_STORE_DOC_ID = "store"

# Profile store

def load_user_profiles_store() -> Dict[str, Any]:
    """
    Load the profile store document from Firestore.

    Returns {"user_profiles": {normalized_email: profile_dict, ...}}.
    """
    doc = _db.collection(_COL_PROFILES).document(_STORE_DOC_ID).get()
    if doc.exists:
        data = doc.to_dict()
        data.setdefault("user_profiles", {})
        return data
    return {"user_profiles": {}}


def save_user_profiles_store(data: Dict[str, Any]) -> None:
    """Persist the in-memory profile store back to Firestore."""
    _db.collection(_COL_PROFILES).document(_STORE_DOC_ID).set(data, merge=True)


# Append-only log writers

def write_activity_preference_log(log_entry: Dict[str, Any]) -> None:
    """Write one activity preference log entry (implicit signal)."""
    log_id: str = log_entry["log_id"]
    _db.collection(_COL_ACTIVITY_LOGS).document(log_id).set(log_entry)


def write_recommendation_feedback_log(feedback_entry: Dict[str, Any]) -> None:
    """Write one recommendation feedback entry (explicit signal)."""
    feedback_id: str = feedback_entry["feedback_id"]
    _db.collection(_COL_FEEDBACK_LOGS).document(feedback_id).set(feedback_entry)


def write_inference_feature_log(log_entry: Dict[str, Any]) -> None:
    """
    Write one inference feature log entry (feature vector at serve time).

    Document ID is "{recommendation_id}_{venue_id}" so it can be looked
    up directly during retraining without a collection scan.
    """
    doc_id = f"{log_entry['recommendation_id']}_{log_entry['venue_id']}"
    _db.collection(_COL_INFERENCE_LOGS).document(doc_id).set(log_entry)
