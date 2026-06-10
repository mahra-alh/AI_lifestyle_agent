"""
Firestore storage adapter.

Matches the actual Firestore structure where each user is stored as their
own document directly in the user_profiles collection:

Collections:
  user_profiles            — one document per user (keyed by normalized email)
  activity_preference_logs — implicit signals (clicked/booked/declined)
  recommendation_feedback  — explicit signals (liked/disliked/neutral + rating)
  inference_feature_logs   — feature vectors written at serve time for retraining
  venues                   — venue metadata (unified_venue_pool), keyed by venue_id
  faiss_corpus             — FAISS text rows, keyed by venue_id; `index` field
                             aligns each doc with the corresponding venues doc
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd

# Collection names
_COL_PROFILES       = "user_profiles"
_COL_ACTIVITY_LOGS  = "activity_preference_logs"
_COL_FEEDBACK_LOGS  = "recommendation_feedback"
_COL_INFERENCE_LOGS = "inference_feature_logs"
_COL_VENUES         = "venues"
_COL_FAISS_CORPUS   = "faiss_corpus"


def _get_db():
    """Return the Firestore client, initializing Firebase on first call."""
    if not firebase_admin._apps:
        cred_path = os.getenv(
            "FIREBASE_SERVICE_ACCOUNT_PATH",
            "config/serviceAccountKey.json",
        )
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
    return firestore.client()


# ---------------------------------------------------------------------------
# Profile store — one document per user, keyed by normalized email
# ---------------------------------------------------------------------------

def load_user_profiles_store() -> Dict[str, Any]:
    """
    Load all user profiles from Firestore into the in-memory store format
    expected by profile_store.py:

        {"user_profiles": {normalized_email: profile_dict, ...}}

    Each document in the user_profiles collection is one user.
    """
    docs = _get_db().collection(_COL_PROFILES).stream()
    profiles = {}
    for doc in docs:
        profiles[doc.id] = doc.to_dict()
    return {"user_profiles": profiles}


def save_user_profiles_store(data: Dict[str, Any]) -> None:
    """
    Persist updated profiles back to Firestore.

    Writes each user profile as its own document keyed by normalized email.
    Uses merge=True so only changed fields are overwritten.
    """
    profiles = data.get("user_profiles", {})
    db = _get_db()
    for email, profile in profiles.items():
        db.collection(_COL_PROFILES).document(email).set(profile, merge=True)


def load_single_profile(user_id: str) -> Optional[Dict[str, Any]]:
    """
    Load one user profile directly by document ID (normalized email).

    More efficient than load_user_profiles_store() when you only need
    one user — avoids reading the entire collection.
    """
    doc = _get_db().collection(_COL_PROFILES).document(user_id).get()
    return doc.to_dict() if doc.exists else None


def save_single_profile(user_id: str, profile: Dict[str, Any]) -> None:
    """Save one user profile document, merging with existing fields."""
    _get_db().collection(_COL_PROFILES).document(user_id).set(profile, merge=True)


# ---------------------------------------------------------------------------
# Venue pool — loaded once, used by the recommendation service
# ---------------------------------------------------------------------------

def load_venue_pool() -> "pd.DataFrame":
    """
    Load all venue documents from the `venues` Firestore collection and
    return them as a pandas DataFrame sorted by the `index` field.

    The `index` field was written by import_venues_to_firestore.js and
    preserves the original row order of unified_venue_pool.csv so that
    positional alignment with faiss_corpus is maintained.
    """
    import pandas as pd

    docs = _get_db().collection(_COL_VENUES).stream()
    rows = [doc.to_dict() for doc in docs]
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Restore original CSV row order so any positional logic is preserved.
    if "index" in df.columns:
        df = df.sort_values("index").reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# FAISS corpus — loaded by build_faiss.py to (re)build the index
# ---------------------------------------------------------------------------

def load_faiss_corpus() -> "pd.DataFrame":
    """
    Load all documents from the `faiss_corpus` Firestore collection and
    return them as a pandas DataFrame sorted by the `index` field.

    The `index` field aligns each row with the corresponding row in the
    `venues` collection so FAISS vector positions match venue_ids.
    """
    import pandas as pd

    docs = _get_db().collection(_COL_FAISS_CORPUS).stream()
    rows = [doc.to_dict() for doc in docs]
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    if "index" in df.columns:
        df = df.sort_values("index").reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Append-only log writers
# ---------------------------------------------------------------------------

def write_activity_preference_log(log_entry: Dict[str, Any]) -> None:
    """Write one activity preference log entry (implicit signal)."""
    log_id: str = log_entry["log_id"]
    _get_db().collection(_COL_ACTIVITY_LOGS).document(log_id).set(log_entry)


def write_recommendation_feedback_log(feedback_entry: Dict[str, Any]) -> None:
    """Write one recommendation feedback entry (explicit signal)."""
    feedback_id: str = feedback_entry["feedback_id"]
    _get_db().collection(_COL_FEEDBACK_LOGS).document(feedback_id).set(feedback_entry)


def write_inference_feature_log(log_entry: Dict[str, Any]) -> None:
    """
    Write one inference feature log entry (feature vector at serve time).

    Document ID is "{recommendation_id}_{venue_id}" so it can be looked
    up directly during retraining without a collection scan.
    """
    doc_id = f"{log_entry['recommendation_id']}_{log_entry['venue_id']}"
    _get_db().collection(_COL_INFERENCE_LOGS).document(doc_id).set(log_entry)


# ---------------------------------------------------------------------------
# Pending bookings — two-step booking confirmation
#
# The agent only PREPARES a booking (status "pending"). The calendar write
# happens later, when the user explicitly confirms via the API. Records are
# keyed by pending_id.
# ---------------------------------------------------------------------------

_COL_PENDING_BOOKINGS = "pending_bookings"


def create_pending_booking(record: Dict[str, Any]) -> None:
    """Store a new pending booking record, keyed by its pending_id."""
    pending_id: str = record["pending_id"]
    _get_db().collection(_COL_PENDING_BOOKINGS).document(pending_id).set(record)


def get_pending_booking(pending_id: str) -> Optional[Dict[str, Any]]:
    """Return one pending booking record, or None if it doesn't exist."""
    doc = _get_db().collection(_COL_PENDING_BOOKINGS).document(pending_id).get()
    return doc.to_dict() if doc.exists else None


def update_pending_booking(pending_id: str, updates: Dict[str, Any]) -> None:
    """Apply partial updates (e.g. status changes) to a pending booking."""
    _get_db().collection(_COL_PENDING_BOOKINGS).document(pending_id).update(updates)


def get_latest_pending_booking_for_user(
    user_id: str,
    max_age_seconds: int = 300,
) -> Optional[Dict[str, Any]]:
    """
    Return the most recent still-pending booking for a user, or None.

    Used by the chat endpoint to attach a confirmation card to the response
    right after the agent prepares a booking. Only records created within
    max_age_seconds are considered so stale bookings never resurface.
    """
    from datetime import datetime, timedelta, timezone

    docs = (
        _get_db()
        .collection(_COL_PENDING_BOOKINGS)
        .where("user_id", "==", user_id)
        .where("status", "==", "pending")
        .stream()
    )
    records = [doc.to_dict() for doc in docs]
    if not records:
        return None

    cutoff = (
        datetime.now(timezone.utc) - timedelta(seconds=max_age_seconds)
    ).isoformat()
    fresh = [r for r in records if r.get("created_at", "") >= cutoff]
    if not fresh:
        return None

    return max(fresh, key=lambda r: r.get("created_at", ""))