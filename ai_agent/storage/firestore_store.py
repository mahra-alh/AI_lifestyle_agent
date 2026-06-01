from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import firebase_admin
from firebase_admin import credentials, firestore

USER_PROFILE_COLLECTION = os.getenv("FIRESTORE_USER_PROFILE_COLLECTION", "user_profiles")
ACTIVITY_LOG_COLLECTION = os.getenv(
    "FIRESTORE_ACTIVITY_LOG_COLLECTION",
    "activity_preferences_log",
)
RECOMMENDATION_FEEDBACK_COLLECTION = os.getenv(
    "FIRESTORE_FEEDBACK_COLLECTION",
    "recommendation_feedback_log",
)

_FIRESTORE_CLIENT = None


def _initialize_firebase_app() -> None:
    """
    Initialize Firebase Admin once for the current process.

    Local development can point at a downloaded service account JSON.
    Cloud Run can rely on the default service account if no file path exists.
    """

    if firebase_admin._apps:
        return

    service_account_path = os.getenv(
        "FIREBASE_SERVICE_ACCOUNT_PATH",
        "backend/config/serviceAccountKey.json",
    )
    service_account_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
    project_id = os.getenv("FIREBASE_PROJECT_ID")

    options: Dict[str, Any] = {}
    if project_id:
        options["projectId"] = project_id

    if service_account_json:
        try:
            service_account_info = json.loads(service_account_json)
        except json.JSONDecodeError as error:
            raise ValueError("FIREBASE_SERVICE_ACCOUNT_JSON must contain valid JSON.") from error

        if not isinstance(service_account_info, dict):
            raise ValueError("FIREBASE_SERVICE_ACCOUNT_JSON must contain a JSON object.")

        cred = credentials.Certificate(service_account_info)
        firebase_admin.initialize_app(cred, options=options or None)
        return

    if service_account_path and Path(service_account_path).exists():
        cred = credentials.Certificate(service_account_path)
        firebase_admin.initialize_app(cred, options=options or None)
        return

    firebase_admin.initialize_app(options=options or None)


def get_firestore_client():
    """
    Return a cached Firestore client.
    """

    global _FIRESTORE_CLIENT

    if _FIRESTORE_CLIENT is not None:
        return _FIRESTORE_CLIENT

    _initialize_firebase_app()
    _FIRESTORE_CLIENT = firestore.client()
    return _FIRESTORE_CLIENT


def load_user_profiles_store() -> Dict[str, Any]:
    """
    Load all user profile documents into the in-memory store shape that the
    rest of the agent already expects.
    """

    client = get_firestore_client()
    user_profiles: Dict[str, Dict[str, Any]] = {}

    for snapshot in client.collection(USER_PROFILE_COLLECTION).stream():
        profile_data = snapshot.to_dict() or {}
        profile_data.setdefault("user_id", snapshot.id)
        user_profiles[snapshot.id] = profile_data

    return {"user_profiles": user_profiles}


def save_user_profiles_store(data: Dict[str, Any]) -> None:
    """
    Persist the in-memory profile store into Firestore documents.

    Activity preference and recommendation feedback logs are stored in their
    own Firestore collections.
    """

    client = get_firestore_client()
    user_profiles = data.get("user_profiles", {})

    if not isinstance(user_profiles, dict):
        raise ValueError("data['user_profiles'] must be a dictionary.")

    for user_id, profile in user_profiles.items():
        if not isinstance(profile, dict):
            continue

        doc_id = str(user_id).strip().lower()
        client.collection(USER_PROFILE_COLLECTION).document(doc_id).set(profile, merge=True)


def write_activity_preference_log(log_entry: Dict[str, Any]) -> None:
    """
    Store one activity preference log entry in Firestore.
    """

    if not isinstance(log_entry, dict):
        raise ValueError("log_entry must be a dictionary.")

    log_id = str(log_entry.get("log_id") or "").strip().lower()
    if not log_id:
        raise ValueError("log_entry['log_id'] is required.")

    client = get_firestore_client()
    client.collection(ACTIVITY_LOG_COLLECTION).document(log_id).set(log_entry, merge=False)


def write_recommendation_feedback_log(feedback_entry: Dict[str, Any]) -> None:
    """
    Store one recommendation feedback entry in Firestore.
    """

    if not isinstance(feedback_entry, dict):
        raise ValueError("feedback_entry must be a dictionary.")

    feedback_id = str(feedback_entry.get("feedback_id") or "").strip().lower()
    if not feedback_id:
        raise ValueError("feedback_entry['feedback_id'] is required.")

    client = get_firestore_client()
    client.collection(RECOMMENDATION_FEEDBACK_COLLECTION).document(feedback_id).set(
        feedback_entry,
        merge=False,
    )
