"""
Profile store layer.

Wraps the Firestore storage calls with lightweight in-memory helpers
so the agent tools stay thin. No @function_tool decorators here.

Firestore structure (one document per user, keyed by normalized email):
    user_profiles/
        taherkaasamani@gmail.com   ← document
        daniagz02@gmail.com        ← document
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ai_agent.storage.firestore_store import (
    load_single_profile,
    load_user_profiles_store,
    save_single_profile,
    save_user_profiles_store,
    write_activity_preference_log,
    write_recommendation_feedback_log,
)
from ai_agent.tools.user_profile_pkg.validators import normalize_user_id

TIMEZONE = "Asia/Dubai"

# ---------------------------------------------------------------------------
# Required fields for a complete profile
# ---------------------------------------------------------------------------

REQUIRED_PROFILE_FIELDS: Dict[str, List[str]] = {
    "home_area":                ["home_area", "dubai_area", "area", "location"],
    "monthly_fun_budget_aed":   ["monthly_fun_budget", "monthly budget", "activity budget"],
    "max_per_activity_aed":     ["maximum budget per activity", "single activity budget"],
    "hobbies":                  ["hobbies", "interests"],
    "preferred_activity_types": ["activity types", "type of activities"],
    "work_days":                ["work days", "working days"],
    "work_start_time":          ["work start", "workday starts"],
    "work_end_time":            ["work end", "workday ends"],
}

PROFILE_INTAKE_FIELDS: List[Dict[str, Any]] = [
    {"field": "home_area",                "ask": "Which Dubai area are you based in?",               "example": "Dubai Marina"},
    {"field": "monthly_fun_budget_aed",   "ask": "What is your monthly activity budget in AED?",     "example": 800},
    {"field": "max_per_activity_aed",     "ask": "What is your maximum budget per activity in AED?", "example": 150},
    {"field": "hobbies",                  "ask": "What are your interests or hobbies?",              "example": ["restaurants", "beach walks", "coffee shops"]},
    {"field": "preferred_activity_types", "ask": "What types of activities do you prefer?",          "example": ["outdoor", "food", "wellness", "social"]},
    {"field": "work_days",                "ask": "Which days do you usually work?",                  "example": ["monday to friday"]},
    {"field": "work_start_time",          "ask": "What time does your workday usually start?",       "example": "09:00"},
    {"field": "work_end_time",            "ask": "What time does your workday usually end?",         "example": "18:00"},
]


# ---------------------------------------------------------------------------
# Timestamp utility
# ---------------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Efficient single-user reads (preferred over loading the full collection)
# ---------------------------------------------------------------------------

def load_profile(user_id: str) -> Optional[Dict[str, Any]]:
    """
    Load one user's profile directly from Firestore by document ID.

    Reads only the single document for this user — much faster than
    loading the whole collection when you only need one profile.
    """
    return load_single_profile(normalize_user_id(user_id))


def save_profile(user_id: str, profile: Dict[str, Any]) -> None:
    """Save one user's profile document directly to Firestore."""
    save_single_profile(normalize_user_id(user_id), profile)


# ---------------------------------------------------------------------------
# Full store helpers (used when iterating over all profiles)
# ---------------------------------------------------------------------------

def load_profile_store() -> Dict[str, Any]:
    """Load all user profiles from Firestore into the in-memory store format."""
    return load_user_profiles_store()


def save_profile_store(data: Dict[str, Any]) -> None:
    """Persist all updated profiles back to Firestore."""
    save_user_profiles_store(data)


# Backward-compatible aliases
load_local_profile_store = load_profile_store
save_local_profile_store = save_profile_store


# ---------------------------------------------------------------------------
# Profile access helpers
# ---------------------------------------------------------------------------

def get_local_profile(data: Dict[str, Any], user_id: str) -> Optional[Dict[str, Any]]:
    """Return a user profile from an already-loaded store dict."""
    return data["user_profiles"].get(normalize_user_id(user_id))


def get_or_create_profile(user_id: str) -> Dict[str, Any]:
    """
    Load a user's profile directly from Firestore.
    Creates and saves a shell profile if the user doesn't exist yet.
    """
    normalized = normalize_user_id(user_id)
    profile = load_single_profile(normalized)

    if profile is None:
        profile = {
            "user_id": normalized,
            "created_at": utc_now_iso(),
            "timezone": TIMEZONE,
            "activity_preferences_log_count": 0,
            "recommendation_feedback_count": 0,
        }
        save_single_profile(normalized, profile)

    return profile


def get_or_create_local_profile(data: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    """
    Return a profile from an already-loaded store dict, creating a shell
    entry in the dict if needed. Used by tools that operate on the full store.
    """
    profiles = data["user_profiles"]
    normalized = normalize_user_id(user_id)

    if normalized not in profiles:
        profiles[normalized] = {
            "user_id": normalized,
            "created_at": utc_now_iso(),
            "timezone": TIMEZONE,
            "activity_preferences_log_count": 0,
            "recommendation_feedback_count": 0,
        }

    return profiles[normalized]


# ---------------------------------------------------------------------------
# Completeness helpers
# ---------------------------------------------------------------------------

def find_missing_profile_fields(profile: Dict[str, Any]) -> List[str]:
    """Return required profile fields that are missing or empty."""
    return [
        field for field in REQUIRED_PROFILE_FIELDS
        if field not in profile or profile[field] in [None, "", []]
    ]


def get_profile_intake_fields(
    missing_fields: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Return field metadata the agent uses to ask the user for profile data."""
    if missing_fields is None:
        return PROFILE_INTAKE_FIELDS
    missing_set = set(missing_fields)
    return [f for f in PROFILE_INTAKE_FIELDS if f["field"] in missing_set]


# ---------------------------------------------------------------------------
# Firestore log writers (thin pass-throughs)
# ---------------------------------------------------------------------------

def persist_activity_log(log_entry: Dict[str, Any]) -> None:
    write_activity_preference_log(log_entry)


def persist_feedback_log(feedback_entry: Dict[str, Any]) -> None:
    write_recommendation_feedback_log(feedback_entry)