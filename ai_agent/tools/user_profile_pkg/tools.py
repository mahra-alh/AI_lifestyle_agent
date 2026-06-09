"""
Agent-facing user profile tools.

Each @function_tool is a thin wrapper that:
1. Logs start/success/error via app_logger.
2. Delegates business logic to profile_store, validators, or ml_export.
3. Returns a structured dict the agent can act on.

No validation or ML logic lives here.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from agents import function_tool

from ai_agent.logger.app_logger import log_tool_error, log_tool_start, log_tool_success
from ai_agent.tools.user_profile_pkg.ml_export import build_ml_profile_export, resolve_area_coordinates
from ai_agent.tools.user_profile_pkg.profile_store import (
    TIMEZONE,
    find_missing_profile_fields,
    get_local_profile,
    get_or_create_local_profile,
    get_profile_intake_fields,
    load_profile,
    persist_activity_log,
    persist_feedback_log,
    save_profile_store,
    utc_now_iso,
    REQUIRED_PROFILE_FIELDS,
)
from ai_agent.tools.user_profile_pkg.validators import (
    VALID_PREFERENCE_SIGNALS,
    VALID_RECOMMENDATION_FEEDBACK,
    normalize_text,
    normalize_user_id,
    validate_coordinate,
    validate_dubai_area,
    validate_email_address,
    validate_int_range,
    validate_positive_number,
    validate_rating,
    validate_work_days,
)


# Internal helper — shared by get_user_profile and update_user_profile

def get_user_profile_data(user_id: str) -> Dict[str, Any]:
    """Read a profile from Firestore and annotate it with completeness metadata."""
    tool_name = "get_user_profile"
    user_id = normalize_user_id(user_id)
    start_time = log_tool_start(tool_name=tool_name, user_id=user_id, data={"operation": "read_profile"})

    try:
        profile = load_profile(user_id)

        if profile is None:
            missing_fields = list(REQUIRED_PROFILE_FIELDS.keys())
            log_tool_success(tool_name=tool_name, start_time=start_time, user_id=user_id, data={"profile_exists": False})
            return {
                "profile_exists": False,
                "profile_complete": False,
                "action_required": "collect_profile_details",
                "missing_fields": missing_fields,
                "required_fields": get_profile_intake_fields(missing_fields),
                "message": (
                    "No user profile found. Ask the user for the required "
                    "profile details, then call update_user_profile to save them."
                ),
            }

        profile = profile.copy()
        missing_fields = find_missing_profile_fields(profile)
        profile["profile_exists"] = True
        profile["profile_complete"] = len(missing_fields) == 0
        profile["missing_fields"] = missing_fields
        profile["ml_profile_export"] = build_ml_profile_export(profile)

        if missing_fields:
            profile["action_required"] = "collect_missing_profile_details"
            profile["required_fields"] = get_profile_intake_fields(missing_fields)

        log_tool_success(
            tool_name=tool_name, start_time=start_time, user_id=user_id,
            data={"profile_exists": True, "returned_fields": list(profile.keys())},
        )
        return profile

    except Exception as error:
        log_tool_error(tool_name=tool_name, start_time=start_time, error=error, user_id=user_id)
        raise


# Agent tools

@function_tool
def get_user_profile(user_id: str) -> Dict[str, Any]:
    """Get the user's saved lifestyle profile."""
    return get_user_profile_data(user_id)


@function_tool
def update_user_profile(
    user_id: str,
    # Identity
    email_address: Optional[str] = None,
    # Budget
    monthly_fun_budget_aed: Optional[int] = None,
    max_per_activity_aed: Optional[int] = None,
    # Location
    home_area: Optional[str] = None,
    work_area: Optional[str] = None,
    preferred_areas: Optional[List[str]] = None,
    max_travel_distance_km: Optional[float] = None,
    # Work schedule
    work_days: Optional[List[str]] = None,
    work_start_time: Optional[str] = None,
    work_end_time: Optional[str] = None,
    # Preferences
    hobbies: Optional[List[str]] = None,
    interests: Optional[List[str]] = None,
    preferred_activity_types: Optional[List[str]] = None,
    preferred_environment: Optional[str] = None,
    # ML profile signals
    budget_encoded: Optional[int] = None,
    travel_distance_encoded: Optional[int] = None,
    weather_pref_encoded: Optional[int] = None,
    pref_morning: Optional[int] = None,
    pref_midday: Optional[int] = None,
    pref_afternoon: Optional[int] = None,
    pref_evening: Optional[int] = None,
    pref_late_night: Optional[int] = None,
    diet_halal: Optional[int] = None,
    diet_vegetarian: Optional[int] = None,
    diet_vegan: Optional[int] = None,
    diet_gluten_free: Optional[int] = None,
    social_friends: Optional[int] = None,
    social_family: Optional[int] = None,
    social_partner: Optional[int] = None,
    social_alone: Optional[int] = None,
    factor_cost: Optional[int] = None,
    factor_distance: Optional[int] = None,
    factor_quality: Optional[int] = None,
    factor_comfort: Optional[int] = None,
    adventure_level_encoded: Optional[int] = None,
    currently_saving_money: Optional[int] = None,
    going_out_frequency_encoded: Optional[int] = None,
    activity_duration_encoded: Optional[int] = None,
    excl_nightlife: Optional[int] = None,
    excl_cultural: Optional[int] = None,
    excl_water: Optional[int] = None,
    excl_outdoor_travel: Optional[int] = None,
    user_latitude: Optional[float] = None,
    user_longitude: Optional[float] = None,
    notes: Optional[str] = None,
) -> Dict[str, Any]:
    """Update the user's lifestyle profile fields."""
    tool_name = "update_user_profile"
    user_id = normalize_user_id(user_id)
    start_time = log_tool_start(tool_name=tool_name, user_id=user_id, data={"operation": "update_profile"})

    try:
        updates: Dict[str, Any] = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "timezone": TIMEZONE,
        }

        if email_address is not None:
            updates["email_address"] = validate_email_address(email_address)
        if monthly_fun_budget_aed is not None:
            if monthly_fun_budget_aed < 0:
                raise ValueError("Monthly budget cannot be negative.")
            updates["monthly_fun_budget_aed"] = monthly_fun_budget_aed
        if max_per_activity_aed is not None:
            if max_per_activity_aed < 0:
                raise ValueError("Max per activity budget cannot be negative.")
            updates["max_per_activity_aed"] = max_per_activity_aed
        if home_area is not None:
            updates["home_area"] = validate_dubai_area(home_area)
        if work_area is not None:
            updates["work_area"] = validate_dubai_area(work_area)
        if preferred_areas is not None:
            updates["preferred_areas"] = [validate_dubai_area(a) for a in preferred_areas]
        if max_travel_distance_km is not None:
            if max_travel_distance_km < 0:
                raise ValueError("Max travel distance cannot be negative.")
            updates["max_travel_distance_km"] = max_travel_distance_km
        if work_days is not None:
            updates["work_days"] = validate_work_days(work_days)
        if work_start_time is not None:
            updates["work_start_time"] = work_start_time
        if work_end_time is not None:
            updates["work_end_time"] = work_end_time
        if hobbies is not None:
            updates["hobbies"] = [h.strip().lower() for h in hobbies]
        if interests is not None:
            updates["interests"] = [i.strip().lower() for i in interests]
        if preferred_activity_types is not None:
            updates["preferred_activity_types"] = [a.strip().lower() for a in preferred_activity_types]
        if preferred_environment is not None:
            updates["preferred_environment"] = preferred_environment.strip().lower()
        if budget_encoded is not None:
            updates["budget_encoded"] = validate_int_range(budget_encoded, "budget_encoded", 0, 2)
        if travel_distance_encoded is not None:
            updates["travel_distance_encoded"] = validate_int_range(travel_distance_encoded, "travel_distance_encoded", 0, 4)
        if weather_pref_encoded is not None:
            updates["weather_pref_encoded"] = validate_int_range(weather_pref_encoded, "weather_pref_encoded", 0, 3)

        # Binary flags (0/1)
        for field_name, value in [
            ("pref_morning", pref_morning), ("pref_midday", pref_midday),
            ("pref_afternoon", pref_afternoon), ("pref_evening", pref_evening),
            ("pref_late_night", pref_late_night), ("diet_halal", diet_halal),
            ("diet_vegetarian", diet_vegetarian), ("diet_vegan", diet_vegan),
            ("diet_gluten_free", diet_gluten_free), ("social_friends", social_friends),
            ("social_family", social_family), ("social_partner", social_partner),
            ("social_alone", social_alone), ("factor_cost", factor_cost),
            ("factor_distance", factor_distance), ("factor_quality", factor_quality),
            ("factor_comfort", factor_comfort), ("currently_saving_money", currently_saving_money),
            ("excl_nightlife", excl_nightlife), ("excl_cultural", excl_cultural),
            ("excl_water", excl_water), ("excl_outdoor_travel", excl_outdoor_travel),
        ]:
            if value is not None:
                updates[field_name] = validate_int_range(value, field_name, 0, 1)

        if adventure_level_encoded is not None:
            updates["adventure_level_encoded"] = validate_int_range(adventure_level_encoded, "adventure_level_encoded", 0, 3)
        if going_out_frequency_encoded is not None:
            updates["going_out_frequency_encoded"] = validate_int_range(going_out_frequency_encoded, "going_out_frequency_encoded", 0, 4)
        if activity_duration_encoded is not None:
            updates["activity_duration_encoded"] = validate_int_range(activity_duration_encoded, "activity_duration_encoded", 0, 3)
        if user_latitude is not None:
            updates["user_latitude"] = validate_coordinate(user_latitude, "user_latitude", -90.0, 90.0)
        if user_longitude is not None:
            updates["user_longitude"] = validate_coordinate(user_longitude, "user_longitude", -180.0, 180.0)
        if notes is not None:
            updates["notes"] = notes.strip()

        saved = get_or_create_profile(user_id)
        saved.update(updates)
        save_profile(user_id, saved)
        missing_fields = find_missing_profile_fields(saved)

        log_tool_success(
            tool_name=tool_name, start_time=start_time, user_id=user_id,
            data={"updated_field_names": list(updates.keys())},
        )
        return {
            "success": True,
            "profile_exists": True,
            "profile_complete": len(missing_fields) == 0,
            "message": "User profile updated successfully.",
            "updated_fields": updates,
            "missing_fields": missing_fields,
            "required_fields": get_profile_intake_fields(missing_fields),
            "ml_profile_export": build_ml_profile_export(saved),
            "next_action": (
                "continue_activity_planning"
                if len(missing_fields) == 0
                else "collect_missing_profile_details"
            ),
        }

    except Exception as error:
        log_tool_error(
            tool_name=tool_name, start_time=start_time, error=error, user_id=user_id,
            data={"operation": "update_profile"},
        )
        raise


@function_tool
def log_activity_preference(
    user_id: str,
    activity_id: str,
    activity_name: str,
    performance_signal: str,
    activity_category: Optional[str] = None,
    activity_area: Optional[str] = None,
    activity_source: Optional[str] = None,
    activity_price_aed: Optional[float] = None,
    activity_latitude: Optional[float] = None,
    activity_longitude: Optional[float] = None,
    user_query: Optional[str] = None,
    session_id: Optional[str] = None,
    recommendation_id: Optional[str] = None,
    model_version: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Log a user's behavior toward an activity.

    Allowed signals: clicked, booked, declined.
    Builds training data for the ML recommender.
    """
    tool_name = "log_activity_preference"
    user_id = normalize_user_id(user_id)
    start_time = log_tool_start(
        tool_name=tool_name, user_id=user_id,
        data={
            "operation": "write_activity_preference_log",
            "activity_id": activity_id,
            "performance_signal": performance_signal,
            "session_id": session_id,
            "recommendation_id": recommendation_id,
        },
    )

    try:
        cleaned_signal = normalize_text(performance_signal)
        if cleaned_signal not in VALID_PREFERENCE_SIGNALS:
            raise ValueError(
                f"Invalid performance_signal: {performance_signal}. "
                f"Use one of: {sorted(VALID_PREFERENCE_SIGNALS)}"
            )
        validate_positive_number(activity_price_aed, "activity_price_aed")

        created_at = utc_now_iso()
        log_id = uuid.uuid4().hex

        log_entry: Dict[str, Any] = {
            "log_id": log_id,
            "user_id": user_id,
            "activity_id": activity_id.strip(),
            "activity_name": activity_name.strip(),
            "performance_signal": cleaned_signal,
            "created_at": created_at,
            "timezone": TIMEZONE,
        }
        if activity_category is not None:
            log_entry["activity_category"] = normalize_text(activity_category)
        if activity_area is not None:
            log_entry["activity_area"] = validate_dubai_area(activity_area)
        if activity_source is not None:
            log_entry["activity_source"] = normalize_text(activity_source)
        if activity_price_aed is not None:
            log_entry["activity_price_aed"] = activity_price_aed
        if activity_latitude is not None:
            log_entry["activity_latitude"] = activity_latitude
        if activity_longitude is not None:
            log_entry["activity_longitude"] = activity_longitude
        if user_query is not None:
            log_entry["user_query"] = user_query.strip()
        if session_id is not None:
            log_entry["session_id"] = session_id.strip()
        if recommendation_id is not None:
            log_entry["recommendation_id"] = recommendation_id.strip()
        if model_version is not None:
            log_entry["model_version"] = model_version.strip()

        profile = get_or_create_profile(user_id)
        profile["activity_preferences_log_count"] = int(profile.get("activity_preferences_log_count", 0)) + 1
        profile["last_activity_preference_log_at"] = created_at
        profile["updated_at"] = created_at
        save_profile(user_id, profile)
        persist_activity_log(log_entry)

        log_tool_success(
            tool_name=tool_name, start_time=start_time, user_id=user_id,
            data={
                "operation": "write_activity_preference_log",
                "activity_id": activity_id,
                "performance_signal": cleaned_signal,
                "log_id": log_id,
            },
        )
        return {
            "success": True,
            "message": "Activity preference logged successfully.",
            "log_id": log_id,
            "log_entry": log_entry,
        }

    except Exception as error:
        log_tool_error(
            tool_name=tool_name, start_time=start_time, error=error, user_id=user_id,
            data={"operation": "write_activity_preference_log", "activity_id": activity_id},
        )
        raise


@function_tool
def log_recommendation_feedback(
    user_id: str,
    recommendation_id: str,
    activity_id: str,
    activity_name: str,
    feedback: str,
    rating: Optional[int] = None,
    feedback_reason: Optional[str] = None,
    recommendation_rank: Optional[int] = None,
    activity_category: Optional[str] = None,
    activity_area: Optional[str] = None,
    model_version: Optional[str] = None,
    session_id: Optional[str] = None,
    user_query: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Log explicit feedback on a recommendation.

    Stores liked/disliked/neutral feedback. Helps evaluate recommendation
    quality and can become labeled data for LightGBM training.
    """
    tool_name = "log_recommendation_feedback"
    user_id = normalize_user_id(user_id)
    start_time = log_tool_start(
        tool_name=tool_name, user_id=user_id,
        data={
            "operation": "write_recommendation_feedback",
            "recommendation_id": recommendation_id,
            "activity_id": activity_id,
            "feedback": feedback,
        },
    )

    try:
        cleaned_feedback = normalize_text(feedback)
        if cleaned_feedback not in VALID_RECOMMENDATION_FEEDBACK:
            raise ValueError(
                f"Invalid feedback: {feedback}. "
                f"Use one of: {sorted(VALID_RECOMMENDATION_FEEDBACK)}"
            )
        validate_rating(rating)
        validate_positive_number(recommendation_rank, "recommendation_rank")

        created_at = utc_now_iso()
        feedback_id = uuid.uuid4().hex

        feedback_entry: Dict[str, Any] = {
            "feedback_id": feedback_id,
            "user_id": user_id,
            "recommendation_id": recommendation_id.strip(),
            "activity_id": activity_id.strip(),
            "activity_name": activity_name.strip(),
            "feedback": cleaned_feedback,
            "created_at": created_at,
            "timezone": TIMEZONE,
        }
        if rating is not None:
            feedback_entry["rating"] = rating
        if feedback_reason is not None:
            feedback_entry["feedback_reason"] = feedback_reason.strip()
        if recommendation_rank is not None:
            feedback_entry["recommendation_rank"] = recommendation_rank
        if activity_category is not None:
            feedback_entry["activity_category"] = normalize_text(activity_category)
        if activity_area is not None:
            feedback_entry["activity_area"] = validate_dubai_area(activity_area)
        if model_version is not None:
            feedback_entry["model_version"] = model_version.strip()
        if session_id is not None:
            feedback_entry["session_id"] = session_id.strip()
        if user_query is not None:
            feedback_entry["user_query"] = user_query.strip()

        profile = get_or_create_profile(user_id)
        profile["recommendation_feedback_count"] = int(profile.get("recommendation_feedback_count", 0)) + 1
        profile["last_recommendation_feedback_at"] = created_at
        profile["updated_at"] = created_at
        save_profile(user_id, profile)
        persist_feedback_log(feedback_entry)

        log_tool_success(
            tool_name=tool_name, start_time=start_time, user_id=user_id,
            data={
                "operation": "write_recommendation_feedback",
                "recommendation_id": recommendation_id,
                "feedback": cleaned_feedback,
                "feedback_id": feedback_id,
            },
        )
        return {
            "success": True,
            "message": "Recommendation feedback logged successfully.",
            "feedback_id": feedback_id,
            "feedback_entry": feedback_entry,
        }

    except Exception as error:
        log_tool_error(
            tool_name=tool_name, start_time=start_time, error=error, user_id=user_id,
            data={"operation": "write_recommendation_feedback", "recommendation_id": recommendation_id},
        )
        raise