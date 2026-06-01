from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
import re
import uuid

from agents import function_tool

from ai_agent.data.dubai_areas import DUBAI_AREAS, get_area_by_user_input
from ai_agent.logger.app_logger import log_tool_start, log_tool_success, log_tool_error
from ai_agent.storage.firestore_store import (
    load_user_profiles_store,
    save_user_profiles_store,
    write_activity_preference_log,
    write_recommendation_feedback_log,
)

# Keep profile storage and validation constants in one place.
TIMEZONE = "Asia/Dubai"

VALID_WORK_DAYS = {
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
}
# These signals can be used to track the behavior of the user towards the activites recommended by the agent. 
# They can be used as implicit preference signals for the ML model and to evaluate recommendation performance over time.
VALID_PREFERENCE_SIGNALS = {
    "clicked",
    "booked",
    "declined",
}
# In the frontend the user should be asked a day after the activity is booked whether they liked it, disliked it, or felt neutral about it. 
# This explicit feedback can be used to evaluate and improve the recommendation model over time.
VALID_RECOMMENDATION_FEEDBACK = {
    "liked",
    "disliked",
    "neutral",
}

REQUIRED_PROFILE_FIELDS = [
    "email_address",
    "home_area",
    "monthly_fun_budget_aed",
    "max_per_activity_aed",
    "work_days",
    "work_start_time",
    "work_end_time",
    "hobbies",
    "preferred_activity_types",
]
# Tells the agent which fields to ask for when creating a new profile or filling in missing details for an existing profile.
# The agent should use the "ask" value to prompt the user and the "example" value to understand the expected format of the answer.
PROFILE_INTAKE_FIELDS = [
    {
        "field": "email_address",
        "ask": "What email should I use for your profile?",
        "example": "name@example.com",
    },
    {
        "field": "home_area",
        "ask": "Which Dubai area are you based in?",
        "example": "Dubai Marina",
    },
    {
        "field": "monthly_fun_budget_aed",
        "ask": "What is your monthly activity budget in AED?",
        "example": 800,
    },
    {
        "field": "max_per_activity_aed",
        "ask": "What is your maximum budget per activity in AED?",
        "example": 150,
    },
    {
        "field": "hobbies",
        "ask": "What are your interests or hobbies?",
        "example": ["restaurants", "beach walks", "coffee shops"],
    },
    {
        "field": "preferred_activity_types",
        "ask": "What types of activities do you prefer?",
        "example": ["outdoor", "food", "wellness", "social"],
    },
    {
        "field": "work_days",
        "ask": "Which days do you usually work?",
        "example": ["monday", "tuesday", "wednesday", "thursday", "friday"],
    },
    {
        "field": "work_start_time",
        "ask": "What time does your workday usually start?",
        "example": "09:00",
    },
    {
        "field": "work_end_time",
        "ask": "What time does your workday usually end?",
        "example": "18:00",
    },
]

# Shared helper functions for timestamps, cleanup, validation, and Firestore-backed storage.
def utc_now_iso() -> str:

    # Return the current UTC timestamp as an ISO string.
    # This is useful for saving consistent timestamps in Firestore.
    return datetime.now(timezone.utc).isoformat()

def normalize_text(value: Optional[str]) -> Optional[str]:
    
    # Clean a text value by removing extra spaces and converting to lowercase.
    if value is None:
        return None
    cleaned = value.strip().lower()
    return cleaned if cleaned else None

def normalize_text_list(values: Optional[List[str]]) -> Optional[List[str]]:
    
    # Clean a list of text values and remove empty items.
    if values is None:
        return None

    cleaned_values = []

    for value in values:
        cleaned = normalize_text(value)
        if cleaned is not None:
            cleaned_values.append(cleaned)

    return cleaned_values

def normalize_user_id(user_id: str) -> str:
    """
    Normalize the profile key so Firestore uses one document per email.
    """

    return user_id.strip().lower()

def validate_email_address(email_address: Optional[str]) -> Optional[str]:

    # Validate and normalize the user's email address.
    # This checks the basic email format only.
    # It does not verify whether the email inbox actually exists.
    if email_address is None:
        return None

    cleaned_email = email_address.strip().lower()

    email_pattern = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"

    if not re.match(email_pattern, cleaned_email):
        raise ValueError(
            f"'{email_address}' is not a valid email address format."
        )

    return cleaned_email

def validate_dubai_area(area: Optional[str]) -> Optional[str]:

    # Validate that the area is part of the accepted Dubai areas list.
    # Returns the canonical area ID stored in DUBAI_AREAS.
    if area is None:
        return None

    cleaned_area = normalize_text(area)

    # Accept canonical area IDs as well as user-facing area names and aliases.
    if cleaned_area in DUBAI_AREAS:
        return cleaned_area

    matched_area = get_area_by_user_input(area)

    if matched_area is None:
        raise ValueError(
            f"'{area}' is not in the supported Dubai areas list. "
            "Use a known Dubai area such as Dubai Marina, Business Bay, JVC, "
            "Downtown Dubai, Deira, etc."
        )

    return matched_area["area_id"]

def validate_work_days(work_days: Optional[List[str]]) -> Optional[List[str]]:

    # Validate and normalize work days.
    cleaned_days = normalize_text_list(work_days)

    if cleaned_days is None:
        return None

    invalid_days = [day for day in cleaned_days if day not in VALID_WORK_DAYS]

    if invalid_days:
        raise ValueError(
            f"Invalid work day(s): {invalid_days}. "
            f"Use one of: {sorted(VALID_WORK_DAYS)}"
        )

    return cleaned_days

def validate_positive_number(
    value: Optional[float],
    field_name: str
) -> Optional[float]:

    # Validate that a numeric value is not negative.
    if value is None:
        return None

    try:
        normalized = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be a number.")

    if normalized < 0:
        raise ValueError(f"{field_name} cannot be negative.")

    return normalized

def validate_rating(rating: Optional[int]) -> Optional[int]:

    # Validate a recommendation rating from 1 to 5.
    if rating is None:
        return None

    if rating < 1 or rating > 5:
        raise ValueError("rating must be between 1 and 5.")

    return rating

def validate_int_range(
    value: Optional[int | bool],
    field_name: str,
    minimum: int,
    maximum: int,
) -> Optional[int]:
    """
    Validate and normalize a small integer flag or encoded value.
    """

    if value is None:
        return None

    try:
        normalized = int(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{field_name} must be an integer between {minimum} and {maximum}."
        )

    if normalized < minimum or normalized > maximum:
        raise ValueError(
            f"{field_name} must be between {minimum} and {maximum}."
        )

    return normalized

def validate_coordinate(
    value: Optional[float],
    field_name: str,
    minimum: float,
    maximum: float,
) -> Optional[float]:
    """
    Validate latitude/longitude style coordinates.
    """

    if value is None:
        return None

    try:
        normalized = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be a number.")

    if normalized < minimum or normalized > maximum:
        raise ValueError(
            f"{field_name} must be between {minimum} and {maximum}."
        )

    return normalized

def _resolve_area_coordinates(area_value: Optional[str]) -> Optional[Dict[str, float]]:
    """
    Convert a Dubai area into approximate coordinates if possible.
    """

    if area_value is None:
        return None

    cleaned_area = normalize_text(area_value)

    if cleaned_area in DUBAI_AREAS:
        area = DUBAI_AREAS[cleaned_area]
        return {
            "latitude": float(area["latitude"]),
            "longitude": float(area["longitude"]),
        }

    matched_area = get_area_by_user_input(area_value)
    if matched_area is None:
        return None

    return {
        "latitude": float(matched_area["latitude"]),
        "longitude": float(matched_area["longitude"]),
    }

def build_ml_profile_export(profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a normalized profile snapshot for the ML layer.

    This keeps feature_builder.py from having to guess at basic profile fields
    when the agent already knows them.
    """

    export = profile.copy()
    export["user_email"] = (
        export.get("user_email")
        or export.get("email_address")
        or export.get("user_id")
        or "unknown@example.com"
    )
    export["email_address"] = export.get("email_address") or export["user_email"]

    def coerce_text_list(field_name: str) -> List[str]:
        value = export.get(field_name)

        if value is None:
            return []

        if isinstance(value, list):
            return [
                str(item).strip().lower()
                for item in value
                if str(item).strip()
            ]

        if isinstance(value, tuple):
            return [
                str(item).strip().lower()
                for item in value
                if str(item).strip()
            ]

        cleaned = str(value).strip().lower()
        return [cleaned] if cleaned else []

    def coerce_flag(field_name: str, default: int = 0) -> int:
        value = export.get(field_name)
        if value is None:
            return default
        try:
            normalized = int(value)
        except (TypeError, ValueError):
            return default
        return 1 if normalized else 0

    def coerce_encoded(field_name: str, minimum: int, maximum: int, default: int) -> int:
        value = export.get(field_name)
        if value is None:
            return default
        try:
            normalized = int(value)
        except (TypeError, ValueError):
            return default
        if normalized < minimum or normalized > maximum:
            return default
        return normalized

    budget_level_map = {
        "free": 0,
        "low": 0,
        "medium": 1,
        "high": 2,
        "unknown": 1,
    }

    if "budget_encoded" not in export or export.get("budget_encoded") is None:
        budget_level = str(export.get("budget_level", "unknown")).lower().strip()
        export["budget_encoded"] = budget_level_map.get(budget_level, 1)
    else:
        export["budget_encoded"] = coerce_encoded("budget_encoded", 0, 2, 1)

    export["budget_level"] = str(export.get("budget_level", "unknown")).lower().strip() or "unknown"

    if "travel_distance_encoded" not in export or export.get("travel_distance_encoded") is None:
        max_distance = export.get("max_travel_distance_km")
        try:
            max_distance = float(max_distance)
        except (TypeError, ValueError):
            max_distance = None

        if max_distance is None:
            export["travel_distance_encoded"] = 2
        elif max_distance <= 3:
            export["travel_distance_encoded"] = 0
        elif max_distance <= 7:
            export["travel_distance_encoded"] = 1
        elif max_distance <= 15:
            export["travel_distance_encoded"] = 2
        elif max_distance <= 25:
            export["travel_distance_encoded"] = 3
        else:
            export["travel_distance_encoded"] = 4
    else:
        export["travel_distance_encoded"] = coerce_encoded("travel_distance_encoded", 0, 4, 2)

    if "weather_pref_encoded" not in export or export.get("weather_pref_encoded") is None:
        preferred_environment = str(export.get("preferred_environment", "")).lower().strip()
        if preferred_environment in {"indoor", "indoor only"}:
            export["weather_pref_encoded"] = 0
        elif preferred_environment in {"outdoor", "outdoor only"}:
            export["weather_pref_encoded"] = 1
        elif preferred_environment in {"mixed", "both", "balanced"}:
            export["weather_pref_encoded"] = 2
        else:
            export["weather_pref_encoded"] = 3
    else:
        export["weather_pref_encoded"] = coerce_encoded("weather_pref_encoded", 0, 3, 3)

    for field in [
        "pref_morning",
        "pref_midday",
        "pref_afternoon",
        "pref_evening",
        "pref_late_night",
        "diet_halal",
        "diet_vegetarian",
        "diet_vegan",
        "diet_gluten_free",
        "social_friends",
        "social_family",
        "social_partner",
        "social_alone",
        "factor_cost",
        "factor_distance",
        "factor_quality",
        "factor_comfort",
        "currently_saving_money",
        "excl_nightlife",
        "excl_cultural",
        "excl_water",
        "excl_outdoor_travel",
    ]:
        export[field] = coerce_flag(field, 0)

    export["preferred_areas"] = coerce_text_list("preferred_areas")
    export["activity_preferences"] = coerce_text_list("activity_preferences")
    export["disliked_activities"] = coerce_text_list("disliked_activities")
    export["hobbies"] = coerce_text_list("hobbies")
    export["interests"] = coerce_text_list("interests")
    export["preferred_activity_types"] = coerce_text_list("preferred_activity_types")
    export["work_days"] = coerce_text_list("work_days")

    export["transport_mode"] = str(export.get("transport_mode", "unknown")).lower().strip() or "unknown"
    export["adventure_level_encoded"] = coerce_encoded(
        "adventure_level_encoded", 0, 3, 1
    )
    if "going_out_frequency_encoded" not in export or export.get("going_out_frequency_encoded") is None:
        activity_log_count = export.get("activity_preferences_log_count")
        if activity_log_count is None:
            activity_log_count = len(export.get("activity_preferences_log", []))

        try:
            activity_log_count = int(activity_log_count)
        except (TypeError, ValueError):
            activity_log_count = 0

        if activity_log_count >= 20:
            export["going_out_frequency_encoded"] = 4
        elif activity_log_count >= 10:
            export["going_out_frequency_encoded"] = 3
        elif activity_log_count >= 4:
            export["going_out_frequency_encoded"] = 2
        elif activity_log_count >= 1:
            export["going_out_frequency_encoded"] = 1
        else:
            export["going_out_frequency_encoded"] = 2
    else:
        export["going_out_frequency_encoded"] = coerce_encoded(
            "going_out_frequency_encoded", 0, 4, 2
        )
    export["activity_duration_encoded"] = coerce_encoded(
        "activity_duration_encoded", 0, 3, 1
    )

    if export.get("user_latitude") is None or export.get("user_longitude") is None:
        resolved_coords = None
        for area_name in [
            export.get("home_area"),
            export.get("work_area"),
            *(export.get("preferred_areas") or []),
        ]:
            resolved_coords = _resolve_area_coordinates(area_name)
            if resolved_coords is not None:
                break

        if resolved_coords is not None:
            if export.get("user_latitude") is None:
                export["user_latitude"] = resolved_coords["latitude"]
            if export.get("user_longitude") is None:
                export["user_longitude"] = resolved_coords["longitude"]

    export["has_exceptions"] = int(
        any(
            export.get(field, 0) == 1
            for field in [
                "diet_halal",
                "diet_vegetarian",
                "diet_vegan",
                "diet_gluten_free",
                "excl_nightlife",
                "excl_cultural",
                "excl_water",
                "excl_outdoor_travel",
                "currently_saving_money",
            ]
        )
    )

    export["ml_profile_ready"] = True
    return export

def find_missing_profile_fields(profile: Dict[str, Any]) -> List[str]:
    # Return required profile fields that are missing or empty.
    missing_fields = []

    # Compare the saved profile against the fields needed for recommendations.
    for field in REQUIRED_PROFILE_FIELDS:
        if field not in profile or profile[field] in [None, "", []]:
            missing_fields.append(field)

    return missing_fields

def load_profile_store() -> Dict[str, Any]:
    # Load the profile store from Firestore.
    return load_user_profiles_store()

def save_profile_store(data: Dict[str, Any]) -> None:
    # Persist the in-memory profile store into Firestore.
    save_user_profiles_store(data)

# Backward-compatible aliases for existing imports and call sites.
load_local_profile_store = load_profile_store
save_local_profile_store = save_profile_store

def get_local_profile(data: Dict[str, Any], user_id: str) -> Optional[Dict[str, Any]]:
    # Return a user profile from the Firestore-backed in-memory store.

    return data["user_profiles"].get(normalize_user_id(user_id))

def get_or_create_local_profile(data: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    
    # Return a user profile, creating a shell profile if needed.
    
    # Create the user's profile container before profile updates or feedback logs are saved.
    profiles = data["user_profiles"]
    normalized_user_id = normalize_user_id(user_id)

    if normalized_user_id not in profiles:
        profiles[normalized_user_id] = {
            "user_id": normalized_user_id,
            "created_at": utc_now_iso(),
            "timezone": TIMEZONE,
            "activity_preferences_log_count": 0,
            "recommendation_feedback_count": 0,
        }

    return profiles[normalized_user_id]

def get_profile_intake_fields(missing_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:

    # Return field metadata the agent can use to ask the user for profile data.
    if missing_fields is None:
        return PROFILE_INTAKE_FIELDS

    # Return only the question metadata for fields that are still missing.
    missing_field_set = set(missing_fields)

    return [
        field_info
        for field_info in PROFILE_INTAKE_FIELDS
        if field_info["field"] in missing_field_set
    ]

# Expose profile lookup to the agent.
@function_tool
def get_user_profile(user_id: str) -> Dict[str, Any]:

    # Get the user's saved lifestyle profile.
    tool_name = "get_user_profile"
    user_id = normalize_user_id(user_id)
    start_time = log_tool_start(
        tool_name=tool_name,
        user_id=user_id,
        data={
            "operation": "read_profile"
        }
    )

    try:
        # Read the Firestore profile store and look up the user by email/user ID.
        data = load_profile_store()
        profile = get_local_profile(data, user_id)

        if profile is None:
            # Tell the agent exactly which fields it must collect for a new user.
            missing_fields = REQUIRED_PROFILE_FIELDS.copy()

            log_tool_success(
                tool_name=tool_name,
                start_time=start_time,
                user_id=user_id,
                data={
                    "profile_exists": False
                }
            )

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

        # Add profile status metadata without modifying the saved profile object.
        profile = profile.copy()
        missing_fields = find_missing_profile_fields(profile)
        profile["profile_exists"] = True
        profile["profile_complete"] = len(missing_fields) == 0
        profile["missing_fields"] = missing_fields
        profile["ml_profile_export"] = build_ml_profile_export(profile)

        if missing_fields:
            # Give the agent a focused set of missing fields to ask for next.
            profile["action_required"] = "collect_missing_profile_details"
            profile["required_fields"] = get_profile_intake_fields(missing_fields)

        log_tool_success(
            tool_name=tool_name,
            start_time=start_time,
            user_id=user_id,
            data={
                "profile_exists": True,
                "returned_fields": list(profile.keys())
            }
        )

        return profile

    except Exception as error:
        log_tool_error(
            tool_name=tool_name,
            start_time=start_time,
            error=error,
            user_id=user_id,
        )
        raise

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

    # Extra notes
    notes: Optional[str] = None,
) -> Dict[str, Any]:

    tool_name = "update_user_profile"
    user_id = normalize_user_id(user_id)
    start_time = log_tool_start(
        tool_name=tool_name,
        user_id=user_id,
        data={
            "operation": "update_profile"
        }
    )

    try:
        # Build a partial update from only the values the user provided.
        profile_updates: Dict[str, Any] = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "timezone": "Asia/Dubai",
        }

        if email_address is not None:
            profile_updates["email_address"] = validate_email_address(email_address)

        if monthly_fun_budget_aed is not None:
            if monthly_fun_budget_aed < 0:
                raise ValueError("Monthly budget cannot be negative.")
            profile_updates["monthly_fun_budget_aed"] = monthly_fun_budget_aed

        if max_per_activity_aed is not None:
            if max_per_activity_aed < 0:
                raise ValueError("Max per activity budget cannot be negative.")
            profile_updates["max_per_activity_aed"] = max_per_activity_aed

        if home_area is not None:
            profile_updates["home_area"] = validate_dubai_area(home_area)

        if work_area is not None:
            profile_updates["work_area"] = validate_dubai_area(work_area)

        if preferred_areas is not None:
            profile_updates["preferred_areas"] = [
                validate_dubai_area(area) for area in preferred_areas
            ]

        if max_travel_distance_km is not None:
            if max_travel_distance_km < 0:
                raise ValueError("Max travel distance cannot be negative.")
            profile_updates["max_travel_distance_km"] = max_travel_distance_km

        if work_days is not None:
            profile_updates["work_days"] = validate_work_days(work_days)

        if work_start_time is not None:
            profile_updates["work_start_time"] = work_start_time

        if work_end_time is not None:
            profile_updates["work_end_time"] = work_end_time

        if hobbies is not None:
            profile_updates["hobbies"] = [
                hobby.strip().lower() for hobby in hobbies
            ]

        if interests is not None:
            profile_updates["interests"] = [
                interest.strip().lower() for interest in interests
            ]

        if preferred_activity_types is not None:
            profile_updates["preferred_activity_types"] = [
                activity.strip().lower() for activity in preferred_activity_types
            ]

        if preferred_environment is not None:
            profile_updates["preferred_environment"] = preferred_environment.strip().lower()

        if budget_encoded is not None:
            profile_updates["budget_encoded"] = validate_int_range(
                budget_encoded, "budget_encoded", 0, 2
            )

        if travel_distance_encoded is not None:
            profile_updates["travel_distance_encoded"] = validate_int_range(
                travel_distance_encoded, "travel_distance_encoded", 0, 4
            )

        if weather_pref_encoded is not None:
            profile_updates["weather_pref_encoded"] = validate_int_range(
                weather_pref_encoded, "weather_pref_encoded", 0, 3
            )

        for field_name, value in [
            ("pref_morning", pref_morning),
            ("pref_midday", pref_midday),
            ("pref_afternoon", pref_afternoon),
            ("pref_evening", pref_evening),
            ("pref_late_night", pref_late_night),
            ("diet_halal", diet_halal),
            ("diet_vegetarian", diet_vegetarian),
            ("diet_vegan", diet_vegan),
            ("diet_gluten_free", diet_gluten_free),
            ("social_friends", social_friends),
            ("social_family", social_family),
            ("social_partner", social_partner),
            ("social_alone", social_alone),
            ("factor_cost", factor_cost),
            ("factor_distance", factor_distance),
            ("factor_quality", factor_quality),
            ("factor_comfort", factor_comfort),
            ("currently_saving_money", currently_saving_money),
            ("excl_nightlife", excl_nightlife),
            ("excl_cultural", excl_cultural),
            ("excl_water", excl_water),
            ("excl_outdoor_travel", excl_outdoor_travel),
        ]:
            if value is not None:
                profile_updates[field_name] = validate_int_range(value, field_name, 0, 1)

        if adventure_level_encoded is not None:
            profile_updates["adventure_level_encoded"] = validate_int_range(
                adventure_level_encoded, "adventure_level_encoded", 0, 3
            )

        if going_out_frequency_encoded is not None:
            profile_updates["going_out_frequency_encoded"] = validate_int_range(
                going_out_frequency_encoded, "going_out_frequency_encoded", 0, 4
            )

        if activity_duration_encoded is not None:
            profile_updates["activity_duration_encoded"] = validate_int_range(
                activity_duration_encoded, "activity_duration_encoded", 0, 3
            )

        if user_latitude is not None:
            profile_updates["user_latitude"] = validate_coordinate(
                user_latitude, "user_latitude", -90.0, 90.0
            )

        if user_longitude is not None:
            profile_updates["user_longitude"] = validate_coordinate(
                user_longitude, "user_longitude", -180.0, 180.0
            )

        if notes is not None:
            profile_updates["notes"] = notes.strip()

        # Merge the update into the Firestore-backed profile store.
        data = load_profile_store()
        saved_profile = get_or_create_local_profile(data, user_id)
        saved_profile.update(profile_updates)
        save_profile_store(data)
        missing_fields = find_missing_profile_fields(saved_profile)

        log_tool_success(
            tool_name=tool_name,
            start_time=start_time,
            user_id=user_id,
            data={
                "updated_field_names": list(profile_updates.keys())
            }
        )

        return {
            "success": True,
            "profile_exists": True,
            "profile_complete": len(missing_fields) == 0,
            "message": "User profile updated successfully.",
            "updated_fields": profile_updates,
            "missing_fields": missing_fields,
            "required_fields": get_profile_intake_fields(missing_fields),
            "ml_profile_export": build_ml_profile_export(saved_profile),
            "next_action": (
                "continue_activity_planning"
                if len(missing_fields) == 0
                else "collect_missing_profile_details"
            ),
        }

    except Exception as error:
        log_tool_error(
            tool_name=tool_name,
            start_time=start_time,
            error=error,
            user_id=user_id,
            data={
                "operation": "update_profile"
            }
        )
        raise

# Expose activity behavior logging to the agent.
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

    Purpose:
    - Builds training data for the ML recommender.
    - Captures implicit and explicit user preference signals.
    - Allowed signals: clicked, booked, declined.
    """

    tool_name = "log_activity_preference"
    user_id = normalize_user_id(user_id)

    start_time = log_tool_start(
        tool_name=tool_name,
        user_id=user_id,
        data={
            "operation": "write_activity_preference_log",
            "activity_id": activity_id,
            "performance_signal": performance_signal,
            "session_id": session_id,
            "recommendation_id": recommendation_id,
            "model_version": model_version,
        }
    )

    try:
        # Validate the behavioral signal before it is written to the profile log.
        cleaned_signal = normalize_text(performance_signal)

        if cleaned_signal not in VALID_PREFERENCE_SIGNALS:
            raise ValueError(
                f"Invalid performance_signal: {performance_signal}. "
                f"Use one of: {sorted(VALID_PREFERENCE_SIGNALS)}"
            )

        validate_positive_number(activity_price_aed, "activity_price_aed")

        created_at = utc_now_iso()
        log_id = uuid.uuid4().hex

        # Build the activity log entry from required and optional activity details.
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

        # Store the activity log entry in Firestore and update the profile counters.
        data = load_profile_store()
        profile = get_or_create_local_profile(data, user_id)
        profile["activity_preferences_log_count"] = int(profile.get("activity_preferences_log_count", 0)) + 1
        profile["last_activity_preference_log_at"] = created_at
        profile["updated_at"] = created_at
        save_profile_store(data)
        write_activity_preference_log(log_entry)

        log_tool_success(
            tool_name=tool_name,
            start_time=start_time,
            user_id=user_id,
            data={
                "operation": "write_activity_preference_log",
                "activity_id": activity_id,
                "performance_signal": cleaned_signal,
                "log_id": log_id,
                "has_recommendation_id": recommendation_id is not None,
                "has_session_id": session_id is not None,
                "has_model_version": model_version is not None,
            }
        )

        return {
            "success": True,
            "message": "Activity preference logged successfully.",
            "log_id": log_id,
            "log_entry": log_entry,
        }

    except Exception as error:
        log_tool_error(
            tool_name=tool_name,
            start_time=start_time,
            error=error,
            user_id=user_id,
            data={
                "operation": "write_activity_preference_log",
                "activity_id": activity_id,
                "performance_signal": performance_signal,
                "session_id": session_id,
                "recommendation_id": recommendation_id,
                "model_version": model_version,
            }
        )
        raise

# Expose explicit recommendation feedback logging to the agent.
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

    Purpose:
    - Stores liked/disliked/neutral feedback.
    - Helps evaluate recommendation quality.
    - Can become labeled data for LightGBM training.
    """

    tool_name = "log_recommendation_feedback"
    user_id = normalize_user_id(user_id)

    start_time = log_tool_start(
        tool_name=tool_name,
        user_id=user_id,
        data={
            "operation": "write_recommendation_feedback",
            "recommendation_id": recommendation_id,
            "activity_id": activity_id,
            "feedback": feedback,
            "session_id": session_id,
            "model_version": model_version,
        }
    )

    try:
        # Validate the feedback label before writing it to the profile.
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

        # Build the feedback entry from the recommendation and optional context.
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

        # Store the feedback entry in Firestore and update the profile counters.
        data = load_profile_store()
        profile = get_or_create_local_profile(data, user_id)
        profile["recommendation_feedback_count"] = int(profile.get("recommendation_feedback_count", 0)) + 1
        profile["last_recommendation_feedback_at"] = created_at
        profile["updated_at"] = created_at
        save_profile_store(data)
        write_recommendation_feedback_log(feedback_entry)

        log_tool_success(
            tool_name=tool_name,
            start_time=start_time,
            user_id=user_id,
            data={
                "operation": "write_recommendation_feedback",
                "recommendation_id": recommendation_id,
                "activity_id": activity_id,
                "feedback": cleaned_feedback,
                "feedback_id": feedback_id,
                "rating": rating,
                "recommendation_rank": recommendation_rank,
                "has_feedback_reason": feedback_reason is not None,
                "has_session_id": session_id is not None,
                "has_model_version": model_version is not None,
            }
        )

        return {
            "success": True,
            "message": "Recommendation feedback logged successfully.",
            "feedback_id": feedback_id,
            "feedback_entry": feedback_entry,
        }

    except Exception as error:
        log_tool_error(
            tool_name=tool_name,
            start_time=start_time,
            error=error,
            user_id=user_id,
            data={
                "operation": "write_recommendation_feedback",
                "recommendation_id": recommendation_id,
                "activity_id": activity_id,
                "feedback": feedback,
                "rating": rating,
                "recommendation_rank": recommendation_rank,
                "session_id": session_id,
                "model_version": model_version,
            }
        )
        raise
