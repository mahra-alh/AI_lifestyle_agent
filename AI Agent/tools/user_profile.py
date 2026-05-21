from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from pathlib import Path
import json
import re
import uuid

from agents import function_tool

from data.dubai_areas import DUBAI_AREAS, get_area_by_user_input
from utils.app_logger import log_tool_start, log_tool_success, log_tool_error

# Keep profile storage and validation constants in one place.
TIMEZONE = "Asia/Dubai"
LOCAL_PROFILE_STORE_PATH = Path(__file__).resolve().parents[1] / "data" / "user_profiles.json"

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

# Shared helper functions for timestamps, cleanup, validation, and local storage.
def utc_now_iso() -> str:

    # Return the current UTC timestamp as an ISO string.
    # This is useful for saving consistent timestamps in the local profile store.
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

    if value < 0:
        raise ValueError(f"{field_name} cannot be negative.")

    return value

def validate_rating(rating: Optional[int]) -> Optional[int]:

    # Validate a recommendation rating from 1 to 5.
    if rating is None:
        return None

    if rating < 1 or rating > 5:
        raise ValueError("rating must be between 1 and 5.")

    return rating

def find_missing_profile_fields(profile: Dict[str, Any]) -> List[str]:
    # Return required profile fields that are missing or empty.
    missing_fields = []

    # Compare the saved profile against the fields needed for recommendations.
    for field in REQUIRED_PROFILE_FIELDS:
        if field not in profile or profile[field] in [None, "", []]:
            missing_fields.append(field)

    return missing_fields

def load_local_profile_store() -> Dict[str, Any]:
    # Load local JSON profile data for testing.

    # Create an empty in-memory store when the JSON file has not been created yet.
    if not LOCAL_PROFILE_STORE_PATH.exists():
        return {"user_profiles": {}}

    with LOCAL_PROFILE_STORE_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if "user_profiles" not in data or not isinstance(data["user_profiles"], dict):
        data["user_profiles"] = {}

    return data

def save_local_profile_store(data: Dict[str, Any]) -> None:
    # Persist local JSON profile data for development/testing.
    # Ensure the data folder exists before writing the local profile store.
    LOCAL_PROFILE_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with LOCAL_PROFILE_STORE_PATH.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)

def get_local_profile(data: Dict[str, Any], user_id: str) -> Optional[Dict[str, Any]]:
    # Return a user profile from the local JSON store.

    return data["user_profiles"].get(user_id)

def get_or_create_local_profile(data: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    
    # Return a user profile, creating a shell profile if needed.
    
    # Create the user's profile container before profile updates or feedback logs are saved.
    profiles = data["user_profiles"]

    if user_id not in profiles:
        profiles[user_id] = {
            "user_id": user_id,
            "created_at": utc_now_iso(),
            "timezone": TIMEZONE,
            "activity_preferences_log": [],
            "recommendation_feedback": [],
        }

    return profiles[user_id]

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
    start_time = log_tool_start(
        tool_name=tool_name,
        user_id=user_id,
        data={
            "operation": "read_profile"
        }
    )

    try:
        # Read the local profile store and look up the user by email/user ID.
        data = load_local_profile_store()
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

    # Extra notes
    notes: Optional[str] = None,
) -> Dict[str, Any]:

    tool_name = "update_user_profile"
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

        if notes is not None:
            profile_updates["notes"] = notes.strip()

        # Merge the update into the local JSON profile store.
        data = load_local_profile_store()
        saved_profile = get_or_create_local_profile(data, user_id)
        saved_profile.update(profile_updates)
        save_local_profile_store(data)
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
def log_activity_performance(
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

    tool_name = "log_activity_performance"

    start_time = log_tool_start(
        tool_name=tool_name,
        user_id=user_id,
        data={
            "operation": "write_activity_performance_log",
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

        # Append the activity log entry to the user's local profile.
        data = load_local_profile_store()
        profile = get_or_create_local_profile(data, user_id)
        profile.setdefault("activity_preferences_log", []).append(log_entry)
        profile["last_activity_preference_log_at"] = created_at
        profile["updated_at"] = created_at
        save_local_profile_store(data)

        log_tool_success(
            tool_name=tool_name,
            start_time=start_time,
            user_id=user_id,
            data={
                "operation": "write_activity_performance_log",
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
            "message": "Activity performance logged successfully.",
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
                "operation": "write_activity_performance_log",
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

        # Append the feedback entry to the user's local profile.
        data = load_local_profile_store()
        profile = get_or_create_local_profile(data, user_id)
        profile.setdefault("recommendation_feedback", []).append(feedback_entry)
        profile["last_recommendation_feedback_at"] = created_at
        profile["updated_at"] = created_at
        save_local_profile_store(data)

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
