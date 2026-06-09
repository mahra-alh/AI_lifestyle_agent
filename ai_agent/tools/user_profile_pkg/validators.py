"""
Input validation and normalization helpers for user profile data.

All functions are pure so they can be imported and tested
independently from the storage or ML export layers.
"""
from __future__ import annotations

import re
from typing import List, Optional

from ai_agent.data.dubai_areas import DUBAI_AREAS, get_area_by_user_input

VALID_WORK_DAYS = {
    "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday",
}

VALID_PREFERENCE_SIGNALS = {"clicked", "booked", "declined"}

VALID_RECOMMENDATION_FEEDBACK = {"liked", "disliked", "neutral"}


# Text functions
def normalize_text(value: Optional[str]) -> Optional[str]:
    """Clean a text value by stripping whitespace and lowercasing."""
    if value is None:
        return None
    cleaned = value.strip().lower()
    return cleaned if cleaned else None


def normalize_text_list(values: Optional[List[str]]) -> Optional[List[str]]:
    """Clean a list of text values and remove empty items."""
    if values is None:
        return None
    return [c for v in values if (c := normalize_text(v)) is not None]


def normalize_user_id(user_id: str) -> str:
    """Normalize the profile key so Firestore uses one document per email."""
    return user_id.strip().lower()


# Field validators
def validate_email_address(email_address: Optional[str]) -> Optional[str]:
    """Validate and normalize an email address (format only, not deliverability)."""
    if email_address is None:
        return None
    cleaned = email_address.strip().lower()
    if not re.match(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$", cleaned):
        raise ValueError(f"'{email_address}' is not a valid email address format.")
    return cleaned


def validate_dubai_area(area: Optional[str]) -> Optional[str]:
    """
    Validate that the area belongs to the supported Dubai areas list.
    Returns the canonical area ID stored in DUBAI_AREAS.
    """
    if area is None:
        return None
    cleaned = normalize_text(area)
    if cleaned in DUBAI_AREAS:
        return cleaned
    matched = get_area_by_user_input(area)
    if matched is None:
        raise ValueError(
            f"'{area}' is not in the supported Dubai areas list. "
            "Use a known Dubai area such as Dubai Marina, Business Bay, JVC, "
            "Downtown Dubai, Deira, etc."
        )
    return matched["area_id"]


def validate_work_days(work_days: Optional[List[str]]) -> Optional[List[str]]:
    """Validate and normalize work days."""
    cleaned = normalize_text_list(work_days)
    if cleaned is None:
        return None
    invalid = [d for d in cleaned if d not in VALID_WORK_DAYS]
    if invalid:
        raise ValueError(
            f"Invalid work day(s): {invalid}. Use one of: {sorted(VALID_WORK_DAYS)}"
        )
    return cleaned


def validate_positive_number(value: Optional[float], field_name: str) -> Optional[float]:
    """Validate that a numeric value is non-negative."""
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
    """Validate a recommendation rating from 1 to 5."""
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
    """Validate and normalize a small integer flag or encoded value."""
    if value is None:
        return None
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be an integer between {minimum} and {maximum}.")
    if normalized < minimum or normalized > maximum:
        raise ValueError(f"{field_name} must be between {minimum} and {maximum}.")
    return normalized


def validate_coordinate(
    value: Optional[float],
    field_name: str,
    minimum: float,
    maximum: float,
) -> Optional[float]:
    """Validate latitude/longitude style coordinates."""
    if value is None:
        return None
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be a number.")
    if normalized < minimum or normalized > maximum:
        raise ValueError(f"{field_name} must be between {minimum} and {maximum}.")
    return normalized
