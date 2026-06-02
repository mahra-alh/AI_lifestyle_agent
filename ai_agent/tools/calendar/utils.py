"""
Timezone parsing, datetime validation, and buffer logic.

No Google API calls are made here; this module is pure Python.
"""
from .config import DEFAULT_TIMEZONE
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

# Timezone helpers
def get_timezone(timezone_str: str) -> ZoneInfo:
    """
    Validate and return an IANA timezone object.

    Args:
        timezone_str: IANA name such as "Asia/Dubai".

    Returns:
        A ZoneInfo object for the requested timezone.

    Raises:
        ValueError: The string is not a recognised IANA timezone.
    """
    try:
        return ZoneInfo(timezone_str)
    except ZoneInfoNotFoundError as error:
        raise ValueError(
            f"Invalid timezone: '{timezone_str}'. "
            "Use an IANA timezone name, for example 'Asia/Dubai' or 'Europe/London'."
        ) from error


def to_timezone_iso(datetime_str: str, timezone_str: str) -> str:
    """
    Convert an RFC 3339 datetime string into the target timezone.

    Args:
        datetime_str: An ISO/RFC 3339 string, e.g. "2026-05-07T14:00:00Z".
        timezone_str: Target IANA timezone name.

    Returns:
        ISO 8601 string in the target timezone, e.g. "2026-05-07T18:00:00+04:00".
    """
    tz = get_timezone(timezone_str)
    parsed = datetime.fromisoformat(datetime_str.replace("Z", "+00:00"))
    return parsed.astimezone(tz).isoformat()


# Datetime parsing
def parse_datetime(
    datetime_value: str,
    timezone_str: str = DEFAULT_TIMEZONE,
) -> datetime:
    """
    Parse a datetime string and attach the default timezone if the string
    has no UTC offset.

    Args:
        datetime_value: ISO-style string such as "2026-05-07T18:00:00" or
                        "2026-05-07T18:00:00+04:00".
        timezone_str:   Fallback timezone when the string has no offset.

    Returns:
        A timezone-aware datetime object.

    Raises:
        ValueError: The string is empty, not a string, or not parseable.
    """
    if not datetime_value or not isinstance(datetime_value, str):
        raise ValueError("datetime_value must be a non-empty string.")

    tz = get_timezone(timezone_str)
    cleaned = datetime_value.strip().replace("Z", "+00:00")

    try:
        dt = datetime.fromisoformat(cleaned)
    except ValueError as error:
        raise ValueError(
            f"Cannot parse datetime: '{datetime_value}'. "
            "Use ISO 8601 format, e.g. '2026-05-07T18:00:00' or "
            "'2026-05-07T18:00:00+04:00'."
        ) from error

    # Attach the local timezone when none is present in the string.
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)

    return dt


# Validation
def validate_time_range(
    start_time: str,
    end_time: str,
    timezone_str: str = DEFAULT_TIMEZONE,
    allow_past: bool = False,
) -> Dict[str, Any]:
    """
    Validate a start/end time pair and return parsed datetimes + RFC 3339 strings.

    Checks:
        - Both strings are valid datetimes.
        - start_time is strictly before end_time.
        - start_time is in the future (unless allow_past=True).

    Args:
        start_time:   Proposed slot start, e.g. "2026-05-07T18:00:00".
        end_time:     Proposed slot end,   e.g. "2026-05-07T20:00:00".
        timezone_str: IANA timezone for naive strings.
        allow_past:   Skip the future-time check (useful for log queries).

    Returns:
        Dict with keys: start_dt, end_dt, start_rfc3339, end_rfc3339.

    Raises:
        ValueError: Any validation rule is violated.
    """
    start_dt = parse_datetime(start_time, timezone_str)
    end_dt = parse_datetime(end_time, timezone_str)

    if start_dt >= end_dt:
        raise ValueError(
            f"end_time ({end_time}) must be after start_time ({start_time})."
        )

    if not allow_past:
        now_utc = datetime.now(timezone.utc)
        if start_dt.astimezone(timezone.utc) < now_utc:
            raise ValueError(
                f"start_time ({start_time}) must be in the future."
            )

    return {
        "start_dt": start_dt,
        "end_dt": end_dt,
        "start_rfc3339": start_dt.isoformat(),
        "end_rfc3339": end_dt.isoformat(),
    }


# Buffer helper
def apply_buffer(
    start_dt: datetime,
    end_dt: datetime,
    buffer_minutes: int = 0,
) -> Dict[str, datetime]:
    """
    Expand a time window by buffer_minutes on each side.

    Use this to account for travel or setup time around an activity
    when checking calendar availability. The actual booked event still
    uses the original start and end times.

    Example:
        Activity 18:00–20:00, buffer_minutes=10
        → FreeBusy check covers 17:50–20:10

    Args:
        start_dt:       Activity start as a timezone-aware datetime.
        end_dt:         Activity end as a timezone-aware datetime.
        buffer_minutes: Minutes to pad on each side. Must be >= 0.

    Returns:
        Dict with keys: checked_start_dt, checked_end_dt.
    """
    if buffer_minutes < 0:
        raise ValueError("buffer_minutes cannot be negative.")

    delta = timedelta(minutes=buffer_minutes)
    return {
        "checked_start_dt": start_dt - delta,
        "checked_end_dt": end_dt + delta,
    }


# Busy-slot normalisation
def normalize_busy_slots(
    busy_slots: List[Dict[str, Any]],
    timezone_str: str,
) -> List[Dict[str, Any]]:
    """
    Convert raw FreeBusy busy-slot times into a single target timezone.

    Google returns busy slots in UTC. Normalising them to the user's
    local timezone makes the agent's reasoning easier and avoids offset
    inference errors.

    Returns:
        List of {"start": ..., "end": ...} dicts in the target timezone.
        Slots that cannot be parsed are kept with their original values.
    """
    normalised: List[Dict[str, Any]] = []

    for slot in busy_slots:
        start_value = slot.get("start")
        end_value = slot.get("end")

        if not start_value or not end_value:
            continue

        try:
            normalised.append(
                {
                    "start": to_timezone_iso(start_value, timezone_str),
                    "end": to_timezone_iso(end_value, timezone_str),
                }
            )
        except Exception:
            # Preserve the raw value so the agent still gets some information.
            normalised.append({"start": start_value, "end": end_value})

    return normalised
