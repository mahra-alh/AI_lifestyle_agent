"""
Tools for checking calendar availability and booking activities.
get_calendar checks whether the user is free.
book_activity books the approved recommendation into Google Calendar.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from agents import function_tool

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build  # type: ignore
from googleapiclient.errors import HttpError  # type: ignore
from ai_agent.logger.app_logger import log_tool_error, log_tool_start, log_tool_success


# Request calendar access because this tool checks availability and creates approved events.
SCOPES = ["https://www.googleapis.com/auth/calendar"]

# Keep local defaults internal so the agent cannot choose credential file paths.
DEFAULT_CALENDAR_ID = "primary"
DEFAULT_TIMEZONE = "Asia/Dubai"
DEFAULT_LOG_DIR = "data/calendar"
DEFAULT_CREDENTIALS_PATH = "credentials.json"
DEFAULT_TOKEN_PATH = "token.json"

# Small service cache so repeated tool calls do not rebuild the Calendar API client every time.
# The cache is keyed by credentials_path and token_path.
_SERVICE_CACHE: Dict[Tuple[str, str], Any] = {}

# Private helper functions
def _get_calendar_service(
    credentials_path: str = DEFAULT_CREDENTIALS_PATH,
    token_path: str = DEFAULT_TOKEN_PATH,
    use_cache: bool = True,
    allow_interactive_auth: bool = False,
):
    
    # Authenticate with Google Calendar and return a Calendar API service.
    # Reuse the service client during one process to avoid repeated setup work.
    cache_key = (credentials_path, token_path)

    if use_cache and cache_key in _SERVICE_CACHE:
        return _SERVICE_CACHE[cache_key]

    # Load an existing OAuth token if this user has already authenticated.
    creds = None

    token_file = Path(token_path)
    credentials_file = Path(credentials_path)

    if token_file.exists():
        creds = Credentials.from_authorized_user_file(
            filename=str(token_file),
            scopes=SCOPES,
        )

    if not creds or not creds.valid:
        # Refresh an expired token, or start a browser login when no valid token exists.
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not allow_interactive_auth:
                raise RuntimeError(
                    "Google Calendar is not authenticated. "
                    "Create or refresh token.json before running the agent, "
                    "or authenticate in an interactive setup."
                )

            if not credentials_file.exists():
                raise FileNotFoundError(
                    f"Missing {credentials_path}. Download OAuth credentials from "
                    "Google Cloud Console and save the file as credentials.json."
                )

            flow = InstalledAppFlow.from_client_secrets_file(
                client_secrets_file=str(credentials_file),
                scopes=SCOPES,
            )
            creds = flow.run_local_server(port=0)

        token_file.write_text(creds.to_json(), encoding="utf-8")

    # Build the Google Calendar API client used by availability and booking calls.
    service = build("calendar", "v3", credentials=creds)

    if use_cache:
        _SERVICE_CACHE[cache_key] = service

    return service

def _get_timezone(timezone_str: str) -> ZoneInfo:
    # Validate and return an IANA timezone object.
    # IANA Timezone: is a collaborative dataset that maps the world's time zones, including historical, current, and planned changes like daylight saving time

    try:
        return ZoneInfo(timezone_str)
    except ZoneInfoNotFoundError as error:
        raise ValueError(
            f"Invalid timezone_str: {timezone_str}. Use an IANA timezone, "
            "for example 'Asia/Dubai'."
        ) from error

def _parse_datetime(
    datetime_value: str,
    timezone_str: str = DEFAULT_TIMEZONE,
) -> datetime:

    # Parse a datetime string and attach the default timezone if missing.
    if not datetime_value or not isinstance(datetime_value, str):
        raise ValueError("Datetime value must be a non-empty string.")

    # Normalize common datetime formats into an aware datetime object.
    timezone_obj = _get_timezone(timezone_str)

    cleaned_value = datetime_value.strip().replace("Z", "+00:00")

    try:
        dt = datetime.fromisoformat(cleaned_value)
    except ValueError as error:
        raise ValueError(
            f"Invalid datetime format: {datetime_value}. Use ISO/RFC3339 style, "
            "for example '2026-05-07T18:00:00' or '2026-05-07T18:00:00+04:00'."
        ) from error

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone_obj)

    return dt

def _to_timezone_iso(datetime_value: str, timezone_str: str) -> str:
    # Convert an RFC3339 datetime string into the requested timezone.
    timezone_obj = _get_timezone(timezone_str)
    parsed = datetime.fromisoformat(datetime_value.replace("Z", "+00:00"))
    return parsed.astimezone(timezone_obj).isoformat()

def _normalize_busy_slots(
    busy_slots: List[Dict[str, Any]],
    timezone_str: str,
) -> List[Dict[str, Any]]:
    # Normalize busy slot boundaries to one timezone so the agent does not infer offsets incorrectly.
    normalized: List[Dict[str, Any]] = []

    for slot in busy_slots:
        start_value = slot.get("start")
        end_value = slot.get("end")

        if not start_value or not end_value:
            continue

        try:
            normalized.append(
                {
                    "start": _to_timezone_iso(start_value, timezone_str),
                    "end": _to_timezone_iso(end_value, timezone_str),
                }
            )
        except Exception:
            # Keep original values if parsing fails for one slot.
            normalized.append(
                {
                    "start": start_value,
                    "end": end_value,
                }
            )

    return normalized

def _validate_time_range(
    start_time: str,
    end_time: str,
    timezone_str: str = DEFAULT_TIMEZONE,
    allow_past: bool = False,
) -> Dict[str, Any]:
    """
    Validate:
    - start_time and end_time are valid datetime strings.
    - start_time is before end_time.
    """

    start_dt = _parse_datetime(start_time, timezone_str)
    end_dt = _parse_datetime(end_time, timezone_str)

    # Reject invalid or past time ranges before calling the Calendar API.
    if start_dt >= end_dt:
        raise ValueError("end_time must be after start_time.")

    if not allow_past:
        now_utc = datetime.now(timezone.utc)
        if start_dt.astimezone(timezone.utc) < now_utc:
            raise ValueError("start_time must be in the future.")

    return {
        "start_dt": start_dt,
        "end_dt": end_dt,
        "start_rfc3339": start_dt.isoformat(),
        "end_rfc3339": end_dt.isoformat(),
    }

def _apply_buffer(
    start_dt: datetime,
    end_dt: datetime,
    buffer_minutes: int = 0,
) -> Dict[str, datetime]:
    """
    Add a buffer around the calendar availability check.
    Example:
        If the requested activity is 18:00-20:00 and buffer_minutes=10,
        the FreeBusy check will use 17:50-20:10.
    The actual booked event still uses the original start and end time.
    """

    if buffer_minutes < 0:
        raise ValueError("buffer_minutes cannot be negative.")

    buffer_delta = timedelta(minutes=buffer_minutes)

    # Expand the checked window so travel or setup time can be considered.
    return {
        "checked_start_dt": start_dt - buffer_delta,
        "checked_end_dt": end_dt + buffer_delta,
    }

def _check_calendar_availability_internal(
    start_time: str,
    end_time: str,
    calendar_id: str = DEFAULT_CALENDAR_ID,
    timezone_str: str = DEFAULT_TIMEZONE,
    credentials_path: str = DEFAULT_CREDENTIALS_PATH,
    token_path: str = DEFAULT_TOKEN_PATH,
    buffer_minutes: int = 0,
    allow_past: bool = False,
) -> Dict[str, Any]:
    """
    Internal calendar availability check used by both tools.
    This avoids calling an SDK-decorated function from inside another SDK tool.
    """
    # Validate the requested range and add any requested buffer.
    validated = _validate_time_range(
        start_time=start_time,
        end_time=end_time,
        timezone_str=timezone_str,
        allow_past=allow_past,
    )

    buffered = _apply_buffer(
        start_dt=validated["start_dt"],
        end_dt=validated["end_dt"],
        buffer_minutes=buffer_minutes,
    )

    checked_start_rfc3339 = buffered["checked_start_dt"].isoformat()
    checked_end_rfc3339 = buffered["checked_end_dt"].isoformat()

    service = _get_calendar_service(
        credentials_path=credentials_path,
        token_path=token_path,
        allow_interactive_auth=False,
    )

    # Ask Google Calendar FreeBusy for busy slots in the checked window.
    freebusy_body = {
        "timeMin": checked_start_rfc3339,
        "timeMax": checked_end_rfc3339,
        "timeZone": timezone_str,
        "items": [
            {
                "id": calendar_id,
            }
        ],
    }

    freebusy_result = service.freebusy().query(body=freebusy_body).execute()

    calendar_info = freebusy_result.get("calendars", {}).get(calendar_id, {})
    calendar_errors = calendar_info.get("errors", [])

    # Return API-level calendar errors in a structured way for the agent.
    if calendar_errors:
        return {
            "status": "error",
            "message": "Google Calendar returned errors while checking availability.",
            "calendar_id": calendar_id,
            "calendar_errors": calendar_errors,
            "is_free": False,
            "busy_slots": [],
            "requested_start_time": validated["start_rfc3339"],
            "requested_end_time": validated["end_rfc3339"],
            "checked_start_time": checked_start_rfc3339,
            "checked_end_time": checked_end_rfc3339,
            "buffer_minutes": buffer_minutes,
        }

    busy_slots = calendar_info.get("busy", [])
    busy_slots_local = _normalize_busy_slots(
        busy_slots=busy_slots,
        timezone_str=timezone_str,
    )
    is_free = len(busy_slots) == 0

    # Convert the FreeBusy response into the compact result the agent needs.
    return {
        "status": "success",
        "calendar_id": calendar_id,
        "timezone": timezone_str,
        "requested_start_time": validated["start_rfc3339"],
        "requested_end_time": validated["end_rfc3339"],
        "checked_start_time": checked_start_rfc3339,
        "checked_end_time": checked_end_rfc3339,
        "buffer_minutes": buffer_minutes,
        "is_free": is_free,
        "busy_slots": busy_slots,
        "busy_slots_local": busy_slots_local,
        "message": (
            "The user is free during this time slot."
            if is_free
            else "The user is busy during this time slot."
        ),
    }

def _rotate_log_if_needed(log_path: Path, max_log_size_mb: int = 5) -> None:
    # Rotate the local JSONL log if it becomes too large.
    if max_log_size_mb <= 0:
        return

    max_size_bytes = max_log_size_mb * 1024 * 1024

    if log_path.exists() and log_path.stat().st_size > max_size_bytes:
        rotated_path = log_path.with_suffix(".jsonl.1")
        if rotated_path.exists():
            rotated_path.unlink()
        shutil.move(str(log_path), str(rotated_path))

def _save_calendar_log(
    data: Dict[str, Any],
    output_dir: str = DEFAULT_LOG_DIR,
    max_log_size_mb: int = 5,
) -> None:
    # Save a local JSONL log of calendar booking results.
    output_folder = Path(output_dir)
    output_folder.mkdir(parents=True, exist_ok=True)

    log_path = output_folder / "calendar_actions_log.jsonl"
    _rotate_log_if_needed(log_path, max_log_size_mb=max_log_size_mb)

    # Append booking results locally for debugging and audit during testing.
    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(data, ensure_ascii=False) + "\n")

def _build_event_description(
    description: str = "",
    recommendation_reason: str = "",
    weather_context: str = "",
    recommendation_id: str = "",
) -> str:
    # Build event description from optional recommendation context.
    description_parts: List[str] = []

    if description:
        description_parts.append(description)

    if recommendation_reason:
        description_parts.append(f"Recommendation reason: {recommendation_reason}")

    if weather_context:
        description_parts.append(f"Weather context: {weather_context}")

    if recommendation_id:
        description_parts.append(f"Recommendation ID: {recommendation_id}")

    description_parts.append("Booked by Lifestyle AI Agent.")

    return "\n\n".join(description_parts)

# Expose calendar availability checks to the agent.
@function_tool
def get_calendar(
    start_time: str,
    end_time: str,
    calendar_id: str = DEFAULT_CALENDAR_ID,
    timezone_str: str = DEFAULT_TIMEZONE,
    buffer_minutes: int = 0,
) -> Dict[str, Any]:
    """
    Check whether the user's Google Calendar is free for the proposed slot time.
    Use this tool before sending data to the ML recommender.
    Args:
        start_time: Proposed slot start time, for example "2026-05-07T18:00:00".
        end_time: Proposed slot end time, for example "2026-05-07T20:00:00".
        calendar_id: Google Calendar ID. Use "primary" for the user's main calendar.
        timezone_str: IANA timezone name. Default is "Asia/Dubai".
        buffer_minutes: Optional buffer around the availability check.

    Returns:
        A dictionary with availability status, busy slots, checked time range, and messages.
    """

    tool_name = "get_calendar"
    start_clock = log_tool_start(
        tool_name=tool_name,
        data={
            "start_time": start_time,
            "end_time": end_time,
            "calendar_id": calendar_id,
            "timezone_str": timezone_str,
            "buffer_minutes": buffer_minutes,
        },
    )

    try:
        # Use the shared internal checker so this tool and booking stay consistent.
        result = _check_calendar_availability_internal(
            start_time=start_time,
            end_time=end_time,
            calendar_id=calendar_id,
            timezone_str=timezone_str,
            buffer_minutes=buffer_minutes,
            allow_past=False,
        )

        if result.get("status") == "success":
            log_tool_success(
                tool_name=tool_name,
                start_time=start_clock,
                data={
                    "calendar_id": calendar_id,
                    "is_free": result.get("is_free"),
                    "busy_slots_count": len(result.get("busy_slots", [])),
                },
            )
        else:
            log_tool_error(
                tool_name=tool_name,
                start_time=start_clock,
                error=RuntimeError(result.get("message", "Calendar availability check returned error status.")),
                data={
                    "calendar_id": calendar_id,
                    "status": result.get("status"),
                },
            )

        return result

    except HttpError as error:
        result = {
            "status": "error",
            "message": f"Google Calendar API error: {error}",
            "is_free": False,
            "busy_slots": [],
        }
        log_tool_error(
            tool_name=tool_name,
            start_time=start_clock,
            error=error,
            data={"calendar_id": calendar_id},
        )
        return result

    except RuntimeError as error:
        result = {
            "status": "error",
            "message": str(error),
            "is_free": False,
            "busy_slots": [],
        }
        log_tool_error(
            tool_name=tool_name,
            start_time=start_clock,
            error=error,
            data={"calendar_id": calendar_id},
        )
        return result

    except Exception as error:
        # Return tool errors to the agent instead of crashing the full run.
        result = {
            "status": "error",
            "message": f"Calendar availability check failed: {error}",
            "is_free": False,
            "busy_slots": [],
        }
        log_tool_error(
            tool_name=tool_name,
            start_time=start_clock,
            error=error,
            data={"calendar_id": calendar_id},
        )
        return result

# Expose approved calendar booking to the agent.
@function_tool
def book_activity(
    user_approved: bool,
    activity_name: str,
    start_time: str,
    end_time: str,
    location: str = "",
    description: str = "",
    recommendation_id: str = "",
    recommendation_reason: str = "",
    weather_context: str = "",
    calendar_id: str = DEFAULT_CALENDAR_ID,
    timezone_str: str = DEFAULT_TIMEZONE,
    reminder_minutes: int = 30,
    use_default_reminders: bool = False,
    buffer_minutes: int = 0,
) -> Dict[str, Any]:
    """
    Book an approved activity into the user's Google Calendar.
    Args:
        user_approved: Must be True. If False, no event is created.
        activity_name: Calendar event title, for example "Dinner at Arabian Tea House".
        start_time: Activity start time, for example "2026-05-07T18:00:00".
        end_time: Activity end time, for example "2026-05-07T20:00:00".
        location: Activity location or address.
        description: Main event description.
        recommendation_id: ID from the ML recommender, if available.
        recommendation_reason: Explanation of why this activity was recommended.
        weather_context: Weather suitability context used for the recommendation.
        calendar_id: Google Calendar ID. Use "primary" for the user's main calendar.
        timezone_str: IANA timezone name. Default is "Asia/Dubai".
        reminder_minutes: Popup reminder time before the event. Valid range is 0 to 40320.
        use_default_reminders: If True, use the calendar's default reminders.
        buffer_minutes: Optional buffer around the final availability check.

    Returns:
        A dictionary with booking status, event ID, event link, and event details.
    """

    tool_name = "book_activity"
    start_clock = log_tool_start(
        tool_name=tool_name,
        data={
            "user_approved": user_approved,
            "activity_name": activity_name,
            "start_time": start_time,
            "end_time": end_time,
            "calendar_id": calendar_id,
            "timezone_str": timezone_str,
        },
    )

    if not user_approved:
        result = {
            "status": "not_created",
            "message": "User did not approve the recommendation. Calendar event was not created.",
        }
        log_tool_success(
            tool_name=tool_name,
            start_time=start_clock,
            data={"status": "not_created"},
        )
        return result

    try:
        if reminder_minutes < 0 or reminder_minutes > 40320:
            raise ValueError("reminder_minutes must be between 0 and 40320.")

        # Re-check availability immediately before booking to reduce double-booking risk.
        availability = _check_calendar_availability_internal(
            start_time=start_time,
            end_time=end_time,
            calendar_id=calendar_id,
            timezone_str=timezone_str,
            buffer_minutes=buffer_minutes,
            allow_past=False,
        )

        if availability.get("status") != "success":
            result = {
                "status": "error",
                "message": "Could not confirm calendar availability before booking.",
                "availability_result": availability,
            }
            log_tool_error(
                tool_name=tool_name,
                start_time=start_clock,
                error=RuntimeError(result["message"]),
                data={"availability_status": availability.get("status")},
            )
            return result

        if not availability.get("is_free"):
            result = {
                "status": "not_created",
                "message": "The selected time slot is no longer free. Activity was not booked.",
                "availability_result": availability,
            }
            log_tool_success(
                tool_name=tool_name,
                start_time=start_clock,
                data={
                    "status": "not_created",
                    "availability_status": availability.get("status"),
                },
            )
            return result

        validated = _validate_time_range(
            start_time=start_time,
            end_time=end_time,
            timezone_str=timezone_str,
            allow_past=False,
        )

        start_rfc3339 = validated["start_rfc3339"]
        end_rfc3339 = validated["end_rfc3339"]

        # Build a Calendar API client after validation passes.
        service = _get_calendar_service(
            credentials_path=DEFAULT_CREDENTIALS_PATH,
            token_path=DEFAULT_TOKEN_PATH,
            allow_interactive_auth=False,
        )

        # Build the event payload that will be inserted into Google Calendar.
        event_body: Dict[str, Any] = {
            "summary": activity_name,
            "location": location,
            "description": _build_event_description(
                description=description,
                recommendation_reason=recommendation_reason,
                weather_context=weather_context,
                recommendation_id=recommendation_id,
            ),
            "start": {
                "dateTime": start_rfc3339,
                "timeZone": timezone_str,
            },
            "end": {
                "dateTime": end_rfc3339,
                "timeZone": timezone_str,
            },
            # Opaque blocks the calendar as busy.
            "transparency": "opaque",
            "extendedProperties": {
                "private": {
                    "source": "Lifestyle AI Agent",
                    "recommendation_id": recommendation_id,
                }
            },
        }

        if use_default_reminders:
            event_body["reminders"] = {"useDefault": True}
        else:
            # Use an explicit popup reminder when default reminders are disabled.
            event_body["reminders"] = {
                "useDefault": False,
                "overrides": [
                    {
                        "method": "popup",
                        "minutes": reminder_minutes,
                    }
                ],
            }

        # Insert the approved activity into the user's selected calendar.
        created_event = (
            service.events()
            .insert(
                calendarId=calendar_id,
                body=event_body,
            )
            .execute()
        )

        # Save the booking result and return a compact confirmation.
        result = {
            "status": "success",
            "message": "Activity booked successfully in Google Calendar.",
            "calendar_id": calendar_id,
            "event_id": created_event.get("id"),
            "event_title": created_event.get("summary"),
            "event_link": created_event.get("htmlLink"),
            "start_time": start_rfc3339,
            "end_time": end_rfc3339,
            "location": location,
            "recommendation_id": recommendation_id,
            "booked_at": datetime.now(timezone.utc).isoformat(),
        }

        _save_calendar_log(result)
        log_tool_success(
            tool_name=tool_name,
            start_time=start_clock,
            data={
                "status": "success",
                "event_id": result.get("event_id"),
                "calendar_id": calendar_id,
            },
        )

        return result

    except HttpError as error:
        result = {
            "status": "error",
            "message": f"Google Calendar API error: {error}",
        }
        log_tool_error(
            tool_name=tool_name,
            start_time=start_clock,
            error=error,
            data={"calendar_id": calendar_id},
        )
        return result

    except RuntimeError as error:
        result = {
            "status": "error",
            "message": str(error),
        }
        log_tool_error(
            tool_name=tool_name,
            start_time=start_clock,
            error=error,
            data={"calendar_id": calendar_id},
        )
        return result

    except Exception as error:
        result = {
            "status": "error",
            "message": f"Failed to book activity: {error}",
        }
        log_tool_error(
            tool_name=tool_name,
            start_time=start_clock,
            error=error,
            data={"calendar_id": calendar_id},
        )
        return result
