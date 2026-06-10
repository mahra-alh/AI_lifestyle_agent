"""
Reads from:
    auth.py  --> get_calendar_service()
    utils.py --> validate_time_range(), apply_buffer(), normalize_busy_slots()

Two public entry points:
    get_calendar(...)      — agent-facing tool: check a specific slot
    list_upcoming_events() — quick human-readable view of your calendar
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3]))  # adds project root to path

from datetime import datetime, timezone
from typing import Any, Dict, List
from agents import function_tool
from ai_agent.tools.calendar.auth import HttpError, get_calendar_service
from ai_agent.tools.calendar.config import DEFAULT_TIMEZONE
from ai_agent.tools.calendar.utils import (
    apply_buffer,
    normalize_busy_slots,
    validate_time_range,
)

# ---------------------------------------------------------------------------
# Public tool: check a specific time slot
# ---------------------------------------------------------------------------

@function_tool
def get_calendar(
    start_time: str,
    end_time: str,
    calendar_id: str = "primary",
    timezone_str: str = DEFAULT_TIMEZONE,
    buffer_minutes: int = 0,
) -> Dict[str, Any]:
    """
    Check whether the user's Google Calendar is free for a proposed slot.
    Call this before passing a time slot to the ML recommender or booking tool.

    Returns:
        Dict with keys:
            status            "success" | "error"
            is_free           bool — True if no conflicts were found.
            busy_slots        Raw UTC busy periods from Google.
            busy_slots_local  Same slots converted to timezone_str.
            message           Human-readable summary.
            … plus timing metadata for debugging.
    """
    return _check_availability(
        start_time=start_time,
        end_time=end_time,
        calendar_id=calendar_id,
        timezone_str=timezone_str,
        buffer_minutes=buffer_minutes,
        allow_past=False,
    )


# ---------------------------------------------------------------------------
# Public helper: list upcoming events (human-facing, not an agent tool)
# ---------------------------------------------------------------------------

def list_upcoming_events(
    max_results: int = 10,
    calendar_id: str = "primary",
    timezone_str: str = DEFAULT_TIMEZONE,
) -> List[Dict[str, Any]]:
    """
    Fetch and return the next N calendar events from now.

    Args:
        max_results:  Maximum number of events to return (default 10).
        calendar_id:  Google Calendar ID to query.
        timezone_str: IANA timezone used to display event times.

    Returns:
        List of dicts, each containing:
            title      Event summary / title.
            start      Start time as a local ISO string.
            end        End time as a local ISO string.
            location   Location string (may be empty).
            link       URL to open the event in Google Calendar.

    Raises:
        HttpError: Google Calendar API returned an error.
        RuntimeError: Not authenticated (run auth.py first).
    """
    service = get_calendar_service(allow_interactive_auth=False)

    now_utc = datetime.now(timezone.utc).isoformat()

    events_result = (
        service.events()
        .list(
            calendarId=calendar_id,
            timeMin=now_utc,
            maxResults=max_results,
            singleEvents=True,
            orderBy="startTime",
            timeZone=timezone_str,
        )
        .execute()
    )

    raw_events = events_result.get("items", [])
    return [_format_event(event, timezone_str) for event in raw_events]


# ---------------------------------------------------------------------------
# Internal: shared FreeBusy query used by get_calendar and booking.py
# ---------------------------------------------------------------------------

def _check_availability(
    start_time: str,
    end_time: str,
    calendar_id: str = "primary",
    timezone_str: str = DEFAULT_TIMEZONE,
    buffer_minutes: int = 0,
    allow_past: bool = False,
) -> Dict[str, Any]:
    """
    Internal FreeBusy check shared by get_calendar() and booking.py.

    booking.py can re-check availability immediately before creating an event
    without triggering the agent tool wrapper again.
    """
    try:
        validated = validate_time_range(
            start_time=start_time,
            end_time=end_time,
            timezone_str=timezone_str,
            allow_past=allow_past,
        )

        buffered = apply_buffer(
            start_dt=validated["start_dt"],
            end_dt=validated["end_dt"],
            buffer_minutes=buffer_minutes,
        )

        checked_start = buffered["checked_start_dt"].isoformat()
        checked_end = buffered["checked_end_dt"].isoformat()

        service = get_calendar_service(allow_interactive_auth=False)

        freebusy_result = (
            service.freebusy()
            .query(
                body={
                    "timeMin": checked_start,
                    "timeMax": checked_end,
                    "timeZone": timezone_str,
                    "items": [{"id": calendar_id}],
                }
            )
            .execute()
        )

        calendar_info = freebusy_result.get("calendars", {}).get(calendar_id, {})
        api_errors = calendar_info.get("errors", [])

        if api_errors:
            return {
                "status": "error",
                "message": "Google Calendar returned errors while checking availability.",
                "calendar_errors": api_errors,
                "is_free": False,
                "busy_slots": [],
                "busy_slots_local": [],
                **_timing_meta(validated, checked_start, checked_end, buffer_minutes),
            }

        busy_slots = calendar_info.get("busy", [])
        is_free = len(busy_slots) == 0

        return {
            "status": "success",
            "calendar_id": calendar_id,
            "timezone": timezone_str,
            "is_free": is_free,
            "busy_slots": busy_slots,
            "busy_slots_local": normalize_busy_slots(busy_slots, timezone_str),
            "message": (
                "The user is free during this time slot."
                if is_free
                else "The user is busy during this time slot."
            ),
            **_timing_meta(validated, checked_start, checked_end, buffer_minutes),
        }

    except HttpError as error:
        return {
            "status": "error",
            "message": f"Google Calendar API error: {error}",
            "is_free": False,
            "busy_slots": [],
        }
    except (RuntimeError, ValueError) as error:
        return {
            "status": "error",
            "message": str(error),
            "is_free": False,
            "busy_slots": [],
        }
    except Exception as error:
        return {
            "status": "error",
            "message": f"Availability check failed: {error}",
            "is_free": False,
            "busy_slots": [],
        }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _timing_meta(
    validated: Dict[str, Any],
    checked_start: str,
    checked_end: str,
    buffer_minutes: int,
) -> Dict[str, Any]:
    """Return timing metadata fields shared across success and error responses."""
    return {
        "requested_start_time": validated["start_rfc3339"],
        "requested_end_time": validated["end_rfc3339"],
        "checked_start_time": checked_start,
        "checked_end_time": checked_end,
        "buffer_minutes": buffer_minutes,
    }


def _format_event(event: Dict[str, Any], timezone_str: str) -> Dict[str, Any]:
    """Convert a raw Google Calendar event dict into a compact display format."""
    start_raw = event.get("start", {})
    end_raw = event.get("end", {})

    # All-day events use "date"; timed events use "dateTime".
    start = start_raw.get("dateTime") or start_raw.get("date", "")
    end = end_raw.get("dateTime") or end_raw.get("date", "")

    return {
        "title": event.get("summary", "(No title)"),
        "start": start,
        "end": end,
        "location": event.get("location", ""),
        "link": event.get("htmlLink", ""),
    }


# ---------------------------------------------------------------------------
# CLI entry point — run to view your upcoming events
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n📅  Upcoming events\n" + "─" * 40)

    try:
        events = list_upcoming_events(max_results=10)
    except RuntimeError as exc:
        print(f"\n[error] {exc}")
        print("Run  python auth.py first to authenticate.\n")
        raise SystemExit(1) from exc

    if not events:
        print("No upcoming events found.")
    else:
        for i, evt in enumerate(events, start=1):
            print(f"\n{i}. {evt['title']}")
            print(f"   Start    : {evt['start']}")
            print(f"   End      : {evt['end']}")
            if evt["location"]:
                print(f"   Location : {evt['location']}")
            if evt["link"]:
                print(f"   Link     : {evt['link']}")

    print("\n" + "─" * 40)