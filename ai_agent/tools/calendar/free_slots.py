"""
get_free_slots — proactively scans a date range and returns available
time windows, rather than checking a single pre-specified slot.

Use this when the user says something like:
  "Find me a free 2-hour window this weekend"
  "When am I free on Friday evening?"
  "What time can I fit something in tomorrow?"

The tool calls the Google Calendar FreeBusy API for the requested date
range, inverts the busy periods, and filters the gaps down to windows
that are at least min_duration_minutes long.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from agents import function_tool

from ai_agent.logger.app_logger import log_tool_error, log_tool_start, log_tool_success
from .auth import get_calendar_service, HttpError
from .config import DEFAULT_TIMEZONE
from .utils import get_timezone


# Reasonable bounds for a single scan to avoid very large API queries.
_MAX_SCAN_DAYS = 7

# Default window the agent looks within each day (avoids suggesting 2am slots).
_DEFAULT_DAY_START_HOUR = 8    # 8:00 am
_DEFAULT_DAY_END_HOUR = 23     # 11:00 pm


@function_tool
def get_free_slots(
    date_from: str,
    date_to: str,
    min_duration_minutes: int = 60,
    calendar_id: str = "primary",
    timezone_str: str = DEFAULT_TIMEZONE,
    day_start_hour: int = _DEFAULT_DAY_START_HOUR,
    day_end_hour: int = _DEFAULT_DAY_END_HOUR,
) -> Dict[str, Any]:
    """
    Find available time windows within a date range.

    Use this when you need to discover when the user is free, rather than
    checking a specific slot. Returns a list of free windows sorted by
    start time, each at least min_duration_minutes long.

    Args:
        date_from:             Start of the scan range (YYYY-MM-DD).
        date_to:               End of the scan range, inclusive (YYYY-MM-DD).
                               Maximum 7 days after date_from.
        min_duration_minutes:  Minimum gap length to include (default 60 min).
        calendar_id:           Google Calendar ID to query (default "primary").
        timezone_str:          IANA timezone for interpreting dates and
                               displaying results (default "Asia/Dubai").
        day_start_hour:        Earliest hour to include in results (default 8).
        day_end_hour:          Latest hour to include in results (default 23).

    Returns:
        Dict with keys:
            status          "success" | "error"
            free_slots      List of {"start": ISO, "end": ISO, "duration_minutes": int}
            busy_slots      Raw busy periods from Google (UTC).
            date_from       The scan start date used.
            date_to         The scan end date used.
            timezone        The timezone used for results.
            message         Human-readable summary.
    """
    tool_name = "get_free_slots"
    start_clock = log_tool_start(
        tool_name=tool_name,
        data={
            "date_from": date_from,
            "date_to": date_to,
            "min_duration_minutes": min_duration_minutes,
            "timezone_str": timezone_str,
        },
    )

    try:
        tz = get_timezone(timezone_str)

        # Parse and validate date range.
        scan_start, scan_end = _parse_date_range(
            date_from, date_to, tz,
            day_start_hour, day_end_hour,
        )

        # Fetch busy periods from Google Calendar FreeBusy API.
        service = get_calendar_service(allow_interactive_auth=False)
        freebusy_result = (
            service.freebusy()
            .query(
                body={
                    "timeMin": scan_start.isoformat(),
                    "timeMax": scan_end.isoformat(),
                    "timeZone": timezone_str,
                    "items": [{"id": calendar_id}],
                }
            )
            .execute()
        )

        calendar_info = freebusy_result.get("calendars", {}).get(calendar_id, {})
        api_errors = calendar_info.get("errors", [])
        if api_errors:
            return _error_response("Google Calendar returned errors.", api_errors)

        busy_slots_raw = calendar_info.get("busy", [])
        busy_periods = _parse_busy_periods(busy_slots_raw)

        # Compute free windows across the scanned days.
        free_slots = _compute_free_slots(
            scan_start=scan_start,
            scan_end=scan_end,
            busy_periods=busy_periods,
            min_duration_minutes=min_duration_minutes,
            tz=tz,
            day_start_hour=day_start_hour,
            day_end_hour=day_end_hour,
        )

        # Format output.
        free_slots_out = [
            {
                "start": slot["start"].isoformat(),
                "end": slot["end"].isoformat(),
                "duration_minutes": int(
                    (slot["end"] - slot["start"]).total_seconds() / 60
                ),
            }
            for slot in free_slots
        ]
        busy_slots_out = [
            {
                "start": _utc_to_local(s["start"], tz),
                "end": _utc_to_local(s["end"], tz),
            }
            for s in busy_slots_raw
        ]

        message = (
            f"Found {len(free_slots_out)} free window(s) "
            f"of at least {min_duration_minutes} minutes "
            f"between {date_from} and {date_to}."
            if free_slots_out
            else (
                f"No free windows of at least {min_duration_minutes} minutes "
                f"found between {date_from} and {date_to}."
            )
        )

        log_tool_success(
            tool_name=tool_name,
            start_time=start_clock,
            data={
                "free_slots_count": len(free_slots_out),
                "busy_slots_count": len(busy_slots_raw),
            },
        )

        return {
            "status": "success",
            "free_slots": free_slots_out,
            "busy_slots": busy_slots_out,
            "date_from": date_from,
            "date_to": date_to,
            "timezone": timezone_str,
            "message": message,
        }

    except (RuntimeError, ValueError) as error:
        log_tool_error(tool_name=tool_name, start_time=start_clock, error=error)
        return _error_response(str(error))

    except HttpError as error:
        log_tool_error(tool_name=tool_name, start_time=start_clock, error=error)
        return _error_response(f"Google Calendar API error: {error}")

    except Exception as error:
        log_tool_error(tool_name=tool_name, start_time=start_clock, error=error)
        return _error_response(f"Free slot scan failed: {error}")


def _parse_date_range(
    date_from: str,
    date_to: str,
    tz: ZoneInfo,
    day_start_hour: int,
    day_end_hour: int,
) -> tuple[datetime, datetime]:
    """Parse and validate the scan date range."""
    try:
        start_date = datetime.fromisoformat(date_from).date()
        end_date = datetime.fromisoformat(date_to).date()
    except ValueError as e:
        raise ValueError(
            f"Invalid date format: {e}. Use YYYY-MM-DD."
        ) from e

    if end_date < start_date:
        raise ValueError("date_to must be on or after date_from.")

    if (end_date - start_date).days > _MAX_SCAN_DAYS:
        end_date = start_date + timedelta(days=_MAX_SCAN_DAYS)

    scan_start = datetime(
        start_date.year, start_date.month, start_date.day,
        day_start_hour, 0, 0, tzinfo=tz,
    )
    scan_end = datetime(
        end_date.year, end_date.month, end_date.day,
        day_end_hour, 0, 0, tzinfo=tz,
    )
    return scan_start, scan_end


def _parse_busy_periods(
    raw_busy: List[Dict[str, str]],
) -> List[Dict[str, datetime]]:
    """Parse raw FreeBusy busy slots into datetime objects."""
    periods = []
    for slot in raw_busy:
        try:
            start = datetime.fromisoformat(slot["start"].replace("Z", "+00:00"))
            end = datetime.fromisoformat(slot["end"].replace("Z", "+00:00"))
            periods.append({"start": start, "end": end})
        except (KeyError, ValueError):
            continue
    return sorted(periods, key=lambda p: p["start"])


def _compute_free_slots(
    scan_start: datetime,
    scan_end: datetime,
    busy_periods: List[Dict[str, datetime]],
    min_duration_minutes: int,
    tz: ZoneInfo,
    day_start_hour: int,
    day_end_hour: int,
) -> List[Dict[str, datetime]]:
    """
    Invert busy periods within the scan window to produce free slots.

    Works day-by-day to enforce the day_start_hour / day_end_hour bounds
    so results never include overnight gaps.
    """
    free_slots: List[Dict[str, datetime]] = []
    min_delta = timedelta(minutes=min_duration_minutes)

    # Iterate day by day.
    current_date = scan_start.date()
    end_date = scan_end.date()

    while current_date <= end_date:
        # The usable window for this day.
        day_start = datetime(
            current_date.year, current_date.month, current_date.day,
            day_start_hour, 0, 0, tzinfo=tz,
        )
        day_end = datetime(
            current_date.year, current_date.month, current_date.day,
            day_end_hour, 0, 0, tzinfo=tz,
        )

        # Clamp to the overall scan window.
        window_start = max(day_start, scan_start)
        window_end = min(day_end, scan_end)

        if window_start >= window_end:
            current_date += timedelta(days=1)
            continue

        # Walk through busy periods that overlap this day.
        cursor = window_start
        for period in busy_periods:
            busy_start = period["start"].astimezone(tz)
            busy_end = period["end"].astimezone(tz)

            # Skip busy periods entirely outside the day window.
            if busy_end <= window_start or busy_start >= window_end:
                continue

            # Gap before this busy period.
            gap_end = min(busy_start, window_end)
            if gap_end > cursor and (gap_end - cursor) >= min_delta:
                free_slots.append({"start": cursor, "end": gap_end})

            cursor = max(cursor, busy_end)

        # Gap after the last busy period for this day.
        if window_end > cursor and (window_end - cursor) >= min_delta:
            free_slots.append({"start": cursor, "end": window_end})

        current_date += timedelta(days=1)

    return free_slots


def _utc_to_local(iso_str: str, tz: ZoneInfo) -> str:
    """Convert a UTC ISO string to local timezone ISO string."""
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.astimezone(tz).isoformat()
    except ValueError:
        return iso_str


def _error_response(
    message: str,
    calendar_errors: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "status": "error",
        "message": message,
        "free_slots": [],
        "busy_slots": [],
    }
    if calendar_errors:
        result["calendar_errors"] = calendar_errors
    return result
