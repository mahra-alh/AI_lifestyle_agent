"""
Agent-facing calendar tools.

Delegates to availability.py and booking.py; adds OpenAI Agents decorators and app logging.
Install third-party deps via requirements.txt in this folder.
"""

from __future__ import annotations

from typing import Any, Dict

from agents import function_tool

from ai_agent.logger.app_logger import log_tool_error, log_tool_start, log_tool_success
from .availability import get_calendar as _get_calendar
from .booking import book_activity as _book_activity
from .utils import DEFAULT_TIMEZONE

DEFAULT_CALENDAR_ID = "primary"


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
        result = _get_calendar(
            start_time=start_time,
            end_time=end_time,
            calendar_id=calendar_id,
            timezone_str=timezone_str,
            buffer_minutes=buffer_minutes,
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
                error=RuntimeError(
                    result.get("message", "Calendar availability check returned error status.")
                ),
                data={
                    "calendar_id": calendar_id,
                    "status": result.get("status"),
                },
            )

        return result

    except Exception as error:
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
    user_approved must be True or no event is created.
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

    try:
        result = _book_activity(
            user_approved=user_approved,
            activity_name=activity_name,
            start_time=start_time,
            end_time=end_time,
            location=location,
            description=description,
            recommendation_id=recommendation_id,
            recommendation_reason=recommendation_reason,
            weather_context=weather_context,
            calendar_id=calendar_id,
            timezone_str=timezone_str,
            reminder_minutes=reminder_minutes,
            use_default_reminders=use_default_reminders,
            buffer_minutes=buffer_minutes,
        )

        status = result.get("status")
        if status in ("success", "not_created"):
            log_tool_success(
                tool_name=tool_name,
                start_time=start_clock,
                data={
                    "status": status,
                    "event_id": result.get("event_id"),
                    "calendar_id": calendar_id,
                },
            )
        else:
            log_tool_error(
                tool_name=tool_name,
                start_time=start_clock,
                error=RuntimeError(result.get("message", "Booking failed.")),
                data={"calendar_id": calendar_id, "status": status},
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
