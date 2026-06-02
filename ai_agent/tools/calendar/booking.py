"""
Reads from:
    auth.py --> get_calendar_service()
    utils.py --> validate_time_range()
    availability.py --> _check_availability()
    logging.py --> save_calendar_log()

One public entry point:
    book_activity(...) — books an approved activity into Google Calendar.

The function re-checks availability immediately before writing the event
to reduce double-booking risk. It will not create an event unless
user_approved=True.

Run this file directly for a dry-run booking test:
    python booking.py
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

# Pipeline imports — all three upstream modules must be working first
from .auth import HttpError, get_calendar_service
from .utils import validate_time_range 
from .availability import _check_availability
from .calendar_logger import save_calendar_log
from .config import DEFAULT_TIMEZONE


# Public tool: book an approved activity

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
    calendar_id: str = "primary",
    timezone_str: str = DEFAULT_TIMEZONE,
    reminder_minutes: int = 30,
    use_default_reminders: bool = False,
    buffer_minutes: int = 0,
) -> Dict[str, Any]:
    """
    Book an approved activity into the user's Google Calendar.

    Will not create an event unless user_approved=True.
    Re-checks availability immediately before booking to reduce
    double-booking risk.

    Args:
        user_approved:        Must be True — safety gate so the agent cannot
                              book without explicit user confirmation.
        activity_name:        Calendar event title, e.g. "Dinner at Arabian Tea House".
        start_time:           Activity start, e.g. "2026-05-07T18:00:00".
        end_time:             Activity end,   e.g. "2026-05-07T20:00:00".
        location:             Venue address or name.
        description:          Main event description.
        recommendation_id:    ID from the ML recommender, if available.
        recommendation_reason: Why this activity was recommended.
        weather_context:      Weather suitability note used in the recommendation.
        calendar_id:          Google Calendar ID. "primary" = main calendar.
        timezone_str:         IANA timezone for naive datetime strings.
        reminder_minutes:     Popup reminder before the event (0–40320 minutes).
        use_default_reminders: Use the calendar's default reminders instead.
        buffer_minutes:       Extra minutes to check around the slot.

    Returns:
        Dict with keys:
            status      "success" | "not_created" | "error"
            message     Human-readable summary.
            event_id    Google Calendar event ID (on success).
            event_link  URL to the event in Google Calendar (on success).
            … plus timing and recommendation metadata.
    """
    # Hard gate — never book without explicit approval.
    if not user_approved:
        return {
            "status": "not_created",
            "message": "User did not approve the activity. No event was created.",
        }

    try:
        if not 0 <= reminder_minutes <= 40320:
            raise ValueError("reminder_minutes must be between 0 and 40320.")

        # Re-check availability right before writing to reduce double-booking.
        availability = _check_availability(
            start_time=start_time,
            end_time=end_time,
            calendar_id=calendar_id,
            timezone_str=timezone_str,
            buffer_minutes=buffer_minutes,
            allow_past=False,
        )

        if availability.get("status") != "success":
            return {
                "status": "error",
                "message": "Could not confirm availability before booking.",
                "availability_result": availability,
            }

        if not availability.get("is_free"):
            return {
                "status": "not_created",
                "message": "The time slot is no longer free. Activity was not booked.",
                "availability_result": availability,
            }

        validated = validate_time_range(
            start_time=start_time,
            end_time=end_time,
            timezone_str=timezone_str,
            allow_past=False,
        )

        service = get_calendar_service(allow_interactive_auth=False)

        event_body = _build_event_body(
            activity_name=activity_name,
            start_rfc3339=validated["start_rfc3339"],
            end_rfc3339=validated["end_rfc3339"],
            timezone_str=timezone_str,
            location=location,
            description=description,
            recommendation_id=recommendation_id,
            recommendation_reason=recommendation_reason,
            weather_context=weather_context,
            reminder_minutes=reminder_minutes,
            use_default_reminders=use_default_reminders,
        )

        created_event = (
            service.events()
            .insert(calendarId=calendar_id, body=event_body)
            .execute()
        )

        result = {
            "status": "success",
            "message": "Activity booked successfully in Google Calendar.",
            "calendar_id": calendar_id,
            "event_id": created_event.get("id"),
            "event_title": created_event.get("summary"),
            "event_link": created_event.get("htmlLink"),
            "start_time": validated["start_rfc3339"],
            "end_time": validated["end_rfc3339"],
            "location": location,
            "recommendation_id": recommendation_id,
            "booked_at": datetime.now(timezone.utc).isoformat(),
        }

        save_calendar_log(result)
        return result

    except HttpError as error:
        return {
            "status": "error",
            "message": f"Google Calendar API error: {error}",
        }
    except (RuntimeError, ValueError) as error:
        return {
            "status": "error",
            "message": str(error),
        }
    except Exception as error:
        return {
            "status": "error",
            "message": f"Failed to book activity: {error}",
        }


# Private helpers
def _build_event_description(
    description: str,
    recommendation_reason: str,
    weather_context: str,
    recommendation_id: str,
) -> str:
    """Assemble the event description from optional recommendation context."""
    parts: List[str] = []

    if description:
        parts.append(description)
    if recommendation_reason:
        parts.append(f"Recommendation reason: {recommendation_reason}")
    if weather_context:
        parts.append(f"Weather context: {weather_context}")
    if recommendation_id:
        parts.append(f"Recommendation ID: {recommendation_id}")

    parts.append("Booked by Lifestyle AI Agent.")

    return "\n\n".join(parts)


def _build_event_body(
    activity_name: str,
    start_rfc3339: str,
    end_rfc3339: str,
    timezone_str: str,
    location: str,
    description: str,
    recommendation_id: str,
    recommendation_reason: str,
    weather_context: str,
    reminder_minutes: int,
    use_default_reminders: bool,
) -> Dict[str, Any]:
    """Build the Google Calendar API event payload."""
    body: Dict[str, Any] = {
        "summary": activity_name,
        "location": location,
        "description": _build_event_description(
            description=description,
            recommendation_reason=recommendation_reason,
            weather_context=weather_context,
            recommendation_id=recommendation_id,
        ),
        "start": {"dateTime": start_rfc3339, "timeZone": timezone_str},
        "end": {"dateTime": end_rfc3339, "timeZone": timezone_str},
        # opaque = block the calendar as busy.
        "transparency": "opaque",
        "extendedProperties": {
            "private": {
                "source": "Lifestyle AI Agent",
                "recommendation_id": recommendation_id,
            }
        },
    }

    if use_default_reminders:
        body["reminders"] = {"useDefault": True}
    else:
        body["reminders"] = {
            "useDefault": False,
            "overrides": [{"method": "popup", "minutes": reminder_minutes}],
        }

    return body


# CLI entry point — dry-run to test booking without agent

if __name__ == "__main__":
    print("\n📅  Booking dry-run\n" + "─" * 40)
    print("Modify the values below and run to test a real booking.\n")

    result = book_activity(
        user_approved=True,
        activity_name="Test Event — Lifestyle AI Agent",
        start_time="2026-06-10T18:00:00",
        end_time="2026-06-10T19:00:00",
        location="Dubai, UAE",
        description="This is a test booking from booking.py.",
        reminder_minutes=30,
    )

    print(f"Status  : {result.get('status')}")
    print(f"Message : {result.get('message')}")

    if result.get("event_link"):
        print(f"Link    : {result.get('event_link')}")

    print("\n" + "─" * 40)
