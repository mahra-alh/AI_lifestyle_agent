"""
Reads from:
    auth.py --> get_calendar_service()
    utils.py --> validate_time_range()
    availability.py --> _check_availability()
    calendar_logger.py --> save_calendar_log()
    firestore_store.py --> pending booking records

Two-step booking flow (the calendar write can only be triggered by an
explicit user click in the UI, never by the model alone):

    1. book_activity(...)            — agent-facing tool: validates the slot
       and creates a PENDING booking record. Nothing is written to the
       calendar at this step.
    2. confirm_pending_booking(...)  — called by the API when the user clicks
       Confirm in the app. Re-checks availability and writes the real event.
       cancel_pending_booking(...)   — called when the user clicks Cancel.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

# Pipeline imports — all three upstream modules must be working first
from agents import function_tool

from .auth import HttpError, get_calendar_service
from .utils import validate_time_range
from .availability import _check_availability
from .calendar_logger import save_calendar_log
from .config import DEFAULT_TIMEZONE

from ai_agent.storage.firestore_store import (
    create_pending_booking,
    get_pending_booking,
    update_pending_booking,
)

# Pending bookings expire if not confirmed within this window.
PENDING_EXPIRY_MINUTES = 15


# Step 1 — prepare a pending booking (no calendar write yet)

def prepare_booking(
    user_id: str,
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
    Validate the slot and store a pending booking awaiting user confirmation.

    Returns a dict with status "pending_confirmation" and the pending_id the
    UI needs to confirm or cancel. Plain function — also wrapped as the
    book_activity agent tool below.
    """
    try:
        if not 0 <= reminder_minutes <= 40320:
            raise ValueError("reminder_minutes must be between 0 and 40320.")

        validated = validate_time_range(
            start_time=start_time,
            end_time=end_time,
            timezone_str=timezone_str,
            allow_past=False,
        )

        # Check availability now so the user is never asked to confirm a
        # slot that is already busy.
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
                "message": "Could not confirm availability before preparing the booking.",
                "availability_result": availability,
            }

        if not availability.get("is_free"):
            return {
                "status": "not_created",
                "message": "The time slot is not free. No booking was prepared.",
                "availability_result": availability,
            }

        now_utc = datetime.now(timezone.utc)
        pending_id = f"pb_{uuid.uuid4().hex[:12]}"

        record: Dict[str, Any] = {
            "pending_id": pending_id,
            "user_id": user_id.strip().lower(),
            "activity_name": activity_name,
            "start_time": validated["start_rfc3339"],
            "end_time": validated["end_rfc3339"],
            "location": location,
            "description": description,
            "recommendation_id": recommendation_id,
            "recommendation_reason": recommendation_reason,
            "weather_context": weather_context,
            "calendar_id": calendar_id,
            "timezone_str": timezone_str,
            "reminder_minutes": reminder_minutes,
            "use_default_reminders": use_default_reminders,
            "buffer_minutes": buffer_minutes,
            "status": "pending",
            "created_at": now_utc.isoformat(),
            "expires_at": (
                now_utc + timedelta(minutes=PENDING_EXPIRY_MINUTES)
            ).isoformat(),
        }

        create_pending_booking(record)

        return {
            "status": "pending_confirmation",
            "pending_id": pending_id,
            "message": (
                "Booking prepared but NOT confirmed yet. A confirmation card is "
                "now shown to the user in the app. Tell the user to confirm or "
                "cancel there. Do not say the booking is complete."
            ),
            "activity_name": activity_name,
            "start_time": validated["start_rfc3339"],
            "end_time": validated["end_rfc3339"],
            "location": location,
            "expires_in_minutes": PENDING_EXPIRY_MINUTES,
        }

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
            "message": f"Failed to prepare booking: {error}",
        }


# Public tool: prepare a booking for user confirmation

@function_tool
def book_activity(
    user_id: str,
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
    Prepare an activity booking for the user's Google Calendar.

    IMPORTANT: this does NOT write to the calendar. It creates a pending
    booking that the user must confirm with the confirmation card shown in
    the app. After calling this, tell the user to confirm or cancel there,
    and never claim the booking is complete.

    Args:
        user_id:              The user's email/ID (the Known user_id).
        activity_name:        Calendar event title, e.g. "Dinner at Arabian Tea House".
        start_time:           Activity start, e.g. "2026-06-12T18:00:00". Must be
                              a time the user explicitly stated.
        end_time:             Activity end,   e.g. "2026-06-12T20:00:00".
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
            status      "pending_confirmation" | "not_created" | "error"
            pending_id  ID of the pending booking (on pending_confirmation).
            message     Human-readable summary to relay to the user.
    """
    return prepare_booking(
        user_id=user_id,
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


# Step 2 — confirm or cancel (called by the API on a user click, never by the model)

def confirm_pending_booking(pending_id: str) -> Dict[str, Any]:
    """
    Execute a pending booking after the user explicitly confirmed in the UI.

    Re-checks availability immediately before writing the event to reduce
    double-booking risk. Updates the pending record's status on success.
    """
    record = get_pending_booking(pending_id)

    if record is None:
        return {"status": "error", "message": "Pending booking not found."}

    if record.get("status") != "pending":
        return {
            "status": "error",
            "message": f"This booking is already {record.get('status')}.",
        }

    expires_at = datetime.fromisoformat(record["expires_at"])
    if datetime.now(timezone.utc) > expires_at:
        update_pending_booking(pending_id, {"status": "expired"})
        return {
            "status": "error",
            "message": "This booking request has expired. Please ask the agent to prepare it again.",
        }

    result = _execute_booking(record)

    if result.get("status") == "success":
        update_pending_booking(
            pending_id,
            {
                "status": "confirmed",
                "confirmed_at": datetime.now(timezone.utc).isoformat(),
                "event_id": result.get("event_id"),
                "event_link": result.get("event_link"),
            },
        )

    return result


def cancel_pending_booking(pending_id: str) -> Dict[str, Any]:
    """Mark a pending booking as cancelled. No calendar event is created."""
    record = get_pending_booking(pending_id)

    if record is None:
        return {"status": "error", "message": "Pending booking not found."}

    if record.get("status") != "pending":
        return {
            "status": "error",
            "message": f"This booking is already {record.get('status')}.",
        }

    update_pending_booking(
        pending_id,
        {
            "status": "cancelled",
            "cancelled_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return {
        "status": "cancelled",
        "message": "Booking cancelled. No event was created.",
        "pending_id": pending_id,
    }


def _execute_booking(record: Dict[str, Any]) -> Dict[str, Any]:
    """Write the calendar event for a confirmed pending booking record."""
    try:
        # Re-check availability right before writing to reduce double-booking.
        availability = _check_availability(
            start_time=record["start_time"],
            end_time=record["end_time"],
            calendar_id=record["calendar_id"],
            timezone_str=record["timezone_str"],
            buffer_minutes=record.get("buffer_minutes", 0),
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

        service = get_calendar_service(allow_interactive_auth=False)

        event_body = _build_event_body(
            activity_name=record["activity_name"],
            start_rfc3339=record["start_time"],
            end_rfc3339=record["end_time"],
            timezone_str=record["timezone_str"],
            location=record.get("location", ""),
            description=record.get("description", ""),
            recommendation_id=record.get("recommendation_id", ""),
            recommendation_reason=record.get("recommendation_reason", ""),
            weather_context=record.get("weather_context", ""),
            reminder_minutes=record.get("reminder_minutes", 30),
            use_default_reminders=record.get("use_default_reminders", False),
        )

        created_event = (
            service.events()
            .insert(calendarId=record["calendar_id"], body=event_body)
            .execute()
        )

        result = {
            "status": "success",
            "message": "Activity booked successfully in Google Calendar.",
            "calendar_id": record["calendar_id"],
            "event_id": created_event.get("id"),
            "event_title": created_event.get("summary"),
            "event_link": created_event.get("htmlLink"),
            "start_time": record["start_time"],
            "end_time": record["end_time"],
            "location": record.get("location", ""),
            "recommendation_id": record.get("recommendation_id", ""),
            "pending_id": record["pending_id"],
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


# CLI entry point — dry-run of the full prepare-then-confirm flow

if __name__ == "__main__":
    print("\n📅  Booking dry-run (two-step flow)\n" + "─" * 40)
    print("Modify the values below and run to test a real booking.\n")

    prepared = prepare_booking(
        user_id="test@example.com",
        activity_name="Test Event — Lifestyle AI Agent",
        start_time="2026-06-12T18:00:00",
        end_time="2026-06-12T19:00:00",
        location="Dubai, UAE",
        description="This is a test booking from booking.py.",
        reminder_minutes=30,
    )

    print(f"Prepare status  : {prepared.get('status')}")
    print(f"Prepare message : {prepared.get('message')}")

    if prepared.get("status") == "pending_confirmation":
        result = confirm_pending_booking(prepared["pending_id"])
        print(f"Confirm status  : {result.get('status')}")
        print(f"Confirm message : {result.get('message')}")
        if result.get("event_link"):
            print(f"Link            : {result.get('event_link')}")

    print("\n" + "─" * 40)
