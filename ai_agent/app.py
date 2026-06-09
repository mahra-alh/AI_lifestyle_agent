"""
ai_agent/app.py
================

Agent entry point. Defines the agent and exposes run_agent() for the
backend to call.
"""
import os
from datetime import datetime
from zoneinfo import ZoneInfo

from agents import Agent, Runner
from dotenv import load_dotenv

from ai_agent.logger.app_logger import create_trace_id, log_event, set_trace_id
from ai_agent.tools.calendar.free_slots import get_free_slots
from old_code.full_calendar_tool import book_activity, get_calendar
from ai_agent.tools.recommendation_tool import get_recommendations
from ai_agent.tools.user_profile_pkg.tools import (
    get_user_profile,
    log_activity_preference,
    log_recommendation_feedback,
    update_user_profile,
)
from ai_agent.tools.weather.tool import get_weather_forecast

load_dotenv()

APP_TIMEZONE = "Asia/Dubai"

# ---------------------------------------------------------------------------
# Agent definition
# ---------------------------------------------------------------------------

agent = Agent(
    name="Lifestyle AI Agent",
    instructions="""
You are a Dubai lifestyle planning assistant.

The user's profile is always complete before they reach this chat — never ask for profile information.
The user's email is already provided as the Known user_id. Never ask for it.

When the user asks for activity suggestions:
1. Call get_user_profile with the known user_id to load their saved preferences.
2. Call get_weather_forecast to check today's weather conditions.
3. Call get_calendar to check calendar availability for the requested time.
4. Call get_ml_recommendations with the profile, weather, and calendar data.
5. Present the ranked results clearly. Ask which option they want to book.
6. Call book_activity only after the user confirms.

If get_ml_recommendations returns no results or an error:
- Tell the user recommendations are temporarily unavailable.
- Do NOT invent or guess activity names. Do not fabricate venues.

When the user reacts to a suggestion, log it:
- Clicks or bookings → log_activity_preference
- Explicit feedback (liked/disliked) → log_recommendation_feedback

Rules:
- Never ask for profile details — the profile is already saved.
- Never invent venue names or activities.
- If the requested time is busy in the calendar, suggest a different free slot.
- If outdoor_comfort_flag is poor_for_outdoor, recommend indoor options only.
- Use Asia/Dubai timezone. 9 pm = 21:00, 9 am = 09:00.
- If the user's time is ambiguous, ask one short clarification.
- Never book without explicit user confirmation.
- Keep responses short, practical, and action-oriented..
""",
    tools=[
        get_user_profile,
        update_user_profile,
        log_activity_preference,
        log_recommendation_feedback,
        get_weather_forecast,
        get_calendar,
        get_free_slots,
        get_recommendations,
        book_activity,
    ],
    model="gpt-4o-mini",
)

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_agent(user_id: str, user_message: str) -> str:
    trace_id = create_trace_id()
    set_trace_id(trace_id)

    log_event(
        event_name="agent_request_started",
        user_id=user_id,
        data={"input_length": len(user_message)},
    )

    try:
        now_dubai = datetime.now(ZoneInfo(APP_TIMEZONE)).isoformat()

        result = Runner.run_sync(
            agent,
            input=(
                f"Known user_id/email for tool calls: {user_id}\n\n"
                f"Current local time ({APP_TIMEZONE}): {now_dubai}\n\n"
                f"User message:\n{user_message}"
            ),
        )

        log_event(
            event_name="agent_request_completed",
            user_id=user_id,
            data={
                "output_length": len(result.final_output),
                "status": "success",
            },
        )
        return result.final_output

    except Exception as error:
        log_event(
            event_name="agent_request_failed",
            level="ERROR",
            user_id=user_id,
            data={
                "status": "failed",
                "error_type": type(error).__name__,
                "error_message": "Agent request failed.",
            },
        )
        raise


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    user_id = input("Email: ").strip()
    if not user_id:
        raise ValueError("Email is required.")

    user_message = input("Message: ").strip()
    if not user_message:
        raise ValueError("Message is required.")

    result = run_agent(user_id=user_id, user_message=user_message)
    print(result)

    decision = input("Accept recommendations? (accept/decline): ").strip().lower()
    if decision in {"accept", "decline"}:
        follow_up = (
            "I accept the recommendations."
            if decision == "accept"
            else "I decline the recommendations."
        )
        print(run_agent(user_id=user_id, user_message=follow_up))


if __name__ == "__main__":
    main()
