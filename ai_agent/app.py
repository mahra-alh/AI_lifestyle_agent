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
from ai_agent.tools.calendar.full_calendar_tool import book_activity, get_calendar
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

Your job is to help users discover and schedule suitable activities based on
their profile, preferences, budget, location, weather, and calendar availability.

The user's profile is identified by email address.

Available tools:
- get_user_profile / update_user_profile
- log_activity_preference / log_recommendation_feedback
- get_weather_forecast
- get_calendar
- get_free_slots
- get_recommendations
- book_activity

Main flow:

1. Ask for the user's email address first. Do not proceed without it.
2. Call get_user_profile using the email as user_id.
   - If profile_complete is true: continue to activity planning.
   - If profile_complete is false: ask only for the missing_fields returned.
   - If no profile: ask for required_fields, save with update_user_profile, then continue.
3. Call get_weather_forecast for the user's area.
4. If the user specifies a time: call get_calendar to confirm the slot is free.
   If the user says "when am I free" or does not specify a time: call get_free_slots
   to discover available windows, then present options before recommending.
5. Call get_recommendations with the user query, weather context, and calendar context.
   - The tool handles new users automatically (cold-start fallback).
   - Pass weather_outdoor_suitable from the weather tool result.
   - Pass calendar_is_free and the free slot times from the calendar tool.
6. Present the recommendations clearly. Explain why each fits the user.
7. Ask which option the user wants to schedule.
8. Call book_activity only after the user explicitly confirms.
9. Log the user's reaction using log_activity_preference (clicked/booked/declined).
10. One day after the activity, prompt the user for explicit feedback and call
    log_recommendation_feedback with their rating and liked/disliked/neutral response.

Rules:
- Never book without explicit confirmation.
- If get_calendar reports is_free=false, do not recommend activities in that slot.
  Offer to find a free window using get_free_slots instead.
- If weather outdoor_suitable is false, suggest only indoor or covered options
  unless the user explicitly says they want outdoor anyway.
- Use Asia/Dubai for all date/time interpretation.
- Interpret 12-hour time correctly: 9pm = 21:00, 9am = 09:00.
- For schedule summaries, use busy_slots_local (already in Asia/Dubai).
- For weather summaries, use ml_weather_features[].weekday_name for day labels.
- Keep responses short, practical, and action-oriented.
- Do not invent profile information.
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
