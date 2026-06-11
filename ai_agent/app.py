import os
import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Any, Literal
from zoneinfo import ZoneInfo
from agents import Agent, Runner, function_tool
from dotenv import load_dotenv
from ai_agent.tools.weather import get_weather_forecast
from ai_agent.tools.calendar.availability import get_calendar
from ai_agent.tools.calendar.booking import book_activity
from ai_agent.tools.calendar.free_slots import get_free_slots
from ai_agent.tools.user_profile_pkg import (
    ml_export,
    get_user_profile,
    update_user_profile,
    log_activity_preference,
    log_recommendation_feedback,
)
from ai_agent.logger.app_logger import create_trace_id, set_trace_id, log_event
from ai_agent.tools.recommendation_tool import get_recommendations

# Load environment variables used by API clients and tools.
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
APP_TIMEZONE = "Asia/Dubai"

# Per-user conversation history. Keyed by normalized user_id (email).
# Stores the full message list returned by Runner.to_input_list() so each
# turn continues the same conversation rather than starting fresh.
_CONVERSATION_HISTORY: dict[str, list] = {}

# Define the agent instructions, available tools, and model.
agent = Agent(
    name="Lifestyle AI Agent",
    instructions="""
You are a Dubai lifestyle planning assistant.

The user's profile is always complete before they reach this chat — never ask for profile information.
The user's email is already provided as the Known user_id. Never ask for it.

When the user asks for activity suggestions:
1. Call get_user_profile with the known user_id to load their saved preferences.
2. Call get_weather_forecast and find the forecast row for the REQUESTED date
   (e.g. tomorrow's row if the user asked about tomorrow, not today's).
3. Call get_calendar to check calendar availability for the requested time.
4. Call get_recommendations with the profile, weather, and calendar data.
5. Present the ranked results clearly. The FIRST line of the message must be
   the requested date's weather in exactly this format:
   Weather for <Weekday>, <YYYY-MM-DD>: <temp>°C, <condition>
   Example: Weather for Friday, 2026-06-12: 34°C, Sunny
   Then list the numbered options.
6. End every recommendation message by asking the user BOTH:
   what they think of the suggestions, and which one they would like
   to add to their calendar.
7. When the user picks an option AND has stated a specific date and time,
   call book_activity. This only PREPARES the booking — it shows a
   confirmation card in the app that the user must tap to finish.
   After calling it, tell the user to confirm or cancel on the card.
   NEVER say the booking is complete; the card does the actual booking.

If get_recommendations returns no results or an error:
- Tell the user recommendations are temporarily unavailable.
- Do NOT invent or guess activity names. Do not fabricate venues.

When the user reacts to a suggestion, log it:
- Clicks or bookings → log_activity_preference
- Explicit feedback (liked/disliked) → log_recommendation_feedback

Feedback is NOT booking consent:
- Statements like "I liked X", "X sounds good", or "great suggestion" are
  feedback only. Log them with log_recommendation_feedback, then ask:
  "Would you like me to book it?" Do not call book_activity for them.

Rules:
- Never ask for profile details — the profile is already saved.
- Never invent venue names or activities.
- If the requested time is busy in the calendar, use get_free_slots to
  suggest alternative free windows.
- If outdoor_comfort_flag is poor_for_outdoor, recommend indoor options only.
- Use Asia/Dubai timezone. 9 pm = 21:00, 9 am = 09:00.
- Never choose a date or time yourself. If the user has not stated one,
  ask one short clarification before calling book_activity.
- Keep responses short, practical, and action-oriented.
""",
    tools=[get_user_profile,
           update_user_profile,
           log_activity_preference,
           log_recommendation_feedback,
           get_recommendations,
           get_weather_forecast,
           get_calendar,
           get_free_slots,
           book_activity
           ],
    model="gpt-4o-mini",
)

async def run_agent(user_id: str, user_message: str):
    # Create a trace ID so logs from this request can be grouped together.
    trace_id = create_trace_id()
    set_trace_id(trace_id)

    log_event(
        event_name="agent_request_started",
        user_id=user_id,
        data={"input_length": len(user_message)},
    )

    try:
        now = datetime.now(ZoneInfo(APP_TIMEZONE))
        tomorrow = now + timedelta(days=1)
        time_context = (
            f"Today is {now.strftime('%A, %Y-%m-%d %H:%M')} ({APP_TIMEZONE}). "
            f"Tomorrow is {tomorrow.strftime('%A, %Y-%m-%d')}."
        )
        existing_history = _CONVERSATION_HISTORY.get(user_id, [])

        if existing_history:
            # Continue the existing conversation. The user_id context is already
            # in the history from the first turn, so only inject the current time.
            run_input = existing_history + [
                {
                    "role": "user",
                    "content": (
                        f"Current local time ({APP_TIMEZONE}): {time_context}\n\n"
                        f"User message:\n{user_message}"
                    ),
                }
            ]
        else:
            # First turn: establish user identity and time context for the whole session.
            run_input = (
                f"Known user_id/email for tool calls: {user_id}\n\n"
                f"Current local time ({APP_TIMEZONE}): {time_context}\n\n"
                f"User message:\n{user_message}"
            )

        result = await Runner.run(agent, input=run_input)

        # Persist the full conversation so the next turn continues from here.
        _CONVERSATION_HISTORY[user_id] = result.to_input_list()

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
    
async def main():
    
    default_email = ""
    user_id = input(f"Email [{default_email}]: ").strip() or default_email

    if not user_id:
        raise ValueError("Email is required.")

    default_message = ""
    user_message = input(f"Message [{default_message}]: ").strip() or default_message

    if not user_message:
        raise ValueError("Message is required.")

    result = await run_agent(user_id=user_id, user_message=user_message)
    print(result)

    default_recommendation_decision = ""
    recommendation_decision = input(
        f"Accept recommendations? (accept/decline) [{default_recommendation_decision}]: "
    ).strip().lower() or default_recommendation_decision

    if recommendation_decision not in {"accept", "decline", ""}:
        raise ValueError("Please enter 'accept' or 'decline'.")

    if recommendation_decision in {"accept", "decline"}:
        follow_up_message = (
            "I accept the recommendations."
            if recommendation_decision == "accept"
            else "I decline the recommendations."
        )
        follow_up_result = await run_agent(user_id=user_id, user_message=follow_up_message)
        print(follow_up_result)

if __name__ == "__main__":
    # Start the local terminal entry point when this file is executed directly.
    asyncio.run(main())
