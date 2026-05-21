import os
from datetime import datetime
from zoneinfo import ZoneInfo
from agents import Agent, Runner
from dotenv import load_dotenv
from ai_agent.tools.weather_tool import get_weather_forecast
from ai_agent.tools.calendar_tool import get_calendar, book_activity
from ai_agent.tools.user_profile import (
    get_user_profile,
    update_user_profile,
    log_activity_performance,
    log_recommendation_feedback,
)
from ai_agent.logger.app_logger import create_trace_id, set_trace_id, log_event

# Load environment variables used by API clients and tools.
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
APP_TIMEZONE = "Asia/Dubai"

# Define the agent instructions, available tools, and model.
agent = Agent(
    name="Lifestyle AI Agent",
    instructions = """
    You are a Dubai lifestyle planning assistant.

Your job is to help users discover and schedule suitable activities based on their profile, preferences, budget, location, weather, and calendar availability.

The user's profile is identified by email address.

Available tools:
- user_profile
- activity_preferences_log
- recommendation_feedback_log
- weather
- get_calendar
- book_activity

The ML recommendation tool is not available yet. Do not call it.
If a complete profile is already provided in the input, do not ask for more profile fields.
Give a recommendation instead of stopping at intake questions.

Main flow:

When the user asks for recommendations or activity planning:

1. Ask for the user's email address first. This is required to look up their profile and preferences. Do not proceed without it.
2. Search the user profile by email using the user_profile tool.
   Use the email address as the user_id argument for profile tools.
3. If the profile exists:
   - If profile_complete is true, use the saved profile and continue the activity planning flow.
   - If profile_complete is false, ask only for the missing_fields returned by the tool.
4. If the profile does not exist:
   - Ask the user to provide the required_fields returned by get_user_profile.
   - Do not create example values for the user.
   - After the user provides the details, save them using update_user_profile.
   - Then continue the original activity planning request.

When suggesting activities:
1. Use the profile data.
2. Check budget and preferences.
3. Check weather for outdoor activities.
4. Check calendar availability for the requested time.
5. If the requested time overlaps a busy slot, do not suggest activities in that window. Ask for an alternate free time or propose free alternatives outside busy slots.
6. If weather outdoor_comfort_flag is poor_for_outdoor, suggest only indoor or covered options unless the user explicitly says they want outdoor anyway.
7. Suggest 5 rule-based options.
8. Ask which option the user wants to schedule.
9. Use book_activity only after confirmation.

When the user expresses a new preference:
- Log it using activity_preferences_log.

When the user reacts to a suggestion:
- Log the feedback using recommendation_feedback_log.

Rules:
- Do not ask for full profile details before checking the email.
- Do not ask for profile details if the profile already exists.
- If the profile exists but is incomplete, ask only for the missing fields.
- Do not invent profile information.
- Do not call the missing ML recommendation tool.
- If the profile is complete, always produce 5 concrete, rule-based activity options.
- Use Asia/Dubai as the default timezone for all date/time interpretation.
- Interpret 12-hour time correctly: 9 pm means 21:00 and 9 am means 09:00.
- If the user time is ambiguous, ask one short clarification before calling calendar tools.
- For schedule summaries, prefer get_calendar.busy_slots_local (already normalized to Asia/Dubai) instead of raw busy_slots offsets.
- For weather summaries, use get_weather_forecast.ml_weather_features[].weekday_name for weekday labels; do not infer weekdays from date strings.
- Calendar hard blocker: if get_calendar reports is_free = false for the requested slot, never recommend activities inside that slot.
- Weather hard blocker: if outdoor_comfort_flag is poor_for_outdoor, default to indoor or covered activities.
- Do not book anything without confirmation.
- Keep responses short, practical, and action-oriented.
""",
    tools=[get_user_profile,
           update_user_profile,
           log_activity_performance,
           log_recommendation_feedback,
           get_weather_forecast,
           get_calendar,
           book_activity
           ],
    model="gpt-4o-mini",
)

def run_agent(user_id: str, user_message: str):
    # Create a trace ID so logs from this request can be grouped together.
    trace_id = create_trace_id()
    set_trace_id(trace_id)

    # Log the incoming request without storing the full message content.
    log_event(
        event_name="agent_request_started",
        user_id=user_id,
        data={
            "input_length": len(user_message),
        }
    )

    try:
        now_dubai = datetime.now(ZoneInfo(APP_TIMEZONE)).isoformat()

        # Run the agent with the known email so profile tools use a stable user ID.
        result = Runner.run_sync(
            agent,
            input=(
                f"Known user_id/email for tool calls: {user_id}\n\n"
                f"Current local time ({APP_TIMEZONE}): {now_dubai}\n\n"
                f"User message:\n{user_message}"
            )
        )

        # Log successful completion and return the final agent response.
        log_event(
            event_name="agent_request_completed",
            user_id=user_id,
            data={
                "output_length": len(result.final_output),
                "status": "success",
            }
        )

        return result.final_output

    except Exception as error:
        # Log failures with enough detail to debug terminal test runs.
        log_event(
            event_name="agent_request_failed",
            level="ERROR",
            user_id=user_id,
            data={
                "status": "failed",
                "error_type": type(error).__name__,
                "error_message": "Agent request failed.",
            }
        )
        raise
    
def main():
    # Read a simple terminal input flow for local PowerShell testing.
    default_email = ""
    user_id = input(f"Email [{default_email}]: ").strip() or default_email

    if not user_id:
        raise ValueError("Email is required.")

    default_message = ""
    user_message = input(f"Message [{default_message}]: ").strip() or default_message

    if not user_message:
        raise ValueError("Message is required.")

    result = run_agent(user_id=user_id, user_message=user_message)
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
        follow_up_result = run_agent(user_id=user_id, user_message=follow_up_message)
        print(follow_up_result)

if __name__ == "__main__":
    # Start the local terminal entry point when this file is executed directly.
    main()
