import os
import asyncio
import uuid
from datetime import datetime
from typing import Any, Literal
from zoneinfo import ZoneInfo
from agents import Agent, Runner, function_tool
from dotenv import load_dotenv
from ai_agent.tools.weather_tool import get_weather_forecast
from ai_agent.tools.calendar_tool import get_calendar, book_activity
from ai_agent.tools.user_profile import (
    build_ml_profile_export,
    get_user_profile_data,
    get_user_profile,
    update_user_profile,
    log_activity_preference,
    log_recommendation_feedback,
)
from ai_agent.logger.app_logger import create_trace_id, set_trace_id, log_event
from ml.recommender_service import LifestyleRecommender
from ml.schemas.recommendation_contracts import (
    CalendarContext,
    RecommendationRequest,
    TimeContext,
    UserProfileContext,
    WeatherContext,
)

# Load environment variables used by API clients and tools.
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
APP_TIMEZONE = "Asia/Dubai"

_ML_RECOMMENDER: LifestyleRecommender | None = None

# Per-user conversation history. Keyed by normalized user_id (email).
# Stores the full message list returned by Runner.to_input_list() so each
# turn continues the same conversation rather than starting fresh.
_CONVERSATION_HISTORY: dict[str, list] = {}


def _get_ml_recommender() -> LifestyleRecommender:
    """
    Lazily initialize the ML recommender so startup stays lightweight.
    """

    global _ML_RECOMMENDER

    if _ML_RECOMMENDER is None:
        _ML_RECOMMENDER = LifestyleRecommender()

    return _ML_RECOMMENDER


@function_tool
def get_ml_recommendations(
    user_id: str,
    user_query: str,
    requested_date: str,
    requested_start_time: str,
    requested_end_time: str | None = None,
    timezone: str = APP_TIMEZONE,
    calendar_checked: bool = False,
    is_free: bool = False,
    free_slot_start: str | None = None,
    free_slot_end: str | None = None,
    busy_reason: str | None = None,
    weather_checked: bool = False,
    condition: str | None = None,
    temperature_celsius: float | None = None,
    outdoor_suitable: bool | None = None,
    weather_risk: Literal["low", "medium", "high", "unknown"] = "unknown",
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Build the ML request from the saved profile and return ranked recommendations.
    """

    profile_result = get_user_profile_data(user_id)

    if not bool(profile_result.get("profile_complete", False)):
        return {
            "success": False,
            "user_id": user_id,
            "profile_complete": False,
            "message": "Profile is incomplete. Collect the missing fields before calling ML recommendations.",
            "missing_fields": profile_result.get("missing_fields", []),
            "required_fields": profile_result.get("required_fields", []),
        }

    used_cached_ml_profile_export = "ml_profile_export" in profile_result
    ml_profile_export = profile_result.get("ml_profile_export") or build_ml_profile_export(profile_result)

    profile = UserProfileContext.model_validate(ml_profile_export)
    request = RecommendationRequest(
        request_id=f"ml_{uuid.uuid4().hex}",
        user_query=user_query,
        profile=profile,
        time_context=TimeContext(
            requested_date=requested_date,
            requested_start_time=requested_start_time,
            requested_end_time=requested_end_time,
            timezone=timezone,
        ),
        calendar_context=CalendarContext(
            calendar_checked=calendar_checked,
            is_free=is_free,
            free_slot_start=free_slot_start,
            free_slot_end=free_slot_end,
            busy_reason=busy_reason,
        ),
        weather_context=WeatherContext(
            weather_checked=weather_checked,
            condition=condition,
            temperature_celsius=temperature_celsius,
            outdoor_suitable=outdoor_suitable,
            weather_risk=weather_risk,
        ),
        top_k=top_k,
    )

# Call the ML recommender and return the response along with the original request for traceability.
    response = _get_ml_recommender().recommend(request)
    response_payload = response.model_dump()

    recommendation_items = []
    for rank, item in enumerate(response_payload.get("items", []), start=1):
        recommendation_items.append(
            {
                "recommendation_id": f"{request.request_id}_{rank}",
                "activity_id": item.get("activity_id"),
                "activity_name": item.get("name"),
                "recommendation_rank": rank,
                "model_version": response_payload.get("model_version"),
                "score": item.get("score"),
                "category": item.get("category"),
                "area": item.get("area"),
                "reason": item.get("reason"),
                "source": response_payload.get("source"),
                "user_query": user_query,
            }
        )

    return {
        "success": True,
        "user_id": user_id,
        "profile_complete": bool(profile_result.get("profile_complete", False)),
        "used_ml_profile_export": used_cached_ml_profile_export,
        "request_id": request.request_id,
        "response": response_payload,
        "recommendation_items": recommendation_items,
    }

# Define the agent instructions, available tools, and model.
agent = Agent(
    name="Lifestyle AI Agent",
    instructions = """
    You are a Dubai lifestyle planning assistant.

Your job is to help users discover and schedule suitable activities based on their profile, preferences, budget, location, weather, and calendar availability.

The user's profile is identified by email address.

Available tools:
get_weather_forecast,
get_calendar, 
book_activity,
build_ml_profile_export,
get_user_profile_data, 
get_user_profile, 
update_user_profile, 
log_activity_preference, 
log_recommendation_feedback

The ML recommendation tool is available. Use it once the profile, calendar,
and weather context are available.
If a complete profile is already provided in the input, do not ask for more profile fields.
Give a recommendation instead of stopping at intake questions.

Main flow:

When the user asks for recommendations or activity planning:

1. The user's email is already provided as Known user_id/email. Never ask the user for their email again.
2. Search the user profile using the known user_id/email.
3. Use the known user_id/email as the user_id argument for all profile tools.
4. If the profile exists:
   - If profile_complete is true, use the saved profile and continue the activity planning flow.
   - If profile_complete is false, ask only for the missing_fields returned by the tool.
5. If the profile does not exist:
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
7. Use the ML recommendation tool to generate ranked options whenever the
   profile is complete and the required context is available.
8. If the ML tool returns no_results or an error, fall back to short rule-based
   options that respect the hard blockers.
9. Ask which option the user wants to schedule.
10. Use book_activity only after confirmation.

When the user expresses a new preference:
- Log it using activity_preferences_log.

When the user reacts to a suggestion:
- Log the feedback using recommendation_feedback_log.

Rules:
- Do not ask for the email. It is already provided as Known user_id/email.
- Do not ask for profile details if the profile already exists.
- If the profile exists but is incomplete, ask only for the missing fields.
- Do not invent profile information.
- Do call the ML recommendation tool when the profile is complete and the
  calendar/weather context is available.
- Use the normalized `ml_profile_export` from the profile tool as the ML input
  shape.
- If the ML tool is unavailable or returns no_results, use rule-based fallback.
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
           log_activity_preference,
           log_recommendation_feedback,
           get_ml_recommendations,
           get_weather_forecast,
           get_calendar,
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
        now_dubai = datetime.now(ZoneInfo(APP_TIMEZONE)).isoformat()
        existing_history = _CONVERSATION_HISTORY.get(user_id, [])

        if existing_history:
            # Continue the existing conversation. The user_id context is already
            # in the history from the first turn, so only inject the current time.
            run_input = existing_history + [
                {
                    "role": "user",
                    "content": (
                        f"Current local time ({APP_TIMEZONE}): {now_dubai}\n\n"
                        f"User message:\n{user_message}"
                    ),
                }
            ]
        else:
            # First turn: establish user identity and time context for the whole session.
            run_input = (
                f"Known user_id/email for tool calls: {user_id}\n\n"
                f"Current local time ({APP_TIMEZONE}): {now_dubai}\n\n"
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
