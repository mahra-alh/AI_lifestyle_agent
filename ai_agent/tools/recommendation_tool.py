"""
Agent-facing recommendation tool.

Wires together the weather context, calendar context, and user profile
that the agent has already collected, then calls the ML recommendation
service and returns ranked venues the agent can present to the user.

The tool is deliberately thin — all ML logic lives in
ml/recommendation_service.py.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from agents import function_tool

from ai_agent.logger.app_logger import log_tool_error, log_tool_start, log_tool_success
from ml.recommendation_service import get_recommendations as _get_recommendations
from ml.schemas.recommendation_contracts import (
    CalendarContext,
    RecommendationRequest,
    TimeContext,
    WeatherContext,
)


@function_tool
def get_recommendations(
    user_id: str,
    user_query: str,
    # Time window
    requested_date: str = "",
    requested_start_time: str = "",
    requested_end_time: Optional[str] = None,
    # Calendar context (from get_calendar tool)
    calendar_is_free: bool = True,
    calendar_free_slot_start: Optional[str] = None,
    calendar_free_slot_end: Optional[str] = None,
    # Weather context (from get_weather_forecast tool)
    weather_outdoor_suitable: Optional[bool] = None,
    weather_temp_avg_c: Optional[float] = None,
    weather_humidity_pct: Optional[float] = None,
    weather_precipitation_mm: Optional[float] = None,
    weather_wind_speed_kmh: Optional[float] = None,
    weather_condition: Optional[str] = None,
    # Optional session tracking
    session_id: Optional[str] = None,
    # Number of results
    top_n: int = 5,
) -> Dict[str, Any]:
    """
    Get personalised activity recommendations for the user.

    Call this after:
      1. get_user_profile — to have the user's preferences loaded.
      2. get_weather_forecast — to know outdoor suitability.
      3. get_calendar — to confirm the requested time is free.

    The tool runs FAISS semantic retrieval followed by LightGBM reranking.
    New users without a profile get a cold-start popularity-based ranking
    automatically — no special handling needed from the agent.

    Args:
        user_id:                 User's email/ID (used to fetch the profile).
        user_query:              The user's natural-language request, e.g.
                                 "something cheap and relaxing near the marina".
        requested_date:          Target date (YYYY-MM-DD).
        requested_start_time:    Target start time (HH:MM or ISO).
        requested_end_time:      Target end time (optional).
        calendar_is_free:        Whether the slot is free (from get_calendar).
        calendar_free_slot_start: Start of confirmed free slot.
        calendar_free_slot_end:   End of confirmed free slot.
        weather_outdoor_suitable: True/False from get_weather_forecast.
        weather_temp_avg_c:       Average temperature for the day.
        weather_humidity_pct:     Average humidity percentage.
        weather_precipitation_mm: Precipitation in mm.
        weather_wind_speed_kmh:   Wind speed in km/h.
        weather_condition:        Textual weather condition, e.g. "Partly cloudy".
        session_id:              Optional session ID for feedback tracking.
        top_n:                   Number of recommendations to return (default 5).

    Returns:
        Dict with keys:
            status                "success" | "error"
            recommendation_id     Unique ID for this recommendation batch.
            recommendations       List of ranked venues (rank, name, score, etc.).
            model_version         Model that produced the ranking.
            is_cold_start         True if the user had no profile.
            message               Human-readable summary for the agent.
    """
    tool_name = "get_recommendations"
    start_clock = log_tool_start(
        tool_name=tool_name,
        user_id=user_id,
        data={
            "user_query": user_query,
            "requested_date": requested_date,
            "calendar_is_free": calendar_is_free,
            "weather_outdoor_suitable": weather_outdoor_suitable,
            "top_n": top_n,
        },
    )

    try:
        # Load the user's profile so the ML layer can personalise rankings.
        from ai_agent.tools.user_profile_pkg.tools import get_user_profile_data
        profile_result = get_user_profile_data(user_id)
        profile = profile_result.get("ml_profile_export") or profile_result

        # Build the structured request.
        request = RecommendationRequest(
            user_id=user_id,
            user_query=user_query,
            profile=profile,
            weather_context=WeatherContext(
                outdoor_suitable=weather_outdoor_suitable,
                outdoor_flag=1 if weather_outdoor_suitable else (0 if weather_outdoor_suitable is False else None),
                temp_avg_c=weather_temp_avg_c,
                humidity_avg_pct=weather_humidity_pct,
                precipitation_mm=weather_precipitation_mm,
                wind_speed_kmh=weather_wind_speed_kmh,
                weather_condition=weather_condition,
            ),
            time_context=TimeContext(
                requested_date=requested_date,
                requested_start_time=requested_start_time,
                requested_end_time=requested_end_time,
            ),
            calendar_context=CalendarContext(
                is_free=calendar_is_free,
                free_slot_start=calendar_free_slot_start,
                free_slot_end=calendar_free_slot_end,
            ),
            session_id=session_id,
        )

        # Run the pipeline
        result = _get_recommendations(request, top_n=top_n)

        log_tool_success(
            tool_name=tool_name,
            start_time=start_clock,
            user_id=user_id,
            data={
                "recommendation_id": result.recommendation_id,
                "recommendations_count": len(result.recommendations),
                "model_version": result.model_version,
                "is_cold_start": result.is_cold_start,
            },
        )

        return {
            "status": "success",
            "recommendation_id": result.recommendation_id,
            "recommendations": result.recommendations,
            "model_version": result.model_version,
            "is_cold_start": result.is_cold_start,
            "message": result.message,
        }

    except Exception as error:
        log_tool_error(
            tool_name=tool_name,
            start_time=start_clock,
            error=error,
            user_id=user_id,
            data={"user_query": user_query},
        )
        return {
            "status": "error",
            "message": f"Recommendation service error: {error}",
            "recommendations": [],
        }
