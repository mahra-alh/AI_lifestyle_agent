"""
Recommendation request/response contracts for the Lifestyle AI Agent.

Purpose:
- Defines the data shape sent from the AI Agent to the ML recommender.
- Defines the data shape returned from the ML recommender to the AI Agent.
- Keeps the agent, backend, and ML model consistent.
"""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field

# Input Context Models
class UserProfileContext(BaseModel):
    """
    User information collected from the profile tool.
    This should come from the user's stored profile or from newly collected answers.
    """

    user_email: str = Field(..., description="User email used as the profile identifier.")

    budget_level: Literal["free", "low", "medium", "high", "unknown"] = Field(
        default="unknown",
        description="User's preferred budget level."
    )

    preferred_areas: list[str] = Field(
        default_factory=list,
        description="Dubai areas preferred by the user, for example Dubai Marina, JBR, Downtown Dubai."
    )

    activity_preferences: list[str] = Field(
        default_factory=list,
        description="User's preferred activity types, hobbies, and interests."
    )

    disliked_activities: list[str] = Field(
        default_factory=list,
        description="Activities the user dislikes or wants to avoid."
    )

    transport_mode: Literal["walking", "car", "taxi", "metro", "unknown"] = Field(
        default="unknown",
        description="Main transport mode used by the user."
    )

    profile_complete: bool = Field(
        default=False,
        description="Whether the minimum required profile fields are available."
    )

class TimeContext(BaseModel):
    """
    Time information extracted from the user's prompt.
    Example prompt: recommend something today at 9 pm.
    """

    requested_date: str = Field(
        ...,
        description="Requested date in YYYY-MM-DD format."
    )

    requested_start_time: str = Field(
        ...,
        description="Requested start time in HH:MM 24-hour format."
    )

    requested_end_time: str | None = Field(
        default=None,
        description="Optional requested end time in HH:MM 24-hour format."
    )

    timezone: str = Field(
        default="Asia/Dubai",
        description="Timezone for the request."
    )

class CalendarContext(BaseModel):
    """
    Calendar availability result from the calendar tool.
    """

    calendar_checked: bool = Field(
        default=False,
        description="Whether the user's calendar was checked."
    )

    is_free: bool = Field(
        default=False,
        description="Whether the user is free during the requested time."
    )

    free_slot_start: str | None = Field(
        default=None,
        description="Free slot start datetime, preferably RFC3339/ISO format."
    )

    free_slot_end: str | None = Field(
        default=None,
        description="Free slot end datetime, preferably RFC3339/ISO format."
    )

    busy_reason: str | None = Field(
        default=None,
        description="Reason if the user is busy."
    )

class WeatherContext(BaseModel):
    """
    Weather result from the weather tool.
    Used to decide indoor/outdoor suitability.
    """

    weather_checked: bool = Field(
        default=False,
        description="Whether weather was checked."
    )

    condition: str | None = Field(
        default=None,
        description="Weather condition, for example clear, rain, dusty, humid."
    )

    temperature_celsius: float | None = Field(
        default=None,
        description="Temperature in Celsius."
    )

    outdoor_suitable: bool | None = Field(
        default=None,
        description="Whether outdoor activities are suitable."
    )

    weather_risk: Literal["low", "medium", "high", "unknown"] = Field(
        default="unknown",
        description="Weather risk level for outdoor activities."
    )

# Main ML Request Model
class RecommendationRequest(BaseModel):
    """
    Main request sent from the AI Agent to the ML recommender.
    """

    request_id: str = Field(
        ...,
        description="Unique ID for tracing this recommendation request."
    )

    user_query: str = Field(
        ...,
        description="Original user prompt."
    )

    profile: UserProfileContext
    time_context: TimeContext
    calendar_context: CalendarContext
    weather_context: WeatherContext

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of recommendations requested."
    )

# Output Models
class RecommendationItem(BaseModel):
    """
    One recommended activity/place/event returned by the ML recommender.
    """

    activity_id: str = Field(..., description="Unique activity or venue ID.")
    name: str = Field(..., description="Recommended activity/place/event name.")

    category: str = Field(
        default="unknown",
        description="Activity category, for example restaurant, cinema, event, outdoor, cultural."
    )

    area: str | None = Field(
        default=None,
        description="Dubai area where the activity is located."
    )

    latitude: float | None = Field(default=None, description="Activity latitude.")
    longitude: float | None = Field(default=None, description="Activity longitude.")

    estimated_budget_level: Literal["free", "low", "medium", "high", "unknown"] = Field(
        default="unknown",
        description="Estimated budget level for this recommendation."
    )

    indoor_outdoor: Literal["indoor", "outdoor", "both", "unknown"] = Field(
        default="unknown",
        description="Whether the activity is indoor, outdoor, or both."
    )

    score: float = Field(
        ...,
        ge=0,
        le=1,
        description="Final recommender score between 0 and 1."
    )

    reason: str = Field(
        ...,
        description="Short explanation of why this was recommended."
    )

    constraints_matched: list[str] = Field(
        default_factory=list,
        description="Constraints matched, for example budget, calendar, weather, location."
    )

    risk_flags: list[str] = Field(
        default_factory=list,
        description="Potential issues, for example far_location, weather_risk, budget_uncertain."
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional extra data for debugging or frontend display."
    )

class RecommendationResponse(BaseModel):
    """
    Response returned from the ML recommender to the AI Agent.
    """

    status: Literal["success", "no_results", "error"] = Field(
        ...,
        description="Overall recommendation status."
    )

    source: Literal["ml", "fallback"] = Field(
        default="ml",
        description="Whether the result came from ML or fallback."
    )

    request_id: str = Field(
        ...,
        description="Same request ID received in RecommendationRequest."
    )

    model_version: str = Field(
        default="unknown",
        description="Version of FAISS/LightGBM model used."
    )

    items: list[RecommendationItem] = Field(
        default_factory=list,
        description="Ranked recommendation results."
    )

    message: str | None = Field(
        default=None,
        description="Optional message for the agent/user."
    )

    fallback_reason: str | None = Field(
        default=None,
        description="Reason fallback was used, if applicable."
    )