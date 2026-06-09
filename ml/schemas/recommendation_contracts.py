"""
Typed request and response objects shared between the agent tool layer
(ai_agent/tools/recommendation_tool.py) and the ML service layer
(ml/recommendation_service.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class WeatherContext:
    """Weather signals passed from the weather tool into the ranker."""
    outdoor_suitable: Optional[bool] = None   # True / False / None (unknown)
    outdoor_flag: Optional[int] = None         # 1 = suitable, 0 = not suitable
    temp_avg_c: Optional[float] = None
    humidity_avg_pct: Optional[float] = None
    precipitation_mm: Optional[float] = None
    wind_speed_kmh: Optional[float] = None
    weather_condition: Optional[str] = None


@dataclass
class TimeContext:
    """Time signals — what window the user is planning for."""
    requested_date: str = ""           # YYYY-MM-DD
    requested_start_time: str = ""     # HH:MM or full ISO
    requested_end_time: Optional[str] = None


@dataclass
class CalendarContext:
    """Free-slot signals from the calendar tool."""
    is_free: bool = True
    free_slot_start: Optional[str] = None
    free_slot_end: Optional[str] = None


@dataclass
class RecommendationRequest:
    """
    Everything the recommendation service needs to score candidates.

    Built by recommendation_tool.py from the agent's available context.
    """
    user_id: str
    user_query: str
    profile: Any = None                          # profile dict or Pydantic model
    weather_context: WeatherContext = field(default_factory=WeatherContext)
    time_context: TimeContext = field(default_factory=TimeContext)
    calendar_context: CalendarContext = field(default_factory=CalendarContext)
    session_id: Optional[str] = None


@dataclass
class Recommendation:
    """A single ranked venue recommendation."""
    rank: int
    recommendation_id: str
    model_version: str
    venue_id: str
    name: str
    model_score: float
    faiss_score: float
    area: Optional[str] = None
    category: Optional[str] = None
    meal_cost_for_one: Optional[float] = None
    budget_level: Optional[str] = None
    description: Optional[str] = None
    has_outdoor_seating: Optional[bool] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None


@dataclass
class RecommendationResult:
    """The full response from the recommendation service."""
    recommendation_id: str
    recommendations: List[Dict[str, Any]]
    model_version: str
    is_cold_start: bool
    message: str
