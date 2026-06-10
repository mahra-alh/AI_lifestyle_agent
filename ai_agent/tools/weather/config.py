# ai_agent/tools/weather/config.py
from __future__ import annotations
import os
from pathlib import Path

from ai_agent.secrets import get_secret, get_secret_optional

#  Weather API

VISUAL_CROSSING_BASE_URL: str = (
    "https://weather.visualcrossing.com/"
    "VisualCrossingWebServices/rest/services/timeline"
)

# Supports both naming conventions
WEATHER_API_KEY: str = (
    get_secret_optional("WEATHER_API_KEY")
    or get_secret_optional("VISUAL_CROSSING_API_KEY")
)

#  Forecast settings 

FORECAST_DAYS: int = 7

REQUESTED_ELEMENTS = ",".join([
    "datetime",
    "temp", "tempmin", "tempmax",
    "feelslike", "feelslikemin", "feelslikemax",
    "humidity", "dew",
    "precip", "precipprob", "precipcover", "preciptype",
    "windspeed", "windgust", "winddir",
    "cloudcover",
    "uvindex",
    "visibility",
    "conditions", "description", "icon",
    "sunrise", "sunset",
])

#  Local storage 

WEATHER_OUTPUT_DIR: Path = Path(__file__).parents[3] / "data" / "weather"

#  Tool identity 

WEATHER_TOOL_NAME: str = "get_weather_forecast"

# Redis (caching) 

REDIS_HOST: str     = get_secret("REDIS_HOST")
REDIS_PORT: int     = int(get_secret_optional("REDIS_PORT", "6379"))
REDIS_USERNAME: str = get_secret_optional("REDIS_USERNAME", "default")
REDIS_PASSWORD: str = get_secret("REDIS_PASSWORD")
REDIS_TTL_SECONDS: int = 6 * 60 * 60  # 6 hours
