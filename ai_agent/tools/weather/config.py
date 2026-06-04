# global variables
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env once here so every module that imports config gets the secrets.
#load_dotenv() # needs to be removed when we push to prod
load_dotenv(dotenv_path=Path(__file__).parents[3] / ".env")

# API stuff
VISUAL_CROSSING_BASE_URL: str = (
    "https://weather.visualcrossing.com/"
    "VisualCrossingWebServices/rest/services/timeline"
)

# Supports both naming conventions (some .env files use the long form)
WEATHER_API_KEY: str = os.getenv("WEATHER_API_KEY") or os.getenv("VISUAL_CROSSING_API_KEY") or ""

# Forecast settings
FORECAST_DAYS: int = 7

REQUESTED_ELEMENTS = ",".join([
    "datetime",
    # actual temperature range: temp / tempmin / tempmax
    "temp",
    "tempmin",
    "tempmax",
    # accounts for heat index or wind chill: feelslike / feelslikemin / feelslikemax
    "feelslike",
    "feelslikemin",
    "feelslikemax",
    # for detecting uncomfortable or heavy-feeling weather: humidity / dew
    "humidity",
    "dew",
    # checks rain amount, rain chance, rain coverage, and type: precip / precipprob / precipcover / preciptype
    "precip",
    "precipprob",
    "precipcover",
    "preciptype",
    # for outdoor safety and comfort: windspeed / windgust / winddir
    "windspeed",
    "windgust",
    "winddir",
    # to estimate sun exposure
    "cloudcover",
    # for outdoor activity flag
    "uvindex",
    # useful for poor weather, fog, dust, or haze
    "visibility",
    # text labels for the agent to explain the decision: conditions / description / icon
    "conditions",
    "description",
    "icon",
    # maybe avoid recommending outdoor activities after dark (not used for now)
    "sunrise",
    "sunset",
])

# local storage
WEATHER_OUTPUT_DIR: Path = Path("data/weather")

# tool identity 
WEATHER_TOOL_NAME: str = "get_weather_forecast"

# Conenct to Redis (caching)
REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
REDIS_USERNAME: str = os.getenv("REDIS_USERNAME", "default")
REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")
REDIS_TTL_SECONDS: int = 6 * 60 * 60  # 6 hours — forecasts don't change faster
