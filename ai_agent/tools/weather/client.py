# this file is responsible for fetching weather data and converting it to be ML ready

from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Dict, List
from urllib.parse import quote
import requests

from ai_agent.tools.weather.cache import get_cached_forecast, set_cached_forecast
from ai_agent.tools.weather.outdoor_flag import calc_outdoor_flag
from ai_agent.tools.weather.config  import (
    FORECAST_DAYS,
    REQUESTED_ELEMENTS,
    VISUAL_CROSSING_BASE_URL,
    WEATHER_API_KEY,
)
from ai_agent.tools.weather.formatters import weekday_name_from_iso_date


# fetch raw api data:
# build the request URL
# send the API request
# validate the response
# return raw JSON data
def _fetch_raw(location_name: str, api_key: str) -> Dict[str, Any]:
    """
    Call the Visual Crossing timeline endpoint and return the parsed JSON body.

    Raises:
        RuntimeError: On any network or HTTP error.
    """

    # convert the location into a URL-safe format
    # Example:"Abu Dhabi" -> "Abu%20Dhabi"
    encoded_location = quote(location_name.strip())

    # build the API URL
    api_url = f"{VISUAL_CROSSING_BASE_URL}/{encoded_location}"

    # define the weather data we want from the API
    params = {
        "unitGroup": "metric",
        "key": api_key,
        "contentType": "json",
        "include": "days,hours",
        "elements": REQUESTED_ELEMENTS,
    }

    try:
        # send the request to the weather API
        response = requests.get(
            api_url,
            params=params,
            timeout=20,
        )

        # raise an exception if the API returned an error
        response.raise_for_status()

    except requests.RequestException as error:

        # convert API/network errors into a cleaner error message
        raise RuntimeError(
            f"Weather API request failed: {error}"
        ) from error

    # convert JSON response into a Python dictionary
    return response.json()


# convert raw API output into structured feature rows
# outdoor flag created here
def _parse_days(
    raw_days: List[Dict[str, Any]],
    location_name: str,
) -> List[Dict[str, Any]]:
    """
    Convert the raw API day objects into consistent feature rows.
    """

    # record when the weather data was retrieved
    retrieved_at = datetime.now(
        timezone.utc
    ).isoformat()

    # store processed weather rows
    daily_rows: List[Dict[str, Any]] = []

    # process one forecast day at a time
    for day in raw_days:

        # extract date
        date = day.get("datetime")

        # temperature features
        temp_avg_c = day.get("temp")
        temp_min_c = day.get("tempmin")
        temp_max_c = day.get("tempmax")

        feelslike_c = day.get("feelslike")
        feelslike_min_c = day.get("feelslikemin")
        feelslike_max_c = day.get("feelslikemax")

        # humidity-related features
        humidity_avg_pct = day.get("humidity")
        dew_point_c = day.get("dew")

        # rain-related features
        precipitation_mm = day.get("precip")
        precipitation_probability_pct = day.get("precipprob")
        precipitation_cover_pct = day.get("precipcover")
        precipitation_type = day.get("preciptype")

        # wind-related features
        wind_speed_kmh = day.get("windspeed")
        wind_gust_kmh = day.get("windgust")
        wind_direction_deg = day.get("winddir")

        # visibility and sun exposure features
        cloud_cover_pct = day.get("cloudcover")
        uv_index = day.get("uvindex")
        visibility_km = day.get("visibility")

        # daylight features
        sunrise = day.get("sunrise")
        sunset = day.get("sunset")

        # weather description
        weather_condition = day.get("conditions")
        weather_description = day.get("description")
        weather_icon = day.get("icon")

        # choose feels-like temperature when available
        # because it better represents outdoor comfort
        effective_temp_c = (
            feelslike_c
            if feelslike_c is not None
            else temp_avg_c
        )

        # target/engineered feature
        # 1 = suitable for outdoor activity
        # 0 = not suitable for outdoor activity
        outdoor_flag = calc_outdoor_flag(
            temp_avg_c=temp_avg_c,
            feelslike_c=feelslike_c,
            humidity_avg_pct=humidity_avg_pct,
            precipitation_mm=precipitation_mm,
            precipitation_probability_pct=precipitation_probability_pct,
            wind_speed_kmh=wind_speed_kmh,
            wind_gust_kmh=wind_gust_kmh,
            uv_index=uv_index,
            visibility_km=visibility_km,
        )

        # create a structured feature row
        daily_rows.append(
            {
                "date": date,
                "weekday_name": weekday_name_from_iso_date(date),
                "location_name": location_name,

                # temperature features
                "temp_avg_c": temp_avg_c,
                "temp_min_c": temp_min_c,
                "temp_max_c": temp_max_c,
                "feelslike_c": feelslike_c,
                "feelslike_min_c": feelslike_min_c,
                "feelslike_max_c": feelslike_max_c,
                "effective_temp_c": effective_temp_c,

                # humidity features
                "humidity_avg_pct": humidity_avg_pct,
                "dew_point_c": dew_point_c,

                # rain features
                "precipitation_mm": precipitation_mm,
                "precipitation_probability_pct": precipitation_probability_pct,
                "precipitation_cover_pct": precipitation_cover_pct,
                "precipitation_type": precipitation_type,

                # wind features
                "wind_speed_kmh": wind_speed_kmh,
                "wind_gust_kmh": wind_gust_kmh,
                "wind_direction_deg": wind_direction_deg,

                # visibility and sun exposure
                "cloud_cover_pct": cloud_cover_pct,
                "uv_index": uv_index,
                "visibility_km": visibility_km,

                # daylight features
                "sunrise": sunrise,
                "sunset": sunset,

                # weather description
                "weather_condition": weather_condition,
                "weather_description": weather_description,
                "weather_icon": weather_icon,

                # target/engineered feature
                # 1 = suitable for outdoor activity
                # 0 = not suitable for outdoor activity
                "outdoor_flag": outdoor_flag,

                # timestamp of retrieval
                "retrieved_at": retrieved_at,
            }
        )

    # return all processed weather rows
    return daily_rows


# convert raw hourly API output into structured feature rows
# hourly data is useful for recommending the best time for outdoor activity
def _parse_hours(
    raw_days: List[Dict[str, Any]],
    location_name: str,
) -> List[Dict[str, Any]]:
    """
    Convert the raw hourly API data into consistent feature rows.
    """

    # record when the weather data was retrieved
    retrieved_at = datetime.now(
        timezone.utc
    ).isoformat()

    # store processed hourly weather rows
    hourly_rows: List[Dict[str, Any]] = []

    # process one forecast day at a time
    for day in raw_days:

        # extract the day date
        date = day.get("datetime")

        # process one forecast hour at a time
        for hour in day.get("hours", []):

            # extract hour time
            hour_time = hour.get("datetime")

            # temperature features
            temp_c = hour.get("temp")
            feelslike_c = hour.get("feelslike")

            # humidity-related feature
            humidity_pct = hour.get("humidity")
            dew_point_c = hour.get("dew")

            # rain-related features
            precipitation_mm = hour.get("precip")
            precipitation_probability_pct = hour.get("precipprob")
            precipitation_type = hour.get("preciptype")

            # wind-related features
            wind_speed_kmh = hour.get("windspeed")
            wind_gust_kmh = hour.get("windgust")
            wind_direction_deg = hour.get("winddir")

            # visibility and sun exposure features
            cloud_cover_pct = hour.get("cloudcover")
            uv_index = hour.get("uvindex")
            visibility_km = hour.get("visibility")

            # weather description
            weather_condition = hour.get("conditions")
            weather_icon = hour.get("icon")

            # choose feels-like temperature when available
            # because it better represents outdoor comfort
            effective_temp_c = (
                feelslike_c
                if feelslike_c is not None
                else temp_c
            )

            # target/engineered feature
            # 1 = suitable for outdoor activity
            # 0 = not suitable for outdoor activity
            outdoor_flag = calc_outdoor_flag(
                temp_avg_c=temp_c,
                feelslike_c=feelslike_c,
                humidity_avg_pct=humidity_pct,
                precipitation_mm=precipitation_mm,
                precipitation_probability_pct=precipitation_probability_pct,
                wind_speed_kmh=wind_speed_kmh,
                wind_gust_kmh=wind_gust_kmh,
                uv_index=uv_index,
                visibility_km=visibility_km,
            )

            # create a structured hourly feature row
            hourly_rows.append(
                {
                    "date": date,
                    "time": hour_time,
                    "datetime": f"{date} {hour_time}",
                    "weekday_name": weekday_name_from_iso_date(date),
                    "location_name": location_name,

                    # temperature features
                    "temp_c": temp_c,
                    "feelslike_c": feelslike_c,
                    "effective_temp_c": effective_temp_c,

                    # humidity features
                    "humidity_pct": humidity_pct,
                    "dew_point_c": dew_point_c,

                    # rain features
                    "precipitation_mm": precipitation_mm,
                    "precipitation_probability_pct": precipitation_probability_pct,
                    "precipitation_type": precipitation_type,

                    # wind features
                    "wind_speed_kmh": wind_speed_kmh,
                    "wind_gust_kmh": wind_gust_kmh,
                    "wind_direction_deg": wind_direction_deg,

                    # visibility and sun exposure
                    "cloud_cover_pct": cloud_cover_pct,
                    "uv_index": uv_index,
                    "visibility_km": visibility_km,

                    # weather description
                    "weather_condition": weather_condition,
                    "weather_icon": weather_icon,

                    # target/engineered feature
                    # 1 = suitable for outdoor activity
                    # 0 = not suitable for outdoor activity
                    "outdoor_flag": outdoor_flag,

                    # timestamp of retrieval
                    "retrieved_at": retrieved_at,
                }
            )

    # return all processed hourly rows
    return hourly_rows


# main entry point for the weather client
# this is the only function that other modules should call.

# Workflow:
# 1. check Redis cache — return immediately if a valid forecast exists
# 2. validate API key
# 3. fetch raw weather data from Visual Crossing
# 4. extract forecast days
# 5. convert into ML-ready daily and hourly rows
# 6. store the result in Redis for future calls
# 7. return structured features
def get_forecast(location_name: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Fetch and parse the next 7-day forecast for a location.

    Checks Redis cache first. On a cache hit the API is never called,
    keeping latency low and avoiding unnecessary quota usage.
    On a cache miss the API is called and the result is stored in Redis
    so the next call for the same location is served from cache.
    """

    # check Redis cache before hitting the API
    # if a valid forecast exists, return it immediately
    cached = get_cached_forecast(location_name)
    if cached is not None:
        return cached

    # verify that an API key exists
    if not WEATHER_API_KEY:
        raise ValueError(
            "Missing WEATHER_API_KEY or "
            "VISUAL_CROSSING_API_KEY in environment. "
            "Add one of these to your .env file."
        )

    # retrieve raw weather data from the API
    api_data = _fetch_raw(
        location_name=location_name,
        api_key=WEATHER_API_KEY,
    )

    # keep only the required forecast days (7 days)
    raw_days = api_data.get("days", [])[:FORECAST_DAYS]

    # validate that forecast data exists
    if not raw_days:
        raise RuntimeError(
            "Weather API returned no forecast data."
        )

    # convert raw weather data into structured ML-ready daily feature rows
    daily_rows = _parse_days(
        raw_days=raw_days,
        location_name=location_name,
    )

    # convert raw weather data into structured ML-ready hourly feature rows
    hourly_rows = _parse_hours(
        raw_days=raw_days,
        location_name=location_name,
    )

    # build the final forecast dict with both daily and hourly features
    forecast = {
        "daily_features": daily_rows,
        "hourly_features": hourly_rows,
    }

    # store the result in Redis so the next call is served from cache
    set_cached_forecast(location_name, forecast)

    # return both daily and hourly features
    return forecast