from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import requests
from agents import function_tool
from dotenv import load_dotenv
from ai_agent.logger.app_logger import log_tool_error, log_tool_start, log_tool_success

load_dotenv()

# Keep the weather API endpoint and local output folder in one place.
VISUAL_CROSSING_BASE_URL = (
    "https://weather.visualcrossing.com/"
    "VisualCrossingWebServices/rest/services/timeline"
)
DEFAULT_WEATHER_OUTPUT_DIR = "data/weather"

def _save_json(data: Dict[str, Any], output_path: Path) -> None:
    # Save the full weather result for local inspection.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

def _save_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    # Save the daily forecast rows in a spreadsheet-friendly format.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def _outdoor_comfort_flag(
    temp_avg_c: Optional[float],
    humidity_avg_pct: Optional[float],
    precipitation_mm: Optional[float],
    wind_speed_kmh: Optional[float],
) -> str:

    # Creates a simple weather comfort signal for recommendations.
    if temp_avg_c is None:
        return "unknown"

    if precipitation_mm is not None and precipitation_mm >= 5:
        return "poor_for_outdoor"

    if wind_speed_kmh is not None and wind_speed_kmh >= 35:
        return "poor_for_outdoor"

    if temp_avg_c >= 45:
        return "poor_for_outdoor"

    if temp_avg_c >= 33 or (humidity_avg_pct is not None and humidity_avg_pct >= 75):
        return "moderate_for_outdoor"

    return "good_for_outdoor"

def _build_agent_weather_context(
    daily_rows: List[Dict[str, Any]],
    location_name: str,
) -> str:
    
    # Creates readable weather context for the SDK Agent.
    summary_lines = [f"7-day weather forecast for {location_name}:"]

    for row in daily_rows:
        summary_lines.append(
            f"- {row['weekday_name']}, {row['date']}: "
            f"{row['weather_condition']}, "
            f"avg temp {row['temp_avg_c']} C, "
            f"humidity {row['humidity_avg_pct']}%, "
            f"precipitation {row['precipitation_mm']} mm, "
            f"wind speed {row['wind_speed_kmh']} km/h, "
            f"outdoor comfort: {row['outdoor_comfort_flag']}."
        )

    return "\n".join(summary_lines)

def _weekday_name_from_iso_date(date_value: Optional[str]) -> str:
    # Convert YYYY-MM-DD into weekday name so downstream text does not infer weekdays incorrectly.
    if not date_value:
        return "Unknown"

    try:
        return datetime.fromisoformat(date_value).strftime("%A")
    except ValueError:
        return "Unknown"

@function_tool
def get_weather_forecast(
    location_name: str = "dubai",
) -> Dict[str, Any]:
    tool_name = "get_weather_forecast"
    start_time = log_tool_start(
        tool_name=tool_name,
        data={"location_name": location_name},
    )

    try:
        # Weather tool for the Agent.
        # This tool calls the Visual Crossing Weather API and stores the next
        # 7 days of weather data.
        # Load the API key from the local environment.
        api_key = os.getenv("WEATHER_API_KEY") or os.getenv("VISUAL_CROSSING_API_KEY")

        if not api_key:
            raise ValueError("Missing WEATHER_API_KEY or VISUAL_CROSSING_API_KEY environment variable.")

        encoded_location = quote(location_name.strip())

        # Build the Visual Crossing timeline request for daily metric forecast data.
        api_url = f"{VISUAL_CROSSING_BASE_URL}/{encoded_location}"

        params = {
            "unitGroup": "metric",
            "key": api_key,
            "contentType": "json",
            "include": "days",
            "elements": ",".join(
                [
                    "datetime",
                    "temp",
                    "tempmin",
                    "tempmax",
                    "humidity",
                    "precip",
                    "windspeed",
                    "conditions",
                    "icon",
                ]
            ),
        }

        try:
            # Call the weather API with a short timeout so the agent does not hang.
            response = requests.get(
                api_url,
                params=params,
                timeout=20,
            )

            response.raise_for_status()
        except requests.RequestException as error:
            raise RuntimeError(f"Weather API request failed: {error}") from error

        api_data = response.json()

        # Keep only the next seven forecast days for the current recommendation flow.
        days = api_data.get("days", [])[:7]
        if not days:
            raise RuntimeError("Weather API returned no forecast data.")

        daily_rows: List[Dict[str, Any]] = []

        retrieved_at = datetime.now(timezone.utc).isoformat()

        # Convert API day objects into consistent feature rows for the agent and future ML use.
        for day in days:
            date = day.get("datetime")
            weekday_name = _weekday_name_from_iso_date(date)
            temp_avg_c = day.get("temp")
            temp_min_c = day.get("tempmin")
            temp_max_c = day.get("tempmax")
            humidity_avg_pct = day.get("humidity")
            precipitation_mm = day.get("precip")
            wind_speed_kmh = day.get("windspeed")
            weather_condition = day.get("conditions")
            weather_icon = day.get("icon")

            outdoor_comfort_flag = _outdoor_comfort_flag(
                temp_avg_c=temp_avg_c,
                humidity_avg_pct=humidity_avg_pct,
                precipitation_mm=precipitation_mm,
                wind_speed_kmh=wind_speed_kmh,
            )

            daily_rows.append(
                {
                    "date": date,
                    "weekday_name": weekday_name,
                    "location_name": location_name,
                    "temp_avg_c": temp_avg_c,
                    "temp_min_c": temp_min_c,
                    "temp_max_c": temp_max_c,
                    "humidity_avg_pct": humidity_avg_pct,
                    "precipitation_mm": precipitation_mm,
                    "wind_speed_kmh": wind_speed_kmh,
                    "weather_condition": weather_condition,
                    "weather_icon": weather_icon,
                    "outdoor_comfort_flag": outdoor_comfort_flag,
                    "retrieved_at": retrieved_at,
                }
            )

        agent_weather_context = _build_agent_weather_context(
            daily_rows=daily_rows,
            location_name=location_name,
        )

        # Convert the location into a safe file name for local forecast outputs.
        safe_location_name = (
            location_name.lower()
            .strip()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )

        output_folder = Path(DEFAULT_WEATHER_OUTPUT_DIR)

        json_path = output_folder / f"{safe_location_name}_7_day_weather.json"
        csv_path = output_folder / f"{safe_location_name}_7_day_weather.csv"

        final_result = {
            "status": "success",
            "source": "Visual Crossing",
            "location_name": location_name,
            "ml_weather_features": daily_rows,
            "agent_weather_context": agent_weather_context,
            "saved_files": {
                "json": str(json_path),
                "csv": str(csv_path),
            },
        }

        # Save forecast artifacts locally and return the structured result to the agent.
        _save_json(final_result, json_path)
        _save_csv(daily_rows, csv_path)

        log_tool_success(
            tool_name=tool_name,
            start_time=start_time,
            data={
                "location_name": location_name,
                "forecast_days": len(daily_rows),
            },
        )

        return final_result

    except Exception as error:
        log_tool_error(
            tool_name=tool_name,
            start_time=start_time,
            error=error,
            data={"location_name": location_name},
        )
        return {
            "status": "error",
            "message": str(error),
            "ml_weather_features": [],
            "agent_weather_context": "Weather forecast is currently unavailable.",
        }
