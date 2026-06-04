# this file acts as the orchestration layer of the weather pipeline
# it coordinates data collection, feature delivery, storage, logging, and agent communication 

from __future__ import annotations
from typing import Any, Dict

# allows the function to be exposed as a callable tool to the AI agent
from agents import function_tool

# logging utilities used to track tool execution.
from ai_agent.logger.app_logger import (
    log_tool_error,
    log_tool_start,
    log_tool_success,
)

# weather-specific modules
# splitting functionality across modules keeps the code clean and modular
from ai_agent.tools.weather.client import get_forecast
from ai_agent.tools.weather.config import WEATHER_OUTPUT_DIR, WEATHER_TOOL_NAME
from ai_agent.tools.weather.formatters import build_agent_weather_context
from ai_agent.tools.weather.storage import safe_location_filename, save_csv, save_json

# register this function as an AI Agent tool
# this is the only function the agent needs to know about
@function_tool
def get_weather_forecast(
    location_name: str = "dubai",
) -> Dict[str, Any]:
    """
    Main weather tool used by the AI agent.

    Responsibilities:
    1. Get forecast data
    2. Build agent-friendly weather summary
    3. Save forecast files
    4. Return structured weather features

    """

    # log the start of tool execution
    start_time = log_tool_start(
        tool_name=WEATHER_TOOL_NAME,
        data={"location_name": location_name},
    )

    try:

        # retrieve the weather forecast from the API
        forecast = get_forecast(location_name=location_name)
        daily_rows = forecast["daily_features"]
        hourly_rows = forecast.get("hourly_features", [])

        # convert weather rows into a text summary (for LLM to understand)
        agent_weather_context = build_agent_weather_context(
            daily_rows=daily_rows,
            location_name=location_name,
        )

        safe_name = safe_location_filename(
            location_name
        )

        # output file paths:
        # weather data will be saved as both JSON and CSV
        json_path = (
            WEATHER_OUTPUT_DIR
            / f"{safe_name}_7_day_weather.json"
        )

        csv_path = (
            WEATHER_OUTPUT_DIR
            / f"{safe_name}_7_day_hourly_weather.csv"
        )

        # build the final response object which contains everything the AI agent needs
        final_result: Dict[str, Any] = {
            "status": "success",
            "source": "Visual Crossing",
            "location_name": location_name,
            "ml_weather_features": daily_rows,
            "ml_hourly_features": hourly_rows,
            "agent_weather_context": agent_weather_context,
            "saved_files": {
                "json": str(json_path),
                "csv": str(csv_path),
            },
        }

        # save the full forecast response
        save_json(
            final_result,
            json_path,
        )

        # save weather rows as CSV
        save_csv(
            daily_rows,
            csv_path,
        )

        # log successful completion
        log_tool_success(
            tool_name=WEATHER_TOOL_NAME,
            start_time=start_time,
            data={
                "location_name": location_name,
                "forecast_days": len(daily_rows),
            },
        )

        # return successful forecast data to the AI agent
        return final_result

    except Exception as error:

        # log any unexpected errors
        # helps diagnose failures in production
        log_tool_error(
            tool_name=WEATHER_TOOL_NAME,
            start_time=start_time,
            error=error,
            data={"location_name": location_name},
        )

        # return a safe fallback response
        # prevents the AI agent from crashing if the weather service fails
        return {
            "status": "error",
            "message": str(error),

            # empty weather features because data retrieval failed
            "ml_weather_features": [],

            # fallback message for the agent.
            "agent_weather_context":
                "Weather forecast is currently unavailable.",
        }