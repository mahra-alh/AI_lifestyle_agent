# this file is essentially the LLM formatting layer of the weather tool.
from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Optional


def weekday_name_from_iso_date(date_value: Optional[str]) -> str:
    """
    Convert a YYYY-MM-DD string into a weekday name.
    """
    # handle missing dates safely
    if not date_value:
        return "Unknown"

    try:
        # convert ISO date into weekday name
        return datetime.fromisoformat(
            date_value
        ).strftime("%A")

    except ValueError:
        # return a safe fallback if the date format is invalid
        return "Unknown"


# build a natural-language weather summary that can be provided directly to the AI agent
def build_agent_weather_context(
    daily_rows: List[Dict[str, Any]],
    location_name: str,
) -> str:
    """
    Produce a plain-text weather summary for the AI agent.
    """

    # create the first line of the summary
    summary_lines = [
        f"7-day weather forecast for {location_name}:"
    ]

    # process one forecast day at a time
    for row in daily_rows:

        # convert binary outdoor flag into
        # human-readable text
        outdoor_status = (
            "Suitable weather for outdoors"
            if row["outdoor_flag"] == 1
            else "Not suitable weather for outdoors"
        )

        # create a consistent weather summary line
        # temperature: helps determine comfort level
        # feels-like temperature: reflects heat index and real-world conditions
        # humidity: high humidity makes weather feel hotter
        # rain probability: for outdoor planning
        # wind speed: affects comfort and safety
        # UV index: indicates sun exposure risk
        # outdoor recommendation: final engineered feature used by the agent
        summary_lines.append(
            f"- {row['weekday_name']}, "
            f"{row['date']}: "
            f"{row['weather_condition']}, "
            f"avg temp {row['temp_avg_c']} C, "
            f"feels like {row['feelslike_c']} C, "
            f"humidity {row['humidity_avg_pct']}%, "
            f"rain probability "
            f"{row['precipitation_probability_pct']}%, "
            f"wind speed "
            f"{row['wind_speed_kmh']} km/h, "
            f"UV index {row['uv_index']}, "
            f"outdoor recommendation: "
            f"{outdoor_status}."
        )

    # combine all lines into one text block that can be sent to the AI agent
    return "\n".join(summary_lines)