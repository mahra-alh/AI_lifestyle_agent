from typing import Optional

def calc_outdoor_flag(
    temp_avg_c: Optional[float],
    feelslike_c: Optional[float],
    humidity_avg_pct: Optional[float],
    precipitation_mm: Optional[float],
    precipitation_probability_pct: Optional[float],
    wind_speed_kmh: Optional[float],
    wind_gust_kmh: Optional[float],
    uv_index: Optional[float],
    visibility_km: Optional[float],
) -> int:
    """
    Return outdoor suitability flag.

    1 = suitable for outdoor activity
    0 = not suitable for outdoor activity
    """


    if temp_avg_c is None and feelslike_c is None:
        return 0

    # choose feels-like temperature when available
    # because it better represents outdoor comfort
    effective_temp_c = (
        feelslike_c
        if feelslike_c is not None
        else temp_avg_c
    )
    # heavy rain makes outdoor activity unsuitable
    if precipitation_mm is not None and precipitation_mm >= 5:
        return 0
    # high rain probability makes outdoor plans risky
    if (
        precipitation_probability_pct is not None
        and precipitation_probability_pct >= 70
    ):
        return 0
    # strong wind is not ideal for outdoor activities
    if wind_speed_kmh is not None and wind_speed_kmh >= 35:
        return 0
    # strong gusts can be unsafe
    if wind_gust_kmh is not None and wind_gust_kmh >= 50:
        return 0
    # poor visibility can indicate fog, dust, or bad weather
    if visibility_km is not None and visibility_km < 3:
        return 0
    # extreme heat is unsafe for outdoor activity
    if effective_temp_c is not None and effective_temp_c >= 42:
        return 0
    # very high humidity makes outdoor activity uncomfortable
    if humidity_avg_pct is not None and humidity_avg_pct >= 85:
        return 0
    # very high UV exposure is risky
    if uv_index is not None and uv_index >= 10:
        return 0
    
    return 1 # default assumption