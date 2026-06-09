"""
ML profile export layer.

Converts a raw Firestore profile dict into a fully normalized, ML-ready
snapshot that feature_builder.py can consume without guessing at field
names or encoding conventions.

All encoding constants are imported from ml.schemas.feature_contract —
the single source of truth shared with the training pipeline.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ai_agent.data.dubai_areas import DUBAI_AREAS, get_area_by_user_input
from ml.schemas.feature_contract import (
    BUDGET_TO_LEVEL,
    TRAVEL_DISTANCE_THRESHOLDS,
    WEATHER_PREF_ANY,
    WEATHER_PREF_INDOOR,
    WEATHER_PREF_MIXED,
    WEATHER_PREF_OUTDOOR,
)

# Area coordinate function (used by both ml_export and user_profile tools)

def resolve_area_coordinates(area_value: Optional[str]) -> Optional[Dict[str, float]]:
    """Convert a Dubai area name into approximate coordinates if possible."""
    if area_value is None:
        return None
    cleaned = area_value.strip().lower()
    if cleaned in DUBAI_AREAS:
        area = DUBAI_AREAS[cleaned]
        return {"latitude": float(area["latitude"]), "longitude": float(area["longitude"])}
    matched = get_area_by_user_input(area_value)
    if matched is None:
        return None
    return {"latitude": float(matched["latitude"]), "longitude": float(matched["longitude"])}


# Public function
def build_ml_profile_export(profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a normalized profile snapshot for the ML layer.

    Keeps feature_builder.py from having to guess at basic profile fields
    when the agent already knows them. All encoded integers are validated
    and clamped to their expected range.
    """
    export = profile.copy()

    # Canonical email field
    export["user_email"] = (
        export.get("user_email")
        or export.get("email_address")
        or export.get("user_id")
        or "unknown@example.com"
    )
    export["email_address"] = export.get("email_address") or export["user_email"]

    # Coercion functions

    def coerce_text_list(field_name: str) -> List[str]:
        value = export.get(field_name)
        if value is None:
            return []
        items = value if isinstance(value, (list, tuple)) else [value]
        return [s for item in items if (s := str(item).strip().lower())]

    def coerce_flag(field_name: str, default: int = 0) -> int:
        value = export.get(field_name)
        if value is None:
            return default
        try:
            return 1 if int(value) else 0
        except (TypeError, ValueError):
            return default

    def coerce_encoded(field_name: str, minimum: int, maximum: int, default: int) -> int:
        value = export.get(field_name)
        if value is None:
            return default
        try:
            n = int(value)
        except (TypeError, ValueError):
            return default
        return n if minimum <= n <= maximum else default

    #  Budget 

    if export.get("budget_encoded") is None:
        budget_level = str(export.get("budget_level", "unknown")).lower().strip()
        export["budget_encoded"] = BUDGET_TO_LEVEL.get(budget_level, 1)
    else:
        export["budget_encoded"] = coerce_encoded("budget_encoded", 0, 2, 1)

    export["budget_level"] = str(export.get("budget_level", "unknown")).lower().strip() or "unknown"

    #  Travel distance 

    if export.get("travel_distance_encoded") is None:
        try:
            max_distance = float(export.get("max_travel_distance_km") or "nan")
        except (TypeError, ValueError):
            max_distance = float("nan")

        if max_distance != max_distance:  # isnan check without importing math
            export["travel_distance_encoded"] = 2
        elif max_distance <= 3:  export["travel_distance_encoded"] = 0
        elif max_distance <= 7:  export["travel_distance_encoded"] = 1
        elif max_distance <= 15: export["travel_distance_encoded"] = 2
        elif max_distance <= 25: export["travel_distance_encoded"] = 3
        else:                    export["travel_distance_encoded"] = 4
    else:
        export["travel_distance_encoded"] = coerce_encoded("travel_distance_encoded", 0, 4, 2)

    #  Weather preference 

    if export.get("weather_pref_encoded") is None:
        env = str(export.get("preferred_environment", "")).lower().strip()
        if env in {"indoor", "indoor only"}:     export["weather_pref_encoded"] = WEATHER_PREF_INDOOR
        elif env in {"outdoor", "outdoor only"}: export["weather_pref_encoded"] = WEATHER_PREF_OUTDOOR
        elif env in {"mixed", "both", "balanced"}: export["weather_pref_encoded"] = WEATHER_PREF_MIXED
        else:                                    export["weather_pref_encoded"] = WEATHER_PREF_ANY
    else:
        export["weather_pref_encoded"] = coerce_encoded("weather_pref_encoded", 0, 3, 3)

    # Binary flags (0/1) 

    for field in [
        "pref_morning", "pref_midday", "pref_afternoon", "pref_evening", "pref_late_night",
        "diet_halal", "diet_vegetarian", "diet_vegan", "diet_gluten_free",
        "social_friends", "social_family", "social_partner", "social_alone",
        "factor_cost", "factor_distance", "factor_quality", "factor_comfort",
        "currently_saving_money",
        "excl_nightlife", "excl_cultural", "excl_water", "excl_outdoor_travel",
    ]:
        export[field] = coerce_flag(field, 0)

    #  Text list fields 

    for field in [
        "preferred_areas", "activity_preferences", "disliked_activities",
        "hobbies", "interests", "preferred_activity_types", "work_days",
    ]:
        export[field] = coerce_text_list(field)

    #  Encoded scalars 

    export["transport_mode"] = str(export.get("transport_mode", "unknown")).lower().strip() or "unknown"
    export["adventure_level_encoded"] = coerce_encoded("adventure_level_encoded", 0, 3, 1)

    if export.get("going_out_frequency_encoded") is None:
        log_count = export.get("activity_preferences_log_count")
        if log_count is None:
            log_count = len(export.get("activity_preferences_log", []))
        try:
            log_count = int(log_count)
        except (TypeError, ValueError):
            log_count = 0

        if log_count >= 20:   export["going_out_frequency_encoded"] = 4
        elif log_count >= 10: export["going_out_frequency_encoded"] = 3
        elif log_count >= 4:  export["going_out_frequency_encoded"] = 2
        elif log_count >= 1:  export["going_out_frequency_encoded"] = 1
        else:                 export["going_out_frequency_encoded"] = 2
    else:
        export["going_out_frequency_encoded"] = coerce_encoded("going_out_frequency_encoded", 0, 4, 2)

    export["activity_duration_encoded"] = coerce_encoded("activity_duration_encoded", 0, 3, 1)

    #  Coordinates 

    if export.get("user_latitude") is None or export.get("user_longitude") is None:
        for area_name in [
            export.get("home_area"),
            export.get("work_area"),
            *(export.get("preferred_areas") or []),
        ]:
            coords = resolve_area_coordinates(area_name)
            if coords is not None:
                if export.get("user_latitude") is None:
                    export["user_latitude"] = coords["latitude"]
                if export.get("user_longitude") is None:
                    export["user_longitude"] = coords["longitude"]
                break

    # Derived exception flag 

    exception_fields = [
        "diet_halal", "diet_vegetarian", "diet_vegan", "diet_gluten_free",
        "excl_nightlife", "excl_cultural", "excl_water", "excl_outdoor_travel",
        "currently_saving_money",
    ]
    export["has_exceptions"] = int(any(export.get(f, 0) == 1 for f in exception_fields))
    export["ml_profile_ready"] = True

    return export
