"""
Feature builder for LightGBM ranking.

Purpose:
- Convert FAISS candidate rows into the numeric feature layout used by the
  LightGBM training notebook.
- Derive user-side signals from the request profile plus the profile-tool
  fields already stored in the app.
- Keep the builder usable even when some profile attributes are missing by
  falling back to safe, model-friendly defaults.
"""
from __future__ import annotations

import re
from datetime import datetime
from math import isnan
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ai_agent.data.dubai_areas import DUBAI_AREAS, calculate_distance_km
from ml.schemas.recommendation_contracts import RecommendationRequest

FEATURE_COLUMNS = [
    "meal_cost_for_one",
    "budget_encoded",
    "serves_alcohol",
    "has_shisha",
    "has_outdoor_seating",
    "is_chain",
    "open_duration_fri",
    "is_open_at_dinner_fri",
    "is_open_at_breakfast_fri",
    "days_open_at_dinner",
    "weekend_open_hours",
    "weekday_open_hours",
    "weekend_open_boost",
    "american",
    "arabic",
    "asian",
    "bar",
    "cafe",
    "chinese",
    "european",
    "fastfood",
    "french",
    "grills",
    "healthy",
    "indian",
    "international",
    "italian",
    "japanese",
    "lebanese",
    "mediterranean",
    "mexican",
    "seafood",
    "turkish",
    "beach_waterfront",
    "cultural_centre",
    "gallery",
    "heritage_site",
    "library",
    "museum",
    "park_attraction",
    "haversine_distance_km",
    "budget_diff",
    "budget_score",
    "distance_score",
    "constraint_score",
    "time_score",
    "weather_score",
    "diet_halal_conflict",
    "diet_veg_conflict",
    "excl_nightlife_conflict",
    "excl_cultural_conflict",
    "excl_water_conflict",
    "outdoor_ok",
    "is_weekend",
    "is_dinner_request",
    "is_holiday",
    "travel_distance_encoded",
    "weather_pref_encoded",
    "pref_morning",
    "pref_midday",
    "pref_afternoon",
    "pref_evening",
    "pref_late_night",
    "diet_halal",
    "diet_vegetarian",
    "diet_vegan",
    "diet_gluten_free",
    "social_friends",
    "social_family",
    "social_partner",
    "social_alone",
    "factor_cost",
    "factor_distance",
    "factor_quality",
    "factor_comfort",
    "adventure_level_encoded",
    "currently_saving_money",
    "going_out_frequency_encoded",
    "activity_duration_encoded",
    "excl_nightlife",
    "excl_cultural",
    "excl_water",
    "excl_outdoor_travel",
    "has_exceptions",
]

_BUDGET_TO_LEVEL = {
    "free": 0,
    "low": 0,
    "medium": 1,
    "high": 2,
    "unknown": 1,
}

_BUDGET_TO_PROXY_AED = {
    "free": 0.0,
    "low": 50.0,
    "medium": 150.0,
    "high": 350.0,
    "unknown": np.nan,
}

TRAVEL_DISTANCE_THRESHOLDS = {
    0: 3.0,
    1: 7.0,
    2: 15.0,
    3: 25.0,
    4: 999.0,
}

WEATHER_PREF_INDOOR = 0
WEATHER_PREF_OUTDOOR = 1
WEATHER_PREF_MIXED = 2
WEATHER_PREF_ANY = 3

TIME_SLOT_FEATURES = (
    "pref_morning",
    "pref_midday",
    "pref_afternoon",
    "pref_evening",
    "pref_late_night",
)

CUISINE_FEATURE_KEYWORDS = {
    "american": ["american", "burger", "bbq", "barbecue", "diner"],
    "arabic": ["arabic", "middle eastern", "shawarma", "mezza", "mezze", "manakish"],
    "asian": ["asian", "wok", "pan asian"],
    "bar": ["bar", "pub", "cocktail", "lounge", "taproom", "beer"],
    "cafe": ["cafe", "coffee", "espresso", "latte", "bakery"],
    "chinese": ["chinese", "dim sum", "noodle", "noodles"],
    "european": ["european", "continental", "brasserie"],
    "fastfood": ["fastfood", "fast food", "burger", "fries", "fried chicken", "drive-thru"],
    "french": ["french", "bistro", "patisserie", "brasserie"],
    "grills": ["grill", "grills", "bbq", "barbecue", "steak", "steakhouse"],
    "healthy": ["healthy", "salad", "organic", "vegan", "vegetarian", "wellness"],
    "indian": ["indian", "curry", "biryani", "tandoor", "naan"],
    "international": ["international", "fusion", "global"],
    "italian": ["italian", "pizza", "pasta", "gelato", "ristorante"],
    "japanese": ["japanese", "sushi", "ramen", "izakaya"],
    "lebanese": ["lebanese", "levant", "hummus", "falafel"],
    "mediterranean": ["mediterranean", "greek", "levant", "med"],
    "mexican": ["mexican", "taco", "burrito", "latino", "latin american"],
    "seafood": ["seafood", "fish", "oyster", "sea bass", "shrimp", "prawn"],
    "turkish": ["turkish", "kebab", "doner", "doner"],
}

VENUE_LOCATION_KEYWORDS = {
    "beach_waterfront": ["beach", "waterfront", "seaside", "marina", "waterside", "by the sea"],
    "cultural_centre": ["cultural centre", "cultural center", "arts center", "arts centre"],
    "gallery": ["gallery", "art gallery"],
    "heritage_site": ["heritage", "historical", "historic", "old town"],
    "library": ["library", "book cafe", "reading room"],
    "museum": ["museum", "exhibition", "expo"],
    "park_attraction": ["park", "garden", "nature", "trail", "outdoor attraction"],
}

EXCLUSION_TO_VENUE_CATS = {
    "excl_nightlife": ["bar"],
    "excl_cultural": [
        "cultural_centre",
        "gallery",
        "heritage_site",
        "library",
        "museum",
    ],
    "excl_water": ["beach_waterfront"],
    "excl_outdoor_travel": ["park_attraction", "beach_waterfront"],
}

def build_lightgbm_feature_frame(
    candidates: list[pd.Series],
    request: RecommendationRequest,
    profile_data: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Build a numeric feature DataFrame for LightGBM.

    Each row corresponds to one FAISS candidate. The frame mirrors the feature
    order used by the training notebook so it can be passed straight into a
    trained LightGBM booster.
    """

    resolved_profile = _resolve_profile_data(request, profile_data)
    user_features = _build_user_feature_map(resolved_profile, request)

    rows: list[dict[str, Any]] = []

    for candidate in candidates:
        text = _candidate_text(candidate)
        text_lower = text.lower()

        venue_features = _build_venue_feature_map(candidate, text_lower)
        interaction_features = _build_interaction_feature_map(
            candidate=candidate,
            text_lower=text_lower,
            request=request,
            user_features=user_features,
            venue_features=venue_features,
            profile=resolved_profile,
        )

        row = {
            **venue_features,
            **interaction_features,
            **user_features,
        }

        rows.append({column: row.get(column, 0) for column in FEATURE_COLUMNS})

    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)

def calculate_rule_rank_score(feature_frame: pd.DataFrame) -> pd.Series:
    """
    Rule-based ranking score used when the LightGBM model artifact is missing.

    The score mirrors the notebook's weighted composite so the fallback remains
    aligned with the training-time logic.
    """

    frame = feature_frame.fillna(0)

    score = (
        0.30 * frame["budget_score"]
        + 0.25 * frame["distance_score"]
        + 0.25 * frame["constraint_score"]
        + 0.10 * frame["time_score"]
        + 0.10 * frame["weather_score"]
        - 0.20 * frame["diet_halal_conflict"]
        - 0.15 * frame["diet_veg_conflict"]
        - 0.10 * frame["excl_nightlife_conflict"]
        - 0.10 * frame["excl_cultural_conflict"]
        - 0.10 * frame["excl_water_conflict"]
    )

    return score.clip(lower=0.0, upper=1.0)


def _build_venue_feature_map(candidate: pd.Series, text_lower: str) -> dict[str, Any]:
    budget_level = _infer_budget_level(candidate, text_lower)
    venue_area = _infer_area_from_text(text_lower)

    venue_latitude = _to_float(_get_value(candidate, ["latitude", "lat"], default=np.nan))
    venue_longitude = _to_float(_get_value(candidate, ["longitude", "lon", "lng"], default=np.nan))

    if (venue_latitude is None or isnan(venue_latitude)) and venue_area is not None:
        venue_latitude = float(venue_area["latitude"])
    if (venue_longitude is None or isnan(venue_longitude)) and venue_area is not None:
        venue_longitude = float(venue_area["longitude"])

    open_signals = _infer_open_signals(text_lower, candidate)

    row = {
        "meal_cost_for_one": _infer_meal_cost_for_one(candidate, budget_level),
        "budget_encoded": _budget_level_to_encoded(budget_level),
        "serves_alcohol": _infer_binary_flag(
            candidate,
            ["serves_alcohol", "alcohol_served", "has_alcohol"],
            text_lower,
            ["alcohol", "cocktail", "wine", "pub", "bar"],
        ),
        "has_shisha": _infer_binary_flag(
            candidate,
            ["has_shisha", "shisha"],
            text_lower,
            ["shisha", "hookah"],
        ),
        "has_outdoor_seating": _infer_binary_flag(
            candidate,
            ["has_outdoor_seating", "outdoor_seating"],
            text_lower,
            ["outdoor", "terrace", "patio", "alfresco", "waterfront", "beach", "garden"],
        ),
        "is_chain": _infer_binary_flag(
            candidate,
            ["is_chain", "chain"],
            text_lower,
            ["nando", "starbucks", "mcdonald", "kfc", "subway", "pizza hut", "tgi friday"],
        ),
        "open_duration_fri": open_signals["open_duration_fri"],
        "is_open_at_dinner_fri": open_signals["is_open_at_dinner_fri"],
        "is_open_at_breakfast_fri": open_signals["is_open_at_breakfast_fri"],
        "days_open_at_dinner": open_signals["days_open_at_dinner"],
        "weekend_open_hours": open_signals["weekend_open_hours"],
        "weekday_open_hours": open_signals["weekday_open_hours"],
        "weekend_open_boost": open_signals["weekend_open_boost"],
        "american": _category_flag(candidate, text_lower, "american"),
        "arabic": _category_flag(candidate, text_lower, "arabic"),
        "asian": _category_flag(candidate, text_lower, "asian"),
        "bar": _category_flag(candidate, text_lower, "bar"),
        "cafe": _category_flag(candidate, text_lower, "cafe"),
        "chinese": _category_flag(candidate, text_lower, "chinese"),
        "european": _category_flag(candidate, text_lower, "european"),
        "fastfood": _category_flag(candidate, text_lower, "fastfood"),
        "french": _category_flag(candidate, text_lower, "french"),
        "grills": _category_flag(candidate, text_lower, "grills"),
        "healthy": _category_flag(candidate, text_lower, "healthy"),
        "indian": _category_flag(candidate, text_lower, "indian"),
        "international": _category_flag(candidate, text_lower, "international"),
        "italian": _category_flag(candidate, text_lower, "italian"),
        "japanese": _category_flag(candidate, text_lower, "japanese"),
        "lebanese": _category_flag(candidate, text_lower, "lebanese"),
        "mediterranean": _category_flag(candidate, text_lower, "mediterranean"),
        "mexican": _category_flag(candidate, text_lower, "mexican"),
        "seafood": _category_flag(candidate, text_lower, "seafood"),
        "turkish": _category_flag(candidate, text_lower, "turkish"),
        "beach_waterfront": _location_flag(candidate, text_lower, "beach_waterfront"),
        "cultural_centre": _location_flag(candidate, text_lower, "cultural_centre"),
        "gallery": _location_flag(candidate, text_lower, "gallery"),
        "heritage_site": _location_flag(candidate, text_lower, "heritage_site"),
        "library": _location_flag(candidate, text_lower, "library"),
        "museum": _location_flag(candidate, text_lower, "museum"),
        "park_attraction": _location_flag(candidate, text_lower, "park_attraction"),
        "_venue_latitude": venue_latitude,
        "_venue_longitude": venue_longitude,
        "_budget_level": budget_level,
    }

    return row

def _build_user_feature_map(
    profile: Mapping[str, Any],
    request: RecommendationRequest,
) -> dict[str, Any]:
    profile_text = " ".join(
        [
            _joined(profile.get("preferred_areas")),
            _joined(profile.get("activity_preferences")),
            _joined(profile.get("disliked_activities")),
            str(profile.get("home_area", "")),
            str(profile.get("work_area", "")),
            _joined(profile.get("hobbies")),
            _joined(profile.get("interests")),
            _joined(profile.get("preferred_activity_types")),
            str(profile.get("preferred_environment", "")),
            str(profile.get("notes", "")),
            request.user_query,
        ]
    ).lower()

    budget_level = str(profile.get("budget_level", "unknown")).lower().strip()
    explicit_budget_encoded = _to_int(profile.get("budget_encoded"))
    user_budget_encoded = (
        max(0, min(explicit_budget_encoded, 2))
        if explicit_budget_encoded is not None
        else _budget_level_to_encoded(budget_level)
    )

    travel_distance_encoded = _encode_travel_distance(profile)
    weather_pref_encoded = _encode_weather_preference(profile, request)
    pref_slots = _encode_time_of_day_preferences(profile, request)
    diet_flags = _infer_diet_flags(profile, profile_text)
    social_flags = _infer_social_flags(profile, profile_text)
    factor_flags = _infer_factor_flags(profile, profile_text, request.user_query)
    adventure_level_encoded = _infer_adventure_level(profile, profile_text)
    currently_saving_money = _infer_currently_saving_money(profile, profile_text)
    going_out_frequency_encoded = _infer_going_out_frequency(profile)
    activity_duration_encoded = _infer_activity_duration(request)
    exclusion_flags = _infer_exclusion_flags(profile, profile_text, request)
    has_exceptions = int(
        any(
            [
                diet_flags["diet_halal"],
                diet_flags["diet_vegetarian"],
                diet_flags["diet_vegan"],
                diet_flags["diet_gluten_free"],
                exclusion_flags["excl_nightlife"],
                exclusion_flags["excl_cultural"],
                exclusion_flags["excl_water"],
                exclusion_flags["excl_outdoor_travel"],
                currently_saving_money == 1,
            ]
        )
    )

    row = {
        "budget_encoded": user_budget_encoded,
        "travel_distance_encoded": travel_distance_encoded,
        "weather_pref_encoded": weather_pref_encoded,
        **pref_slots,
        **diet_flags,
        **social_flags,
        **factor_flags,
        "adventure_level_encoded": adventure_level_encoded,
        "currently_saving_money": currently_saving_money,
        "going_out_frequency_encoded": going_out_frequency_encoded,
        "activity_duration_encoded": activity_duration_encoded,
        **exclusion_flags,
        "has_exceptions": has_exceptions,
    }

    return row

def _build_interaction_feature_map(
    candidate: pd.Series,
    text_lower: str,
    request: RecommendationRequest,
    user_features: Mapping[str, Any],
    venue_features: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    user_location = _resolve_user_location(profile)
    venue_lat = venue_features.get("_venue_latitude")
    venue_lon = venue_features.get("_venue_longitude")

    if user_location is None:
        distance_km = np.nan
    elif venue_lat is None or venue_lon is None or _is_missing_number(venue_lat) or _is_missing_number(venue_lon):
        distance_km = np.nan
    else:
        distance_km = calculate_distance_km(
            float(user_location["latitude"]),
            float(user_location["longitude"]),
            float(venue_lat),
            float(venue_lon),
        )

    user_budget_level = int(user_features.get("budget_encoded", 1))
    venue_budget_level = int(venue_features.get("budget_encoded", 1))
    budget_score = _budget_score(user_budget_level, venue_budget_level)

    budget_diff = abs(user_budget_level - venue_budget_level)
    travel_pref = int(user_features.get("travel_distance_encoded", 2))
    distance_score = _distance_score(distance_km, travel_pref)

    diet_halal_conflict = int(
        int(user_features.get("diet_halal", 0)) == 1
        and int(venue_features.get("serves_alcohol", 0)) == 1
    )
    diet_veg_conflict = int(
        (int(user_features.get("diet_vegetarian", 0)) == 1
         or int(user_features.get("diet_vegan", 0)) == 1)
        and int(venue_features.get("grills", 0)) == 1
        and int(venue_features.get("healthy", 0)) == 0
    )
    excl_nightlife_conflict = int(
        int(user_features.get("excl_nightlife", 0)) == 1
        and int(venue_features.get("bar", 0)) == 1
    )
    excl_cultural_conflict = int(
        int(user_features.get("excl_cultural", 0)) == 1
        and any(int(venue_features.get(cat, 0)) == 1 for cat in EXCLUSION_TO_VENUE_CATS["excl_cultural"])
    )
    excl_water_conflict = int(
        int(user_features.get("excl_water", 0)) == 1
        and int(venue_features.get("beach_waterfront", 0)) == 1
    )

    constraint_score = 0.0 if any(
        [
            diet_halal_conflict,
            diet_veg_conflict,
            excl_nightlife_conflict,
            excl_cultural_conflict,
            excl_water_conflict,
        ]
    ) else 1.0

    time_score = _time_score(user_features, venue_features, text_lower, constraint_score)
    weather_score = _weather_score(
        weather_pref=int(user_features.get("weather_pref_encoded", WEATHER_PREF_ANY)),
        venue_outdoor=int(venue_features.get("has_outdoor_seating", 0)),
    )

    outdoor_ok = int(request.weather_context.outdoor_suitable is not False)
    is_weekend = _is_weekend(request.time_context.requested_date)
    is_dinner_request = _is_dinner_request(request.time_context.requested_start_time)
    is_holiday = int(_to_int(profile.get("is_holiday")) or 0)

    return {
        "haversine_distance_km": distance_km,
        "budget_diff": budget_diff,
        "budget_score": budget_score,
        "distance_score": distance_score,
        "constraint_score": constraint_score,
        "time_score": time_score,
        "weather_score": weather_score,
        "diet_halal_conflict": diet_halal_conflict,
        "diet_veg_conflict": diet_veg_conflict,
        "excl_nightlife_conflict": excl_nightlife_conflict,
        "excl_cultural_conflict": excl_cultural_conflict,
        "excl_water_conflict": excl_water_conflict,
        "outdoor_ok": outdoor_ok,
        "is_weekend": is_weekend,
        "is_dinner_request": is_dinner_request,
        "is_holiday": is_holiday,
    }

def _resolve_profile_data(
    request: RecommendationRequest,
    profile_data: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = request.profile.model_dump()

    if profile_data:
        resolved.update(dict(profile_data))

    return resolved

def _candidate_text(candidate: pd.Series) -> str:
    possible_columns = [
        "faiss_text",
        "description",
        "text",
        "name",
        "title",
        "category",
        "primary_category",
        "area",
        "location_area",
    ]

    values = []

    for column in possible_columns:
        if column in candidate.index and pd.notna(candidate[column]):
            values.append(str(candidate[column]))

    return " ".join(values)

def _infer_budget_level(candidate: pd.Series, text_lower: str) -> str:
    possible_columns = [
        "budget_level",
        "estimated_budget_level",
        "budget",
        "price_level",
        "cost_level",
    ]

    for column in possible_columns:
        if column in candidate.index and pd.notna(candidate[column]):
            value = str(candidate[column]).lower().strip()
            if value in _BUDGET_TO_LEVEL:
                return value

    for level in ("free", "low", "medium", "high"):
        if re.search(rf"\b{re.escape(level)}\b", text_lower):
            return level

    return "unknown"

def _infer_meal_cost_for_one(candidate: pd.Series, budget_level: str) -> float:
    for column in ["meal_cost_for_one", "price", "price_aed", "cost"]:
        if column in candidate.index and pd.notna(candidate[column]):
            value = _to_float(candidate[column])
            if value is not None:
                return float(value)

    return float(_BUDGET_TO_PROXY_AED.get(budget_level, np.nan))

def _budget_level_to_encoded(level: str) -> int:
    return int(_BUDGET_TO_LEVEL.get(str(level).lower().strip(), 1))

def _budget_score(user_budget_level: int, venue_budget_level: int) -> float:
    diff = abs(int(user_budget_level) - int(venue_budget_level))
    return {0: 1.0, 1: 0.5, 2: 0.0}.get(min(diff, 2), 0.0)

def _user_budget_aed_proxy(profile: Mapping[str, Any]) -> float | None:
    for field in ("max_per_activity_aed", "monthly_fun_budget_aed"):
        value = _to_float(profile.get(field))
        if value is not None and not isnan(value):
            return float(value)

    budget_level = str(profile.get("budget_level", "unknown")).lower().strip()
    proxy = _BUDGET_TO_PROXY_AED.get(budget_level, np.nan)
    if isinstance(proxy, float) and isnan(proxy):
        return np.nan
    return float(proxy)

def _distance_score(distance_km: float | None, travel_pref: int) -> float:
    if distance_km is None or (isinstance(distance_km, float) and isnan(distance_km)):
        return 0.3

    if travel_pref == 4:
        return 1.0

    max_km = TRAVEL_DISTANCE_THRESHOLDS.get(travel_pref, 15.0)

    if distance_km <= max_km * 0.33:
        return 1.0
    if distance_km <= max_km * 0.66:
        return 0.6
    if distance_km <= max_km:
        return 0.3
    return 0.0

def _time_score(
    user_features: Mapping[str, Any],
    venue_features: Mapping[str, Any],
    text_lower: str,
    constraint_score: float,
) -> float:
    if constraint_score <= 0:
        return 0.0

    user_slots = {name: int(user_features.get(name, 0)) for name in TIME_SLOT_FEATURES}

    venue_open_dinner = int(venue_features.get("is_open_at_dinner_fri", 0))
    venue_open_breakfast = int(venue_features.get("is_open_at_breakfast_fri", 0))
    venue_open_duration = float(venue_features.get("open_duration_fri", 0.0) or 0.0)

    if user_slots["pref_morning"]:
        return 1.0 if venue_open_breakfast else 0.0
    if user_slots["pref_midday"]:
        return 1.0 if venue_open_duration >= 4.0 else 0.5 if venue_open_duration > 0 else 0.0
    if user_slots["pref_afternoon"]:
        return 1.0 if venue_open_duration >= 4.0 else 0.5 if venue_open_duration > 0 else 0.0
    if user_slots["pref_evening"]:
        return 1.0 if venue_open_dinner else 0.0
    if user_slots["pref_late_night"]:
        if venue_open_dinner:
            return 1.0
        if "late" in text_lower or "night" in text_lower:
            return 0.8
        return 0.0

    return 0.5

def _weather_score(weather_pref: int, venue_outdoor: int) -> float:
    if weather_pref == WEATHER_PREF_ANY:
        return 1.0
    if weather_pref == WEATHER_PREF_MIXED:
        return 0.8
    if weather_pref == WEATHER_PREF_OUTDOOR:
        return 1.0 if venue_outdoor else 0.4
    if weather_pref == WEATHER_PREF_INDOOR:
        return 0.4 if venue_outdoor else 1.0
    return 0.8

def _infer_open_signals(text_lower: str, candidate: pd.Series) -> dict[str, float]:
    def candidate_number(names: Sequence[str], default: float = 0.0) -> float:
        for name in names:
            if name in candidate.index and pd.notna(candidate[name]):
                value = _to_float(candidate[name])
                if value is not None and not isnan(value):
                    return float(value)
        return default

    has_breakfast = _contains_any(text_lower, ["breakfast", "brunch", "morning", "sunrise"])
    has_dinner = _contains_any(text_lower, ["dinner", "evening", "night", "late", "supper"])
    has_all_day = _contains_any(text_lower, ["24/7", "24 hours", "open 24", "all day", "all-day"])
    mentions_weekend = _contains_any(text_lower, ["weekend", "friday", "saturday", "sunday"])

    explicit_open_duration = candidate_number(["open_duration_fri", "weekend_open_hours"], default=np.nan)
    if not isnan(explicit_open_duration):
        open_duration = explicit_open_duration
    elif has_all_day:
        open_duration = 12.0
    elif has_dinner and has_breakfast:
        open_duration = 10.0
    elif has_dinner:
        open_duration = 8.0
    elif has_breakfast:
        open_duration = 6.0
    elif _contains_any(text_lower, ["cafe", "coffee"]):
        open_duration = 8.0
    else:
        open_duration = 0.0

    weekday_open_hours = candidate_number(["weekday_open_hours"], default=open_duration)
    weekend_open_hours = candidate_number(
        ["weekend_open_hours"],
        default=open_duration * (1.1 if mentions_weekend else 1.0),
    )

    if weekday_open_hours == 0.0 and open_duration > 0:
        weekday_open_hours = open_duration
    if weekend_open_hours == 0.0 and open_duration > 0:
        weekend_open_hours = open_duration * (1.1 if mentions_weekend else 1.0)

    if weekday_open_hours > 0:
        weekend_open_boost = weekend_open_hours / weekday_open_hours
    else:
        weekend_open_boost = 1.0 if weekend_open_hours > 0 else 0.0

    days_open_at_dinner = candidate_number(["days_open_at_dinner"], default=0.0)
    if days_open_at_dinner == 0.0:
        days_open_at_dinner = 7.0 if _contains_any(text_lower, ["daily", "every day", "7 days"]) else (1.0 if has_dinner else 0.0)

    return {
        "open_duration_fri": float(open_duration),
        "is_open_at_dinner_fri": int(has_dinner or open_duration >= 6.0),
        "is_open_at_breakfast_fri": int(has_breakfast),
        "days_open_at_dinner": float(days_open_at_dinner),
        "weekend_open_hours": float(weekend_open_hours),
        "weekday_open_hours": float(weekday_open_hours),
        "weekend_open_boost": float(weekend_open_boost),
    }

def _infer_diet_flags(profile: Mapping[str, Any], profile_text: str) -> dict[str, int]:
    explicit_flags = {
        "diet_halal": _to_int(profile.get("diet_halal")),
        "diet_vegetarian": _to_int(profile.get("diet_vegetarian")),
        "diet_vegan": _to_int(profile.get("diet_vegan")),
        "diet_gluten_free": _to_int(profile.get("diet_gluten_free")),
    }

    if all(value is not None for value in explicit_flags.values()):
        return {key: int(bool(value)) for key, value in explicit_flags.items()}

    explicit = {str(x).lower().strip() for x in _as_list(profile.get("diet_preferences"))}
    text = profile_text.lower()

    return {
        "diet_halal": int(_contains_any(text, ["halal"]) or "halal" in explicit),
        "diet_vegetarian": int(_contains_any(text, ["vegetarian", "veg"]) or "vegetarian" in explicit),
        "diet_vegan": int(_contains_any(text, ["vegan"]) or "vegan" in explicit),
        "diet_gluten_free": int(_contains_any(text, ["gluten free", "gluten-free", "gf"]) or "gluten-free" in explicit or "gluten free" in explicit),
    }

def _infer_social_flags(profile: Mapping[str, Any], profile_text: str) -> dict[str, int]:
    explicit_flags = {
        "social_friends": _to_int(profile.get("social_friends")),
        "social_family": _to_int(profile.get("social_family")),
        "social_partner": _to_int(profile.get("social_partner")),
        "social_alone": _to_int(profile.get("social_alone")),
    }

    if all(value is not None for value in explicit_flags.values()):
        return {key: int(bool(value)) for key, value in explicit_flags.items()}

    text = profile_text.lower()
    keywords = {
        "social_friends": ["with friends", "friends", "group", "hangout"],
        "social_family": ["family", "kids", "children", "parents"],
        "social_partner": ["partner", "date", "couple", "romantic"],
        "social_alone": ["alone", "solo", "myself", "independent"],
    }

    flags = {name: 0 for name in keywords}

    for name, terms in keywords.items():
        if _contains_any(text, terms):
            flags[name] = 1

    if not any(flags.values()):
        preferred_activity_types = " ".join(_as_list(profile.get("preferred_activity_types"))).lower()
        if _contains_any(preferred_activity_types, ["family"]):
            flags["social_family"] = 1
        elif _contains_any(preferred_activity_types, ["date", "partner", "couple"]):
            flags["social_partner"] = 1
        elif _contains_any(preferred_activity_types, ["friends", "social", "group"]):
            flags["social_friends"] = 1

    return flags

def _infer_factor_flags(profile: Mapping[str, Any], profile_text: str, user_query: str) -> dict[str, int]:
    explicit_flags = {
        "factor_cost": _to_int(profile.get("factor_cost")),
        "factor_distance": _to_int(profile.get("factor_distance")),
        "factor_quality": _to_int(profile.get("factor_quality")),
        "factor_comfort": _to_int(profile.get("factor_comfort")),
    }

    if all(value is not None for value in explicit_flags.values()):
        return {key: int(bool(value)) for key, value in explicit_flags.items()}

    text = f"{profile_text} {user_query}".lower()

    factor_cost = int(
        _contains_any(text, ["budget", "cheap", "affordable", "save money", "low cost", "cost"])
        or str(profile.get("budget_level", "unknown")).lower().strip() in {"free", "low"}
    )
    factor_distance = int(
        _contains_any(text, ["nearby", "close", "distance", "short drive", "walkable"])
        or _to_float(profile.get("max_travel_distance_km")) is not None
        or bool(profile.get("home_area"))
    )
    factor_quality = int(_contains_any(text, ["best", "top", "quality", "high rated", "rated", "premium"]))
    factor_comfort = int(_contains_any(text, ["comfortable", "chill", "relax", "cozy", "indoor", "covered"]))

    return {
        "factor_cost": factor_cost,
        "factor_distance": factor_distance,
        "factor_quality": factor_quality,
        "factor_comfort": factor_comfort,
    }

def _infer_adventure_level(profile: Mapping[str, Any], profile_text: str) -> int:
    explicit = profile.get("adventure_level_encoded")
    if explicit is not None:
        value = _to_int(explicit)
        if value is not None:
            return max(0, min(value, 3))

    text = profile_text.lower()
    if _contains_any(text, ["adventure", "thrill", "extreme", "hiking", "climbing"]):
        return 3
    if _contains_any(text, ["explore", "outdoors", "active"]):
        return 2
    if _contains_any(text, ["relax", "calm", "chill"]):
        return 0
    return 1

def _infer_currently_saving_money(profile: Mapping[str, Any], profile_text: str) -> int:
    explicit = profile.get("currently_saving_money")
    if explicit is not None:
        return int(bool(explicit))

    text = profile_text.lower()
    if str(profile.get("budget_level", "unknown")).lower().strip() in {"free", "low"}:
        return 1
    return int(_contains_any(text, ["saving money", "budget", "cheap", "affordable", "low cost"]))

def _infer_going_out_frequency(profile: Mapping[str, Any]) -> int:
    explicit = profile.get("going_out_frequency_encoded")
    value = _to_int(explicit)
    if value is not None:
        return max(0, min(value, 4))

    log_count = profile.get("activity_preferences_log_count")
    if log_count is None:
        logs = profile.get("activity_preferences_log")
        if isinstance(logs, list):
            log_count = len(logs)

    count = _to_int(log_count) or 0
    if count >= 20:
        return 4
    if count >= 10:
        return 3
    if count >= 4:
        return 2
    if count >= 1:
        return 1

    return 2

def _infer_activity_duration(request: RecommendationRequest) -> int:
    explicit = _to_int(getattr(request.profile, "activity_duration_encoded", None))
    if explicit is not None:
        return max(0, min(explicit, 3))

    start = _parse_time(request.time_context.requested_start_time)
    end = _parse_time(request.time_context.requested_end_time) if request.time_context.requested_end_time else None

    if start is not None and end is not None:
        duration_hours = _time_diff_hours(start, end)
    elif request.calendar_context.free_slot_start and request.calendar_context.free_slot_end:
        duration_hours = _datetime_diff_hours(
            request.calendar_context.free_slot_start,
            request.calendar_context.free_slot_end,
        )
    else:
        duration_hours = None

    if duration_hours is None:
        return 1
    if duration_hours <= 2:
        return 0
    if duration_hours <= 4:
        return 1
    if duration_hours <= 6:
        return 2
    return 3

def _infer_exclusion_flags(
    profile: Mapping[str, Any],
    profile_text: str,
    request: RecommendationRequest,
) -> dict[str, int]:
    explicit_flags = {
        "excl_nightlife": _to_int(profile.get("excl_nightlife")),
        "excl_cultural": _to_int(profile.get("excl_cultural")),
        "excl_water": _to_int(profile.get("excl_water")),
        "excl_outdoor_travel": _to_int(profile.get("excl_outdoor_travel")),
    }

    if all(value is not None for value in explicit_flags.values()):
        return {key: int(bool(value)) for key, value in explicit_flags.items()}

    text = f"{profile_text} {request.user_query}".lower()
    preferred_environment = str(profile.get("preferred_environment", "")).lower().strip()

    excl_nightlife = int(_contains_any(text, ["nightlife", "club", "party", "bar", "drinking", "alcohol"]))
    excl_cultural = int(_contains_any(text, ["museum", "gallery", "heritage", "cultural", "library", "art"]))
    excl_water = int(_contains_any(text, ["water", "beach", "waterfront", "seaside", "marina", "boat"]))
    excl_outdoor_travel = int(
        preferred_environment in {"indoor", "indoor only"}
        or _contains_any(text, ["avoid outdoor", "indoors", "indoor only"])
    )

    return {
        "excl_nightlife": excl_nightlife,
        "excl_cultural": excl_cultural,
        "excl_water": excl_water,
        "excl_outdoor_travel": excl_outdoor_travel,
    }

def _encode_travel_distance(profile: Mapping[str, Any]) -> int:
    explicit = profile.get("travel_distance_encoded")
    value = _to_int(explicit)
    if value is not None:
        return max(0, min(value, 4))

    max_distance = _to_float(profile.get("max_travel_distance_km"))
    if max_distance is None or isnan(max_distance):
        return 2

    if max_distance <= 3:
        return 0
    if max_distance <= 7:
        return 1
    if max_distance <= 15:
        return 2
    if max_distance <= 25:
        return 3
    return 4

def _encode_weather_preference(profile: Mapping[str, Any], request: RecommendationRequest | None = None) -> int:
    explicit = profile.get("weather_pref_encoded")
    value = _to_int(explicit)
    if value is not None:
        return max(0, min(value, 3))

    preferred_environment = str(profile.get("preferred_environment", "")).lower().strip()
    if preferred_environment in {"indoor", "indoor only"}:
        return WEATHER_PREF_INDOOR
    if preferred_environment in {"outdoor", "outdoor only"}:
        return WEATHER_PREF_OUTDOOR
    if preferred_environment in {"mixed", "both", "balanced"}:
        return WEATHER_PREF_MIXED

    if request is not None and request.weather_context.outdoor_suitable is False:
        return WEATHER_PREF_INDOOR

    return WEATHER_PREF_ANY

def _encode_time_of_day_preferences(
    profile: Mapping[str, Any],
    request: RecommendationRequest,
) -> dict[str, int]:
    explicit = {slot: _to_int(profile.get(slot)) for slot in TIME_SLOT_FEATURES}
    if all(value is not None for value in explicit.values()):
        return {slot: int(bool(explicit[slot])) for slot in TIME_SLOT_FEATURES}

    requested_time = _parse_time(request.time_context.requested_start_time)

    if requested_time is None:
        return {slot: 0 for slot in TIME_SLOT_FEATURES}

    hour = requested_time.hour

    if 5 <= hour < 11:
        active_slot = "pref_morning"
    elif 11 <= hour < 14:
        active_slot = "pref_midday"
    elif 14 <= hour < 17:
        active_slot = "pref_afternoon"
    elif 17 <= hour < 23:
        active_slot = "pref_evening"
    else:
        active_slot = "pref_late_night"

    return {slot: int(slot == active_slot) for slot in TIME_SLOT_FEATURES}

def _category_flag(candidate: pd.Series, text_lower: str, feature_name: str) -> int:
    if feature_name in candidate.index and pd.notna(candidate[feature_name]):
        return int(bool(_to_int(candidate[feature_name]) or _to_float(candidate[feature_name])))

    keywords = CUISINE_FEATURE_KEYWORDS.get(feature_name, [])
    return int(_contains_any(text_lower, keywords))


def _location_flag(candidate: pd.Series, text_lower: str, feature_name: str) -> int:
    if feature_name in candidate.index and pd.notna(candidate[feature_name]):
        return int(bool(_to_int(candidate[feature_name]) or _to_float(candidate[feature_name])))

    keywords = VENUE_LOCATION_KEYWORDS.get(feature_name, [])
    return int(_contains_any(text_lower, keywords))

def _infer_binary_flag(
    candidate: pd.Series,
    candidate_columns: Sequence[str],
    text_lower: str,
    keywords: Sequence[str],
) -> int:
    for column in candidate_columns:
        if column in candidate.index and pd.notna(candidate[column]):
            value = _to_int(candidate[column])
            if value is not None:
                return int(bool(value))
            return int(str(candidate[column]).strip().lower() in {"true", "yes", "y"})

    return int(_contains_any(text_lower, keywords))

def _infer_area_from_text(text_lower: str) -> dict[str, Any] | None:
    aliases: list[tuple[str, dict[str, Any]]] = []
    for area in DUBAI_AREAS.values():
        for alias in area.get("aliases", []):
            aliases.append((alias, area))

    # Prefer longer aliases first so "dubai marina" wins over "marina".
    for alias, area in sorted(aliases, key=lambda item: len(item[0]), reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", text_lower):
            return area

    return None

def _resolve_user_location(profile: Mapping[str, Any]) -> dict[str, Any] | None:
    user_lat = _to_float(profile.get("user_latitude"))
    user_lon = _to_float(profile.get("user_longitude"))

    if user_lat is not None and user_lon is not None and not isnan(user_lat) and not isnan(user_lon):
        return {"latitude": float(user_lat), "longitude": float(user_lon)}

    for area_name in _as_list(profile.get("preferred_areas")) + [profile.get("home_area"), profile.get("work_area")]:
        if not area_name:
            continue
        resolved = _resolve_area(area_name)
        if resolved is not None:
            return resolved

    return None

def _resolve_area(area_value: Any) -> dict[str, Any] | None:
    if area_value is None:
        return None

    cleaned = str(area_value).strip().lower()
    if cleaned in DUBAI_AREAS:
        area = DUBAI_AREAS[cleaned]
        return {"latitude": area["latitude"], "longitude": area["longitude"]}

    for area_id, area in DUBAI_AREAS.items():
        aliases = [area_id, *area.get("aliases", [])]
        if cleaned in aliases or any(cleaned == alias for alias in aliases):
            return {"latitude": area["latitude"], "longitude": area["longitude"]}

    return None

def _is_weekend(requested_date: str) -> int:
    try:
        parsed = datetime.fromisoformat(requested_date).date()
    except ValueError:
        return 0

    # Dubai leisure planning often treats Friday as a weekend-adjacent day.
    return int(parsed.weekday() >= 4)

def _is_dinner_request(requested_start_time: str) -> int:
    parsed = _parse_time(requested_start_time)
    if parsed is None:
        return 0
    return int(parsed.hour >= 17 or parsed.hour < 2)

def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None

    text = str(value).strip()
    for fmt in ("%H:%M", "%H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue

    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None

def _time_diff_hours(start: datetime, end: datetime) -> float | None:
    start_minutes = start.hour * 60 + start.minute
    end_minutes = end.hour * 60 + end.minute
    diff = end_minutes - start_minutes
    if diff <= 0:
        return None
    return diff / 60.0

def _datetime_diff_hours(start: str, end: str) -> float | None:
    try:
        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end)
    except ValueError:
        return None

    diff = (end_dt - start_dt).total_seconds() / 3600.0
    return diff if diff > 0 else None

def _contains_any(text: str, phrases: Sequence[str]) -> bool:
    normalized = text.lower()
    for phrase in phrases:
        if not phrase:
            continue
        if re.search(rf"\b{re.escape(phrase.lower())}\b", normalized):
            return True
    return False

def _joined(values: Any) -> str:
    if values is None:
        return ""
    if isinstance(values, str):
        return values
    if isinstance(values, Sequence):
        return " ".join(str(value) for value in values if value is not None)
    return str(values)

def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]

def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())

def _get_value(candidate: pd.Series, possible_columns: Sequence[str], default: Any = None) -> Any:
    for column in possible_columns:
        if column in candidate.index and pd.notna(candidate[column]):
            return candidate[column]
    return default

def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None

def _is_missing_number(value: Any) -> bool:
    numeric = _to_float(value)
    return numeric is None or isnan(numeric)
