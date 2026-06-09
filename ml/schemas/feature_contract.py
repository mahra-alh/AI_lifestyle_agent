from __future__ import annotations

from typing import Any, Mapping, Sequence

# Shared encoding maps
# (previously duplicated in feature_builder.py AND user_profile.py)

BUDGET_TO_LEVEL: dict[str, int] = {
    "free": 0,
    "low": 0,
    "medium": 1,
    "high": 2,
    "unknown": 1,
}

BUDGET_TO_PROXY_AED: dict[str, float] = {
    "free": 0.0,
    "low": 60.0,
    "medium": 120.0,
    "high": 250.0,
    "unknown": float("nan"),
}

# Keys are travel_distance_encoded values (0–4); values are max km thresholds.
TRAVEL_DISTANCE_THRESHOLDS: dict[int, float] = {
    0: 3.0,
    1: 7.0,
    2: 15.0,
    3: 25.0,
    4: 999.0,
}

# Weather preference encoded values
WEATHER_PREF_INDOOR: int = 0
WEATHER_PREF_OUTDOOR: int = 1
WEATHER_PREF_MIXED: int = 2
WEATHER_PREF_ANY: int = 3

# Feature column groups
# (grouped for readability; FEATURE_COLUMNS is the authoritative ordered list)
#
# Do NOT re-order, insert, or rename without retraining the model.
# Keep this identical on the training and serving sides.

VENUE_STATIC_COLUMNS: list[str] = [
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
]

# The 27 venue categories (unprefixed feature names).
VENUE_CATEGORY_COLUMNS: list[str] = [
    "american", "arabic", "asian", "bar", "cafe", "chinese", "european",
    "fastfood", "french", "grills", "healthy", "indian", "international",
    "italian", "japanese", "lebanese", "mediterranean", "mexican", "seafood",
    "turkish", "beach_waterfront", "cultural_centre", "gallery",
    "heritage_site", "library", "museum", "park_attraction",
]

# User × venue interaction features, computed per (user, venue) pair.
INTERACTION_COLUMNS: list[str] = [
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
]

CONTEXT_COLUMNS: list[str] = [
    "outdoor_ok",
    "is_weekend",
    "is_dinner_request",
    "is_holiday",
]

USER_PROFILE_COLUMNS: list[str] = [
    "travel_distance_encoded", "weather_pref_encoded",
    "pref_morning", "pref_midday", "pref_afternoon", "pref_evening",
    "pref_late_night", "diet_halal", "diet_vegetarian", "diet_vegan",
    "diet_gluten_free", "social_friends", "social_family", "social_partner",
    "social_alone", "factor_cost", "factor_distance", "factor_quality",
    "factor_comfort", "adventure_level_encoded", "currently_saving_money",
    "going_out_frequency_encoded", "activity_duration_encoded",
    "excl_nightlife", "excl_cultural", "excl_water", "excl_outdoor_travel",
    "has_exceptions",
]

# Authoritative ordered list — matches the order feature_builder.py emits.
FEATURE_COLUMNS: list[str] = (
    VENUE_STATIC_COLUMNS
    + VENUE_CATEGORY_COLUMNS
    + INTERACTION_COLUMNS
    + CONTEXT_COLUMNS
    + USER_PROFILE_COLUMNS
)

# Metadata !!!
# Map each canonical category feature name -> the column it lives under in
# the venue pool CSV (use this when reading the pool to avoid constant-0 features).
#   df["american"] = df["cat_american"]   # correct
#   df.get("american", 0)                 # wrong: always 0
VENUE_CATEGORY_SOURCE: dict[str, str] = {
    name: f"cat_{name}" for name in VENUE_CATEGORY_COLUMNS
}

# Venue-static features that do NOT exist in unified_venue_pool.csv and must
# be engineered before training/serving (otherwise they are silently 0/NaN).
DERIVED_VENUE_COLUMNS: set[str] = {"days_open_at_dinner", "weekend_open_boost"}

# Context features that are hardcoded constants in the current notebook.
# Constant-in-training means the model cannot learn a response to them yet.
CONSTANT_IN_TRAINING_COLUMNS: set[str] = {"outdoor_ok", "is_weekend", "is_holiday"}

# Sub-scores that are leaked from the rule label into the feature matrix.
# Exclude these once the label is a real user outcome (clicked/booked/liked)
# rather than the synthetic rule. Use training_feature_columns(drop_leaked=True).
LEAKED_RULE_FEATURES: set[str] = {
    "budget_score", "distance_score", "constraint_score",
    "time_score", "weather_score",
}

# LightGBM categorical features (unordered).
CATEGORICAL_FEATURES: list[str] = ["budget_encoded"]

# functions

def training_feature_columns(*, drop_leaked: bool = False) -> list[str]:
    """Return the ordered feature list to train on."""
    if not drop_leaked:
        return list(FEATURE_COLUMNS)
    return [c for c in FEATURE_COLUMNS if c not in LEAKED_RULE_FEATURES]


def categorical_indices(columns: Sequence[str] | None = None) -> list[int]:
    """Positional indices of CATEGORICAL_FEATURES within columns (default FEATURE_COLUMNS)."""
    columns = list(columns) if columns is not None else list(FEATURE_COLUMNS)
    return [columns.index(c) for c in CATEGORICAL_FEATURES if c in columns]


def diff_columns(
    actual: Sequence[str],
    expected: Sequence[str] | None = None,
) -> dict[str, Any]:
    """
    Compare an actual column list against the contract.

    Returns a report dict: missing (in expected, not actual), unexpected
    (in actual, not expected), order_mismatch (same set, wrong order), and ok.
    """
    expected = list(expected) if expected is not None else list(FEATURE_COLUMNS)
    actual = list(actual)
    exp_set, act_set = set(expected), set(actual)
    missing = [c for c in expected if c not in act_set]
    unexpected = [c for c in actual if c not in exp_set]
    order_mismatch = not missing and not unexpected and actual != expected
    return {
        "ok": not missing and not unexpected and not order_mismatch,
        "missing": missing,
        "unexpected": unexpected,
        "order_mismatch": order_mismatch,
        "expected_count": len(expected),
        "actual_count": len(actual),
    }


def assert_booster_matches(
    model: Any,
    *,
    expected: Sequence[str] | None = None,
) -> None:
    """
    Fail loudly at load time if a trained model's feature names/order do not
    match the contract.
    """
    names = _extract_feature_names(model)
    if names is None:
        raise ValueError(
            "Could not read feature names from the model. Train with a pandas "
            "DataFrame (so names are stored), or save the column list next to "
            "the artifact and compare it explicitly with diff_columns()."
        )
    report = diff_columns(names, expected)
    if not report["ok"]:
        raise ValueError(
            "Model feature contract mismatch — refusing to serve.\n"
            f"  missing from model:   {report['missing']}\n"
            f"  unexpected in model:  {report['unexpected']}\n"
            f"  order mismatch:       {report['order_mismatch']}\n"
            f"  expected {report['expected_count']} cols, "
            f"model has {report['actual_count']}."
        )


def _extract_feature_names(model: Any) -> list[str] | None:
    booster = getattr(model, "booster_", model)
    fn = getattr(booster, "feature_name", None)
    if callable(fn):
        names = fn()
        # LightGBM uses placeholder names Column_0.. when trained on a numpy array.
        if names and not all(str(n).startswith("Column_") for n in names):
            return list(names)
    names = getattr(model, "feature_name_", None)
    if names:
        return list(names)
    return None


__all__ = [
    "FEATURE_COLUMNS",
    "VENUE_STATIC_COLUMNS",
    "VENUE_CATEGORY_COLUMNS",
    "INTERACTION_COLUMNS",
    "CONTEXT_COLUMNS",
    "USER_PROFILE_COLUMNS",
    "VENUE_CATEGORY_SOURCE",
    "DERIVED_VENUE_COLUMNS",
    "CONSTANT_IN_TRAINING_COLUMNS",
    "LEAKED_RULE_FEATURES",
    "CATEGORICAL_FEATURES",
    "BUDGET_TO_LEVEL",
    "BUDGET_TO_PROXY_AED",
    "TRAVEL_DISTANCE_THRESHOLDS",
    "WEATHER_PREF_INDOOR",
    "WEATHER_PREF_OUTDOOR",
    "WEATHER_PREF_MIXED",
    "WEATHER_PREF_ANY",
    "training_feature_columns",
    "categorical_indices",
    "diff_columns",
    "assert_booster_matches",
]
