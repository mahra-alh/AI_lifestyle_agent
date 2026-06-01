import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.metrics.pairwise import haversine_distances

warnings.filterwarnings("ignore")

# Paths
_HERE = Path(__file__).parent
VENUE_POOL_PATH = _HERE / "../../data/unified_venue_pool.csv"
SURVEY_PATH     = _HERE / "../../data/augmented_combined.csv"
ARTIFACTS_DIR   = _HERE / "../../artifacts"

# Feature definitions
TRAVEL_DISTANCE_THRESHOLDS = {0: 3, 1: 7, 2: 15, 3: 25, 4: 999}

EXCLUSION_TO_VENUE_CATS = {
    "excl_nightlife":      ["cat_bar"],
    "excl_cultural":       ["cat_museum", "cat_gallery", "cat_heritage_site",
                            "cat_cultural_centre", "cat_library"],
    "excl_water":          ["cat_beach_waterfront"],
    "excl_outdoor_travel": ["cat_park_attraction", "cat_beach_waterfront"],
}

USER_VENUE_FEATURES = [
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

CONTEXT_FEATURES = [
    "outdoor_ok",
    "is_weekend",
    "is_dinner_request",
    "is_holiday",
]

USER_PROFILE_FEATURES = [
    "budget_encoded", "travel_distance_encoded", "weather_pref_encoded",
    "pref_morning", "pref_midday", "pref_afternoon", "pref_evening",
    "pref_late_night", "diet_halal", "diet_vegetarian", "diet_vegan",
    "diet_gluten_free", "social_friends", "social_family", "social_partner",
    "social_alone", "factor_cost", "factor_distance", "factor_quality",
    "factor_comfort", "adventure_level_encoded", "currently_saving_money",
    "going_out_frequency_encoded", "activity_duration_encoded",
    "excl_nightlife", "excl_cultural", "excl_water", "excl_outdoor_travel",
    "has_exceptions",
]

def build_venue_static_features(venue_df: pd.DataFrame) -> list[str]:
    cuisine_cats = [c for c in venue_df.columns if c.startswith("cat_")]
    return (
        ["meal_cost_for_one", "budget_encoded"]
        + ["serves_alcohol", "has_shisha", "has_outdoor_seating"]
        + ["is_chain"]
        + [
            "open_duration_fri", "is_open_at_dinner_fri", "is_open_at_breakfast_fri",
            "days_open_at_dinner", "weekend_open_hours", "weekday_open_hours",
            "weekend_open_boost",
        ]
        + cuisine_cats
    )

def build_all_features(venue_static: list[str]) -> list[str]:
    """Deduplicated ordered feature list used by the model."""
    combined = venue_static + USER_VENUE_FEATURES + CONTEXT_FEATURES + USER_PROFILE_FEATURES
    return list(dict.fromkeys(combined))  # preserve order, drop duplicates

# Core helpers
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres between two lat/lon points."""
    p1 = np.radians([lat1, lon1])
    p2 = np.radians([lat2, lon2])
    return haversine_distances([p1], [p2])[0][0] * 6371


def compute_relevance_score(user_row: pd.Series, venue_row: pd.Series) -> float:
    """Synthetic relevance score [0, 1] for a (user, venue) pair.

    Components:
        budget_match    30% — encoded budget tier match
        distance_score  25% — haversine vs user's stated travel threshold
        constraint_ok   25% — hard violations (halal, exclusions) give 0
        time_score      10% — venue hours vs user's preferred time slot
        weather_score   10% — outdoor seating vs weather preference
    """
    # Budget match
    user_budget  = int(user_row.get("budget_encoded", 1))
    venue_budget = int(venue_row.get("budget_encoded", 1))
    budget_score = {0: 1.0, 1: 0.5, 2: 0.0}[abs(user_budget - venue_budget)]

    # Distance score
    user_lat    = float(user_row.get("user_latitude",  25.1))
    user_lon    = float(user_row.get("user_longitude", 55.2))
    venue_lat   = float(venue_row.get("latitude",  float("nan")))
    venue_lon   = float(venue_row.get("longitude", float("nan")))
    travel_pref = int(user_row.get("travel_distance_encoded", 2))
    max_km      = TRAVEL_DISTANCE_THRESHOLDS[travel_pref]

    dist_km = haversine_km(user_lat, user_lon, venue_lat, venue_lon)
    if pd.isna(dist_km):
        distance_score = 0.3
    elif travel_pref == 4:
        distance_score = 1.0
    elif dist_km <= max_km * 0.33:
        distance_score = 1.0
    elif dist_km <= max_km * 0.66:
        distance_score = 0.6
    elif dist_km <= max_km:
        distance_score = 0.3
    else:
        distance_score = 0.0

    # Constraint score (hard — any violation → 0)
    constraint_score = 1.0
    if user_row.get("diet_halal", 0) == 1 and venue_row.get("serves_alcohol", 0) == 1:
        constraint_score = 0.0
    if constraint_score > 0:
        if user_row.get("diet_vegetarian", 0) == 1 or user_row.get("diet_vegan", 0) == 1:
            if venue_row.get("cat_grills", 0) == 1 and venue_row.get("cat_healthy", 0) == 0:
                constraint_score = 0.0
    if constraint_score > 0:
        for excl_col, blocked_cats in EXCLUSION_TO_VENUE_CATS.items():
            if user_row.get(excl_col, 0) == 1:
                if any(venue_row.get(cat, 0) == 1 for cat in blocked_cats):
                    constraint_score = 0.0
                    break

    # Time slot match
    time_score = 0.5
    if constraint_score > 0:
        pref_evening   = int(user_row.get("pref_evening", 0))
        pref_morning   = int(user_row.get("pref_morning", 0))
        open_dinner    = int(venue_row.get("is_open_at_dinner_fri", 1))
        open_morning   = int(venue_row.get("is_open_at_breakfast_fri", 1))
        if pref_evening and open_dinner:
            time_score = 1.0
        elif pref_morning and open_morning:
            time_score = 1.0
        elif pref_evening and not open_dinner:
            time_score = 0.0
        elif pref_morning and not open_morning:
            time_score = 0.0

    # Weather match
    weather_pref   = int(user_row.get("weather_pref_encoded", 3))
    venue_outdoor  = int(venue_row.get("has_outdoor_seating", 0))
    if weather_pref == 3:
        weather_score = 1.0
    elif weather_pref == 2:
        weather_score = 0.8
    elif weather_pref == 1:
        weather_score = 1.0 if venue_outdoor else 0.4
    else:
        weather_score = 0.4 if venue_outdoor else 1.0

    score = (
        0.30 * budget_score
        + 0.25 * distance_score
        + 0.25 * constraint_score
        + 0.10 * time_score
        + 0.10 * weather_score
    )
    return round(min(score, 1.0), 4)

# Training data construction
def build_training_pairs(
    survey: pd.DataFrame,
    venues: pd.DataFrame,
    venue_static_features: list[str],
    sample_per_user: int = 200,
    keep_top: int = 10,
    keep_bottom: int = 10,
) -> pd.DataFrame:
    """Build synthetic (user, venue) training pairs.

    For each user: score a random sample of venues, keep the top-K and
    bottom-K by relevance score to maximise label contrast.
    """
    print(f"Building training pairs: {len(survey)} users × sample {sample_per_user} → keep {keep_top + keep_bottom}")
    np.random.seed(42)
    rows = []

    for user_idx, user in survey.iterrows():
        sampled = venues.sample(n=min(sample_per_user, len(venues)), random_state=user_idx)

        scored = [
            (idx, compute_relevance_score(user, venue))
            for idx, venue in sampled.iterrows()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        selected = scored[:keep_top] + scored[-keep_bottom:]

        for venue_idx, label in selected:
            venue = venues.loc[venue_idx]

            # Interaction features
            dist_km = haversine_km(
                float(user["user_latitude"]),
                float(user["user_longitude"]),
                float(venue.get("latitude", float("nan"))),
                float(venue.get("longitude", float("nan"))),
            )
            user_budget  = int(user["budget_encoded"])
            venue_budget = int(venue.get("budget_encoded", 1))
            budget_diff  = abs(user_budget - venue_budget)
            budget_score = {0: 1.0, 1: 0.5, 2: 0.0}[budget_diff]

            travel_pref    = int(user["travel_distance_encoded"])
            max_km         = TRAVEL_DISTANCE_THRESHOLDS[travel_pref]
            if pd.isna(dist_km):
                distance_score = 0.3
            elif travel_pref == 4:
                distance_score = 1.0
            elif dist_km <= max_km * 0.33:
                distance_score = 1.0
            elif dist_km <= max_km * 0.66:
                distance_score = 0.6
            elif dist_km <= max_km:
                distance_score = 0.3
            else:
                distance_score = 0.0

            diet_halal_conflict     = int(user["diet_halal"] == 1 and venue.get("serves_alcohol", 0) == 1)
            diet_veg_conflict       = int(
                (user["diet_vegetarian"] == 1 or user["diet_vegan"] == 1)
                and venue.get("cat_grills", 0) == 1
                and venue.get("cat_healthy", 0) == 0
            )
            excl_nightlife_conflict = int(user["excl_nightlife"] == 1 and venue.get("cat_bar", 0) == 1)
            excl_cultural_conflict  = int(
                user["excl_cultural"] == 1
                and any(venue.get(c, 0) == 1 for c in [
                    "cat_museum", "cat_gallery", "cat_heritage_site",
                    "cat_cultural_centre", "cat_library",
                ])
            )
            excl_water_conflict     = int(user["excl_water"] == 1 and venue.get("cat_beach_waterfront", 0) == 1)
            constraint_score        = 0.0 if any([
                diet_halal_conflict, diet_veg_conflict,
                excl_nightlife_conflict, excl_cultural_conflict, excl_water_conflict,
            ]) else 1.0

            pref_evening   = int(user["pref_evening"])
            pref_morning   = int(user["pref_morning"])
            open_dinner    = int(venue.get("is_open_at_dinner_fri", 1))
            open_morning   = int(venue.get("is_open_at_breakfast_fri", 1))
            if pref_evening and open_dinner:
                time_score = 1.0
            elif pref_morning and open_morning:
                time_score = 1.0
            elif pref_evening and not open_dinner:
                time_score = 0.0
            elif pref_morning and not open_morning:
                time_score = 0.0
            else:
                time_score = 0.5

            weather_pref   = int(user["weather_pref_encoded"])
            venue_outdoor  = int(venue.get("has_outdoor_seating", 0))
            if weather_pref == 3:
                weather_score = 1.0
            elif weather_pref == 2:
                weather_score = 0.8
            elif weather_pref == 1:
                weather_score = 1.0 if venue_outdoor else 0.4
            else:
                weather_score = 0.4 if venue_outdoor else 1.0

            row = {
                "user_id":  user_idx,
                "venue_id": venue["venue_id"],
                "label":    label,
                # interaction
                "haversine_distance_km":    dist_km,
                "budget_diff":              budget_diff,
                "budget_score":             budget_score,
                "distance_score":           distance_score,
                "constraint_score":         constraint_score,
                "time_score":               time_score,
                "weather_score":            weather_score,
                "diet_halal_conflict":      diet_halal_conflict,
                "diet_veg_conflict":        diet_veg_conflict,
                "excl_nightlife_conflict":  excl_nightlife_conflict,
                "excl_cultural_conflict":   excl_cultural_conflict,
                "excl_water_conflict":      excl_water_conflict,
                # context (simulated)
                "outdoor_ok":               1,
                "is_weekend":               1,
                "is_dinner_request":        pref_evening,
                "is_holiday":               0,
            }
            for feat in venue_static_features:
                row[feat] = venue.get(feat, 0)
            for feat in USER_PROFILE_FEATURES:
                row[feat] = user.get(feat, 0)

            rows.append(row)

        if (user_idx + 1) % 100 == 0:
            print(f"  {user_idx + 1}/{len(survey)} users processed...")

    df = pd.DataFrame(rows)
    print(f"\nTraining pairs built: {len(df):,}")
    print(f"Label distribution:")
    print(f"  Mean:  {df['label'].mean():.3f}  Std: {df['label'].std():.3f}")
    print(f"  >= 0.7 (positive): {(df['label'] >= 0.7).sum()}")
    print(f"  0.3–0.7 (neutral): {((df['label'] >= 0.3) & (df['label'] < 0.7)).sum()}")
    print(f"  < 0.3  (negative): {(df['label'] < 0.3).sum()}")
    return df


def split_data(
    train_df: pd.DataFrame,
    all_features: list[str],
    test_frac: float = 0.2,
):
    """User-aware train/test split (no user leaks across sets)."""
    unique_users = train_df["user_id"].unique()
    n_test       = max(1, int(len(unique_users) * test_frac))
    np.random.seed(42)
    test_users   = set(np.random.choice(unique_users, n_test, replace=False))
    train_users  = [u for u in unique_users if u not in test_users]

    train_set = train_df[train_df["user_id"].isin(train_users)].copy()
    test_set  = train_df[train_df["user_id"].isin(test_users)].copy()

    missing = [f for f in all_features if f not in train_df.columns]
    if missing:
        raise ValueError(f"Features missing from training data: {missing}")

    X_train = train_set[all_features].loc[:, ~train_set[all_features].columns.duplicated()]
    X_test  = test_set[all_features].loc[:, ~test_set[all_features].columns.duplicated()]
    y_train = train_set["label"]
    y_test  = test_set["label"]

    print(f"Train: {len(train_set)} pairs ({len(train_users)} users)")
    print(f"Test:  {len(test_set)} pairs ({len(test_users)} users)")
    print(f"Features: {X_train.shape[1]}")
    return X_train, X_test, y_train, y_test

# Model training
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    cat_features: list[str] | None = None,
) -> LGBMRegressor:
    """Train a LightGBM regression model and return it."""
    if cat_features is None:
        cat_features = [f for f in ["budget_encoded", "location_cluster"] if f in X_train.columns]

    model = LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        min_child_samples=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        categorical_feature=cat_features,
        eval_set=[(X_train, y_train, "training"), (X_test, y_test, "validation")],
        callbacks=[
            lgb.early_stopping(20, verbose=False),
            lgb.log_evaluation(50),
        ],
    )

    train_mse = ((model.predict(X_train) - y_train) ** 2).mean()
    test_mse  = ((model.predict(X_test)  - y_test)  ** 2).mean()
    print(f"\nBest iteration: {model.best_iteration_}")
    print(f"Train MSE: {train_mse:.4f}")
    print(f"Test  MSE: {test_mse:.4f}")
    return model

# Artifact persistence
def save_artifacts(
    model: LGBMRegressor,
    all_features: list[str],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> None:
    """Save model binary, feature list, and metadata JSON."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path    = artifacts_dir / "lightgbm_model.txt"
    features_path = artifacts_dir / "lightgbm_model_features.json"
    metadata_path = artifacts_dir / "lightgbm_manifest.json"

    model.booster_.save_model(str(model_path))

    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(all_features, f, indent=2)

    train_mse = float(((model.predict(X_train) - y_train) ** 2).mean())
    test_mse  = float(((model.predict(X_test)  - y_test)  ** 2).mean())
    cat_features = [f for f in ["budget_encoded", "location_cluster"] if f in X_train.columns]

    metadata = {
        "model_file":         str(model_path),
        "features_file":      str(features_path),
        "n_features":         len(all_features),
        "feature_names":      all_features,
        "categorical_features": cat_features,
        "best_iteration":     int(model.best_iteration_) if model.best_iteration_ else None,
        "train_rows":         int(len(X_train)),
        "test_rows":          int(len(X_test)),
        "train_mse":          train_mse,
        "test_mse":           test_mse,
        "model_type":         "LGBMRegressor",
        "target":             "synthetic relevance score",
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nArtifacts saved to {artifacts_dir}/")
    print(f"  Model:    {model_path.name}")
    print(f"  Features: {features_path.name}")
    print(f"  Manifest: {metadata_path.name}")

# Entry point
def main():
    # Load data
    venues = pd.read_csv(VENUE_POOL_PATH)
    venues.drop(columns=["faiss_text"], inplace=True, errors="ignore")
    survey = pd.read_csv(SURVEY_PATH)
    print(f"Venues: {venues.shape}  |  Survey: {survey.shape}")

    # Feature lists
    venue_static = build_venue_static_features(venues)
    all_features = build_all_features(venue_static)
    print(f"Feature counts — static: {len(venue_static)}, interaction: {len(USER_VENUE_FEATURES)}, "
          f"context: {len(CONTEXT_FEATURES)}, profile: {len(USER_PROFILE_FEATURES)}, "
          f"total (deduped): {len(all_features)}")

    # Build training data
    train_df = build_training_pairs(survey, venues, venue_static)

    # Split
    X_train, X_test, y_train, y_test = split_data(train_df, all_features)

    # Train
    model = train_model(X_train, X_test, y_train, y_test)

    # Save
    save_artifacts(model, all_features, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
