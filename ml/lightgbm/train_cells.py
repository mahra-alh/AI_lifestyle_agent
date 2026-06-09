"""
LightGBM ranker training script — production version.

Two label modes controlled by --real-labels flag:

  Synthetic bootstrap (default, --real-labels not set):
    Builds training pairs from the survey CSV + venue pool using the
    rule-based relevance score as a label. Use this to get a first model
    artifact before any real feedback has been collected.

  Real labels (--real-labels):
    Loads inference feature logs + feedback logs from Firestore via
    ml.feedback_join.load_training_frame(). Use this once the system
    has served enough requests to produce meaningful training signal.

Usage
-----
# Synthetic bootstrap (first model):
python -m ml.lightgbm.train_cells \
    --survey    ml/data/augmented_combined.csv \
    --venue-pool ml/data/unified_venue_pool.csv \
    --model-out  ml/models/lgbm_ranker.pkl \
    --version    v2025.06.08

# Real labels (after the system has been serving):
python -m ml.lightgbm.train_cells \
    --survey    ml/data/augmented_combined.csv \
    --venue-pool ml/data/unified_venue_pool.csv \
    --model-out  ml/models/lgbm_ranker.pkl \
    --version    v2025.06.08 \
    --real-labels

# Promote the registered version to production:
python -m ml.lightgbm.train_cells --promote v2025.06.08
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
import hashlib


import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from ml.schemas import feature_contract as fc
from ml.registry.model_registry import ModelRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Haversine helper (used only in the synthetic-pair builder)

def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    if any(pd.isna(x) for x in (lat1, lon1, lat2, lon2)):
        return np.nan
    r = np.radians([lat1, lon1, lat2, lon2])
    dlat, dlon = r[2] - r[0], r[3] - r[1]
    a = np.sin(dlat / 2) ** 2 + np.cos(r[0]) * np.cos(r[2]) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))



# Venue feature builder (training side)
# Mirrors the derived-column logic in feature_builder.py exactly.

def build_venue_features(venues: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the raw venue pool into the contract feature layout.

    Categories are mapped from cat_* → canonical name so they are never
    constant-0 in training. Derived columns (days_open_at_dinner,
    weekend_open_boost) are engineered here; feature_builder.py at serve
    time must mirror these exact formulas.
    """
    v = pd.DataFrame(index=venues.index)

    # Static venue columns that exist directly in the pool.
    for col in [c for c in fc.VENUE_STATIC_COLUMNS if c not in fc.DERIVED_VENUE_COLUMNS]:
        v[col] = pd.to_numeric(venues.get(col), errors="coerce")

    # Engineered derived columns.
    dinner_cols = [c for c in venues.columns if c.startswith("is_open_at_dinner_")]
    v["days_open_at_dinner"] = (
        venues[dinner_cols].fillna(0).astype(float).sum(axis=1)
        if dinner_cols else 0.0
    )
    wkd = pd.to_numeric(venues.get("weekday_open_hours"), errors="coerce").fillna(0.0)
    wke = pd.to_numeric(venues.get("weekend_open_hours"), errors="coerce").fillna(0.0)
    v["weekend_open_boost"] = wke - wkd

    # Category one-hots via the cat_ source mapping.
    for feat, source in fc.VENUE_CATEGORY_SOURCE.items():
        v[feat] = pd.to_numeric(venues.get(source), errors="coerce").fillna(0).astype(int)

    # Carry venue_id and coordinates for the pair builder.
    v["venue_id"] = venues["venue_id"].values
    v["latitude"] = pd.to_numeric(venues.get("latitude"), errors="coerce")
    v["longitude"] = pd.to_numeric(venues.get("longitude"), errors="coerce")
    return v


# Pair interaction + synthetic label

_EXCL_TO_CATS = {
    "excl_nightlife":     ["bar"],
    "excl_cultural":      ["cultural_centre", "gallery", "heritage_site", "library", "museum"],
    "excl_water":         ["beach_waterfront"],
    "excl_outdoor_travel": ["park_attraction", "beach_waterfront"],
}


def _pair_interaction(user: pd.Series, vrow: pd.Series) -> dict:
    """Compute interaction features for one (user, venue) pair."""
    dist = _haversine_km(
        user.get("user_latitude"), user.get("user_longitude"),
        vrow.get("latitude"), vrow.get("longitude"),
    )
    travel_pref = int(user.get("travel_distance_encoded", 2))
    max_km = fc.TRAVEL_DISTANCE_THRESHOLDS.get(travel_pref, 15.0)

    if pd.isna(dist):
        distance_score = 0.3
    elif travel_pref == 4:
        distance_score = 1.0
    elif dist <= max_km * 0.33:
        distance_score = 1.0
    elif dist <= max_km * 0.66:
        distance_score = 0.6
    elif dist <= max_km:
        distance_score = 0.3
    else:
        distance_score = 0.0

    ub = int(user.get("budget_encoded", 1))
    vb = int(vrow.get("budget_encoded", 1))
    budget_diff = abs(ub - vb)
    budget_score = {0: 1.0, 1: 0.5, 2: 0.0}.get(budget_diff, 0.0)

    diet_halal_conflict     = int(user.get("diet_halal", 0) == 1 and vrow.get("serves_alcohol", 0) == 1)
    diet_veg_conflict       = int(
        (user.get("diet_vegetarian", 0) == 1 or user.get("diet_vegan", 0) == 1)
        and vrow.get("grills", 0) == 1 and vrow.get("healthy", 0) == 0
    )
    excl_nightlife_conflict = int(user.get("excl_nightlife", 0) == 1 and vrow.get("bar", 0) == 1)
    excl_cultural_conflict  = int(
        user.get("excl_cultural", 0) == 1
        and any(vrow.get(c, 0) == 1 for c in _EXCL_TO_CATS["excl_cultural"])
    )
    excl_water_conflict     = int(user.get("excl_water", 0) == 1 and vrow.get("beach_waterfront", 0) == 1)
    constraint_score = 0.0 if any([
        diet_halal_conflict, diet_veg_conflict,
        excl_nightlife_conflict, excl_cultural_conflict, excl_water_conflict,
    ]) else 1.0

    if user.get("pref_evening", 0) and vrow.get("is_open_at_dinner_fri", 1):
        time_score = 1.0
    elif user.get("pref_morning", 0) and vrow.get("is_open_at_breakfast_fri", 1):
        time_score = 1.0
    elif user.get("pref_evening", 0) and not vrow.get("is_open_at_dinner_fri", 1):
        time_score = 0.0
    elif user.get("pref_morning", 0) and not vrow.get("is_open_at_breakfast_fri", 1):
        time_score = 0.0
    else:
        time_score = 0.5

    wp      = int(user.get("weather_pref_encoded", 3))
    outdoor = int(vrow.get("has_outdoor_seating", 0))
    weather_score = {3: 1.0, 2: 0.8}.get(
        wp, (1.0 if outdoor else 0.4) if wp == 1 else (0.4 if outdoor else 1.0)
    )

    return {
        "haversine_distance_km": dist,
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
    }


def _synthetic_label(inter: dict) -> float:
    score = (
        0.30 * inter["budget_score"]
        + 0.25 * inter["distance_score"]
        + 0.25 * inter["constraint_score"]
        + 0.10 * inter["time_score"]
        + 0.10 * inter["weather_score"]
    )
    return round(min(score, 1.0), 4)


def build_synthetic_pairs(
    survey: pd.DataFrame,
    venue_feats: pd.DataFrame,
    sample_per_user: int = 200,
    keep_top: int = 10,
    keep_bottom: int = 10,
) -> pd.DataFrame:
    """
    Build bootstrap training pairs from the survey + venue pool.

    For each user, samples up to sample_per_user venues, scores them
    with the rule label, and keeps the top-k and bottom-k to give the
    model both positive and negative examples.
    """
    rows = []
    vf = venue_feats.reset_index(drop=True)

    for uidx, user in survey.iterrows():
        sample = vf.sample(n=min(sample_per_user, len(vf)), random_state=int(uidx))
        scored = [
            (i, _synthetic_label(_pair_interaction(user, vf.loc[i])))
            for i in sample.index
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        selected = scored[:keep_top] + scored[-keep_bottom:]

        for i, label in selected:
            vrow = vf.loc[i]
            inter = _pair_interaction(user, vrow)
            row: dict = {"user_id": uidx, "venue_id": vrow["venue_id"], "label": label}

            for c in fc.VENUE_STATIC_COLUMNS + fc.VENUE_CATEGORY_COLUMNS:
                row[c] = vrow.get(c, 0)
            row.update(inter)
            row.update({
                "outdoor_ok": 1,
                "is_weekend": 1,
                "is_dinner_request": int(user.get("pref_evening", 0)),
                "is_holiday": 0,
            })
            for c in fc.USER_PROFILE_COLUMNS:
                row[c] = user.get(c, 0)

            rows.append(row)

    return pd.DataFrame(rows)


# Training
def train(
    survey_path: str,
    venue_path: str,
    model_out: str,
    version: str,
    use_real_labels: bool,
    notes: str | None = None,
) -> dict:
    """
    Run the full training pipeline and register the artifact.

    Returns the registry metadata dict for the trained version.
    """
    log.info("=== LightGBM training  version=%s  real_labels=%s ===", version, use_real_labels)

    # Load data
    log.info("Loading venue pool: %s", venue_path)
    venues = pd.read_csv(venue_path)
    venue_feats = build_venue_features(venues)
    log.info("Venue features ready: %d venues × %d columns", *venue_feats.shape)

    # Build training frame
    if use_real_labels:
        log.info("Loading real labels from Firestore via feedback_join...")
        from ml.feedback_join import load_training_frame
        train_df = load_training_frame(drop_leaked=True)
        label_source = "real"
    else:
        log.info("Loading survey: %s", survey_path)
        survey = pd.read_csv(survey_path)
        log.info("Building synthetic pairs: %d users", len(survey))
        train_df = build_synthetic_pairs(survey, venue_feats)
        label_source = "synthetic"

    log.info("Training rows: %d", len(train_df))

    # Contract verification
    feature_cols = fc.training_feature_columns(drop_leaked=use_real_labels)
    missing = [c for c in feature_cols if c not in train_df.columns]
    if missing:
        log.error("train_df is missing contract columns: %s", missing)
        sys.exit(1)

    report = fc.diff_columns(
        [c for c in train_df.columns if c in feature_cols],
        feature_cols,
    )
    if not report["ok"]:
        log.error("Feature contract mismatch: %s", report)
        sys.exit(1)

    log.info(
        "Contract OK — %d features%s",
        len(feature_cols),
        " (leaked sub-scores dropped)" if use_real_labels else " (synthetic bootstrap)",
    )

    # Train / test split (by user so no data leakage)
    users = train_df["user_id"].unique()
    rng = np.random.default_rng(42)
    test_users = set(rng.choice(users, max(1, int(len(users) * 0.2)), replace=False))
    train_mask = ~train_df["user_id"].isin(test_users)

    X_train = train_df.loc[train_mask, feature_cols].copy()
    y_train = train_df.loc[train_mask, "label"].values
    X_test  = train_df.loc[~train_mask, feature_cols].copy()
    y_test  = train_df.loc[~train_mask, "label"].values

    log.info(
        "Split: %d train rows (%d users) / %d test rows (%d users)",
        len(X_train), len(users) - len(test_users),
        len(X_test), len(test_users),
    )

    # Encode LightGBM categorical features.
    cat_features = [c for c in fc.CATEGORICAL_FEATURES if c in feature_cols]
    for c in cat_features:
        X_train[c] = X_train[c].astype("int").astype("category")
        X_test[c]  = X_test[c].astype("int").astype("category")

    # Fit
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
        eval_set=[(X_test, y_test)],
        callbacks=[
            lgb.early_stopping(20, verbose=False),
            lgb.log_evaluation(50),
        ],
    )

    test_mse = float(((model.predict(X_test) - y_test) ** 2).mean())
    log.info("best_iteration: %d", model.best_iteration_)
    log.info("test MSE: %.6f", test_mse)

    if not use_real_labels:
        log.warning(
            "Low MSE reflects the model re-deriving the rule (leaked sub-scores are "
            "in the features). Treat as a sanity check, not a quality metric. "
            "Real quality = NDCG on held-out feedback."
        )

    # Feature importances (top 15)
    imp = (
        pd.Series(model.feature_importances_, index=feature_cols)
        .sort_values(ascending=False)
        .head(15)
    )
    log.info("Top 15 feature importances:\n%s", imp.to_string())

    # Save artifact
    artifact_path = Path(model_out)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, artifact_path)
    log.info("Model saved: %s", artifact_path)

    features_json_path = Path(str(artifact_path) + ".features.json")
    with features_json_path.open("w") as f:
        json.dump(
            {
                "version": version,
                "feature_columns": feature_cols,
                "feature_set_hash": fc.feature_set_hash(),
                "label_source": label_source,
                "trained_at": datetime.now(timezone.utc).isoformat(),
            },
            f,
            indent=2,
        )
    log.info("Feature manifest saved: %s", features_json_path)

    # Round-trip contract assertion.
    reloaded = joblib.load(artifact_path)
    fc.assert_booster_matches(reloaded, expected=feature_cols)
    log.info("Round-trip contract assertion passed.")

    # Register with model registry
    try:
        registry = ModelRegistry()
        metadata = registry.register(
            version=version,
            artifact_path=str(artifact_path),
            feature_set_hash=fc.feature_set_hash(),
            training_rows=len(train_df),
            label_source=label_source,
            metrics={"test_mse": test_mse, "best_iteration": model.best_iteration_},
            notes=notes,
            status="staging",
        )
        log.info(
            "Registered in model registry as '%s' (status: staging). "
            "Run with --promote %s to make it active.",
            version, version,
        )
        return metadata
    except RuntimeError as e:
        log.warning("Skipping model registry: %s", e)
        return {}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train or promote a LightGBM ranking model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    p.add_argument("--survey",      default="ai_agent/data/augmented_combined.csv",  help="Path to user survey CSV")
    p.add_argument("--venue-pool",  default="ai_agent/data/unified_venue_pool.csv",  help="Path to venue pool CSV")
    p.add_argument("--model-out",   default="ml/models/lgbm_ranker.pkl",       help="Output path for the model artifact")
    p.add_argument("--version",     default=datetime.now(timezone.utc).strftime("v%Y.%m.%d"), help="Version string for the registry")
    p.add_argument("--real-labels", action="store_true", help="Use real Firestore feedback instead of synthetic labels")
    p.add_argument("--notes",       default=None, help="Free-text notes stored in the registry")
    p.add_argument("--promote",     default=None, metavar="VERSION", help="Promote an already-registered version to active and exit")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Promote-only mode: no training needed
    if args.promote:
        try:
            registry = ModelRegistry()
            registry.promote(args.promote)
            log.info("Promoted '%s' to active production.", args.promote)
        except RuntimeError as e:
            log.warning("Skipping model registry: %s", e)
        return

    train(
        survey_path=args.survey,
        venue_path=args.venue_pool,
        model_out=args.model_out,
        version=args.version,
        use_real_labels=args.real_labels,
        notes=args.notes,
    )


if __name__ == "__main__":
    main()
