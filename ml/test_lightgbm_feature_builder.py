import math

import pandas as pd

from ml.lightgbm.feature_builder import (
    FEATURE_COLUMNS,
    build_lightgbm_feature_frame,
    calculate_rule_rank_score,
)
from ml.schemas.recommendation_contracts import (
    CalendarContext,
    RecommendationRequest,
    TimeContext,
    UserProfileContext,
    WeatherContext,
)


def test_lightgbm_feature_builder_derives_expected_columns_and_signals():
    request = RecommendationRequest(
        request_id="feature_builder_test_001",
        user_query="recommend something near Dubai Marina this evening",
        profile=UserProfileContext(
            user_email="test@example.com",
            budget_level="medium",
            home_area="Dubai Marina",
            preferred_areas=["Dubai Marina"],
            max_per_activity_aed=200,
            max_travel_distance_km=10,
            preferred_environment="indoor",
            activity_preferences=["coffee", "outdoor"],
            disliked_activities=["nightlife"],
            transport_mode="metro",
            profile_complete=True,
        ),
        time_context=TimeContext(
            requested_date="2026-05-27",
            requested_start_time="21:00",
            requested_end_time="23:00",
            timezone="Asia/Dubai",
        ),
        calendar_context=CalendarContext(
            calendar_checked=True,
            is_free=True,
            free_slot_start="2026-05-27T20:30:00+04:00",
            free_slot_end="2026-05-27T23:30:00+04:00",
        ),
        weather_context=WeatherContext(
            weather_checked=True,
            condition="clear",
            temperature_celsius=31.0,
            outdoor_suitable=True,
            weather_risk="low",
        ),
        top_k=5,
    )

    candidates = [
        pd.Series(
            {
                "venue_id": "venue_001",
                "faiss_text": "maze specialty coffee al barsha south 1 desserts coffee cafe low",
                "_similarity_score": 0.84,
            }
        )
    ]

    feature_frame = build_lightgbm_feature_frame(candidates, request)

    assert list(feature_frame.columns) == FEATURE_COLUMNS
    assert len(feature_frame) == 1
    assert feature_frame.loc[0, "budget_encoded"] == 1
    assert feature_frame.loc[0, "cafe"] == 1
    assert feature_frame.loc[0, "pref_evening"] == 1
    assert feature_frame.loc[0, "is_dinner_request"] == 1
    assert feature_frame.loc[0, "travel_distance_encoded"] == 2
    assert feature_frame.loc[0, "weather_pref_encoded"] == 0
    assert feature_frame.loc[0, "budget_score"] == 0.5
    assert feature_frame.loc[0, "constraint_score"] == 1.0
    assert not math.isnan(float(feature_frame.loc[0, "haversine_distance_km"]))

    rule_scores = calculate_rule_rank_score(feature_frame)
    assert len(rule_scores) == 1
    assert 0.0 <= float(rule_scores.iloc[0]) <= 1.0
