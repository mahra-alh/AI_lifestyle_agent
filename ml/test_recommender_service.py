from ml.recommender_service import LifestyleRecommender
from ml.schemas.recommendation_contracts import (
    RecommendationRequest,
    UserProfileContext,
    TimeContext,
    CalendarContext,
    WeatherContext,
)


def test_recommender_service():
    request = RecommendationRequest(
        request_id="test_phase_3_001",
        user_query="recommend something today at 9 pm",
        profile=UserProfileContext(
            user_email="test@example.com",
            budget_level="medium",
            preferred_areas=["Dubai Marina", "JBR"],
            activity_preferences=["food", "cinema", "outdoor"],
            disliked_activities=[],
            transport_mode="metro",
            profile_complete=True,
        ),
        time_context=TimeContext(
            requested_date="2026-05-26",
            requested_start_time="21:00",
            timezone="Asia/Dubai",
        ),
        calendar_context=CalendarContext(
            calendar_checked=True,
            is_free=True,
            free_slot_start="2026-05-26T20:30:00+04:00",
            free_slot_end="2026-05-26T23:00:00+04:00",
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

    recommender = LifestyleRecommender()
    response = recommender.recommend(request)

    print("STATUS:", response.status)
    print("SOURCE:", response.source)
    print("MODEL VERSION:", response.model_version)
    print("MESSAGE:", response.message)
    print("FALLBACK REASON:", response.fallback_reason)

    print("\nITEMS:")
    for item in response.items:
        print(item.model_dump())


if __name__ == "__main__":
    test_recommender_service()