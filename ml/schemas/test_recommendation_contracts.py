from ml.schemas.recommendation_contracts import (
    RecommendationRequest,
    UserProfileContext,
    TimeContext,
    CalendarContext,
    WeatherContext,
    RecommendationResponse,
    RecommendationItem,
)


def test_contracts():
    request = RecommendationRequest(
        request_id="test_001",
        user_query="recommend something today at 9 pm",
        profile=UserProfileContext(
            user_email="test@example.com",
            budget_level="medium",
            preferred_areas=["Dubai Marina", "JBR"],
            activity_preferences=["food", "cinema", "outdoor"],
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

    response = RecommendationResponse(
        status="success",
        source="ml",
        request_id=request.request_id,
        model_version="faiss_v1_lightgbm_v1",
        items=[
            RecommendationItem(
                activity_id="place_001",
                name="Example Marina Dinner",
                category="restaurant",
                area="Dubai Marina",
                latitude=25.0800,
                longitude=55.1400,
                estimated_budget_level="medium",
                indoor_outdoor="indoor",
                score=0.91,
                reason="Matches your budget, preferred area, food interest, and available time.",
                constraints_matched=["budget", "location", "calendar", "preference"],
                risk_flags=[],
            )
        ],
        message="Recommendations generated successfully.",
    )

    print("REQUEST OK:")
    print(request.model_dump())

    print("\nRESPONSE OK:")
    print(response.model_dump())


if __name__ == "__main__":
    test_contracts()