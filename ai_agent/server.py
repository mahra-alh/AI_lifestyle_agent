from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

from ai_agent.app import run_agent, _get_ml_recommender
from ai_agent.tools.calendar.auth import get_calendar_auth_status
from ai_agent.tools.user_profile import log_recommendation_feedback, get_user_profile_data, update_user_profile_data

APP_NAME = "AI Lifestyle Agent API"
APP_VERSION = "0.1.0"

class ProfileSetupRequest(BaseModel):
    user_id: str
    home_area: str
    monthly_fun_budget_aed: int
    max_per_activity_aed: int
    hobbies: list[str] = []
    interests: list[str] = []
    preferred_activity_types: list[str] = []
    work_days: list[str] = []
    work_start_time: str = "09:00"
    work_end_time: str = "18:00"
    preferred_environment: str = "mixed"
    adventure_level_encoded: int = 1
    social_alone: int = 0
    social_partner: int = 0
    social_friends: int = 0
    social_family: int = 0
    diet_halal: int = 0
    diet_vegetarian: int = 0
    diet_vegan: int = 0
    diet_gluten_free: int = 0

class ChatRequest(BaseModel):
    user_id: str = Field(..., description="Stable user identifier, usually the email address.")
    user_message: str = Field(..., description="User message to send to the agent.")

class ChatResponse(BaseModel):
    success: bool
    user_id: str
    response: str | dict[str, Any]

class FeedbackRequest(BaseModel):
    user_id: str
    recommendation_id: str
    activity_id: str
    activity_name: str
    feedback: str
    rating: int | None = None
    feedback_reason: str | None = None
    recommendation_rank: int | None = None
    activity_category: str | None = None
    activity_area: str | None = None
    model_version: str | None = None
    session_id: str | None = None
    user_query: str | None = None

def _dependency_snapshot() -> dict[str, Any]:
    recommender_status = _get_ml_recommender().get_status()
    weather_api_key_present = bool(
        os.getenv("WEATHER_API_KEY") or os.getenv("VISUAL_CROSSING_API_KEY")
    )
    openai_api_key_present = bool(os.getenv("OPENAI_API_KEY"))
    calendar_auth = get_calendar_auth_status()

    ready = (
        openai_api_key_present
        and weather_api_key_present
        and recommender_status["ready"]
    )

    booking_ready = ready and calendar_auth["booking_ready"]
    if calendar_auth["booking_required"]:
        ready = ready and calendar_auth["booking_ready"]

    return {
        "ready": ready,
        "booking_ready": booking_ready,
        "openai_api_key_present": openai_api_key_present,
        "weather_api_key_present": weather_api_key_present,
        "calendar_auth": calendar_auth,
        "recommender": recommender_status,
    }


app = FastAPI(title=APP_NAME, version=APP_VERSION)

allowed_origins_env = os.getenv("CORS_ORIGINS", "").strip()
allowed_origins = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]

if allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

@app.get("/")
def root() -> dict[str, Any]:
    snapshot = _dependency_snapshot()
    return {
        "service": APP_NAME,
        "version": APP_VERSION,
        "status": "ok",
        "utc_now": datetime.now(timezone.utc).isoformat(),
        "ready": snapshot["ready"],
    }

@app.get("/health")
def health() -> dict[str, Any]:
    snapshot = _dependency_snapshot()
    return {
        "service": APP_NAME,
        "version": APP_VERSION,
        "status": "healthy",
        "utc_now": datetime.now(timezone.utc).isoformat(),
        "dependencies": snapshot,
    }

@app.get("/ready")
def ready() -> dict[str, Any]:
    snapshot = _dependency_snapshot()
    if not snapshot["ready"]:
        raise HTTPException(status_code=503, detail=snapshot)

    return {
        "service": APP_NAME,
        "version": APP_VERSION,
        "status": "ready",
        "dependencies": snapshot,
    }

@app.post("/api/profile")
def setup_profile(request: ProfileSetupRequest) -> dict[str, Any]:
    user_id = request.user_id.strip().lower()
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id cannot be empty.")
    try:
        return update_user_profile_data(
            user_id=user_id,
            home_area=request.home_area,
            monthly_fun_budget_aed=request.monthly_fun_budget_aed,
            max_per_activity_aed=request.max_per_activity_aed,
            hobbies=request.hobbies,
            interests=request.interests,
            preferred_activity_types=request.preferred_activity_types,
            work_days=request.work_days,
            work_start_time=request.work_start_time,
            work_end_time=request.work_end_time,
            preferred_environment=request.preferred_environment,
            adventure_level_encoded=request.adventure_level_encoded,
            social_alone=request.social_alone,
            social_partner=request.social_partner,
            social_friends=request.social_friends,
            social_family=request.social_family,
            diet_halal=request.diet_halal,
            diet_vegetarian=request.diet_vegetarian,
            diet_vegan=request.diet_vegan,
            diet_gluten_free=request.diet_gluten_free,
        )
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

@app.get("/api/profile/{user_id}")
def profile(user_id: str) -> dict[str, Any]:
    cleaned_user_id = user_id.strip().lower()

    if not cleaned_user_id:
        raise HTTPException(
            status_code=400,
            detail="user_id cannot be empty.",
        )

    try:
        return get_user_profile_data(cleaned_user_id)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

@app.post("/api/feedback")
def feedback(request: FeedbackRequest) -> dict[str, Any]:
    user_id = request.user_id.strip().lower()

    if not user_id:
        raise HTTPException(status_code=400, detail="user_id cannot be empty.")

    try:
        return log_recommendation_feedback(
            user_id=user_id,
            recommendation_id=request.recommendation_id,
            activity_id=request.activity_id,
            activity_name=request.activity_name,
            feedback=request.feedback,
            rating=request.rating,
            feedback_reason=request.feedback_reason,
            recommendation_rank=request.recommendation_rank,
            activity_category=request.activity_category,
            activity_area=request.activity_area,
            model_version=request.model_version,
            session_id=request.session_id,
            user_query=request.user_query,
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    user_id = request.user_id.strip().lower()
    user_message = request.user_message.strip()

    if not user_id or not user_message:
        raise HTTPException(
            status_code=400,
            detail="user_id and user_message cannot be empty.",
        )

    try:
        result = await run_agent(user_id=user_id, user_message=user_message)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    return ChatResponse(success=True, user_id=user_id, response=result)