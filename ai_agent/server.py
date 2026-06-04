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
from ai_agent.tools.calendar_tool import get_calendar_auth_status
from ai_agent.tools.user_profile import log_recommendation_feedback,  get_user_profile_data

APP_NAME = "AI Lifestyle Agent API"
APP_VERSION = "0.1.0"

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