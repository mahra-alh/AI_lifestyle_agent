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

APP_NAME = "AI Lifestyle Agent API"
APP_VERSION = "0.1.0"


class ChatRequest(BaseModel):
    user_id: str = Field(..., description="Stable user identifier, usually the email address.")
    user_message: str = Field(..., description="User message to send to the agent.")


class ChatResponse(BaseModel):
    success: bool
    user_id: str
    response: str | dict[str, Any]


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


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    user_id = request.user_id.strip().lower()
    user_message = request.user_message.strip()

    if not user_id or not user_message:
        raise HTTPException(
            status_code=400,
            detail="user_id and user_message cannot be empty.",
        )

    try:
        result = run_agent(user_id=user_id, user_message=user_message)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    return ChatResponse(success=True, user_id=user_id, response=result)
