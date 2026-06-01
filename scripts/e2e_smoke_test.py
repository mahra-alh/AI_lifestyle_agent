from __future__ import annotations

import json
import os
import sys
from typing import Any

import requests


BACKEND_URL = os.getenv("E2E_BACKEND_URL", "http://localhost:5000").rstrip("/")
USER_EMAIL = os.getenv("E2E_USER_EMAIL", "demo@example.com")
USER_MESSAGE = os.getenv(
    "E2E_USER_MESSAGE",
    (
        "My email is {email}. Recommend something for tonight in Dubai from "
        "20:00 to 22:00. Check my profile, calendar, weather, and use the ML "
        "recommender. Do not book anything yet."
    ),
).format(email=USER_EMAIL)
BOOKING_CONFIRMATION_MESSAGE = os.getenv("E2E_BOOKING_CONFIRMATION_MESSAGE", "").strip()


class SmokeFailure(RuntimeError):
    pass


def request_json(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
    url = f"{BACKEND_URL}{path}"
    response = requests.request(method, url, timeout=180, **kwargs)

    try:
        payload = response.json()
    except ValueError as error:
        raise SmokeFailure(f"{method} {path} returned non-JSON response: {response.text[:300]}") from error

    if response.status_code >= 500:
        raise SmokeFailure(f"{method} {path} failed with {response.status_code}: {json.dumps(payload, indent=2)}")

    return {
        "status_code": response.status_code,
        "body": payload,
    }


def assert_condition(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)


def check_backend_health() -> dict[str, Any]:
    result = request_json("GET", "/health")
    body = result["body"]
    dependencies = body.get("dependencies", {})

    assert_condition(result["status_code"] == 200, "Backend /health did not return HTTP 200.")
    assert_condition(dependencies.get("python_agent_reachable") is True, "Backend cannot reach Python AI service.")

    return body


def check_backend_readiness() -> dict[str, Any]:
    result = request_json("GET", "/ready")
    body = result["body"]

    assert_condition(result["status_code"] in {200, 503}, "Backend /ready returned an unexpected HTTP status.")
    return body


def check_agent_chat(user_message: str) -> dict[str, Any]:
    result = request_json(
        "POST",
        "/api/chat",
        json={
            "user_id": USER_EMAIL,
            "user_message": user_message,
        },
    )
    body = result["body"]

    assert_condition(result["status_code"] == 200, f"Chat request failed with HTTP {result['status_code']}.")
    assert_condition(body.get("success") is True, "Chat response did not report success=true.")
    assert_condition(bool(str(body.get("response", "")).strip()), "Chat response was empty.")

    return body


def summarize_health(health: dict[str, Any]) -> dict[str, Any]:
    python_agent = health.get("python_agent", {})
    dependencies = python_agent.get("dependencies", {})
    recommender = dependencies.get("recommender", {})
    calendar_auth = dependencies.get("calendar_auth", {})

    return {
        "backend_ok": health.get("ok"),
        "python_agent_ready": dependencies.get("ready"),
        "booking_ready": dependencies.get("booking_ready"),
        "calendar_auth": {
            "booking_required": calendar_auth.get("booking_required"),
            "booking_ready": calendar_auth.get("booking_ready"),
            "credentials_source": calendar_auth.get("credentials_source"),
            "token_source": calendar_auth.get("token_source"),
        },
        "recommender": {
            "ready": recommender.get("ready"),
            "faiss_ready": recommender.get("faiss_ready"),
            "lightgbm_ready": recommender.get("lightgbm_ready"),
            "lightgbm_fallback_active": recommender.get("lightgbm_fallback_active"),
            "artifact_errors": recommender.get("artifact_errors"),
            "artifact_warnings": recommender.get("artifact_warnings"),
        },
    }


def main() -> int:
    print(f"Smoke target: {BACKEND_URL}")
    print(f"Smoke user: {USER_EMAIL}")

    health = check_backend_health()
    readiness = check_backend_readiness()
    chat = check_agent_chat(USER_MESSAGE)

    booking = None
    if BOOKING_CONFIRMATION_MESSAGE:
        booking = check_agent_chat(BOOKING_CONFIRMATION_MESSAGE)

    print(
        json.dumps(
            {
                "health_summary": summarize_health(health),
                "ready_status": readiness.get("status"),
                "chat_response_preview": str(chat.get("response", ""))[:500],
                "booking_response_preview": str(booking.get("response", ""))[:500] if booking else None,
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SmokeFailure as error:
        print(f"SMOKE TEST FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
