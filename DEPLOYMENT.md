# Deployment Guide

## Services

- `python-agent`: FastAPI app in `ai_agent/server.py`
- `backend`: Node gateway in `backend/server.js`
- `frontend`: React app in `frontend/`
- Node 20+ for the backend gateway

## Required environment

Copy `.env.example` to `.env` and set the values for:

- `OPENAI_API_KEY`
- `WEATHER_API_KEY` or `VISUAL_CROSSING_API_KEY`
- `REACT_APP_API_URL`
- `REACT_APP_FIREBASE_*` variables
- `GOOGLE_CALENDAR_CREDENTIALS_JSON` and `GOOGLE_CALENDAR_TOKEN_JSON` for Google Calendar booking in Cloud Run
- `FIREBASE_SERVICE_ACCOUNT_JSON` or Cloud Run application default credentials for Firestore
- `REQUIRE_LIGHTGBM_ARTIFACTS=true` if deployment should fail readiness when the LightGBM model cannot load
- `REQUIRE_CALENDAR_AUTH=true` if deployment should fail readiness when Google Calendar booking is not configured

For local development, file paths are also supported:

- `GOOGLE_CALENDAR_CREDENTIALS_PATH=credentials.json`
- `GOOGLE_CALENDAR_TOKEN_PATH=token.json`
- `FIREBASE_SERVICE_ACCOUNT_PATH=backend/config/serviceAccountKey.json`

Do not commit real `.env`, `credentials.json`, `token.json`, or service account files.
For Cloud Run, store secret values in Secret Manager and expose them to the service as environment variables.

## Google Calendar booking auth

The Calendar tools do not open an interactive browser login in normal app execution.
That means production must provide auth before the service starts.

For local development:

- Place OAuth client credentials at `GOOGLE_CALENDAR_CREDENTIALS_PATH`.
- Generate or provide the OAuth token at `GOOGLE_CALENDAR_TOKEN_PATH`.

For Cloud Run:

- Store the OAuth client JSON in Secret Manager and expose it as `GOOGLE_CALENDAR_CREDENTIALS_JSON`.
- Store the authorized user token JSON in Secret Manager and expose it as `GOOGLE_CALENDAR_TOKEN_JSON`.
- Set `REQUIRE_CALENDAR_AUTH=true` if calendar booking is part of the production demo.

The `/health` and `/ready` responses include `calendar_auth` without exposing secret values.
If `REQUIRE_CALENDAR_AUTH=true`, `/ready` fails unless both credentials and token config are present.

## Local start

```bash
pip install -r requirements.txt
cd backend
npm install
cd ..
uvicorn ai_agent.server:app --reload --host 0.0.0.0 --port 8000
```

In a second terminal:

```bash
cd backend
npm start
```

## Docker

```bash
docker compose up --build
```

## ML artifacts

The Python AI service loads runtime model files from `ml/artifacts/`.

Required for FAISS retrieval:

- `ml/artifacts/faiss_index.bin`
- `ml/artifacts/faiss_lookup.csv`

Required for trained LightGBM ranking:

- `ml/artifacts/lightgbm_model.txt`
- `ml/artifacts/lightgbm_manifest.json`
- `ml/artifacts/lightgbm_model_features.json`

If FAISS is missing or cannot load, `/ready` fails because recommendations cannot be generated.
If LightGBM is missing or its manifest does not match the serving feature columns, the app falls back to rule-based ranking. Set `REQUIRE_LIGHTGBM_ARTIFACTS=true` in deployment to make `/ready` fail instead.

The local service URLs are:

- Python AI service: `http://localhost:8000`
- Node backend gateway: `http://localhost:5000`
- Backend-to-agent URL: `PYTHON_AGENT_URL=http://python-agent:8000` inside Docker Compose

For Cloud Run, deploy the Python AI service and set the Node backend's
`PYTHON_AGENT_URL` to the Python service URL, for example:

```bash
PYTHON_AGENT_URL=https://your-python-agent-service-xxxxx.a.run.app
```

## Health checks

- Python service `GET /health` reports AI service and dependency status.
- Python service `GET /ready` reports whether the agent can serve traffic.
- Python service `POST /api/chat` runs the agent.
- Backend `GET /api/test` verifies the gateway is reachable.
- Backend `POST /api/chat` forwards chat requests to the Python service.

## End-to-end smoke test

Use the smoke test after starting the Python service and backend locally, or after deploying them to Cloud Run.
The test goes through the same path the frontend uses:

```text
Backend /api/chat -> Python /api/chat -> agent tools -> recommender/calendar/weather
```

Local run:

```bash
python scripts/e2e_smoke_test.py
```

Cloud Run run:

```bash
E2E_BACKEND_URL=https://your-backend-service-xxxxx.a.run.app python scripts/e2e_smoke_test.py
```

Useful test variables:

- `E2E_BACKEND_URL`: backend URL to test.
- `E2E_USER_EMAIL`: test user email. This user should already have a complete profile in Firestore.
- `E2E_USER_MESSAGE`: recommendation request prompt.
- `E2E_BOOKING_CONFIRMATION_MESSAGE`: optional follow-up prompt to confirm booking one recommendation.

For a full booking test, set `REQUIRE_CALENDAR_AUTH=true` and provide Google Calendar credentials/token through local files or Secret Manager env vars before running the smoke test.
