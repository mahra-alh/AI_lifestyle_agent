# AI Lifestyle Agent

An AI-powered lifestyle assistant that recommends meaningful free-time activities and books the selected option directly into the user's Google Calendar. The system combines personal preferences, budget, time availability, location, live weather, and an ML ranking model to generate personalised recommendations for Dubai.

---

## How it works

1. The user sends a message through the frontend (e.g. "Suggest something for tonight after 8pm").
2. The Node.js backend gateway forwards the request to the Python AI service.
3. The AI Agent (OpenAI `gpt-4o-mini`) collects the user's profile from Firestore, checks Google Calendar availability, and fetches a 7-day weather forecast.
4. When the profile and context are ready, the agent calls the ML recommender, which uses FAISS semantic search to retrieve candidate venues and LightGBM to rank them.
5. The agent presents the ranked options and, after user confirmation, books the chosen activity in Google Calendar.

---

## Architecture

```
Browser (React)
  └─► Node.js backend gateway  (port 5000)
         └─► Python AI service  (port 8000)
                ├── Agent (OpenAI Agents SDK)
                │     ├── user_profile tool  → Firestore
                │     ├── get_calendar tool  → Google Calendar API
                │     ├── get_weather tool   → Visual Crossing API
                │     └── get_ml_recommendations → ML recommender
                └── ML recommender
                      ├── FAISS semantic search  (ml/artifacts/)
                      └── LightGBM re-ranker     (ml/artifacts/)
```

---

## Folder structure

```
AI_lifestyle_agent/
├── ai_agent/
│   ├── app.py               # Agent definition and run_agent()
│   ├── server.py            # FastAPI server (entry point for Python service)
│   ├── tools/
│   │   ├── calendar_tool.py # Google Calendar availability + booking
│   │   ├── user_profile.py  # Firestore profile read/write + ML export
│   │   └── weather_tool.py  # Visual Crossing weather forecast
│   ├── storage/
│   │   └── firestore_store.py  # Firestore read/write helpers
│   ├── logger/
│   │   └── app_logger.py    # Structured JSONL logger
│   └── data/
│       └── dubai_areas.py   # Dubai area coordinates lookup
│
├── ml/
│   ├── recommender_service.py       # Main ML recommender (FAISS + LightGBM)
│   ├── schemas/
│   │   └── recommendation_contracts.py  # Pydantic request/response models
│   ├── lightgbm/
│   │   └── feature_builder.py       # LightGBM feature engineering
│   ├── faiss/
│   │   ├── semantic_search.py       # FAISSRetriever class (used by tests)
│   │   └── build_faiss_artifacts.py # Script to rebuild FAISS index from corpus
│   ├── artifacts/                   # Runtime model files (not committed to git)
│   │   ├── faiss_index.bin
│   │   ├── faiss_lookup.csv
│   │   ├── faiss_manifest.json
│   │   ├── lightgbm_model.txt
│   │   ├── lightgbm_manifest.json
│   │   └── lightgbm_model_features.json
│   ├── data/
│   │   └── faiss_corpus.csv         # Venue text corpus for FAISS (committed)
│   └── notebooks/
│       ├── faiss/build_faiss.py     # Notebook-derived FAISS build script
│       └── lightgbm/light_test.py   # Notebook-derived LightGBM training script
│
├── backend/
│   ├── server.js            # Node.js gateway (routes frontend requests to Python)
│   ├── firebaseAdmin.js     # Firebase Admin SDK initialisation
│   ├── import_venues_to_firestore.js  # One-time data import script
│   └── config/
│       └── serviceAccountKey.json   # Placeholder only — use env vars in production
│
├── frontend/
│   └── src/
│       ├── App.js           # React app (connection test UI)
│       └── firebase.js      # Firebase client config
│
├── scripts/
│   ├── e2e_smoke_test.py    # End-to-end smoke test (backend → agent → ML)
│   └── download_embedding_model.py  # Pre-download embedding model for offline use
│
├── Dockerfile               # Python AI service container
├── docker-compose.yml       # Runs python-agent + backend together
├── requirements.txt         # Python dependencies
└── DEPLOYMENT.md            # Deployment guide (Cloud Run, secrets, health checks)
```

---

## Prerequisites

- Python 3.12+
- Node.js 20+
- A Firebase project with Firestore enabled
- An OpenAI API key
- A Visual Crossing API key (free tier works)
- Google Calendar OAuth credentials (optional — needed only for calendar booking)

---

## Environment variables

Copy `.env.example` to `.env` and fill in the values:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | OpenAI API key for the agent |
| `WEATHER_API_KEY` | Yes | Visual Crossing API key |
| `FIREBASE_PROJECT_ID` | Yes | Firebase project ID |
| `FIREBASE_SERVICE_ACCOUNT_JSON` | Production | Firebase Admin service account JSON (for Cloud Run) |
| `FIREBASE_SERVICE_ACCOUNT_PATH` | Local dev | Path to service account JSON file |
| `FIRESTORE_USER_PROFILE_COLLECTION` | No | Firestore collection name (default: `user_profiles`) |
| `GOOGLE_CALENDAR_CREDENTIALS_JSON` | Optional | Google Calendar OAuth client JSON (for Cloud Run) |
| `GOOGLE_CALENDAR_TOKEN_JSON` | Optional | Google Calendar OAuth token JSON (for Cloud Run) |
| `GOOGLE_CALENDAR_CREDENTIALS_PATH` | Optional | Path to OAuth credentials file (local dev) |
| `GOOGLE_CALENDAR_TOKEN_PATH` | Optional | Path to OAuth token file (local dev) |
| `REQUIRE_CALENDAR_AUTH` | No | Set to `true` to fail readiness if calendar auth is missing |
| `REQUIRE_LIGHTGBM_ARTIFACTS` | No | Set to `true` to fail readiness if LightGBM model is missing |
| `CORS_ORIGINS` | No | Allowed frontend origins (comma-separated) |
| `PORT` | No | Node backend port (default: `5000`) |
| `PYTHON_AGENT_URL` | No | URL of Python AI service (default: `http://127.0.0.1:8000`) |
| `REACT_APP_API_URL` | Frontend | Backend URL for the React app |
| `REACT_APP_FIREBASE_*` | Frontend | Firebase web config values |

> **Security note:** Never commit `.env`, `credentials.json`, `token.json`, or any real service account JSON to git.
> These files are already listed in `.gitignore`. For Cloud Run, store secrets in Secret Manager.

---

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Node backend dependencies

```bash
cd backend
npm install
cd ..
```

### 3. Install frontend dependencies (optional, for local UI)

```bash
cd frontend
npm install
cd ..
```

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY, WEATHER_API_KEY, Firebase variables
```

### 5. Set up Firebase credentials (local dev)

Place your Firebase service account JSON file at `backend/config/serviceAccountKey.json`, or set `FIREBASE_SERVICE_ACCOUNT_JSON` in `.env`.

### 6. Set up Google Calendar (optional)

Place your OAuth credentials at `credentials.json` (root of the project), then run the calendar tool once interactively to generate `token.json`. For production, store both as Secret Manager secrets.

---

## Run locally

Start the Python AI service:

```bash
uvicorn ai_agent.server:app --reload --host 0.0.0.0 --port 8000
```

In a separate terminal, start the Node backend gateway:

```bash
cd backend
npm start
```

Optionally start the frontend:

```bash
cd frontend
npm start
```

Service URLs:
- Python AI service: `http://localhost:8000`
- Node backend gateway: `http://localhost:5000`
- Frontend: `http://localhost:3000`

---

## Run with Docker

```bash
docker compose up --build
```

This starts the Python AI service on port 8000 and the Node backend on port 5000.
Environment variables are loaded from `.env` via `env_file` in `docker-compose.yml`.

---

## ML artifacts

The recommender loads pre-built artifacts from `ml/artifacts/` at startup.

**Required for FAISS semantic search:**
- `ml/artifacts/faiss_index.bin`
- `ml/artifacts/faiss_lookup.csv`

**Required for LightGBM ranking (optional — falls back to rule-based ranking if missing):**
- `ml/artifacts/lightgbm_model.txt`
- `ml/artifacts/lightgbm_manifest.json`
- `ml/artifacts/lightgbm_model_features.json`

The FAISS index (`.bin`) and lookup table (`.csv`) are excluded from git by `.gitignore` because they are large binary/CSV files.
The LightGBM model files (`.txt`, `.json`) are not gitignored and can be committed.

Both artifact sets must be present in `ml/artifacts/` before the service can serve recommendations.

**Build the FAISS index** (required before first run and after updating `ml/data/faiss_corpus.csv`):

```bash
python ml/faiss/build_faiss_artifacts.py
```

**To retrain the LightGBM model** (requires survey data — see `ml/notebooks/lightgbm/light_test.py`):

```bash
python ml/notebooks/lightgbm/light_test.py
```

---

## Run tests

Unit tests (no running services required):

```bash
# Test Pydantic contracts
python ml/schemas/test_recommendation_contracts.py

# Test LightGBM feature builder (no model artifact needed)
python -m pytest ml/test_lightgbm_feature_builder.py -v

# Test full recommender (requires ml/artifacts/)
python ml/test_recommender_service.py
```

---

## End-to-end smoke test

Requires both services running locally (or on Cloud Run):

```bash
python scripts/e2e_smoke_test.py
```

Override defaults with environment variables:

```bash
E2E_BACKEND_URL=http://localhost:5000 \
E2E_USER_EMAIL=your@email.com \
python scripts/e2e_smoke_test.py
```

---

## Health checks

| Endpoint | Service | Description |
|---|---|---|
| `GET /health` | Python (8000) | AI service + dependency status |
| `GET /ready` | Python (8000) | Returns 200 if agent can serve traffic, 503 if not |
| `POST /api/chat` | Python (8000) | Send a message to the agent |
| `GET /health` | Node (5000) | Gateway + Python service reachability |
| `GET /api/test` | Node (5000) | Quick gateway check |
| `POST /api/chat` | Node (5000) | Forwards to Python `/api/chat` |

---

## Deployment notes

See [DEPLOYMENT.md](DEPLOYMENT.md) for:
- Cloud Run deployment steps
- Secret Manager configuration for credentials
- Google Calendar auth setup for production
- FAISS and LightGBM artifact loading behaviour
- Full smoke test instructions for production

---

## Remaining before first production deployment

1. **Build FAISS artifacts** (gitignored, must be present in `ml/artifacts/`):
   ```bash
   python ml/faiss/build_faiss_artifacts.py
   ```
2. **Commit LightGBM artifacts** (in `ml/artifacts/` — not gitignored, ready to commit once verified)
3. Set real `OPENAI_API_KEY` in Cloud Run environment
4. Set real `WEATHER_API_KEY` in Cloud Run environment
5. Create Firebase project, enable Firestore, download service account JSON → store in Secret Manager as `FIREBASE_SERVICE_ACCOUNT_JSON`
6. Set `FIREBASE_PROJECT_ID` in Cloud Run environment
7. Authorize Google Calendar OAuth credentials → store both JSONs in Secret Manager as `GOOGLE_CALENDAR_CREDENTIALS_JSON` and `GOOGLE_CALENDAR_TOKEN_JSON`
8. Set `REACT_APP_FIREBASE_*` variables and `REACT_APP_API_URL` for the frontend
9. Deploy frontend to Firebase Hosting or similar
