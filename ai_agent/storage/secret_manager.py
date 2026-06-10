"""
Fetches secrets from Google Cloud Secret Manager and injects them into
os.environ so the rest of the codebase reads them exactly as before via
os.getenv() — no other files need to change.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile

log = logging.getLogger(__name__)

#  GCP project ID
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "ai-lifestyle-agent-7e96c")

# Map each Secret Manager secret name to the env var the codebase expects.
# Secrets whose value is a JSON service-account file are handled separately.
_PLAIN_SECRETS: dict[str, str] = {
    "OPENAI_API_KEY":  "OPENAI_API_KEY",
    "WEATHER_API_KEY": "WEATHER_API_KEY",
    "REDIS_PASSWORD":  "REDIS_PASSWORD",
}

# JSON service-account secrets — written to temp files, path injected into env.
_JSON_FILE_SECRETS: dict[str, str] = {
    "FIREBASE_SERVICE_ACCOUNT":     "FIREBASE_SERVICE_ACCOUNT_PATH",
    "BUCKET_SERVICE_ACCOUNT":       "GCS_SERVICE_ACCOUNT_FILE",
    "GOOGLE_CALENDAR_CREDENTIALS":  "GOOGLE_CALENDAR_CREDENTIALS_PATH",
    "GOOGLE_CALENDAR_TOKEN":        "GOOGLE_CALENDAR_TOKEN_PATH",
}

# Temp files created during the process lifetime.
# Stored here so they are not garbage-collected before the process exits.
_tmp_files: list[tempfile.NamedTemporaryFile] = []


def _access_secret(client, secret_name: str) -> str:
    """
    Access the latest version of a secret and return its string value.
    """
    name = f"projects/{GCP_PROJECT_ID}/secrets/{secret_name}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("utf-8")


def _write_temp_json(content: str, suffix: str) -> str:
    """
    Write a JSON string to a named temp file that persists for the process
    lifetime and return the file path.
    """
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=suffix,
        delete=False,
        encoding="utf-8",
    )
    tmp.write(content)
    tmp.flush()
    _tmp_files.append(tmp)
    return tmp.name


def load_secrets(force_local: bool = False) -> None:
    """
    Load all secrets from GCP Secret Manager into os.environ.

    Falls back to python-dotenv (.env file) when:
        - force_local=True is passed, or
        - the google-cloud-secret-manager package is not installed, or
        - the GCP metadata server / ADC credentials are not available
          (i.e. running on a developer laptop without gcloud auth).

    This means the same codebase works locally (via .env) and in
    production (via Secret Manager) without any changes.
    """
    if force_local:
        _load_dotenv_fallback()
        return

    try:
        from google.cloud import secretmanager
        from google.api_core.exceptions import NotFound, PermissionDenied

        client = secretmanager.SecretManagerServiceClient()

        log.info("Loading secrets from GCP Secret Manager (project: %s)...", GCP_PROJECT_ID)

        # Plain string secrets → inject directly into os.environ
        for secret_name, env_key in _PLAIN_SECRETS.items():
            try:
                value = _access_secret(client, secret_name)
                os.environ[env_key] = value
                log.info("  ✓ %s", secret_name)
            except NotFound:
                log.warning("  ✗ %s not found in Secret Manager — skipping.", secret_name)
            except PermissionDenied:
                log.warning("  ✗ %s — permission denied.", secret_name)

        # JSON file secrets → write to temp file, inject path into os.environ
        for secret_name, env_key in _JSON_FILE_SECRETS.items():
            try:
                json_content = _access_secret(client, secret_name)
                # Validate it's real JSON before writing
                json.loads(json_content)
                tmp_path = _write_temp_json(json_content, suffix=f"_{secret_name}.json")
                os.environ[env_key] = tmp_path
                log.info("  ✓ %s → %s", secret_name, tmp_path)
            except NotFound:
                log.warning("  ✗ %s not found in Secret Manager — skipping.", secret_name)
            except PermissionDenied:
                log.warning("  ✗ %s — permission denied.", secret_name)
            except json.JSONDecodeError:
                log.warning("  ✗ %s — value is not valid JSON.", secret_name)

        log.info("Secrets loaded successfully.")

    except ImportError:
        log.warning(
            "google-cloud-secret-manager not installed. "
            "Falling back to .env file."
        )
        _load_dotenv_fallback()

    except Exception as exc:
        log.warning(
            "Could not connect to Secret Manager (%s). "
            "Falling back to .env file.",
            exc,
        )
        _load_dotenv_fallback()


def _load_dotenv_fallback() -> None:
    """Load secrets from the local .env file (development only)."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        log.info("Loaded secrets from .env file (local development mode).")
    except ImportError:
        log.warning("python-dotenv not installed — no secrets loaded.")