"""
Central helper for reading secrets from Google Cloud Secret Manager.

How it works:
- On first call, detects the GCP project ID automatically (works on
  Cloud Run).
- Caches every fetched secret in memory so Secret Manager is only
  called once per process, not on every request.
- Falls back gracefully to environment variables if Secret Manager is
  unreachable (e.g., local dev without GCP credentials). Set
  USE_SECRET_MANAGER=false in the local .env to skip it entirely.
"""
from __future__ import annotations

import os
import logging
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

#  Configuration

# Set USE_SECRET_MANAGER=false in your local .env to use plain env-vars instead
_USE_SECRET_MANAGER: bool = (
    os.getenv("USE_SECRET_MANAGER", "true").strip().lower() not in {"false", "0", "no", "off"}
)

# Override with GCP_PROJECT_ID env var if auto-detection fails
_PROJECT_ID: Optional[str] = os.getenv("GCP_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")


#  Internal helpers 

@lru_cache(maxsize=1)
def _get_client():
    """Return a cached Secret Manager client (imported lazily)."""
    from google.cloud import secretmanager  # type: ignore
    return secretmanager.SecretManagerServiceClient()


@lru_cache(maxsize=1)
def _resolve_project_id() -> str:
    """
    Resolve the GCP project ID.
    Order of preference:
      1. GCP_PROJECT_ID / GOOGLE_CLOUD_PROJECT env var
      2. Metadata server (Cloud Run)
    """
    if _PROJECT_ID:
        return _PROJECT_ID
    try:
        import urllib.request
        req = urllib.request.Request(
            "http://metadata.google.internal/computeMetadata/v1/project/project-id",
            headers={"Metadata-Flavor": "Google"},
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.read().decode()
    except Exception as exc:
        raise RuntimeError(
            "Cannot determine GCP project ID. "
            "Set GCP_PROJECT_ID or GOOGLE_CLOUD_PROJECT in your environment."
        ) from exc


# Per-process in-memory cache  { secret_name: secret_value }
_cache: dict[str, str] = {}


#  Public API 

def get_secret(name: str, version: str = "latest") -> str:
    """
    Fetch a secret value by name.
    """
    # 1. Return from in-process cache
    if name in _cache:
        return _cache[name]

    # 2. Try Secret Manager (unless explicitly disabled)
    if _USE_SECRET_MANAGER:
        try:
            project_id = _resolve_project_id()
            client = _get_client()
            secret_path = f"projects/{project_id}/secrets/{name}/versions/{version}"
            response = client.access_secret_version(request={"name": secret_path})
            value = response.payload.data.decode("utf-8").strip()
            _cache[name] = value
            logger.debug("Loaded secret '%s' from Secret Manager.", name)
            return value
        except Exception as exc:
            logger.warning(
                "Secret Manager lookup failed for '%s': %s — falling back to env var.",
                name, exc,
            )

    # 3. Fall back to environment variable
    env_value = os.getenv(name)
    if env_value is not None:
        _cache[name] = env_value
        logger.debug("Loaded secret '%s' from environment variable.", name)
        return env_value

    raise RuntimeError(
        f"Secret '{name}' not found in Secret Manager or environment variables."
    )


def get_secret_optional(name: str, default: str = "") -> str:
    """Like get_secret(), but returns `default` instead of raising on failure."""
    try:
        return get_secret(name)
    except RuntimeError:
        return default
