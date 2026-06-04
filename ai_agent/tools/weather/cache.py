from __future__ import annotations
import json
import logging
from typing import Any, Dict, Optional

import redis
from ai_agent.tools.weather.config import (
    REDIS_HOST,
    REDIS_PASSWORD,
    REDIS_PORT,
    REDIS_TTL_SECONDS,
    REDIS_USERNAME,
)

logger = logging.getLogger(__name__)

# Redis client 
def _make_client() -> redis.Redis:
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        username=REDIS_USERNAME,
        password=REDIS_PASSWORD,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5,
    )


# module-level client, created once, reused across calls.
_client: Optional[redis.Redis] = None


def _get_client() -> redis.Redis:
    """Return the module-level Redis client, creating it if needed."""
    global _client
    if _client is None:
        _client = _make_client()
    return _client


# cache key

def _cache_key(location_name: str) -> str:
    """
    Build a namespaced Redis key for a location forecast.

    Example: "weather:forecast:dubai"
    """
    return f"weather:forecast:{location_name.lower().strip().replace(' ', '_')}"


# public interface 

def get_cached_forecast(location_name: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a cached forecast for the given location.

    Returns the parsed forecast dict on a cache hit, or None on a miss
    or any Redis error. Errors are logged but never raised so a cache
    failure never breaks the agent.
    """
    try:
        client = _get_client()
        key = _cache_key(location_name)
        raw = client.get(key)

        if raw is None:
            logger.debug("Cache miss for key: %s", key)
            return None

        logger.debug("Cache hit for key: %s", key)
        return json.loads(raw)

    except redis.RedisError as error:
        logger.warning("Redis read error for '%s': %s", location_name, error)
        return None

    except json.JSONDecodeError as error:
        logger.warning("Cache JSON decode error for '%s': %s", location_name, error)
        return None


def set_cached_forecast(
    location_name: str,
    forecast: Dict[str, Any],
) -> bool:
    """
    Store a forecast in Redis with the configured TTL.

    Returns True on success, False on any Redis error.
    Errors are logged but never raised.
    """
    try:
        client = _get_client()
        key = _cache_key(location_name)
        client.setex(key, REDIS_TTL_SECONDS, json.dumps(forecast))
        logger.debug("Cached forecast for key: %s (TTL: %ds)", key, REDIS_TTL_SECONDS)
        return True

    except redis.RedisError as error:
        logger.warning("Redis write error for '%s': %s", location_name, error)
        return False


def invalidate_cached_forecast(location_name: str) -> bool:
    """
    Delete the cached forecast for a location.

    Useful for forcing a fresh API fetch without waiting for TTL expiry.
    Returns True if the key existed and was deleted, False otherwise.
    """
    try:
        client = _get_client()
        key = _cache_key(location_name)
        deleted = client.delete(key)
        return bool(deleted)

    except redis.RedisError as error:
        logger.warning("Redis delete error for '%s': %s", location_name, error)
        return False


def is_cache_healthy() -> bool:
    """
    Ping Redis to confirm the connection is alive.

    Used during startup or health checks. Returns False on any error
    rather than raising, so callers can degrade gracefully.
    """
    try:
        return _get_client().ping()
    except redis.RedisError:
        return False