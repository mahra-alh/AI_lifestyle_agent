import json
import logging
import os
import time
import traceback
import uuid
import hashlib
from contextvars import ContextVar
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional

# Stores one trace_id per user request.
# This lets all tools share the same trace_id during one agent run.
_trace_id_context: ContextVar[Optional[str]] = ContextVar(
    "trace_id",
    default=None
)

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

APP_LOG_PATH = LOG_DIR / "app.jsonl"

DEBUG_LOGGING_ENABLED = os.getenv("DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}

class JsonFormatter(logging.Formatter):

    # Converts normal Python log records into JSON lines.
    # Each log entry becomes one JSON object on one line.
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add custom fields if they exist
        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)

        return json.dumps(log_data, ensure_ascii=False)

def get_logger(name: str = "lifestyle_agent") -> logging.Logger:

    # Returns a shared application logger.
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    file_handler = RotatingFileHandler(
        APP_LOG_PATH,
        maxBytes=5_000_000,   # 5 MB
        backupCount=5,        # keep 5 old log files
        encoding="utf-8"
    )

    file_handler.setFormatter(JsonFormatter())
    logger.addHandler(file_handler)

    # Prevent duplicated logs from root logger
    logger.propagate = False

    return logger

def create_trace_id() -> str:
    # Creates a unique trace_id for one full user request.
    return f"trace_{uuid.uuid4().hex}"

def set_trace_id(trace_id: str) -> None:
    # Saves trace_id into context so all tools can access it.
    _trace_id_context.set(trace_id)

def get_trace_id() -> str:

    # Returns current trace_id.
    # If none exists, creates a new one.
    trace_id = _trace_id_context.get()

    if trace_id is None:
        trace_id = create_trace_id()
        set_trace_id(trace_id)

    return trace_id

def sanitize_data(data: Optional[Dict[str, Any]]) -> Dict[str, Any]:

    # Removes sensitive fields before writing logs.
    # Never log API keys, passwords, tokens, private keys, or full email addresses.
    if not data:
        return {}

    blocked_keys = {
        "password",
        "api_key",
        "openai_api_key",
        "private_key",
        "token",
        "access_token",
        "refresh_token",
        "authorization",
        "email_address",
        "email",
    }

    def _is_sensitive_key(key: str) -> bool:
        cleaned_key = key.lower()
        if cleaned_key in blocked_keys:
            return True
        sensitive_fragments = ("token", "secret", "password", "authorization", "api_key", "email")
        return any(fragment in cleaned_key for fragment in sensitive_fragments)

    def _sanitize_value(value: Any) -> Any:
        if isinstance(value, dict):
            safe_dict: Dict[str, Any] = {}
            for nested_key, nested_value in value.items():
                key_str = str(nested_key)
                if _is_sensitive_key(key_str):
                    safe_dict[key_str] = "[REDACTED]"
                else:
                    safe_dict[key_str] = _sanitize_value(nested_value)
            return safe_dict

        if isinstance(value, list):
            return [_sanitize_value(item) for item in value]

        return value

    return _sanitize_value(data)

def _hash_user_id(user_id: str) -> str:
    # Keep user correlation in logs without storing raw email/user identifiers.
    normalized_user_id = user_id.strip().lower()
    return hashlib.sha256(normalized_user_id.encode("utf-8")).hexdigest()

def _safe_error_message(error: Exception, max_len: int = 300) -> str:
    # Avoid leaking long/sensitive exception details into logs.
    raw_message = str(error).strip()
    if not raw_message:
        return type(error).__name__
    if len(raw_message) > max_len:
        return raw_message[:max_len] + "...[truncated]"
    return raw_message

def log_event(
    event_name: str,
    level: str = "INFO",
    user_id: Optional[str] = None,
    tool_name: Optional[str] = None,
    data: Optional[Dict[str, Any]] = None,
) -> None:
    # Writes a structured JSONL log event.
    logger = get_logger()

    extra_data = {
        "trace_id": get_trace_id(),
        "event_name": event_name,
    }

    if user_id:
        extra_data["user_id_hash"] = _hash_user_id(user_id)

    if tool_name:
        extra_data["tool_name"] = tool_name

    if data:
        extra_data["data"] = sanitize_data(data)

    log_method = getattr(logger, level.lower(), logger.info)

    log_method(
        event_name,
        extra={"extra_data": extra_data}
    )

def log_tool_start(
    tool_name: str,
    user_id: Optional[str] = None,
    data: Optional[Dict[str, Any]] = None,
) -> float:
    # Logs tool start and returns start time.
    start_time = time.perf_counter()

    log_event(
        event_name="tool_started",
        level="INFO",
        user_id=user_id,
        tool_name=tool_name,
        data=data,
    )

    return start_time

def log_tool_success(
    tool_name: str,
    start_time: float,
    user_id: Optional[str] = None,
    data: Optional[Dict[str, Any]] = None,
) -> None:

    # Logs successful tool completion.
    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

    success_data = {
        "duration_ms": duration_ms,
        "status": "success",
    }

    if data:
        success_data.update(data)

    log_event(
        event_name="tool_completed",
        level="INFO",
        user_id=user_id,
        tool_name=tool_name,
        data=success_data,
    )

def log_tool_error(
    tool_name: str,
    start_time: float,
    error: Exception,
    user_id: Optional[str] = None,
    data: Optional[Dict[str, Any]] = None,
) -> None:

    # Logs tool failure.
    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

    error_data = {
        "duration_ms": duration_ms,
        "status": "failed",
        "error_type": type(error).__name__,
        "error_message": _safe_error_message(error),
    }

    if DEBUG_LOGGING_ENABLED:
        error_data["traceback"] = traceback.format_exc()

    if data:
        error_data.update(data)

    log_event(
        event_name="tool_failed",
        level="ERROR",
        user_id=user_id,
        tool_name=tool_name,
        data=error_data,
    )
