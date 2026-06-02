"""
Handles local JSONL audit logging for all calendar actions.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
from .config import DEFAULT_LOG_DIR,DEFAULT_MAX_LOG_SIZE_MB


# Default log directory relative to the project root.
DEFAULT_LOG_DIR = "data/logs"
DEFAULT_MAX_LOG_SIZE_MB = 5


# Public interface

def save_calendar_log(
    data: Dict[str, Any],
    output_dir: str = DEFAULT_LOG_DIR,
    max_log_size_mb: int = DEFAULT_MAX_LOG_SIZE_MB,
) -> None:
    """
    Append a calendar action result to a local JSONL log file.

    Each line in the log is one JSON object — easy to parse, grep, or
    load into a dataframe for debugging and audit during testing.

    Args:
        data:             Dict to log (booking result, error, etc.).
        output_dir:       Directory where the log file is written.
        max_log_size_mb:  Rotate the log when it exceeds this size in MB.
                          Set to 0 to disable rotation.
    """
    output_folder = Path(output_dir)
    output_folder.mkdir(parents=True, exist_ok=True)

    log_path = output_folder / "calendar_actions_log.jsonl"

    _rotate_log_if_needed(log_path, max_log_size_mb=max_log_size_mb)

    # Stamp every entry with a UTC write time for traceability.
    entry = {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        **data,
    }

    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_calendar_log(
    output_dir: str = DEFAULT_LOG_DIR,
) -> list[Dict[str, Any]]:
    """
    Read and return all entries from the JSONL log file.

    Useful for inspecting past bookings during development.

    Args:
        output_dir: Directory where the log file lives.

    Returns:
        List of dicts, one per logged action. Empty list if no log exists.
    """
    log_path = Path(output_dir) / "calendar_actions_log.jsonl"

    if not log_path.exists():
        return []

    entries: list[Dict[str, Any]] = []

    with log_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    # Skip corrupted lines rather than crashing the reader.
                    continue

    return entries


# Private helpers

def _rotate_log_if_needed(log_path: Path, max_log_size_mb: int) -> None:
    """
    Rotate the log file if it exceeds max_log_size_mb.

    The current log is moved to calendar_actions_log.jsonl.1
    and a fresh log starts. Only one rotated copy is kept.

    Args:
        log_path:         Path to the active log file.
        max_log_size_mb:  Size threshold in MB. 0 disables rotation.
    """
    if max_log_size_mb <= 0:
        return

    max_bytes = max_log_size_mb * 1024 * 1024

    if log_path.exists() and log_path.stat().st_size > max_bytes:
        rotated_path = log_path.with_suffix(".jsonl.1")
        if rotated_path.exists():
            rotated_path.unlink()
        shutil.move(str(log_path), str(rotated_path))
        print(f"[logging] Log rotated → {rotated_path}")


# CLI entry point — run to inspect past bookings

if __name__ == "__main__":
    print("\n📋  Calendar action log\n" + "─" * 40)

    entries = read_calendar_log()

    if not entries:
        print("No log entries found.")
    else:
        for i, entry in enumerate(entries, start=1):
            print(f"\n{i}. [{entry.get('logged_at', '?')}]")
            print(f"   Status   : {entry.get('status', '?')}")
            print(f"   Event    : {entry.get('event_title', '?')}")
            print(f"   Start    : {entry.get('start_time', '?')}")
            print(f"   End      : {entry.get('end_time', '?')}")
            if entry.get("event_link"):
                print(f"   Link     : {entry.get('event_link')}")

    print("\n" + "─" * 40)
