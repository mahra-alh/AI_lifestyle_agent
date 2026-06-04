from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def save_json(data: Dict[str, Any], output_path: Path) -> None:
    """
    Persist the full weather API result as a formatted JSON file.

    The parent directory is created automatically if it does not exist.
    Intended for local inspection and future ML feature pipelines.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def save_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Persist the daily forecast rows as a CSV file.

    Column order follows the key order of the first row dict, so callers
    should ensure consistent key ordering (Python 3.7+ dicts are ordered).
    No-ops silently when rows is empty.
    """
    if not rows:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_location_filename(location_name: str) -> str:
    """
    Convert a location string into a filesystem-safe filename stem.

    Example: "Abu Dhabi/UAE" -> "abu_dhabi_uae"
    """
    return (
        location_name.lower()
        .strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )
