"""
Google Cloud Service client helpers used across the project.

All files in the bucket use these two functions:
    upload_bytes()   — write an in-memory buffer directly to GCS
    download_bytes() — read a GCS file into an in-memory buffer
"""
from __future__ import annotations
import io
import os
from typing import Optional

from google.cloud import storage

SERVICE_ACCOUNT_FILE = os.getenv(
    "GCS_SERVICE_ACCOUNT_FILE",
    "ai_agent/storage/bucket-service-acc.json",
)
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "ai_lifestyle_agent_bucket")

# GCS paths for FAISS artifacts
GCS_FAISS_INDEX_PATH  = "models/faiss_index.bin"
GCS_FAISS_LOOKUP_PATH = "models/faiss_lookup.csv"

# GCS paths for LightGBM artifacts
GCS_LGBM_MODEL_PATH    = "models/lgbm_ranker.pkl"
GCS_LGBM_FEATURES_PATH = "models/lgbm_ranker.pkl.features.json"


def _get_bucket() -> storage.Bucket:
    """Return the GCS bucket client."""
    client = storage.Client.from_service_account_json(SERVICE_ACCOUNT_FILE)
    return client.bucket(BUCKET_NAME)


def upload_bytes(data: bytes, gcs_path: str, content_type: str = "application/octet-stream") -> None:
    """
    Upload raw bytes directly to GCS — no local file needed.

    Args:
        data:         The bytes to upload.
        gcs_path:     Destination path in the bucket, e.g. "models/faiss_index.bin".
        content_type: MIME type of the content.
    """
    bucket = _get_bucket()
    blob   = bucket.blob(gcs_path)
    blob.upload_from_file(io.BytesIO(data), content_type=content_type)
    print(f"Uploaded → gs://{BUCKET_NAME}/{gcs_path}  ({len(data):,} bytes)")


def download_bytes(gcs_path: str) -> bytes:
    """
    Download a GCS file into memory and return raw bytes — no local file created.

    Args:
        gcs_path: Path in the bucket, e.g. "models/faiss_index.bin".

    Returns:
        The file contents as bytes.

    Raises:
        FileNotFoundError: The path does not exist in the bucket.
    """
    bucket = _get_bucket()
    blob   = bucket.blob(gcs_path)

    if not blob.exists():
        raise FileNotFoundError(
            f"gs://{BUCKET_NAME}/{gcs_path} not found. "
            "Run build_faiss.py or train_cells.py first."
        )

    data = blob.download_as_bytes()
    print(f"Downloaded ← gs://{BUCKET_NAME}/{gcs_path}  ({len(data):,} bytes)")
    return data


def list_files(prefix: Optional[str] = None) -> list[str]:
    """List all files in the bucket, optionally filtered by prefix."""
    bucket = _get_bucket()
    blobs  = bucket.list_blobs(prefix=prefix)
    return [blob.name for blob in blobs]