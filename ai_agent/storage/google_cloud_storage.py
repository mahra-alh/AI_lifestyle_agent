"""
Google Cloud Storage client helpers used across the project.
"""
from __future__ import annotations

import io
import logging
from typing import Optional

from google.cloud import storage  # type: ignore

from ai_agent.secrets import get_secret_optional

logger = logging.getLogger(__name__)

#  Config 

BUCKET_NAME = "ai_lifestyle_agent_bucket"



# GCS paths for FAISS artifacts
GCS_FAISS_INDEX_PATH  = "models/faiss_index.bin"
GCS_FAISS_LOOKUP_PATH = "models/faiss_lookup.csv"

# GCS paths for LightGBM artifacts
GCS_LGBM_MODEL_PATH    = "models/lgbm_ranker.pkl"
GCS_LGBM_FEATURES_PATH = "models/lgbm_ranker.pkl.features.json"


#  Internal 

def _get_bucket() -> storage.Bucket:
    """
    Return the GCS bucket client using Application Default Credentials.
    """
    client = storage.Client()     
    return client.bucket(BUCKET_NAME)


#  Public API 

def upload_bytes(
    data: bytes,
    gcs_path: str,
    content_type: str = "application/octet-stream",
) -> None:
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
    logger.info("Uploaded → gs://%s/%s  (%s bytes)", BUCKET_NAME, gcs_path, f"{len(data):,}")


def download_bytes(gcs_path: str) -> bytes:
    """
    Download a GCS file into memory and return raw bytes — no local file created.

    Args:
        gcs_path: Path in the bucket, e.g. "models/faiss_index.bin".

    Returns:
        The file contents as bytes.
    """
    bucket = _get_bucket()
    blob   = bucket.blob(gcs_path)
    buffer = io.BytesIO()
    blob.download_to_file(buffer)
    buffer.seek(0)
    data = buffer.read()
    logger.info("Downloaded ← gs://%s/%s  (%s bytes)", BUCKET_NAME, gcs_path, f"{len(data):,}")
    return data
