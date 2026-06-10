"""
ml/faiss/semantic_search.py
============================

Production FAISS retriever.

Loads the index and lookup table directly from GCS into memory on first
use — no local files created or required.
"""
from __future__ import annotations

import io
import os
import tempfile
from typing import Optional

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from ai_agent.storage.google_cloud_storage import (
    GCS_FAISS_INDEX_PATH,
    GCS_FAISS_LOOKUP_PATH,
    download_bytes,
)

EMBEDDING_MODEL = os.getenv("FAISS_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ---------------------------------------------------------------------------
# Lazy singleton
# ---------------------------------------------------------------------------

_retriever: Optional["FAISSRetriever"] = None


def get_retriever() -> "FAISSRetriever":
    """Return the shared FAISSRetriever, initializing it on first call."""
    global _retriever
    if _retriever is None:
        _retriever = FAISSRetriever()
    return _retriever


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class FAISSRetriever:
    """
    Wraps a FAISS flat-IP index and the lookup DataFrame.

    On initialization, downloads both artifacts from GCS into memory.
    No files are written to disk.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        print("Initializing FAISSRetriever — loading from GCS...")

        # Load embedding model
        self.model = SentenceTransformer(model_name)

        # Load FAISS index from GCS bytes
        # faiss.read_index requires a file path, so we use a temp file
        # that is deleted immediately after loading.
        index_bytes = download_bytes(GCS_FAISS_INDEX_PATH)
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=True) as tmp:
            tmp.write(index_bytes)
            tmp.flush()
            self.index = faiss.read_index(tmp.name)

        # Load lookup CSV from GCS bytes directly into DataFrame
        lookup_bytes = download_bytes(GCS_FAISS_LOOKUP_PATH)
        self.lookup_df = pd.read_csv(io.BytesIO(lookup_bytes))

        print(
            f"FAISSRetriever ready: "
            f"{self.index.ntotal} vectors, "
            f"{len(self.lookup_df)} venues"
        )

    def search(self, query: str, k: int = 50) -> pd.DataFrame:
        """
        Retrieve the top-k most semantically similar venues for a query.

        Args:
            query: Natural-language query, e.g.
                   "cheap outdoor brunch near the marina".
            k:     Number of candidates to retrieve before LightGBM reranks.

        Returns:
            DataFrame with a `faiss_score` column, sorted descending.
        """
        xq = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        scores, indices = self.index.search(xq, k)

        valid_mask    = indices[0] >= 0
        valid_indices = indices[0][valid_mask]
        valid_scores  = scores[0][valid_mask]

        results = self.lookup_df.iloc[valid_indices].copy().reset_index(drop=True)
        results["faiss_score"] = valid_scores

        return results.sort_values("faiss_score", ascending=False).reset_index(drop=True)