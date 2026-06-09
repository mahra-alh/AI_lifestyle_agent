"""
Production FAISS retriever.

Loads model and index once at startup (lazy singleton) and exposes a
single search() method used by the recommendation service.

Artifact paths are resolved from environment variables so they can be
pointed at local files during development and at GCS/mounted paths in
production without code changes.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Artifact paths — override via environment variables

_DEFAULT_BASE = Path(__file__).parents[2] / "models"

FAISS_INDEX_PATH = Path(
    os.getenv("FAISS_INDEX_PATH", str(_DEFAULT_BASE / "faiss_index.bin"))
)
FAISS_LOOKUP_PATH = Path(
    os.getenv("FAISS_LOOKUP_PATH", str(_DEFAULT_BASE / "faiss_lookup.csv"))
)
EMBEDDING_MODEL_NAME = os.getenv(
    "FAISS_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
)

_retriever: Optional["FAISSRetriever"] = None


def get_retriever() -> "FAISSRetriever":
    """Return the shared FAISSRetriever, initializing it on first call."""
    global _retriever
    if _retriever is None:
        _retriever = FAISSRetriever()
    return _retriever


class FAISSRetriever:
    """
    Wraps a FAISS flat-IP index and the lookup DataFrame.

    The index must have been built with normalized embeddings (cosine
    similarity) using build_faiss.py.
    """

    def __init__(
        self,
        index_path: Path = FAISS_INDEX_PATH,
        lookup_path: Path = FAISS_LOOKUP_PATH,
        model_name: str = EMBEDDING_MODEL_NAME,
    ) -> None:
        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at '{index_path}'. "
                "Run ml/faiss/build_faiss.py first."
            )
        if not lookup_path.exists():
            raise FileNotFoundError(
                f"FAISS lookup CSV not found at '{lookup_path}'. "
                "Run ml/faiss/build_faiss.py first."
            )

        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(str(index_path))
        self.lookup_df = pd.read_csv(lookup_path)

    def search(
        self,
        query: str,
        k: int = 50,
    ) -> pd.DataFrame:
        """
        Retrieve the top-k most semantically similar venues for a query.

        Args:
            query: Natural-language query from the user, e.g.
                   "cheap outdoor brunch near the marina".
            k:     Number of candidates to retrieve. Should be larger than
                   the final number of results (e.g. 50) so LightGBM has
                   enough candidates to rerank down to 5.

        Returns:
            DataFrame of candidate rows from the lookup CSV, with an added
            `faiss_score` column (cosine similarity, higher = more similar).
            Rows are sorted by faiss_score descending.
        """
        # Embed the query with the same model used to build the index.
        xq = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        # Search the FAISS index.
        scores, indices = self.index.search(xq, k)

        # Build results DataFrame.
        valid_mask = indices[0] >= 0  # FAISS returns -1 for padding
        valid_indices = indices[0][valid_mask]
        valid_scores = scores[0][valid_mask]

        results = self.lookup_df.iloc[valid_indices].copy().reset_index(drop=True)
        results["faiss_score"] = valid_scores

        return results.sort_values("faiss_score", ascending=False).reset_index(drop=True)
