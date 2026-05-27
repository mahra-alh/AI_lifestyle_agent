"""
Build FAISS artifacts for the Lifestyle AI Agent.

Input:
- ml/data/faiss_corpus.csv

Required columns:
- venue_id
- faiss_text

Output:
- ml/artifacts/faiss_index.bin
- ml/artifacts/faiss_lookup.csv
- ml/artifacts/faiss_manifest.json
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

def build_faiss_artifacts(
    input_path: str | Path = "ml/data/faiss_corpus.csv",
    output_dir: str | Path = "ml/artifacts",
    model_name: str = MODEL_NAME,
    batch_size: int = 64,
) -> None:
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = output_dir / "faiss_index.bin"
    lookup_path = output_dir / "faiss_lookup.csv"
    manifest_path = output_dir / "faiss_manifest.json"

    if not input_path.exists():
        raise FileNotFoundError(f"Input corpus not found: {input_path}")

    print(f"Reading corpus from: {input_path}")
    df = pd.read_csv(input_path)

    required_columns = {"venue_id", "faiss_text"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    print(f"Original rows: {len(df):,}")

    # Keep only rows with usable text.
    df = df.copy()
    df["faiss_text"] = df["faiss_text"].fillna("").astype(str).str.strip()
    df = df[df["faiss_text"] != ""].reset_index(drop=True)

    # Keep the row order stable. FAISS position must match lookup CSV row.
    df.insert(0, "faiss_position", range(len(df)))

    print(f"Usable rows after cleaning: {len(df):,}")

    if len(df) == 0:
        raise ValueError("No usable rows found after cleaning faiss_text.")

    texts = df["faiss_text"].tolist()

    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    print("Creating embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be 2D. Got shape: {embeddings.shape}")

    num_rows, embedding_dim = embeddings.shape

    print(f"Embedding shape: {embeddings.shape}")

    # Because embeddings are normalized, inner product works like cosine similarity.
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings)

    print(f"FAISS index vectors: {index.ntotal:,}")

    if index.ntotal != len(df):
        raise ValueError(
            f"Index row count mismatch. Index has {index.ntotal}, lookup has {len(df)}"
        )

    print(f"Saving FAISS index to: {index_path}")
    faiss.write_index(index, str(index_path))

    print(f"Saving lookup CSV to: {lookup_path}")
    df.to_csv(lookup_path, index=False)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "index_path": str(index_path),
        "lookup_path": str(lookup_path),
        "model_name": model_name,
        "row_count": int(len(df)),
        "embedding_dimension": int(embedding_dim),
        "index_type": "IndexFlatIP",
        "normalized_embeddings": True,
        "required_columns": sorted(list(required_columns)),
    }

    print(f"Saving manifest to: {manifest_path}")
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    print("\nFAISS artifacts built successfully.")
    print(f"- {index_path}")
    print(f"- {lookup_path}")
    print(f"- {manifest_path}")


if __name__ == "__main__":
    build_faiss_artifacts()