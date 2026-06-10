"""
ml/faiss/build_faiss.py
========================

Builds the FAISS index from the faiss_corpus collection in Firestore
and uploads the artifacts directly to GCS — no local files created.

Flow:
    Firestore (faiss_corpus collection)
        ↓  load_faiss_corpus()
    Build embeddings (SentenceTransformer)
        ↓
    Build FAISS index
        ↓
    Serialize to bytes in memory
        ↓
    Upload to GCS  →  gs://bucket/models/faiss_index.bin
                  →  gs://bucket/models/faiss_lookup.csv

Usage:
    python -m ml.faiss.build_faiss
"""

import io
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from ai_agent.storage.firestore_store import load_faiss_corpus
from ai_agent.storage.google_cloud_storage import (
    GCS_FAISS_INDEX_PATH,
    GCS_FAISS_LOOKUP_PATH,
    upload_bytes,
)

# ---------------------------------------------------------------------------
# Step 1 — Load corpus from Firestore
# ---------------------------------------------------------------------------

print("Loading faiss_corpus from Firestore...")
df = load_faiss_corpus()

if df.empty:
    raise RuntimeError(
        "faiss_corpus collection is empty. "
        "Run import_venues_to_firestore.js first."
    )

print(f"Loaded {len(df)} rows.")
sentences = df["faiss_text"].tolist()

# ---------------------------------------------------------------------------
# Step 2 — Build embeddings
# ---------------------------------------------------------------------------

print("Building embeddings...")
model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(
    sentences,
    convert_to_numpy=True,
    normalize_embeddings=True,    # cosine similarity via inner product
    show_progress_bar=True,
).astype(np.float32)

print(f"Embeddings shape: {embeddings.shape}")

# ---------------------------------------------------------------------------
# Step 3 — Build FAISS index
# ---------------------------------------------------------------------------

print("Building FAISS index...")
d     = embeddings.shape[1]
index = faiss.IndexFlatIP(d)
index.add(embeddings)

print(f"Index built: {index.ntotal} vectors, dimension {d}")

# ---------------------------------------------------------------------------
# Step 4 — Serialize to bytes in memory (no local files)
# ---------------------------------------------------------------------------

# Serialize FAISS index to bytes
index_buffer = io.BytesIO()
faiss.write_index(index, faiss.PyCallbackIOWriter(index_buffer.write))
index_bytes = index_buffer.getvalue()

# Serialize lookup CSV to bytes
lookup_bytes = df.to_csv(index=False).encode("utf-8")

# ---------------------------------------------------------------------------
# Step 5 — Upload directly to GCS
# ---------------------------------------------------------------------------

print("\nUploading to GCS...")
upload_bytes(index_bytes,  GCS_FAISS_INDEX_PATH,  content_type="application/octet-stream")
upload_bytes(lookup_bytes, GCS_FAISS_LOOKUP_PATH, content_type="text/csv")

print("\nDone.")
print(f"  Index  → gs://bucket/{GCS_FAISS_INDEX_PATH}")
print(f"  Lookup → gs://bucket/{GCS_FAISS_LOOKUP_PATH}")