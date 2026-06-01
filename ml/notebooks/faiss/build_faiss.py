import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
CORPUS_PATH   = _HERE / "../../data/faiss_corpus.csv"
ARTIFACTS_DIR = _HERE / "../../artifacts"

MODEL_NAME = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_corpus(corpus_path: Path = CORPUS_PATH) -> pd.DataFrame:
    """Load the FAISS text corpus and validate required columns."""
    df = pd.read_csv(corpus_path)
    missing = [c for c in ("venue_id", "faiss_text") if c not in df.columns]
    if missing:
        raise ValueError(f"faiss_corpus.csv is missing columns: {missing}")
    print(f"Corpus loaded: {len(df):,} rows")
    print("Sample faiss_text strings:")
    for i in range(min(5, len(df))):
        print(f"  [{i}] {df['faiss_text'].iloc[i]}")
    return df


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
def encode_corpus(
    sentences: list[str],
    model_name: str = MODEL_NAME,
) -> tuple[SentenceTransformer, np.ndarray]:
    """Encode a list of sentences into normalised float32 embeddings.

    Returns the loaded model (reused for query encoding) and the embedding matrix.
    Normalised embeddings let IndexFlatIP behave as cosine similarity.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        sentences,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    print(f"Embedding matrix: {embeddings.shape}")
    return model, embeddings


# ---------------------------------------------------------------------------
# Index construction
# ---------------------------------------------------------------------------
def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build a flat inner-product FAISS index from the embedding matrix."""
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    print(f"FAISS index built: {index.ntotal:,} vectors (dim={d})")
    return index


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
def smoke_test(
    model: SentenceTransformer,
    index: faiss.IndexFlatIP,
    lookup_df: pd.DataFrame,
    query: str = "cheap arabic restaurant",
    k: int = 5,
) -> None:
    """Run a quick sanity search and print the top-k results."""
    xq = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    scores, indices = index.search(xq, k)
    results = lookup_df.iloc[indices[0]].copy()
    results["similarity_score"] = scores[0]
    print(f"\nSmoke test — query: '{query}'")
    print(results[["faiss_text", "similarity_score"]].to_string(index=False))


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------
def save_artifacts(
    index: faiss.IndexFlatIP,
    lookup_df: pd.DataFrame,
    embeddings: np.ndarray,
    corpus_path: Path = CORPUS_PATH,
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> None:
    """Write index binary, lookup CSV, and manifest JSON to artifacts_dir."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    index_path   = artifacts_dir / "faiss_index.bin"
    lookup_path  = artifacts_dir / "faiss_lookup.csv"
    manifest_path = artifacts_dir / "faiss_manifest.json"

    faiss.write_index(index, str(index_path))
    lookup_df.to_csv(lookup_path, index=False)

    manifest = {
        "created_at_utc":       datetime.now(timezone.utc).isoformat(),
        "input_path":           str(corpus_path),
        "index_path":           str(index_path),
        "lookup_path":          str(lookup_path),
        "model_name":           MODEL_NAME,
        "row_count":            int(index.ntotal),
        "embedding_dimension":  int(embeddings.shape[1]),
        "index_type":           "IndexFlatIP",
        "normalized_embeddings": True,
        "required_columns":     ["faiss_text", "venue_id"],
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nArtifacts saved to {artifacts_dir}/")
    print(f"  Index:    {index_path.name}  ({index_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  Lookup:   {lookup_path.name}  ({len(lookup_df):,} rows)")
    print(f"  Manifest: {manifest_path.name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    df = load_corpus()

    # lookup_df keeps venue_id + faiss_text; row position = FAISS index slot
    lookup_df = df[["venue_id", "faiss_text"]].reset_index(drop=True)
    sentences = lookup_df["faiss_text"].tolist()

    model, embeddings = encode_corpus(sentences)
    index = build_index(embeddings)

    smoke_test(model, index, lookup_df)
    save_artifacts(index, lookup_df, embeddings)


if __name__ == "__main__":
    main()
