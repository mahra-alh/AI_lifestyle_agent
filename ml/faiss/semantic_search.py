from pathlib import Path

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class FAISSRetriever:

    def __init__(self, artifacts_dir=None):
        ml_dir = Path(__file__).resolve().parents[1]
        artifacts_path = Path(artifacts_dir) if artifacts_dir else ml_dir / "artifacts"

        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = faiss.read_index(str(artifacts_path / "faiss_index.bin"))
        self.lookup_df = pd.read_csv(artifacts_path / "faiss_lookup.csv")

    def search(self, query, k=5):
        # convert query into embedding
        xq = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)

        scores, indices = self.index.search(xq,k) # search FAISS
        results = self.lookup_df.iloc[indices[0]].copy() # retrieve matching rows
        results["similarity_score"] = scores[0] # add similarity scores
        return results
