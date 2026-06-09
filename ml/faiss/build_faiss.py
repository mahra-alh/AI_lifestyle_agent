# load the data
# generate embeddings
# build the FAISS index
# save artifacts
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path


# load the data
_BASE = Path(__file__).parents[2]  # project root
df = pd.read_csv(_BASE / "ai_agent" / "data" / "faiss_corpus.csv")


sentences = df["faiss_text"].tolist() 

model = SentenceTransformer("all-MiniLM-L6-v2") #initialise our model

sentence_embedding = model.encode(
    sentences, 
    convert_to_numpy=True, # convert to numpy arrays
    normalize_embeddings=True  # behaves like cosine similiarity
    ).astype(np.float32) # prevent datatype incompatibility issues

d = sentence_embedding.shape[1] # dimension
index = faiss.IndexFlatIP(d) # initilize the index
index.add(sentence_embedding) # add our vectors

faiss.write_index(index, str(_BASE / "models" / "faiss_index.bin"))
df.to_csv(_BASE / "models" / "faiss_lookup.csv", index=False)

print("FAISS index built successfully")