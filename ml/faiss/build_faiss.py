# load the data
# generate embeddings
# build the FAISS index
# save artifacts
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# load the data
df = pd.read_csv("../data/faiss_corpus.csv")

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
faiss.write_index(index, "../models/faiss_index.bin") # save the index
df.to_csv( "../models/faiss_lookup.csv", index=False)

print("FAISS index built successfully")