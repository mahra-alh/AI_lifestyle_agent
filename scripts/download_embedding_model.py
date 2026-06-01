from pathlib import Path
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# repo root = parent of scripts/
REPO_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = REPO_ROOT / "models" / "embeddings" / "all-MiniLM-L6-v2"

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Saving model to: {OUTPUT_DIR}")
    model.save_pretrained(str(OUTPUT_DIR))

    print("Embedding model downloaded and saved successfully.")

if __name__ == "__main__":
    main()