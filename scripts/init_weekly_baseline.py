from src.embedder import embed_abstracts
import numpy as np

# Path to your simulated "weekly" abstracts (we made this for testing)
WEEKLY_JSONL = "data/raw_weekly.jsonl"
OUT_PATH = "models/scibert_best/weekly_embeddings.npy"

embeddings, pmids = embed_abstracts(WEEKLY_JSONL)
np.save(OUT_PATH, embeddings)
print(f"Saved {len(embeddings)} embeddings to {OUT_PATH}")
