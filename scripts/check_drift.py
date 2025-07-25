from src.embedder import embed_abstracts
from scipy.spatial.distance import cosine
import numpy as np
import orjson, os

OLD_EMBED_PATH = "models/scibert_best/weekly_embeddings.npy"
NEW_PMIDS_PATH = "data/pmids_weekly.json"

# 1. Pull recent PubMed abstracts (reuse your harvest + fetch pipeline)
os.system("python src/harvest_pubmed.py --days 7 --out data/pmids_weekly.json")
os.system("python src/fetch_abstracts.py --in data/pmids_weekly.json --out data/raw_weekly.jsonl")

# 2. Tokenize and embed
new_embs, _ = embed_abstracts("data/raw_weekly.jsonl")

# 3. Load last embeddings
old_embs = np.load(OLD_EMBED_PATH)

# 4. Compare distributions (average embedding + cosine sim)
drift_score = cosine(old_embs.mean(axis=0), new_embs.mean(axis=0))

print(f"Weekly drift score: {drift_score:.4f}")
if drift_score > 0.3:
    print("🚨 Drift detected! Trigger retrain.")
    # optionally retrain or raise alert
else:
    print("✅ No significant drift.")

# 5. Save current as new reference
np.save(OLD_EMBED_PATH, new_embs)
