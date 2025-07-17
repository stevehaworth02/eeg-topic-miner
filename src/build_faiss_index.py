import os
import torch
import faiss
import numpy as np
from datasets import load_from_disk
from transformers import AutoModel

# ----- CONFIG -----
MODEL_CKPT = "models/scibert_best/best_model.pt"
TOKENIZED_DATA_DIR = "data/tokenised"
FAISS_INDEX_PATH = "models/scibert_best/faiss.index"
META_PATH = "models/scibert_best/faiss_meta.npy"

# ----- LOAD MODEL -----
class EncoderOnly(torch.nn.Module):
    def __init__(self, encoder_name="allenai/scibert_scivocab_uncased"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.encoder.eval()  # set to eval mode

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            return self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]

device = "cuda" if torch.cuda.is_available() else "cpu"
model = EncoderOnly().to(device)
# Load only the encoder weights (ignore heads if they exist)
state_dict = torch.load(MODEL_CKPT, map_location=device)
model.encoder.load_state_dict(
    {k.replace("encoder.", ""): v for k, v in state_dict.items() if k.startswith("encoder.")},
    strict=False
)
model.eval()

# ----- LOAD DATA -----
ds = load_from_disk(TOKENIZED_DATA_DIR)
ds.set_format(type="python")   # for safety

embeddings = []
meta = []

# ----- EMBED ALL ABSTRACTS -----
batch_size = 64
for i in range(0, len(ds), batch_size):
    batch_ds = ds.select(range(i, min(i+batch_size, len(ds))))
    batch_dict = batch_ds[:]
    if isinstance(batch_dict, dict):  # dict of lists → list of dicts
        keys = list(batch_dict.keys())
        vals = list(batch_dict.values())
        batch = [dict(zip(keys, v)) for v in zip(*vals)]
    else:
        batch = batch_dict
    input_ids = torch.tensor([x['input_ids'] for x in batch]).to(device)
    attn_mask = torch.tensor([x['attention_mask'] for x in batch]).to(device)
    embs = model(input_ids, attn_mask).cpu().numpy()
    embeddings.append(embs)
    meta.extend([x.get('pmid', f"row_{i+j}") for j, x in enumerate(batch)])

embeddings = np.concatenate(embeddings, axis=0)
print(f"Embeddings shape: {embeddings.shape}")

# ----- BUILD FAISS INDEX -----
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, FAISS_INDEX_PATH)
np.save(META_PATH, np.array(meta, dtype=object))

print(f"FAISS index saved to {FAISS_INDEX_PATH}")
print(f"Metadata saved to {META_PATH}")

# ----- Optional: Quick Sanity Test -----
# Search for the nearest neighbors of the first abstract
D, I = index.search(embeddings[:1], 5)
print("Nearest neighbors for the first abstract:", I[0])
print("PMIDs for neighbors:", np.array(meta)[I[0]])
