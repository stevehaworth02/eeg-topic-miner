import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel

MODEL_CKPT = "models/scibert_best/best_model.pt"
FAISS_INDEX_PATH = "models/scibert_best/faiss.index"
META_PATH = "models/scibert_best/faiss_meta.npy"
ENCODER_NAME = "allenai/scibert_scivocab_uncased"
TOP_N = 5  # number of results to show

# ----- Load model and tokenizer -----
class EncoderOnly(torch.nn.Module):
    def __init__(self, encoder_name=ENCODER_NAME):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.encoder.eval()
    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            return self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME)
model = EncoderOnly().to(device)
state_dict = torch.load(MODEL_CKPT, map_location=device)
model.encoder.load_state_dict(
    {k.replace("encoder.", ""): v for k, v in state_dict.items() if k.startswith("encoder.")},
    strict=False
)
model.eval()

# ----- Load FAISS index and metadata -----
index = faiss.read_index(FAISS_INDEX_PATH)
meta = np.load(META_PATH, allow_pickle=True)

# ----- Query loop -----
def embed_query(text):
    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attn_mask = inputs["attention_mask"].to(device)
    emb = model(input_ids, attn_mask).cpu().numpy()
    return emb

def print_results(I, D):
    print("\nTop results:")
    for rank, (idx, dist) in enumerate(zip(I[0], D[0]), 1):
        print(f"Rank {rank}: PMID: {meta[idx]} (distance: {dist:.4f})")

if __name__ == "__main__":
    print("EEG-Topic-Miner Semantic Search")
    print("Type your query and press Enter (type 'exit' to quit)\n")
    while True:
        query = input("Query> ")
        if query.strip().lower() in ("exit", "quit"):
            break
        if not query.strip():
            print("Please enter a non-empty query.")
            continue
        try:
            emb = embed_query(query)
            D, I = index.search(emb, TOP_N)
            print_results(I, D)
        except Exception as e:
            print("Error searching index:", str(e))

