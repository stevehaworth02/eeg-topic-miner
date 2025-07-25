# src/embedder.py
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

ENCODER_NAME = "allenai/scibert_scivocab_uncased"


class EncoderOnly(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(ENCODER_NAME)
        self.encoder.eval()

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            return self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).last_hidden_state[:, 0]


def embed_abstracts(jsonl_path: str) -> tuple[np.ndarray, list[str]]:
    """Read a JSONL of abstracts, return (embeddings, pmids)."""
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EncoderOnly().to(device)

    texts: list[str] = []
    pmids: list[str] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["abstract"])
            pmids.append(obj.get("pmid", "unknown"))

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    input_ids = inputs["input_ids"].to(device)
    attn_mask = inputs["attention_mask"].to(device)

    embs = model(input_ids, attn_mask).cpu().numpy()
    return embs, pmids
