#!/usr/bin/env python
"""
Brick #3  –  weak-label + parallel tokenisation
Outputs Hugging Face Arrow shards in data/tokenised/
"""
import os, json, argparse
from pathlib import Path
from typing import Dict

import ray
from datasets import Dataset
from transformers import AutoTokenizer
from ray.data import read_json

# ── 1. Weak-label keyword maps ───────────────────────────────
TOPIC_KWS: Dict[str, list[str]] = {
    "seizure": ["seizure", "ictal", "epilep"],
    "sleep":   ["sleep", "psg", "hypnogram", " nrem", " rem ", "insomnia"],
    "bci":     ["brain computer interface", "bci", "ssvep", "p300", "motor imagery"],
}
TOPIC2ID = {lbl: i for i, lbl in enumerate(TOPIC_KWS)}
TOPIC2ID["unlabeled"] = len(TOPIC2ID)

DL_KWS = [
    "deep learning", "convolutional neural", "cnn", "convnet",
    "transformer", "attention mechanism", "bert",
    "lstm", "gru", "recurrent neural", "autoencoder",
]

def weak_topic(text: str) -> str:
    txt = text.lower()
    for label, kws in TOPIC_KWS.items():
        if any(kw in txt for kw in kws):
            return label
    return "unlabeled"

def has_deep_learning(text: str) -> int:
    txt = text.lower()
    return int(any(kw in txt for kw in DL_KWS))

# ── 2. Tokeniser setup ───────────────────────────────────────
TOKENIZER = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")

def tok_batch(batch):
    tok = TOKENIZER(batch["text"],
                    padding="max_length",
                    truncation=True,
                    max_length=256)
    tok["topic_id"] = [TOPIC2ID[t] for t in batch["topic"]]
    tok["uses_dl"]  = batch["uses_dl"]
    return tok

# ── 3. Pipeline ──────────────────────────────────────────────
def main(raw_jsonl: str, out_dir: str, workers: int = 4):
    ray.init(num_cpus=workers, include_dashboard=False)

    ds = read_json(raw_jsonl)
    ds = ds.map(lambda r: {
        "text": (r["title"] + " " + r["abstract"]).strip(),
        "topic": weak_topic(r["title"] + " " + r["abstract"]),
        "uses_dl": has_deep_learning(r["title"] + " " + r["abstract"]),
    })

    hf_ds = Dataset.from_pandas(ds.to_pandas())
    hf_ds = hf_ds.map(tok_batch,
                      batched=True,
                      batch_size=2048,
                      num_proc=workers,
                      remove_columns=["text", "topic", "uses_dl"])

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    hf_ds.save_to_disk(out_dir)
    ray.shutdown()
    print(f"Tokenised dataset with topic + DL flags saved → {out_dir}")

# ── 4. CLI ───────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_jsonl", default="data/raw.jsonl")
    ap.add_argument("--out_dir",  default="data/tokenised")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    main(args.raw_jsonl, args.out_dir, args.workers)
