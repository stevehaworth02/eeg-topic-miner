#!/usr/bin/env python
"""
Drop‑in replacement for preprocess_tokenize that uses plain
Python (no Ray) so it works on Windows without Raylet.
CLI flags are identical to the original script.
"""
import argparse
from pathlib import Path
from typing import Dict, List
import orjson
from transformers import AutoTokenizer
from datasets import Dataset

# ── 1. Weak-label keyword maps ───────────────────────────────
TOPIC_KWS: Dict[str, List[str]] = {
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

# ── 2. Labeling functions ───────────────────────────────────
def weak_topic(text: str) -> str:
    txt = text.lower()
    for label, kws in TOPIC_KWS.items():
        if any(kw in txt for kw in kws):
            return label
    return "unlabeled"

def has_deep_learning(text: str) -> int:
    txt = text.lower()
    return int(any(kw in txt for kw in DL_KWS))

# ── 3. Tokenizer setup ───────────────────────────────────────
TOKENIZER = AutoTokenizer.from_pretrained(
    "allenai/scibert_scivocab_uncased", use_fast=True
)

def main(raw_jsonl: str, out_dir: str, workers: int = 1):
    records = []
    with open(raw_jsonl, 'rb') as rf:
        for line in rf:
            obj = orjson.loads(line)
            abstr = obj.get('abstract', '').strip()
            # skip if there's no abstract
            if not abstr:
                continue

            txt = (obj.get('title', '') + ' ' + abstr).strip()
            records.append({
                'text': txt,
                'topic': weak_topic(txt),
                'uses_dl': has_deep_learning(txt),
                'pmid': obj.get('pmid', None),
                'title': obj.get('title', ''),
            })

    if not records:
        print("No records with abstracts found. Exiting.")
        return

    # tokenize in one batch (small enough locally)
    texts = [r['text'] for r in records]
    topics = [TOPIC2ID[r['topic']] for r in records]
    dls    = [r['uses_dl'] for r in records]
    pmids  = [r['pmid'] for r in records]
    titles = [r['title'] for r in records]

    toks = TOKENIZER(
        texts,
        padding='max_length',
        truncation=True,
        max_length=256,
        return_tensors='np'
    )

    # build HuggingFace dataset
    ds = Dataset.from_dict({
        'input_ids':      toks['input_ids'],
        'attention_mask': toks['attention_mask'],
        'topic_id':       topics,
        'uses_dl':        dls,
        'pmid':           pmids,
        'title':          titles,
    })

    # save to disk
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(out_dir)

    uri = out_path.resolve().as_uri()
    print(f"✔ Tokenised dataset saved → {out_dir}")
    print(f"🔗  You can browse it at: {uri}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw_jsonl', required=True,
                    help="Path to data/raw.jsonl")
    ap.add_argument('--out_dir',   required=True,
                    help="Directory to write the tokenised dataset")
    ap.add_argument('--workers',   type=int, default=1,
                    help="(unused) kept for CLI compatibility")
    args = ap.parse_args()
    main(args.raw_jsonl, args.out_dir, args.workers)
