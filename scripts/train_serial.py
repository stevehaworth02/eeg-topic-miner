#!/usr/bin/env python
"""
Serial fine-tuning for SciBERT without Ray Tune (for Windows)

Run:
  python scripts/train_serial.py \
      --data_dir data/tokenised \
      --out_dir models/scibert_best \
      --batch 16 --epochs 3 --lr 2e-5 --wd 0.01
"""
import os
import argparse
import platform
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, accuracy_score
from datasets import load_from_disk
from transformers import AutoModel, AdamW, get_cosine_schedule_with_warmup

def get_device():
    # Force CPU on Windows due to Raylet issues, otherwise auto-detect
    if platform.system() == "Windows":
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

device = get_device()
print(f"▶ using device: {device}")

N_LABELS = 4
class MultiHeadClassifier(nn.Module):
    def __init__(self, encoder_name="allenai/scibert_scivocab_uncased"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size
        self.topic_head = nn.Linear(hidden, N_LABELS)
        self.dl_head    = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h_cls = outputs.last_hidden_state[:, 0]  # [CLS] token
        return self.topic_head(h_cls), self.dl_head(h_cls).squeeze(-1)

def split_dataset(ds, train_frac=0.8, val_frac=0.1, seed=42):
    y = ds["topic_id"]
    sss = StratifiedShuffleSplit(n_splits=1, train_size=train_frac, random_state=seed)
    train_idx, hold_idx = next(sss.split(np.zeros(len(y)), y))
    val_size = int(val_frac * len(ds))
    val_idx  = hold_idx[:val_size]
    test_idx = hold_idx[val_size:]
    return Subset(ds, train_idx), Subset(ds, val_idx), Subset(ds, test_idx)

def main(data_dir, out_dir, batch, epochs, lr, wd):
    ds = load_from_disk(data_dir)
    train_ds, val_ds, _ = split_dataset(ds)

    def collate(batch):
        x = {
            "input_ids":     torch.tensor([b["input_ids"] for b in batch], dtype=torch.long).to(device),
            "attention_mask":torch.tensor([b["attention_mask"] for b in batch], dtype=torch.long).to(device)
        }
        y_topic = torch.tensor([b["topic_id"] for b in batch], dtype=torch.long).to(device)
        y_dl    = torch.tensor([b["uses_dl"]  for b in batch], dtype=torch.float32).to(device)
        return x, y_topic, y_dl

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,  collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=64,   shuffle=False, collate_fn=collate)

    model = MultiHeadClassifier().to(device)
    opt   = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = get_cosine_schedule_with_warmup(
        opt,
        num_warmup_steps=0,
        num_training_steps=len(train_loader) * epochs
    )
    ce, bce = nn.CrossEntropyLoss(), nn.BCEWithLogitsLoss()

    best_f1 = 0.0
    os.makedirs(out_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        for x, yt, yd in train_loader:
            opt.zero_grad()
            logit_t, logit_d = model(**x)
            loss = ce(logit_t, yt) + bce(logit_d, yd)
            loss.backward()
            opt.step()
            sched.step()

        model.eval()
        preds, trues = [], []
        d_preds, d_trues = [], []
        with torch.no_grad():
            for x, yt, yd in val_loader:
                lt, ld = model(**x)
                preds  += lt.argmax(-1).cpu().tolist()
                trues  += yt.cpu().tolist()
                d_preds += (torch.sigmoid(ld) > 0.5).cpu().int().tolist()
                d_trues += yd.cpu().int().tolist()

        topic_f1 = f1_score(trues, preds, average="macro")
        dl_acc   = accuracy_score(d_trues, d_preds)
        print(f"Epoch {epoch+1}/{epochs} — topic_f1: {topic_f1:.4f}, dl_acc: {dl_acc:.4f}")

        if topic_f1 > best_f1:
            best_f1 = topic_f1
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pt"))

    print(f"✔ saved best checkpoint (topic_f1={best_f1:.4f}) → {out_dir}/best_model.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/tokenised")
    ap.add_argument("--out_dir",  default="models/scibert_best")
    ap.add_argument("--batch",    type=int,   default=16)
    ap.add_argument("--epochs",   type=int,   default=3)
    ap.add_argument("--lr",       type=float, default=2e-5)
    ap.add_argument("--wd",       type=float, default=0.01)
    args = ap.parse_args()
    main(args.data_dir, args.out_dir, args.batch, args.epochs, args.lr, args.wd)
