#!/usr/bin/env python
"""
Brick 4 – Fine-tune SciBERT with Ray Tune (two-head classifier)

  • topic_head  (4-class soft-max)     -> topic_id
  • dl_head     (binary sigmoid)       -> uses_dl

Run:
  python src/train.py --data_dir data/tokenised --out_dir models/scibert_best
"""
from pathlib import Path
import os, argparse, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, accuracy_score
from datasets import load_from_disk
from transformers import (
    AutoModel, AdamW, get_cosine_schedule_with_warmup
)
import ray
from ray import tune
from ray.air import session

N_LABELS = 4  # seizure / sleep / bci / unlabeled


# ───────────────────────── model ─────────────────────────────
class MultiHeadClassifier(nn.Module):
    def __init__(self, encoder_name="allenai/scibert_scivocab_uncased"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size
        self.topic_head = nn.Linear(hidden, N_LABELS)
        self.dl_head    = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask):
        h_cls = self.encoder(input_ids,
                             attention_mask=attention_mask
                            ).last_hidden_state[:, 0]   # [CLS]
        return self.topic_head(h_cls), self.dl_head(h_cls).squeeze(-1)


# ─────────── stratified 80/10/10 split (no trunc. error) ─────
def split_dataset(ds, train_frac=0.8, val_frac=0.1, seed=42):
    y = ds["topic_id"]
    sss = StratifiedShuffleSplit(n_splits=1, train_size=train_frac,
                                 random_state=seed)
    train_idx, hold_idx = next(sss.split(np.zeros(len(y)), y))
    val_size = int(val_frac * len(ds))
    val_idx  = hold_idx[:val_size]
    test_idx = hold_idx[val_size:]
    return (Subset(ds, train_idx),
            Subset(ds, val_idx),
            Subset(ds, test_idx))


# ──────────────── training loop for Ray Tune ────────────────
def train_loop(config):
    ds = load_from_disk(config["data_dir"])
    train_ds, val_ds, _ = split_dataset(ds, seed=42)

    def collate(batch):
        keys = ("input_ids", "attention_mask")
        x = {k: torch.tensor([b[k] for b in batch]) for k in keys}
        y_topic = torch.tensor([b["topic_id"] for b in batch])
        y_dl    = torch.tensor([b["uses_dl"]  for b in batch], dtype=torch.float32)
        return x, y_topic, y_dl

    train_loader = DataLoader(train_ds, batch_size=config["batch"],
                              shuffle=True, collate_fn=collate)
    val_loader   = DataLoader(val_ds, batch_size=64,
                              shuffle=False, collate_fn=collate)

    model = MultiHeadClassifier().to("cuda")
    opt   = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["wd"])
    sched = get_cosine_schedule_with_warmup(
        opt, num_warmup_steps=0,
        num_training_steps=len(train_loader)*config["epochs"]
    )
    ce, bce = nn.CrossEntropyLoss(), nn.BCEWithLogitsLoss()

    for epoch in range(config["epochs"]):
        model.train()
        for x, yt, yd in train_loader:
            x = {k: v.to("cuda") for k, v in x.items()}
            yt, yd = yt.to("cuda"), yd.to("cuda")
            opt.zero_grad()
            logit_t, logit_d = model(**x)
            loss = ce(logit_t, yt) + bce(logit_d, yd)
            loss.backward(); opt.step(); sched.step()

        # ── validation ──
        model.eval()
        pt, tt, pd, td = [], [], [], []
        with torch.no_grad():
            for x, yt, yd in val_loader:
                x = {k: v.to("cuda") for k, v in x.items()}
                lt, ld = model(**x)
                pt += lt.argmax(-1).cpu().tolist()
                tt += yt.tolist()
                pd += (torch.sigmoid(ld) > 0.5).cpu().int().tolist()
                td += yd.int().tolist()

        session.report({
            "topic_f1": f1_score(tt, pt, average="macro"),
            "dl_acc":  accuracy_score(td, pd),
            "epoch":   epoch
        })

    ckpt_dir = session.get_checkpoint().get_directory()
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "model.pt"))


# ────────────── CLI & Ray Tune sweep setup ───────────────────
def main(data_dir, out_dir):
    data_dir = Path(data_dir).resolve().as_posix()
    out_dir  = Path(out_dir).resolve().as_posix()

    search = {
        "lr":     tune.loguniform(5e-6, 5e-5),
        "batch":  tune.choice([16, 32]),
        "epochs": tune.choice([3, 4]),
        "wd":     tune.uniform(0.0, 0.1),
        "data_dir": data_dir,
    }

    tuner = tune.Tuner(
        train_loop,
        param_space=search,
        tune_config=tune.TuneConfig(
            metric="topic_f1", mode="max", num_samples=8
        ),
        run_config=ray.air.RunConfig(name="scibert_multitask")  # default ~/ray_results/
    )
    results = tuner.fit()
    best_ckpt = results.get_best_result("topic_f1", "max").checkpoint
    best_ckpt.to_directory(out_dir)
    print(f"✓ Best checkpoint saved → {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/tokenised")
    ap.add_argument("--out_dir",  default="models/scibert_best")
    args = ap.parse_args()
    main(args.data_dir, args.out_dir)
