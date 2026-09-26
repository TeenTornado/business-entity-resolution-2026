"""Cross-encoder re-scorer for uncertain pairs (runs on a machine with a GPU / Apple MPS).

A small multilingual transformer (intfloat/multilingual-e5-small, MIT licence, 118 M
parameters) reads both records' raw name + address text jointly and predicts match /
no-match. Its multilingual pretraining covers French words and address conventions
that the hand-built features only learn from US/India training pairs. Only pairs the
LightGBM pipeline is unsure about are scored; blend.py combines both scores on
validation and decides whether the blend is used at all.

  python3 ce.py train --data ce_data --out ce_model          # ~30-40 min on an M4
  python3 ce.py score --data ce_data --model ce_model        # writes ce_data/*_ce.npy
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE = "intfloat/multilingual-e5-small"  # --base intfloat/multilingual-e5-base for the larger model


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batches(df, tok, bs, max_len, shuffle=False, seed=0):
    idx = np.random.RandomState(seed).permutation(len(df)) if shuffle else np.arange(len(df))
    a, b = df.text_a.values, df.text_b.values
    for lo in range(0, len(idx), bs):
        j = idx[lo:lo + bs]
        enc = tok(list(a[j]), list(b[j]), truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        yield j, enc


def cmd_train(a):
    dev = device()
    df = pd.read_parquet(f"{a.data}/ce_train.parquet")
    if a.max_pairs and len(df) > a.max_pairs:
        df = df.sample(a.max_pairs, random_state=0).reset_index(drop=True)
    print(f"train pairs {len(df)} positives {df.label.mean():.3f} device {dev}", flush=True)
    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForSequenceClassification.from_pretrained(a.base, num_labels=1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * int(np.ceil(len(df) / a.bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.06)
    y = torch.tensor(df.label.values, dtype=torch.float32)
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    t0, step = time.time(), 0
    for ep in range(a.epochs):
        for j, enc in batches(df, tok, a.bs, a.max_len, shuffle=True, seed=ep):
            enc = {k: v.to(dev) for k, v in enc.items()}
            loss = lossf(model(**enc).logits.squeeze(-1), y[j].to(dev))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 200 == 0:
                el = time.time() - t0
                print(f"step {step}/{steps} loss {loss.item():.4f} {step * a.bs / el:.0f} pairs/s "
                      f"eta {el / step * (steps - step) / 60:.1f} min", flush=True)
    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print("saved", a.out, flush=True)


@torch.no_grad()
def cmd_score(a):
    dev = device()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(a.model).to(dev).eval()
    for name in ("val", "test"):
        df = pd.read_parquet(f"{a.data}/ce_{name}.parquet")
        out = np.zeros(len(df), np.float32)
        t0 = time.time()
        for j, enc in batches(df, tok, a.bs * 4, a.max_len):
            enc = {k: v.to(dev) for k, v in enc.items()}
            out[j] = torch.sigmoid(model(**enc).logits.squeeze(-1)).float().cpu().numpy()
        np.save(f"{a.data}/{name}_ce.npy", out)
        print(f"scored {name}: {len(df)} pairs in {(time.time() - t0) / 60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "score"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="ce_model")
    ap.add_argument("--model", default="ce_model")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--max-pairs", type=int, default=0, help="subsample training pairs (0 = all)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--max-len", type=int, default=96)
    a = ap.parse_args()
    cmd_train(a) if a.cmd == "train" else cmd_score(a)


if __name__ == "__main__":
    main()
