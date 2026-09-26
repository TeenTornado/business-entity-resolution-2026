"""Cross-encoder re-scorer for uncertain pairs (runs on a machine with a GPU / Apple MPS).

A multilingual transformer (default intfloat/multilingual-e5-large, MIT licence, 560 M
parameters, XLM-RoBERTa-large backbone) reads both records' raw name + address text
jointly and predicts match / no-match. Its multilingual pretraining covers French words
and address conventions that the hand-built features only learn from US/India training
pairs. Only pairs the LightGBM pipeline is unsure about are scored; ce_blend.py combines
both scores on validation and decides whether the blend is used at all.

Long runs are resumable: a checkpoint is written every --ckpt-every steps and `train`
continues from the latest one automatically.

  python3 ce.py train --data ce_data --out ce_model
  python3 ce.py score --data ce_data --model ce_model      # writes ce_data/{val,test}_ce.npy
"""
import argparse
import glob
import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE = "intfloat/multilingual-e5-large"


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encode(tok, a, b, max_len):
    return tok(list(a), list(b), truncation=True, max_length=max_len, padding=True, return_tensors="pt")


def latest_ckpt(out):
    c = sorted(glob.glob(f"{out}/ckpt_*"), key=lambda p: int(p.rsplit("_", 1)[1]))
    return c[-1] if c else None


def cmd_train(a):
    dev = device()
    df = pd.read_parquet(f"{a.data}/ce_train.parquet")
    if a.max_pairs and len(df) > a.max_pairs:
        df = df.sample(a.max_pairs, random_state=0).reset_index(drop=True)
    print(f"train pairs {len(df)} positives {df.label.mean():.3f} device {dev} model {a.base}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForSequenceClassification.from_pretrained(a.base, num_labels=1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    per_epoch = int(np.ceil(len(df) / a.bs))
    steps = a.epochs * per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.06)
    start = 0
    ck = latest_ckpt(a.out)
    if ck:
        st = torch.load(f"{ck}/state.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start = st["step"]
        print(f"resumed from {ck} at step {start}", flush=True)
    y = df.label.values.astype(np.float32)
    A, B = df.text_a.values, df.text_b.values
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    t0, done = time.time(), 0
    for step in range(start, steps):
        ep, k = divmod(step, per_epoch)
        idx = np.random.RandomState(ep).permutation(len(df))[k * a.bs:(k + 1) * a.bs]
        enc = {kk: v.to(dev) for kk, v in encode(tok, A[idx], B[idx], a.max_len).items()}
        loss = lossf(model(**enc).logits.squeeze(-1), torch.from_numpy(y[idx]).to(dev))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        done += 1
        if (step + 1) % 100 == 0:
            el = time.time() - t0
            print(f"step {step + 1}/{steps} loss {loss.item():.4f} {done * a.bs / el:.0f} pairs/s "
                  f"eta {el / done * (steps - step - 1) / 3600:.2f} h", flush=True)
        if (step + 1) % a.ckpt_every == 0 or step + 1 == steps:
            path = f"{a.out}/ckpt_{step + 1}"
            os.makedirs(path, exist_ok=True)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "step": step + 1}, f"{path}/state.pt")
            for old in sorted(glob.glob(f"{a.out}/ckpt_*"), key=lambda p: int(p.rsplit("_", 1)[1]))[:-2]:
                for f in glob.glob(f"{old}/*"):
                    os.remove(f)
                os.rmdir(old)
            print(f"checkpoint {path}", flush=True)
    model.save_pretrained(f"{a.out}/final")
    tok.save_pretrained(f"{a.out}/final")
    print("TRAINING DONE", flush=True)


@torch.no_grad()
def cmd_score(a):
    dev = device()
    src = a.model
    if not os.path.exists(f"{src}/final/config.json"):
        # score from the latest checkpoint (e.g. when time runs out before the end of training)
        ck = latest_ckpt(src)
        tok = AutoTokenizer.from_pretrained(a.base)
        model = AutoModelForSequenceClassification.from_pretrained(a.base, num_labels=1)
        model.load_state_dict(torch.load(f"{ck}/state.pt", map_location="cpu", weights_only=False)["model"])
        print(f"scoring with checkpoint {ck}", flush=True)
    else:
        tok = AutoTokenizer.from_pretrained(f"{src}/final")
        model = AutoModelForSequenceClassification.from_pretrained(f"{src}/final")
    model = model.to(dev).eval()
    for name in ("val", "test"):
        df = pd.read_parquet(f"{a.data}/ce_{name}.parquet")
        out = np.zeros(len(df), np.float32)
        t0 = time.time()
        bs = a.bs * 4
        for lo in range(0, len(df), bs):
            enc = {k: v.to(dev) for k, v in encode(tok, df.text_a.values[lo:lo + bs], df.text_b.values[lo:lo + bs],
                                                  a.max_len).items()}
            out[lo:lo + bs] = torch.sigmoid(model(**enc).logits.squeeze(-1)).float().cpu().numpy()
            if (lo // bs) % 200 == 0:
                print(f"{name}: {lo + bs}/{len(df)} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        np.save(f"{a.data}/{name}_ce.npy", out)
        print(f"SCORED {name}: {len(df)} pairs in {(time.time() - t0) / 60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "score"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="ce_model")
    ap.add_argument("--model", default="ce_model")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--max-pairs", type=int, default=0, help="subsample training pairs (0 = all)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=96)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    a = ap.parse_args()
    cmd_train(a) if a.cmd == "train" else cmd_score(a)


if __name__ == "__main__":
    main()
