"""v2 cross-encoder data: v1 training pairs + French pseudo-labelled test pairs (from the
blended ce1 scores), and a wider uncertain band for val/test scoring."""
import argparse
import glob
import os

import numpy as np
import pandas as pd

from ce_blend import logit
from ce_export import texts
from io_utils import read_p1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--c1-dir", required=True)
    ap.add_argument("--v1", required=True, help="v1 ce_data dir (ce_train.parquet, *_idx.npy, *_ce.npy)")
    ap.add_argument("--test-feat", required=True)
    ap.add_argument("--test-scores", required=True)
    ap.add_argument("--w", default="0.34095672,0.72499883,0.35208133", help="ce1 blend: w0,w_p2,w_ce")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lo", type=float, default=0.02)
    ap.add_argument("--hi", type=float, default=0.98)
    ap.add_argument("--n-fr", type=int, default=150000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    w0, w1, w2 = map(float, a.w.split(","))
    s = np.load(a.test_scores).astype(np.float64)
    ti, tce = np.load(f"{a.v1}/test_idx.npy"), np.load(f"{a.v1}/test_ce.npy")
    pb = s.copy()
    pb[ti] = 1 / (1 + np.exp(-(w0 + w1 * logit(s[ti]) + w2 * logit(tce))))
    b = pd.concat([pd.read_parquet(f, columns=["i1", "i2"]) for f in sorted(glob.glob(f"{a.test_feat}/block*.parquet"))],
                  ignore_index=True)
    fr = read_p1(a.work, "test", ["country"]).country.values[b.i1.values] == "France"
    rng = np.random.RandomState(0)
    pos = np.flatnonzero(fr & (pb >= 0.98))
    neg = np.flatnonzero(fr & (pb <= 0.02) & (pb >= 0.001))
    pos = rng.choice(pos, min(len(pos), a.n_fr // 2), replace=False)
    neg = rng.choice(neg, min(len(neg), a.n_fr // 2), replace=False)
    t1, t23 = texts(a.work, "test")
    sel = np.r_[pos, neg]
    frdf = pd.DataFrame({"text_a": t1[b.i1.values[sel]], "text_b": t23[b.i2.values[sel]],
                         "label": np.r_[np.ones(len(pos)), np.zeros(len(neg))].astype(np.int8)})
    tr = pd.concat([pd.read_parquet(f"{a.v1}/ce_train.parquet"), frdf], ignore_index=True)
    tr = tr.sample(frac=1.0, random_state=0).reset_index(drop=True)
    tr.to_parquet(f"{a.out}/ce_train.parquet", compression="zstd")
    print(f"train {len(tr)} (France pseudo: {len(pos)} pos, {len(neg)} neg)", flush=True)
    u = (s > a.lo) & (s < a.hi)
    np.save(f"{a.out}/test_idx.npy", np.flatnonzero(u))
    pd.DataFrame({"text_a": t1[b.i1.values[u]], "text_b": t23[b.i2.values[u]]}).to_parquet(
        f"{a.out}/ce_test.parquet", compression="zstd")
    print(f"test pairs to score {int(u.sum())}", flush=True)
    del t1, t23
    v = pd.read_parquet(f"{a.c1_dir}/val_pairs.parquet")
    uv = (v.p2 > a.lo) & (v.p2 < a.hi)
    np.save(f"{a.out}/val_idx.npy", np.flatnonzero(uv.values))
    v1, v23 = texts(a.work, "train")
    pd.DataFrame({"text_a": v1[v.i1.values[uv]], "text_b": v23[v.i2.values[uv]]}).to_parquet(
        f"{a.out}/ce_val.parquet", compression="zstd")
    print(f"val pairs to score {int(uv.sum())}", flush=True)


if __name__ == "__main__":
    main()
