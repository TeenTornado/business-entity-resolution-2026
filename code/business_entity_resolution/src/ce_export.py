"""Export text pairs for the cross-encoder (ce.py): hard training pairs, uncertain
validation pairs (with c1's stage-2 scores) and uncertain test pairs."""
import argparse
import glob
import os
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd

from io_utils import read_p1, read_p23


def texts(work, split):
    cols = ["raw_name", "raw_addr", "country"]
    P1, P23 = read_p1(work, split, cols), read_p23(work, split, cols)
    f = lambda P: (P.raw_name.str.strip() + " | " + P.raw_addr.str.strip() + " | " + P.country).values  # noqa: E731
    return f(P1), f(P23)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--c1-dir", required=True, help="stage-2 dir with stage1_fold*.txt, decision.json, val_pairs.parquet")
    ap.add_argument("--test-feat", required=True)
    ap.add_argument("--test-scores", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=800000)
    ap.add_argument("--prune-thr", type=float, default=0.004)
    ap.add_argument("--lo", type=float, default=0.005)
    ap.add_argument("--hi", type=float, default=0.995)
    ap.add_argument("--max-train", type=int, default=600000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import json
    cols = json.load(open(f"{a.c1_dir}/decision.json"))["feat_cols"]
    cache = f"{a.work}/train_pairs_n{a.n_train}_p{a.prune_thr}.parquet"
    tc, tr_ids, va, vb = pickle.load(open(cache + ".meta", "rb"))[:4]
    # ---- training pairs: out-of-fold stage-1 score in the uncertain band, plus some easy ones
    c = pd.read_parquet(cache, columns=["i1", "i2", "label"] + cols)
    c = c[c.i1.isin(tr_ids)].reset_index(drop=True)
    fold = np.random.RandomState(7).rand(int(c.i1.max()) + 1) < 0.5
    f_of = fold[c.i1.values]
    p = np.zeros(len(c), np.float32)
    for k in (0, 1):
        m = lgb.Booster(model_file=f"{a.c1_dir}/stage1_fold{k}.txt")
        oof = f_of != bool(k)
        p[oof] = m.predict(c.loc[oof, cols], num_threads=4)
    c = c[["i1", "i2", "label"]]
    rng = np.random.RandomState(0)
    hard = (p > 0.01) & (p < 0.99)
    take = hard | (rng.rand(len(c)) < 0.05)
    c = c[take].reset_index(drop=True)
    if len(c) > a.max_train:
        c = c.sample(a.max_train, random_state=0).reset_index(drop=True)
    t1, t23 = texts(a.work, "train")
    pd.DataFrame({"text_a": t1[c.i1.values], "text_b": t23[c.i2.values], "label": c.label.values.astype(np.int8)}
                 ).to_parquet(f"{a.out}/ce_train.parquet", compression="zstd")
    print(f"train pairs {len(c)} (hard {int(hard.sum())}) positives {c.label.mean():.3f}", flush=True)
    # ---- validation pairs (c1 stage-2 scores)
    v = pd.read_parquet(f"{a.c1_dir}/val_pairs.parquet")
    u = (v.p2 > a.lo) & (v.p2 < a.hi)
    np.save(f"{a.out}/val_idx.npy", np.flatnonzero(u.values))
    pd.DataFrame({"text_a": t1[v.i1.values[u]], "text_b": t23[v.i2.values[u]]}).to_parquet(
        f"{a.out}/ce_val.parquet", compression="zstd")
    print(f"val uncertain pairs {int(u.sum())} of {len(v)}", flush=True)
    del t1, t23
    # ---- test pairs
    b = pd.concat([pd.read_parquet(f, columns=["i1", "i2"]) for f in sorted(glob.glob(f"{a.test_feat}/block*.parquet"))],
                  ignore_index=True)
    s = np.load(a.test_scores)
    u = (s > a.lo) & (s < a.hi)
    np.save(f"{a.out}/test_idx.npy", np.flatnonzero(u))
    t1, t23 = texts(a.work, "test")
    pd.DataFrame({"text_a": t1[b.i1.values[u]], "text_b": t23[b.i2.values[u]]}).to_parquet(
        f"{a.out}/ce_test.parquet", compression="zstd")
    print(f"test uncertain pairs {int(u.sum())} of {len(b)}", flush=True)


if __name__ == "__main__":
    main()
