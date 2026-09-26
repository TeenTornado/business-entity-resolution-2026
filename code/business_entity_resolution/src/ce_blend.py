"""Blend the cross-encoder with the stage-2 LightGBM scores on uncertain pairs.

A logistic blend logit(p) = w0 + w1*logit(p2) + w2*logit(p_ce) is fitted on validation
half A (uncertain pairs only); the threshold is re-tuned on half A with the one-owner
rule; half B is reported next to the unblended c1 score on the same pairs. Pairs outside
the uncertain band keep their stage-2 score.
"""
import argparse
import glob
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import scoring
from io_utils import read_p1, read_p23
from train import tune


def logit(p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--c1-dir", required=True)
    ap.add_argument("--ce-data", required=True)
    ap.add_argument("--test-feat", required=True)
    ap.add_argument("--test-scores", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--n-train", type=int, default=800000)
    ap.add_argument("--prune-thr", type=float, default=0.004)
    a = ap.parse_args()
    cache = f"{a.work}/train_pairs_n{a.n_train}_p{a.prune_thr}.parquet"
    tc, _, va_a, va_b = pickle.load(open(cache + ".meta", "rb"))[:4]
    v = pd.read_parquet(f"{a.c1_dir}/val_pairs.parquet")
    vi = np.load(f"{a.ce_data}/val_idx.npy")
    ce = np.load(f"{a.ce_data}/val_ce.npy")
    X = np.c_[logit(v.p2.values[vi]), logit(ce)]
    A = v.half.values[vi] == 0
    lr = LogisticRegression(C=10.0).fit(X[A], v.label.values[vi][A])
    print("blend weights", lr.intercept_, lr.coef_, flush=True)
    pb = v.p2.values.astype(np.float64).copy()
    pb[vi] = lr.predict_proba(X)[:, 1]
    res = {}
    for name, p in (("c1", v.p2.values), ("blend", pb)):
        vA, vB = v[v.half == 0].reset_index(drop=True), v[v.half == 1].reset_index(drop=True)
        (fA, (thr, _)), _ = tune(vA, p[v.half.values == 0], va_a, tc)
        kB = scoring.decide(vB, p[v.half.values == 1], thr)
        fB, _ = scoring.macro_f05(va_b, vB, kB, tc)
        yB = vB.label.values == 1
        res[name] = (thr, fB)
        print(f"VALIDATION {name}: thr {thr:.4f} half A F={fA:.5f} -> half B macro F0.5 = {fB:.5f} | "
              f"TP {int((kB & yB).sum())} FP {int((kB & ~yB).sum())} FN {int((~kB & yB).sum())}", flush=True)
    # ---- test
    b = pd.concat([pd.read_parquet(f, columns=["i1", "i2"]) for f in sorted(glob.glob(f"{a.test_feat}/block*.parquet"))],
                  ignore_index=True)
    s = np.load(a.test_scores).astype(np.float64)
    ti = np.load(f"{a.ce_data}/test_idx.npy")
    tce = np.load(f"{a.ce_data}/test_ce.npy")
    s[ti] = lr.predict_proba(np.c_[logit(s[ti]), logit(tce)])[:, 1]
    keep = scoring.decide(b, s, res["blend"][0])
    keep1 = scoring.decide(b, np.load(a.test_scores), 0.7375)
    P1 = read_p1(a.work, "test", ["entity_id", "country"])
    ct = P1.country.values[b.i1.values]
    for c in ("US", "India", "France"):
        m = ct == c
        print(f"test {c}: c1 {int(keep1[m].sum())} blend {int(keep[m].sum())} "
              f"(only c1 {int((keep1 & ~keep & m).sum())}, only blend {int((keep & ~keep1 & m).sum())})", flush=True)
    os.makedirs(a.out, exist_ok=True)
    from predict import write_lists
    ids23 = read_p23(a.work, "test", ["entity_id"]).entity_id.values
    sel = b[keep]
    write_lists(f"{a.out}/matching_results.tsv", "matched_entity_ids", P1.entity_id.values,
                sel.i1.values, ids23[sel.i2.values])
    import shutil
    shutil.copy(a.candidates, f"{a.out}/candidate_pairs.tsv")
    print(f"wrote {a.out}: pairs {int(keep.sum())}; blend beats c1 on half B: {res['blend'][1] > res['c1'][1]}")


if __name__ == "__main__":
    main()
