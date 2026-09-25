"""Train the pairwise matcher on the training split and tune the decision rule
for macro F0.5 on a held-out set of S1 entities."""
import argparse
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

import features
import scoring

FIELD_COLS = ["entity_id", "country"] + features.FIELDS


def load(work, split):
    P1 = pd.read_parquet(f"{work}/{split}_p1.parquet", columns=FIELD_COLS)
    P23 = pd.concat([pd.read_parquet(f"{work}/{split}_p2.parquet", columns=FIELD_COLS),
                     pd.read_parquet(f"{work}/{split}_p3.parquet", columns=FIELD_COLS)], ignore_index=True)
    return P1, P23


def add_labels(cand, P1, P23, gt):
    pairs = gt.assign(m=gt.matched_entity_ids.str.split(",")).explode("m")
    pairs = pairs[pairs.m.notna() & (pairs.m != "")]
    owner = pd.Series(pd.Index(P1.entity_id).get_indexer(pairs.source1_entity_id.values), index=pairs.m.values)
    own_i2 = owner.reindex(P23.entity_id.values).fillna(-1).astype(np.int64).values
    cand["label"] = (own_i2[cand.i2.values] == cand.i1.values).astype(np.int8)
    tcount = pairs.groupby("source1_entity_id").size()
    truth_counts = pd.Series(tcount.values, index=pd.Index(P1.entity_id).get_indexer(tcount.index))
    return cand, truth_counts


def prepare_candidates(cand, ids23):
    cand["i1"] = cand.i1.astype(np.int32)
    cand["i2"] = cand.i2.astype(np.int32)
    cand["src3"] = ids23.str.startswith("S3").values[cand.i2.values].astype(np.int8)
    cand = scoring.context_features(cand)
    for c in cand.columns:
        if cand[c].dtype == np.float64:
            cand[c] = cand[c].astype(np.float32)
    return cand


def split_ids(n, seed=0, n_train=300000, n_val=150000):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    return np.sort(perm[:n_train]), np.sort(perm[n_train:n_train + n_val])


PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=127, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, num_threads=4)


def tune(cand_v, p, s1_val, truth_counts):
    best = (-1, None)
    grid = []
    for thr in np.arange(0.3, 0.951, 0.0125):
        for rel in (0.0,):
            keep = scoring.decide(cand_v, p, thr, one_owner=True, rel=rel)
            f, _ = scoring.macro_f05(s1_val, cand_v, keep, truth_counts)
            grid.append((round(float(thr), 3), rel, f))
            if f > best[0]:
                best = (f, (float(thr), rel))
    return best, grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--n-train", type=int, default=300000)
    ap.add_argument("--n-val", type=int, default=150000)
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--reuse-features", action="store_true")
    ap.add_argument("--stage2", action="store_true", help="adopt stage-2 model if it wins on validation")
    a = ap.parse_args()
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:7.0f}s] {m}", flush=True)  # noqa: E731

    ids = pd.read_parquet(f"{a.work}/train_p1.parquet", columns=["entity_id"]).entity_id
    ids23 = pd.concat([pd.read_parquet(f"{a.work}/train_p{s}.parquet", columns=["entity_id"]) for s in (2, 3)],
                      ignore_index=True).entity_id
    gt = pd.read_parquet(f"{a.work}/train_gt.parquet")
    cand = prepare_candidates(pd.read_parquet(f"{a.work}/train_cand.parquet"), ids23)
    cand, truth_counts = add_labels(cand, ids.to_frame(), ids23.to_frame(), gt)
    log(f"candidates={len(cand)} positives={cand.label.sum()} total_true={truth_counts.sum()} "
        f"pair_recall={cand.label.sum() / truth_counts.sum():.4f}")
    tr_ids, va_ids = split_ids(len(ids), n_train=a.n_train, n_val=a.n_val)
    # val half A: early stopping + stage-2 training; val half B: untouched, used for
    # threshold tuning and the reported validation score
    rng = np.random.RandomState(1)
    half = rng.rand(len(va_ids)) < 0.5
    va_a, va_b = va_ids[half], va_ids[~half]
    cand = cand[cand.i1.isin(np.concatenate([tr_ids, va_ids]))].reset_index(drop=True)
    del ids, ids23
    P1, P23 = load(a.work, "train")
    cand = scoring.freq_features(cand, P1, P23)
    cache = f"{a.work}/feat_pairs.parquet"
    if a.reuse_features and os.path.exists(cache):
        F = pd.read_parquet(cache)
        F = cand[["i1", "i2"]].merge(F, on=["i1", "i2"], how="left").drop(columns=["i1", "i2"])
        log("reused cached pairwise features")
    else:
        tables = features.build_df_tables([P1, P23])
        log("df tables built")
        F = features.compute(cand, P1, P23, tables)
        pd.concat([cand[["i1", "i2"]], F], axis=1).to_parquet(cache)
        log(f"features: {len(cand)} pairs")
    del P1, P23
    cand = pd.concat([cand, F.set_index(cand.index)], axis=1)
    del F
    feat_cols = features.FEATURE_NAMES + scoring.CONTEXT_COLS + scoring.FREQ_COLS
    tr = cand[cand.i1.isin(tr_ids)]
    vA = cand[cand.i1.isin(va_a)].reset_index(drop=True)
    vB = cand[cand.i1.isin(va_b)].reset_index(drop=True)
    del cand

    dtr = lgb.Dataset(tr[feat_cols], tr.label)
    dva = lgb.Dataset(vA[feat_cols], vA.label, reference=dtr)
    m1 = lgb.train(PARAMS, dtr, num_boost_round=a.rounds, valid_sets=[dva],
                   callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)])
    del dtr, dva, tr
    log(f"stage1 trained, best_iter={m1.best_iteration}")
    pA = m1.predict(vA[feat_cols], num_iteration=m1.best_iteration)
    pB = m1.predict(vB[feat_cols], num_iteration=m1.best_iteration)
    (f1, (thr1, _)), grid1 = tune(vB, pB, va_b, truth_counts)
    log(f"VALIDATION (half B) stage-1 macro F0.5 = {f1:.5f} at thr={thr1}")

    # stage 2: stage-1 probabilities of competing candidates as extra features
    cols2 = feat_cols + scoring.PROB_COLS
    vA = scoring.prob_context(vA, pA)
    vB = scoring.prob_context(vB, pB)
    p2 = dict(PARAMS, num_leaves=63)
    m2 = lgb.train(p2, lgb.Dataset(vA[cols2], vA.label), num_boost_round=600)
    qB = m2.predict(vB[cols2])
    (f2, (thr2, _)), grid2 = tune(vB, qB, va_b, truth_counts)
    log(f"VALIDATION (half B) stage-2 macro F0.5 = {f2:.5f} at thr={thr2}")
    base_keep = scoring.decide(vB, pB, 0.5, one_owner=False)
    log(f"(stage-1, plain p>=0.5, no one-owner rule: {scoring.macro_f05(va_b, vB, base_keep, truth_counts)[0]:.5f})")

    os.makedirs(a.model_dir, exist_ok=True)
    m1.save_model(f"{a.model_dir}/lgb_stage1.txt", num_iteration=m1.best_iteration)
    # stage 2 is only adopted when explicitly requested: its competitor features are
    # computed within a validation subset, so they are less complete than at test time
    use2 = a.stage2 and f2 > f1
    if use2:
        m2.save_model(f"{a.model_dir}/lgb_stage2.txt")
    json.dump({"thr": thr2 if use2 else thr1, "rel": 0.0, "stage2": bool(use2),
               "val_f05_stage1": f1, "val_f05_stage2": f2, "feat_cols": feat_cols, "feat_cols2": cols2,
               "grid_stage1": grid1, "grid_stage2": grid2}, open(f"{a.model_dir}/decision.json", "w"), indent=1)
    imp = pd.Series(m1.feature_importance("gain"), index=feat_cols).sort_values(ascending=False)
    print(imp.head(30).to_string())
    _, detail = scoring.macro_f05(va_b, vB, scoring.decide(vB, qB if use2 else pB, thr2 if use2 else thr1), truth_counts)
    detail.to_parquet(f"{a.model_dir}/val_detail.parquet")


if __name__ == "__main__":
    main()
