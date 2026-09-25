"""Train the pairwise matcher on the training split and tune the decision rule
for macro F0.5 on a held-out set of S1 entities."""
import argparse
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

import blocking
import features
import scoring
from io_utils import read_p1, read_p23

FIELD_COLS = ["entity_id", "country"] + features.FIELDS


def load(work, split):
    return read_p1(work, split, FIELD_COLS), read_p23(work, split, FIELD_COLS)


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


PRUNER_PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=200,
                     feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=4)


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
    ap.add_argument("--prune-thr", type=float, default=0.01, help="pruner probability cut for the candidate set")
    a = ap.parse_args()
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:7.0f}s] {m}", flush=True)  # noqa: E731

    ids = read_p1(a.work, "train", ["entity_id"]).entity_id
    ids23 = read_p23(a.work, "train", ["entity_id"]).entity_id
    gt = pd.read_parquet(f"{a.work}/train_gt.parquet")
    cand = prepare_candidates(pd.read_parquet(f"{a.work}/train_cand.parquet"), ids23)
    cand, truth_counts = add_labels(cand, ids.to_frame(), ids23.to_frame(), gt)
    log(f"candidates={len(cand)} positives={cand.label.sum()} total_true={truth_counts.sum()} "
        f"pair_recall={cand.label.sum() / truth_counts.sum():.4f}")
    tr_ids, va_ids = split_ids(len(ids), n_train=a.n_train, n_val=a.n_val)
    # val half A: early stopping + threshold tuning; val half B: untouched, reported score
    rng = np.random.RandomState(1)
    half = rng.rand(len(va_ids)) < 0.5
    va_a, va_b = va_ids[half], va_ids[~half]
    cand = cand[cand.i1.isin(np.concatenate([tr_ids, va_ids]))].reset_index(drop=True)
    del ids, ids23
    P1, P23 = load(a.work, "train")
    cand = scoring.freq_features(cand, P1, P23)
    cand = scoring.vocab_features(cand, P1, P23)
    sib_path = f"{a.work}/siblings.json"
    offsets = json.load(open(sib_path))["offsets"] if os.path.exists(sib_path) else []
    cand = scoring.sibling_features(cand, P1, P23, offsets)
    nc, ac = blocking.split_cosines(P1, P23, cand)
    cand = scoring.pruner_features(cand, P1, P23, nc, ac)
    del nc, ac
    log("context / frequency / pruner features done")

    # ---- stage A: candidate pruner (cheap features only) -> the candidate set
    pcols = scoring.PRUNER_COLS + scoring.VOCAB_COLS + scoring.SIB_COLS
    is_tr, is_a, is_b = (cand.i1.isin(x).values for x in (tr_ids, va_a, va_b))
    pr = lgb.train(PRUNER_PARAMS, lgb.Dataset(cand.loc[is_tr, pcols], cand.label[is_tr]), num_boost_round=800,
                   valid_sets=[lgb.Dataset(cand.loc[is_a, pcols], cand.label[is_a])],
                   callbacks=[lgb.early_stopping(30), lgb.log_evaluation(200)])
    q = pr.predict(cand[pcols], num_iteration=pr.best_iteration)
    kept = q >= a.prune_thr
    tot_b = truth_counts.reindex(va_b).fillna(0).sum()
    log(f"pruner: keep q>={a.prune_thr}: candidates/S1 (half B) {kept[is_b].sum() / len(va_b):.2f} "
        f"(from {is_b.sum() / len(va_b):.2f}), pair recall {cand.label[is_b & kept].sum() / tot_b:.4f} "
        f"(from {cand.label[is_b].sum() / tot_b:.4f})")
    cand = cand[kept].reset_index(drop=True)

    # ---- stage B: pairwise matcher on the pruned candidate set
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
    feat_cols = features.FEATURE_NAMES + scoring.CONTEXT_COLS + scoring.FREQ_COLS + scoring.VOCAB_COLS + \
        scoring.SIB_COLS + ["pr_name_cos", "pr_addr_cos"]
    tr = cand[cand.i1.isin(tr_ids)]
    vA = cand[cand.i1.isin(va_a)].reset_index(drop=True)
    vB = cand[cand.i1.isin(va_b)].reset_index(drop=True)
    del cand

    dtr = lgb.Dataset(tr[feat_cols], tr.label)
    dva = lgb.Dataset(vA[feat_cols], vA.label, reference=dtr)
    m1 = lgb.train(PARAMS, dtr, num_boost_round=a.rounds, valid_sets=[dva],
                   callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)])
    del dtr, dva, tr
    log(f"matcher trained, best_iter={m1.best_iteration}")
    pA = m1.predict(vA[feat_cols], num_iteration=m1.best_iteration)
    pB = m1.predict(vB[feat_cols], num_iteration=m1.best_iteration)
    (fA, (thr, _)), grid = tune(vA, pA, va_a, truth_counts)
    fB, detail = scoring.macro_f05(va_b, vB, scoring.decide(vB, pB, thr), truth_counts)
    (fB_oracle, (thr_o, _)), _ = tune(vB, pB, va_b, truth_counts)
    log(f"VALIDATION (half B, threshold {thr:.3f} tuned on half A) macro F0.5 = {fB:.5f} "
        f"(oracle threshold {thr_o:.3f}: {fB_oracle:.5f})")

    os.makedirs(a.model_dir, exist_ok=True)
    pr.save_model(f"{a.model_dir}/lgb_pruner.txt", num_iteration=pr.best_iteration)
    m1.save_model(f"{a.model_dir}/lgb_stage1.txt", num_iteration=m1.best_iteration)
    json.dump({"thr": thr, "rel": 0.0, "prune_thr": a.prune_thr, "offsets": offsets, "val_f05": fB, "feat_cols": feat_cols,
               "pruner_cols": pcols, "grid_half_a": grid}, open(f"{a.model_dir}/decision.json", "w"), indent=1)
    imp = pd.Series(m1.feature_importance("gain"), index=feat_cols).sort_values(ascending=False)
    print(imp.head(30).to_string())
    detail.to_parquet(f"{a.model_dir}/val_detail.parquet")

if __name__ == "__main__":
    main()
