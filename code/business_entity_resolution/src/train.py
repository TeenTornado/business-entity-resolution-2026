"""Train the pairwise matcher on the training split and tune the decision rule
for macro F0.5 on a held-out set of S1 entities."""
import argparse
import json
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


def split_ids(n, seed=0, n_train=300000, n_val=150000):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    return np.sort(perm[:n_train]), np.sort(perm[n_train:n_train + n_val])


PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, num_threads=4)


def tune(cand_v, p, s1_val, truth_counts):
    best = (-1, None)
    grid = []
    for thr in np.arange(0.2, 0.96, 0.025):
        for rel in (0.0, 0.3, 0.5):
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
    a = ap.parse_args()
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:7.0f}s] {m}", flush=True)  # noqa: E731

    P1, P23 = load(a.work, "train")
    gt = pd.read_parquet(f"{a.work}/train_gt.parquet")
    cand = pd.read_parquet(f"{a.work}/train_cand.parquet")
    cand["src3"] = P23.entity_id.str.startswith("S3").values[cand.i2.values].astype(np.int8)
    cand = scoring.context_features(cand)
    cand, truth_counts = add_labels(cand, P1, P23, gt)
    log(f"candidates={len(cand)} positives={cand.label.sum()} total_true={truth_counts.sum()} "
        f"pair_recall={cand.label.sum() / truth_counts.sum():.4f}")

    tr_ids, va_ids = split_ids(len(P1), n_train=a.n_train, n_val=a.n_val)
    tables = features.build_df_tables([P1, P23])
    log("df tables built")
    parts = {}
    for name, ids in (("train", tr_ids), ("val", va_ids)):
        c = cand[cand.i1.isin(ids)].reset_index(drop=True)
        F = features.compute(c, P1, P23, tables)
        parts[name] = pd.concat([c, F], axis=1)
        log(f"features {name}: {len(c)} pairs")
    feat_cols = features.FEATURE_NAMES + scoring.CONTEXT_COLS
    tr, va = parts["train"], parts["val"]
    tr.to_parquet(f"{a.work}/feat_train.parquet")
    va.to_parquet(f"{a.work}/feat_val.parquet")

    dtr = lgb.Dataset(tr[feat_cols], tr.label)
    dva = lgb.Dataset(va[feat_cols], va.label, reference=dtr)
    model = lgb.train(PARAMS, dtr, num_boost_round=2000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    p = model.predict(va[feat_cols], num_iteration=model.best_iteration)
    log(f"trained, best_iter={model.best_iteration}")

    (f, (thr, rel)), grid = tune(va, p, va_ids, truth_counts)
    log(f"VALIDATION macro F0.5 = {f:.5f} at thr={thr} rel={rel}")
    base_keep = scoring.decide(va, p, 0.5, one_owner=False)
    log(f"(plain p>=0.5 without one-owner rule: {scoring.macro_f05(va_ids, va, base_keep, truth_counts)[0]:.5f})")

    import os
    os.makedirs(a.model_dir, exist_ok=True)
    model.save_model(f"{a.model_dir}/lgb_stage1.txt", num_iteration=model.best_iteration)
    json.dump({"thr": thr, "rel": rel, "val_f05": f, "feat_cols": feat_cols,
               "grid": grid}, open(f"{a.model_dir}/decision.json", "w"), indent=1)
    imp = pd.Series(model.feature_importance("gain"), index=feat_cols).sort_values(ascending=False)
    print(imp.head(30).to_string())


if __name__ == "__main__":
    main()
