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


def add_candidate_features(cand, P1, P23, offsets, pruner=None, prune_thr=None, pruner_cols=None,
                           chunk=250000, log=print, spill=None):
    """Cheap candidate-level features (shared by train/predict).
    Graph features that need all S1s competing for a record (name-rank) are computed
    once on the full retrieval graph; everything else runs in S1 chunks. If a pruner
    is given, only its survivors are kept, which bounds memory on the test set."""
    nc, ac = blocking.split_cosines(P1, P23, cand)
    cand["pr_name_cos"] = nc
    cand["pr_addr_cos"] = ac
    del nc, ac
    cand = scoring.pruner_features(cand, P1, P23, None, None)
    cand = scoring.name_graph_features(cand)
    bounds = np.searchsorted(cand.i1.values, np.arange(0, len(P1) + chunk, chunk))
    parts = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if hi <= lo:
            continue
        c = cand.iloc[lo:hi].copy()
        c = scoring.freq_features(c, P1, P23)
        c = scoring.vocab_features(c, P1, P23)
        c = scoring.sibling_features(c, P1, P23, offsets)
        if pruner is not None:
            q = pruner.predict(c[pruner_cols], num_threads=4)
            c = c[q >= prune_thr]
        c = _downcast(c)
        if spill is not None:
            c.to_parquet(f"{spill}/part{len(parts):04d}.parquet")
            parts.append(None)
        else:
            parts.append(c)
        log(f"  candidate features: {hi}/{len(cand)} rows")
    if spill is not None:
        return None  # caller frees the retrieval graph, then reads the spilled chunks
    return pd.concat(parts, ignore_index=True)


def _downcast(df):
    for col in df.columns:
        if df[col].dtype == np.float64:
            df[col] = df[col].astype(np.float32)
        elif df[col].dtype == np.int64 and col not in ("i1", "i2"):
            df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def read_spill(spill):
    import glob
    import shutil
    out = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{spill}/part*.parquet"))], ignore_index=True)
    shutil.rmtree(spill)
    return out


PRUNER_FEATURES = scoring.PRUNER_COLS + scoring.VOCAB_COLS + scoring.SIB_COLS + scoring.NAME_GRAPH_COLS
MATCHER_FEATURES = (features.FEATURE_NAMES + scoring.CONTEXT_COLS + scoring.FREQ_COLS + scoring.VOCAB_COLS
                    + scoring.SIB_COLS + scoring.NAME_GRAPH_COLS
                    + ["pr_name_cos", "pr_addr_cos", "pr_name_eq", "pr_addr_eq", "pr_num_eq"])


def tune(cand_v, p, s1_val, truth_counts, veto=False):
    best = (-1, None)
    grid = []
    for thr in np.arange(0.3, 0.951, 0.0125):
        for rel in (0.0,):
            keep = scoring.decide(cand_v, p, thr, one_owner=True, rel=rel)
            if veto:
                keep &= ~scoring.sibling_veto(cand_v, keep)
            f, _ = scoring.macro_f05(s1_val, cand_v, keep, truth_counts)
            grid.append((round(float(thr), 3), rel, f))
            if f > best[0]:
                best = (f, (float(thr), rel))
    return best, grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--n-train", type=int, default=400000)
    ap.add_argument("--n-val", type=int, default=150000)
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--ablate-pool", action="store_true", help="also train a matcher without pool-statistic features")
    ap.add_argument("--prune-thr", type=float, default=0.01, help="pruner probability cut for the candidate set")
    a = ap.parse_args()
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:7.0f}s] {m}", flush=True)  # noqa: E731

    ids = read_p1(a.work, "train", ["entity_id"]).entity_id
    ids23 = read_p23(a.work, "train", ["entity_id"]).entity_id
    gt = pd.read_parquet(f"{a.work}/train_gt.parquet")
    cand = prepare_candidates(pd.read_parquet(f"{a.work}/train_cand.parquet"), ids23)
    cand, truth_counts = add_labels(cand, ids.to_frame(), ids23.to_frame(), gt)
    log(f"candidates={len(cand)} ({len(cand) / len(ids):.1f}/S1) positives={cand.label.sum()} "
        f"total_true={truth_counts.sum()} pair_recall={cand.label.sum() / truth_counts.sum():.4f}")
    tr_ids, va_ids = split_ids(len(ids), n_train=a.n_train, n_val=a.n_val)
    # val half A: early stopping, threshold and rule selection; half B: untouched, reported score
    half = np.random.RandomState(1).rand(len(va_ids)) < 0.5
    va_a, va_b = va_ids[half], va_ids[~half]
    cand = cand[cand.i1.isin(np.concatenate([tr_ids, va_ids]))].reset_index(drop=True)
    del ids, ids23
    P1, P23 = load(a.work, "train")
    sib_path = f"{a.work}/siblings.json"
    offsets = json.load(open(sib_path))["offsets"] if os.path.exists(sib_path) else []
    spill = f"{a.work}/train_candfeat_spill"
    os.makedirs(spill, exist_ok=True)
    add_candidate_features(cand, P1, P23, offsets, log=log, spill=spill)
    del cand
    scoring._MEMO.clear()
    cand = read_spill(spill)
    log("candidate-level features done")

    # ---- stage A: candidate pruner (cheap features only) -> the candidate set
    pcols = PRUNER_FEATURES
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
    tables = features.build_df_tables([P1, P23])
    F = features.compute(cand, P1, P23, tables)
    del P1, P23, tables
    cand = pd.concat([cand, F.set_index(cand.index)], axis=1)
    del F
    log(f"pairwise features: {len(cand)} pairs")
    tr = cand[cand.i1.isin(tr_ids)]
    vA = cand[cand.i1.isin(va_a)].reset_index(drop=True)
    vB = cand[cand.i1.isin(va_b)].reset_index(drop=True)
    del cand

    results = {}
    variants = {"full": MATCHER_FEATURES}
    if a.ablate_pool:
        variants["no_pool_stats"] = [c for c in MATCHER_FEATURES if c not in scoring.POOL_COLS]
    for name, cols in variants.items():
        m = lgb.train(PARAMS, lgb.Dataset(tr[cols], tr.label), num_boost_round=a.rounds,
                      valid_sets=[lgb.Dataset(vA[cols], vA.label)],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)])
        pA = m.predict(vA[cols], num_iteration=m.best_iteration)
        pB = m.predict(vB[cols], num_iteration=m.best_iteration)
        for veto in (False, True):
            (fA, (thr, _)), grid = tune(vA, pA, va_a, truth_counts, veto=veto)
            kB = scoring.decide(vB, pB, thr)
            if veto:
                kB &= ~scoring.sibling_veto(vB, kB)
            fB, detail = scoring.macro_f05(va_b, vB, kB, truth_counts)
            results[(name, veto)] = dict(fA=fA, fB=fB, thr=thr, model=m, cols=cols, grid=grid, detail=detail)
            log(f"VALIDATION [{name}, sibling_veto={veto}] threshold {thr:.4f} (tuned on half A, F={fA:.5f}) "
                f"-> half B macro F0.5 = {fB:.5f}")
    best_key = max(results, key=lambda k: results[k]["fA"])
    best = results[best_key]
    log(f"selected on half A: {best_key} -> reported half B F0.5 = {best['fB']:.5f}")

    os.makedirs(a.model_dir, exist_ok=True)
    pr.save_model(f"{a.model_dir}/lgb_pruner.txt", num_iteration=pr.best_iteration)
    best["model"].save_model(f"{a.model_dir}/lgb_stage1.txt", num_iteration=best["model"].best_iteration)
    json.dump({"thr": best["thr"], "rel": 0.0, "prune_thr": a.prune_thr, "offsets": offsets,
               "sibling_veto": bool(best_key[1]), "variant": best_key[0], "val_f05": best["fB"],
               "feat_cols": best["cols"], "pruner_cols": pcols, "grid_half_a": best["grid"],
               "all_results": {f"{k[0]}|veto={k[1]}": {"fA": v["fA"], "fB": v["fB"], "thr": v["thr"]}
                               for k, v in results.items()}},
              open(f"{a.model_dir}/decision.json", "w"), indent=1)
    imp = pd.Series(best["model"].feature_importance("gain"), index=best["cols"]).sort_values(ascending=False)
    print(imp.head(30).to_string())
    best["detail"].to_parquet(f"{a.model_dir}/val_detail.parquet")


if __name__ == "__main__":
    main()
