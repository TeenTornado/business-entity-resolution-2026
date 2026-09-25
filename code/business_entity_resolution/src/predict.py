"""Run blocking + matching on the test split and write the two submission files."""
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
from train import load, prepare_candidates


def write_lists(path, header, P1_ids, i1, ids):
    """Write one row per S1 entity with a comma-separated list (possibly empty).
    Streams rows so 50M+ candidate ids never become one big pandas object."""
    order = np.argsort(i1, kind="stable")
    i1s, idss = np.asarray(i1)[order], np.asarray(ids, dtype=object)[order]
    bounds = np.searchsorted(i1s, np.arange(len(P1_ids) + 1))
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{header}\n")
        for r, s1 in enumerate(P1_ids):
            lo, hi = bounds[r], bounds[r + 1]
            f.write(s1 + "\t" + ",".join(dict.fromkeys(idss[lo:hi])) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--cap", type=int, default=3000)
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--block", type=int, default=150000, help="S1 rows scored per block")
    a = ap.parse_args()
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:7.0f}s] {m}", flush=True)  # noqa: E731
    os.makedirs(a.out, exist_ok=True)
    cfg = json.load(open(f"{a.model_dir}/decision.json"))

    P1, P23 = load(a.work, a.split)
    cand_path = f"{a.work}/{a.split}_cand.parquet"
    if os.path.exists(cand_path):
        cand = pd.read_parquet(cand_path)
    else:
        cand = blocking.run(P1, P23, cap=a.cap, k=a.k, log=log)
        cand.to_parquet(cand_path)
    log(f"retrieval candidates (blocking top-{a.k}): {len(cand)}")
    cand = prepare_candidates(cand, P23.entity_id)
    cand = scoring.freq_features(cand, P1, P23)
    cand = scoring.vocab_features(cand, P1, P23)
    if "prune_thr" in cfg:
        # candidate pruner: cheap blocking-stage features only; its survivors ARE the candidate set
        nc, ac = blocking.split_cosines(P1, P23, cand)
        cand = scoring.pruner_features(cand, P1, P23, nc, ac)
        del nc, ac
        pr = lgb.Booster(model_file=f"{a.model_dir}/lgb_pruner.txt")
        q = pr.predict(cand[cfg["pruner_cols"]], num_threads=4)
        cand = cand[q >= cfg["prune_thr"]].reset_index(drop=True)
        log(f"pruned candidate set: {len(cand)} pairs ({len(cand) / len(P1):.2f} per S1)")
    write_lists(f"{a.out}/candidate_pairs.tsv", "candidate_entity_ids", P1.entity_id.values,
                cand.i1.values, P23.entity_id.values[cand.i2.values])
    tables = features.build_df_tables([P1, P23])
    log("df tables built")
    m1 = lgb.Booster(model_file=f"{a.model_dir}/lgb_stage1.txt")
    use2 = bool(cfg.get("stage2"))
    spill = f"{a.work}/{a.split}_featblocks"
    os.makedirs(spill, exist_ok=True)
    p = np.zeros(len(cand), np.float32)
    # pass 1: pairwise features + stage-1 score, in blocks of S1 rows to bound memory
    bounds = np.searchsorted(cand.i1.values, np.arange(0, len(P1) + a.block, a.block))
    blocks = [(lo, hi) for lo, hi in zip(bounds[:-1], bounds[1:]) if hi > lo]
    for b, (lo, hi) in enumerate(blocks):
        c = cand.iloc[lo:hi]
        F = features.compute(c, P1, P23, tables)
        X = pd.concat([c, F], axis=1)[cfg["feat_cols"]]
        p[lo:hi] = m1.predict(X, num_threads=4)
        if use2:
            X.to_parquet(f"{spill}/block{b}.parquet")
        log(f"stage-1 scored pairs {hi}/{len(cand)}")
    del tables
    if use2:
        # pass 2: competing-candidate probabilities need the whole candidate graph
        ctx = scoring.prob_context(cand[["i1", "i2"]].copy(), p)[scoring.PROB_COLS]
        m2 = lgb.Booster(model_file=f"{a.model_dir}/lgb_stage2.txt")
        p2 = np.zeros(len(cand), np.float32)
        for b, (lo, hi) in enumerate(blocks):
            X = pd.read_parquet(f"{spill}/block{b}.parquet")
            X = pd.concat([X, ctx.iloc[lo:hi]], axis=1)[cfg["feat_cols2"]]
            p2[lo:hi] = m2.predict(X, num_threads=4)
            os.remove(f"{spill}/block{b}.parquet")
            log(f"stage-2 scored pairs {hi}/{len(cand)}")
        p = p2
    np.save(f"{a.work}/{a.split}_scores.npy", p)
    keep = scoring.decide(cand, p, cfg["thr"], one_owner=True, rel=cfg["rel"])
    sel = cand[keep]
    write_lists(f"{a.out}/matching_results.tsv", "matched_entity_ids", P1.entity_id.values,
                sel.i1.values, P23.entity_id.values[sel.i2.values])
    n_match = sel.groupby("i1").size()
    log(f"matches written: pairs={len(sel)} S1 with >=1 match={len(n_match)}/{len(P1)} "
        f"mean list={len(sel) / len(P1):.3f}")
    stats = cand.assign(keep=keep).groupby(P1.country.values[cand.i1.values]).keep.agg(["size", "sum"])
    print(stats.to_string())


if __name__ == "__main__":
    main()
