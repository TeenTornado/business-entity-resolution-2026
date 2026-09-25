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
from train import load


def write_lists(path, header, P1_ids, i1, ids):
    """Write one row per S1 entity with a comma-separated list (possibly empty)."""
    df = pd.DataFrame({"i1": i1, "id": ids})
    lists = df.groupby("i1").id.agg(lambda x: ",".join(dict.fromkeys(x)))
    col = lists.reindex(np.arange(len(P1_ids))).fillna("").values
    out = pd.DataFrame({"source1_entity_id": P1_ids, header: col})
    out.to_csv(path, sep="\t", index=False, lineterminator="\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--cap", type=int, default=3000)
    ap.add_argument("--k", type=int, default=30)
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
    log(f"candidates: {len(cand)}")
    write_lists(f"{a.out}/candidate_pairs.tsv", "candidate_entity_ids", P1.entity_id.values,
                cand.i1.values, P23.entity_id.values[cand.i2.values])

    cand["src3"] = P23.entity_id.str.startswith("S3").values[cand.i2.values].astype(np.int8)
    cand = scoring.context_features(cand)
    tables = features.build_df_tables([P1, P23])
    F = features.compute(cand, P1, P23, tables)
    cand = pd.concat([cand, F], axis=1)
    log("features done")

    m1 = lgb.Booster(model_file=f"{a.model_dir}/lgb_stage1.txt")
    p = m1.predict(cand[cfg["feat_cols"]], num_threads=4)
    if cfg.get("stage2"):
        cand = scoring.prob_context(cand, p)
        m2 = lgb.Booster(model_file=f"{a.model_dir}/lgb_stage2.txt")
        p = m2.predict(cand[cfg["feat_cols2"]], num_threads=4)
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
