"""Stage 2: collective (group-aware) re-scoring of the stage-1 candidate pairs.

Stage 1 judges every (S1, record) pair on its own. Its dominant loss on validation
is recall on true copies that carry no address: 4.4% of all true copies have an
empty address (97.7% of empty-address records are true copies), yet stage 1 accepts
only ~56% of them, because a bare name matches several S1 entities (about 10
candidate S1s per empty-address record). Those misses are ~55% of all lost F0.5.

But an S1 has ~3.5 copies and the others usually match confidently. Copies of one
entity inside one source share that source's rendering of it (casing, spelling
noise, address format), so a hard copy resembles the entity's other copies far more
than it resembles the S1 reference. Stage 2 compares each candidate record with the
S1's confident copies ("anchors") and with the anchors of the record's best
competing S1, then re-scores the pair with those contrastive features.

Stage-1 scores for the training rows are out-of-fold (2 folds over training S1s), so
stage 2 learns from honest scores. Validation half A tunes; half B is reported.

  python3 collective.py train   --work $W --model-dir $W/m1 --out-dir $W/c1
  python3 collective.py predict --work $W --model-dir $W/m1 --out-dir $W/c1 \
      --test-feat $W/feat_r1 --out $W/outc1 --candidates $W/out1/candidate_pairs.tsv
"""
import argparse
import glob
import json
import os
import pickle
import shutil
import time
from multiprocessing import Pool

import lightgbm as lgb
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

import scoring
from io_utils import read_p1, read_p23
from train import PARAMS, tune

REC_COLS = ["entity_id", "n_all", "n_core", "a_toks", "a_nums", "raw_name", "raw_addr"]
AGG = ["n", "n_src", "nm", "core", "core_eq", "raw", "raw_src", "ad", "num"]
COLL_COLS = ([f"ca_{x}" for x in AGG] + [f"cc_{x}" for x in AGG]
             + ["c_p1", "c_rank", "c_comp_p", "c_margin", "c_nr", "c_emp", "c_is_anchor",
                "c_d_nm", "c_d_raw", "c_d_core", "c_d_ad"])
_G = {}


def log(m, t0=time.time()):
    print(f"[{time.time() - t0:7.0f}s] {m}", flush=True)


class Packed:
    """Strings in one bytes buffer + offsets: forked workers read them without the
    copy-on-write blow-up that millions of Python str objects cause."""

    def __init__(self, strings):
        enc = [x.encode("utf-8") for x in strings]
        self.off = np.zeros(len(enc) + 1, np.int64)
        np.cumsum([len(x) for x in enc], out=self.off[1:])
        self.buf = b"".join(enc)

    def __getitem__(self, i):
        return self.buf[self.off[i]:self.off[i + 1]].decode("utf-8")


def records(work, split, keep):
    """Text fields of the S2/S3 records `keep` (sorted pool indices), compactly indexed."""
    R = {}
    for key, col in (("n_all", "n_all"), ("n_core", "n_core"), ("a_toks", "a_toks"), ("nums", "a_nums"),
                     ("raw", "raw_name")):
        v = read_p23(work, split, [col])[col].values[keep]
        R[key] = Packed(x.strip() if key == "raw" else x for x in v)
    R["src"] = read_p23(work, split, ["entity_id"]).entity_id.str.startswith("S3").values[keep].astype(np.int8)
    R["emp"] = (read_p23(work, split, ["raw_addr"]).raw_addr.str.strip() == "").values[keep]
    return R


def _agg(r, anchors):
    """Aggregate similarity of record r to a set of anchor records (NaN if none)."""
    out = np.full(len(AGG), np.nan, np.float32)
    anchors = [a for a in anchors if a != r]
    out[0] = len(anchors)
    if not anchors:
        return out
    G = _G
    src, nm, core, raw, ad, nums = G["src"][r], G["n_all"][r], G["n_core"][r], G["raw"][r], G["a_toks"][r], G["nums"][r]
    ns = set(nums.split())
    n_src = 0
    b_nm = b_core = b_raw = b_raw_src = -1.0
    b_ad = b_num = -1.0
    eq = 0
    for a in anchors:
        same = G["src"][a] == src
        n_src += same
        b_nm = max(b_nm, fuzz.token_set_ratio(nm, G["n_all"][a]))
        c = fuzz.ratio(core, G["n_core"][a])
        b_core = max(b_core, c)
        eq |= core != "" and core == G["n_core"][a]
        rw = fuzz.ratio(raw, G["raw"][a])  # case-sensitive: keeps the source's styling
        b_raw = max(b_raw, rw)
        if same:
            b_raw_src = max(b_raw_src, rw)
        if ad and G["a_toks"][a]:
            b_ad = max(b_ad, fuzz.token_set_ratio(ad, G["a_toks"][a]))
        na = G["nums"][a]
        if ns and na:
            sa = set(na.split())
            b_num = max(b_num, len(ns & sa) / len(ns | sa))
    out[1:] = [n_src, b_nm, b_core, eq, b_raw, b_raw_src, b_ad, b_num]
    out[out < 0] = np.nan
    return out


def _work(bounds):
    lo, hi = bounds
    G = _G
    ptr, anc = G["ptr"], G["anc"]
    res = np.empty((hi - lo, 2 * len(AGG)), np.float32)
    for k in range(lo, hi):
        s, r, c = G["s"][k], G["r"][k], G["c"][k]
        res[k - lo, :len(AGG)] = _agg(r, anc[ptr[s]:ptr[s + 1]])
        res[k - lo, len(AGG):] = _agg(r, anc[ptr[c]:ptr[c + 1]]) if c >= 0 else np.nan
    return res


def collective_features(i1, i2, p1, work, split, anchor_thr=0.5, workers=4, chunk=50000):
    """Contrastive group features for every pair (i1 -> i2) given stage-1 scores p1."""
    i1 = np.asarray(i1, np.int64)
    uniq, i2 = np.unique(np.asarray(i2, np.int64), return_inverse=True)
    R = records(work, split, uniq)
    p1 = np.asarray(p1, np.float32)
    n = len(i1)
    # best and second-best S1 for every record
    o = np.lexsort((-p1, i2))
    i2s = i2[o]
    first = np.r_[True, i2s[1:] != i2s[:-1]]
    gid = np.cumsum(first) - 1
    starts = np.flatnonzero(first)
    sizes = np.diff(np.r_[starts, n])
    top1_i1, top1_p = i1[o][starts], p1[o][starts]
    has2 = sizes > 1
    top2_i1 = np.full(len(starts), -1, np.int64)
    top2_p = np.zeros(len(starts), np.float32)
    top2_i1[has2] = i1[o][starts[has2] + 1]
    top2_p[has2] = p1[o][starts[has2] + 1]
    g = np.empty(n, np.int64)
    g[o] = gid
    is_top = top1_i1[g] == i1
    comp = np.where(is_top, top2_i1[g], top1_i1[g])
    comp_p = np.where(is_top, top2_p[g], top1_p[g])
    is_anchor = is_top & (p1 >= anchor_thr)
    # anchors grouped by S1 (CSR)
    ai = np.flatnonzero(is_anchor)
    ai = ai[np.argsort(i1[ai], kind="stable")]
    n1 = int(max(i1.max(), comp.max())) + 2
    ptr = np.searchsorted(i1[ai], np.arange(n1 + 1))
    _G.update(R)
    _G.update(ptr=ptr, anc=i2[ai], s=i1, r=i2, c=comp)
    bounds = [(lo, min(lo + chunk, n)) for lo in range(0, n, chunk)]
    with Pool(workers) as pool:
        parts = pool.map(_work, bounds, chunksize=1)
    _G.clear()
    F = pd.DataFrame(np.vstack(parts), columns=COLL_COLS[:2 * len(AGG)])
    F["c_p1"] = p1
    F["c_rank"] = pd.Series(-p1).groupby(i1).rank(method="first").values.astype(np.float32)
    F["c_comp_p"] = comp_p
    F["c_margin"] = p1 - comp_p
    F["c_nr"] = sizes[g].astype(np.float32)
    F["c_emp"] = R["emp"][i2].astype(np.float32)
    F["c_is_anchor"] = is_anchor.astype(np.float32)
    for x in ("nm", "raw", "core", "ad"):
        F[f"c_d_{x}"] = F[f"ca_{x}"].fillna(-1) - F[f"cc_{x}"].fillna(-1)
    return F[COLL_COLS]


def cmd_train(a):
    cfg = json.load(open(f"{a.model_dir}/decision.json"))
    cols = cfg["feat_cols"]
    cache = f"{a.work}/train_pairs_n{a.n_train}_p{a.prune_thr}.parquet"
    truth_counts, tr_ids, va_a, va_b = pickle.load(open(cache + ".meta", "rb"))[:4]
    os.makedirs(a.out_dir, exist_ok=True)
    cand = pd.read_parquet(cache, columns=["i1", "i2", "label"] + cols)
    if a.max_s1:  # quick/low-memory run on a subset of S1 entities
        tr_ids, va_a, va_b = tr_ids[:a.max_s1], va_a[:a.max_s1 // 4], va_b[:a.max_s1 // 4]
        cand = cand[cand.i1.isin(np.concatenate([tr_ids, va_a, va_b]))].reset_index(drop=True)
    # The cached pairs cover only the train+val S1 subset. A record whose true S1 lies
    # outside it has lost its rightful competitor, which never happens on test (every
    # S1 is present). Drop those pairs so competitor/anchor features mean the same thing.
    gt = pd.read_parquet(f"{a.work}/train_gt.parquet")
    pairs = gt.assign(m=gt.matched_entity_ids.str.split(",")).explode("m")
    pairs = pairs[pairs.m.notna() & (pairs.m != "")]
    i1_all = pd.Index(read_p1(a.work, "train", ["entity_id"]).entity_id)
    owner = pd.Series(i1_all.get_indexer(pairs.source1_entity_id.values), index=pairs.m.values)
    owner = owner.reindex(read_p23(a.work, "train", ["entity_id"]).entity_id.values).fillna(-1).astype(np.int64).values
    in_u = np.zeros(len(i1_all), bool)
    in_u[np.concatenate([tr_ids, va_a, va_b])] = True
    o = owner[cand.i2.values]
    orphan = (o >= 0) & ~in_u[np.maximum(o, 0)]
    log(f"dropping {int(orphan.sum())} pairs whose record belongs to an S1 outside the subset")
    cand = cand[~orphan].reset_index(drop=True)
    del gt, pairs, owner, o, orphan
    for c in cols:
        if cand[c].dtype == np.float64:
            cand[c] = cand[c].astype(np.float32)
    log(f"pairs {len(cand)}")
    # ---- out-of-fold stage-1 scores (2 folds over training S1s)
    fold = np.random.RandomState(7).rand(int(cand.i1.max()) + 1) < 0.5
    is_tr = cand.i1.isin(tr_ids).values
    is_a = cand.i1.isin(va_a).values
    is_b = cand.i1.isin(va_b).values
    f_of = fold[cand.i1.values]
    p1 = np.zeros(len(cand), np.float32)
    dva = lgb.Dataset(cand.loc[is_a, cols], cand.label.values[is_a], free_raw_data=False)
    for k in (0, 1):
        path = f"{a.out_dir}/stage1_fold{k}.txt"
        fit = is_tr & (f_of == bool(k))
        if os.path.exists(path):
            m = lgb.Booster(model_file=path)
        else:
            m = lgb.train(PARAMS, lgb.Dataset(cand.loc[fit, cols], cand.label.values[fit]),
                          num_boost_round=a.rounds, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)])
            m.save_model(path, num_iteration=m.best_iteration)
        oof = is_tr & (f_of != bool(k))
        p1[oof] = m.predict(cand.loc[oof, cols], num_threads=a.workers)
        val = is_a | is_b
        p1[val] += 0.5 * m.predict(cand.loc[val, cols], num_threads=a.workers)
        log(f"stage-1 fold {k} done")
    del dva
    F = collective_features(cand.i1.values, cand.i2.values, p1, a.work, "train", a.anchor_thr, a.workers)
    log("collective features done")
    cand = pd.concat([cand, F], axis=1)
    del F
    s2cols = cols + COLL_COLS
    vA = cand[is_a].reset_index(drop=True)
    vB = cand[is_b].reset_index(drop=True)
    # stage-1 baseline on the same folds (fold-mean scores), for an apples-to-apples comparison
    (fA1, (thr1, _)), _ = tune(vA, vA.c_p1.values, va_a, truth_counts)
    fB1, _ = scoring.macro_f05(va_b, vB, scoring.decide(vB, vB.c_p1.values, thr1), truth_counts)
    log(f"VALIDATION stage-1 (fold mean) thr {thr1:.4f} -> half B macro F0.5 = {fB1:.5f}")
    params = dict(PARAMS, num_threads=a.workers, learning_rate=0.05)
    m2 = lgb.train(params, lgb.Dataset(cand.loc[is_tr, s2cols], cand.label.values[is_tr]),
                   num_boost_round=a.rounds, valid_sets=[lgb.Dataset(vA[s2cols], vA.label)],
                   callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)])
    pA = m2.predict(vA[s2cols], num_iteration=m2.best_iteration)
    pB = m2.predict(vB[s2cols], num_iteration=m2.best_iteration)
    (fA, (thr, _)), grid = tune(vA, pA, va_a, truth_counts)
    kB = scoring.decide(vB, pB, thr)
    fB, detail = scoring.macro_f05(va_b, vB, kB, truth_counts)
    log(f"VALIDATION stage-2 collective thr {thr:.4f} (half A F={fA:.5f}) -> half B macro F0.5 = {fB:.5f}")
    emp = vB.c_emp.values == 1
    y = vB.label.values == 1
    k1 = scoring.decide(vB, vB.c_p1.values, thr1)
    for nm, msk in (("empty-addr", emp), ("has-addr", ~emp)):
        log(f"  {nm}: recall stage1 {k1[y & msk].mean():.4f} -> stage2 {kB[y & msk].mean():.4f} | "
            f"false pairs stage1 {int((k1 & ~y & msk).sum())} -> stage2 {int((kB & ~y & msk).sum())}")
    m2.save_model(f"{a.out_dir}/stage2.txt", num_iteration=m2.best_iteration)
    json.dump({"thr": thr, "anchor_thr": a.anchor_thr, "feat_cols": cols, "s2_cols": s2cols, "val_f05": fB,
               "val_f05_stage1": fB1, "grid_half_a": grid}, open(f"{a.out_dir}/decision.json", "w"), indent=1)
    detail.to_parquet(f"{a.out_dir}/val_detail.parquet")
    imp = pd.Series(m2.feature_importance("gain"), index=s2cols).sort_values(ascending=False)
    print(imp.head(25).to_string())


def cmd_predict(a):
    cfg = json.load(open(f"{a.out_dir}/decision.json"))
    cols, s2cols = cfg["feat_cols"], cfg["s2_cols"]
    folds = [lgb.Booster(model_file=f"{a.out_dir}/stage1_fold{k}.txt") for k in (0, 1)]
    blocks = sorted(glob.glob(f"{a.test_feat}/block*.parquet"))
    parts = []
    for b in blocks:
        x = pd.read_parquet(b, columns=["i1", "i2"] + cols)
        p = np.mean([m.predict(x[cols], num_threads=a.workers) for m in folds], axis=0).astype(np.float32)
        parts.append(pd.DataFrame({"i1": x.i1.values, "i2": x.i2.values, "p1": p}))
    base = pd.concat(parts, ignore_index=True)
    log(f"test pairs {len(base)}")
    F = collective_features(base.i1.values, base.i2.values, base.p1.values, a.work, a.split, cfg["anchor_thr"],
                            a.workers)
    m2 = lgb.Booster(model_file=f"{a.out_dir}/stage2.txt")
    p2 = np.zeros(len(base), np.float32)
    off = 0
    for b in blocks:
        x = pd.read_parquet(b, columns=cols)
        X = pd.concat([x.reset_index(drop=True), F.iloc[off:off + len(x)].reset_index(drop=True)], axis=1)
        p2[off:off + len(x)] = m2.predict(X[s2cols], num_threads=a.workers)
        off += len(x)
    np.save(f"{a.out_dir}/test_scores_stage2.npy", p2)
    keep = scoring.decide(base, p2, cfg["thr"], one_owner=True)
    k1 = scoring.decide(base, base.p1.values, cfg["thr"], one_owner=True)
    from predict import write_lists
    os.makedirs(a.out, exist_ok=True)
    P1 = read_p1(a.work, a.split, ["entity_id", "country"])
    ids23 = read_p23(a.work, a.split, ["entity_id"]).entity_id.values
    sel = base[keep]
    write_lists(f"{a.out}/matching_results.tsv", "matched_entity_ids", P1.entity_id.values,
                sel.i1.values, ids23[sel.i2.values])
    if a.candidates:
        shutil.copy(a.candidates, f"{a.out}/candidate_pairs.tsv")
    ctry = P1.country.values[base.i1.values]
    st = pd.DataFrame({"country": ctry, "stage1": k1, "stage2": keep, "emp": (F.c_emp.values == 1) & keep})
    print(st.groupby("country")[["stage1", "stage2", "emp"]].sum().to_string())
    log(f"matches written: pairs={int(keep.sum())} (stage-1 rule would give {int(k1.sum())}), "
        f"mean list={keep.sum() / len(P1):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "predict"])
    ap.add_argument("--work", required=True)
    ap.add_argument("--model-dir", help="stage-1 model dir (train: for feat_cols)")
    ap.add_argument("--out-dir", required=True, help="stage-2 model dir")
    ap.add_argument("--n-train", type=int, default=800000, help="which cached training pairs to use")
    ap.add_argument("--prune-thr", type=float, default=0.004)
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--anchor-thr", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-s1", type=int, default=0, help="train on a subset of training S1s (0 = all)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--test-feat", help="test pair features saved by predict.py (block*.parquet)")
    ap.add_argument("--out", help="output dir for matching_results.tsv")
    ap.add_argument("--candidates", help="candidate_pairs.tsv from the same stage-1 run (copied to --out)")
    a = ap.parse_args()
    cmd_train(a) if a.cmd == "train" else cmd_predict(a)


if __name__ == "__main__":
    import multiprocessing as _mp
    _mp.set_start_method("fork", force=True)  # macOS defaults to spawn; workers rely on fork-inherited globals
    main()
