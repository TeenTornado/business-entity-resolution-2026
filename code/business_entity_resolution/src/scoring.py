"""Candidate-level context features, decision rules and the F0.5 metric."""
import numpy as np
import pandas as pd


def context_features(cand):
    """Features derived from the whole candidate graph (cheap, vectorised).

    cand must have columns i1, i2, blk_score, blk_rank (sorted by i1, score desc).
    """
    g1 = cand.groupby("i1").blk_score
    top = g1.transform("max")
    cand["blk_rel"] = cand.blk_score / top.clip(lower=1e-6)
    cand["blk_n1"] = g1.transform("size").astype(np.int16)
    second = cand.blk_score.where(cand.blk_rank == 1).groupby(cand.i1).transform("max").fillna(0)
    cand["blk_top_gap"] = top - second
    # reverse direction: how does this S1 rank among all S1s that chose this S2/S3 record?
    g2 = cand.groupby("i2").blk_score
    cand["rev_n"] = g2.transform("size").astype(np.int16)
    cand["rev_rank"] = g2.rank(ascending=False, method="first").astype(np.int16) - 1
    best2 = g2.transform("max")
    # best competing score among OTHER S1 entities
    s_sorted = cand[["i2", "blk_score"]].sort_values(["i2", "blk_score"], ascending=[True, False])
    second2 = s_sorted.groupby("i2").blk_score.nth(1)
    sec_map = pd.Series(second2.values, index=s_sorted.loc[second2.index, "i2"].values)
    sec = cand.i2.map(sec_map).fillna(0).values
    cand["rev_best_other"] = np.where(cand.blk_score.values >= best2.values, sec, best2.values)
    cand["rev_margin"] = cand.blk_score - cand.rev_best_other
    return cand


CONTEXT_COLS = ["blk_score", "blk_rank", "blk_rel", "blk_n1", "blk_top_gap", "rev_n", "rev_rank", "rev_best_other", "rev_margin", "src3"]


def prob_context(cand, p):
    """Second-order features from first-pass probabilities (used by the stage-2 model)."""
    cand = cand.assign(p1=p)
    g1 = cand.groupby("i1").p1
    cand["p1_max_i1"] = g1.transform("max")
    cand["p1_rank_i1"] = g1.rank(ascending=False, method="first") - 1
    cand["p1_sum_i1"] = g1.transform("sum")
    cand["p1_n_hi_i1"] = (cand.p1 > 0.5).groupby(cand.i1).transform("sum")
    g2 = cand.groupby("i2").p1
    mx2 = g2.transform("max")
    cand["p1_rank_i2"] = g2.rank(ascending=False, method="first") - 1
    s = cand[["i2", "p1"]].sort_values(["i2", "p1"], ascending=[True, False])
    sec = s.groupby("i2").p1.nth(1)
    sec_map = pd.Series(sec.values, index=s.loc[sec.index, "i2"].values)
    sec_v = cand.i2.map(sec_map).fillna(0).values
    cand["p1_other_i2"] = np.where(cand.p1.values >= mx2.values, sec_v, mx2.values)
    return cand


PROB_COLS = ["p1", "p1_max_i1", "p1_rank_i1", "p1_sum_i1", "p1_n_hi_i1", "p1_rank_i2", "p1_other_i2"]


def decide(cand, p, thr, one_owner=True, rel=0.0):
    """Return boolean mask of accepted pairs."""
    keep = p >= thr
    if rel > 0:
        mx = pd.Series(p).groupby(cand.i1.values).transform("max").values
        keep &= p >= rel * mx
    if one_owner:
        # each S2/S3 record belongs to at most one S1 entity: keep argmax only
        s = pd.DataFrame({"i2": cand.i2.values, "p": p})
        best = s.groupby("i2").p.transform("max").values
        keep &= p >= best
    return keep


def macro_f05(s1_index, cand, keep, truth_counts):
    """Macro F0.5 over all S1 rows in s1_index.

    cand needs columns i1, label; truth_counts: Series i1 -> number of true matches.
    """
    sel = cand[keep]
    tp = sel.groupby("i1").label.sum()
    npred = sel.groupby("i1").size()
    df = pd.DataFrame(index=pd.Index(s1_index, name="i1"))
    df["T"] = truth_counts.reindex(df.index).fillna(0).values
    df["P"] = npred.reindex(df.index).fillna(0).values
    df["TP"] = tp.reindex(df.index).fillna(0).values
    denom = 0.25 * df["T"] + df["P"]
    f = np.where(denom > 0, 1.25 * df["TP"] / denom.where(denom > 0, 1), 1.0)
    return float(f.mean()), df.assign(f=f)


def freq_features(cand, P1, P23):
    """How common is a name / address within its country? A unique name makes a
    name-only (empty address) match safe; a very common one makes it risky."""
    def codes(col):
        k1 = P1.country.values.astype(object) + "|" + P1[col].values.astype(object)
        k2 = P23.country.values.astype(object) + "|" + P23[col].values.astype(object)
        c, _ = pd.factorize(np.concatenate([k1, k2]))
        c1, c2 = c[: len(k1)], c[len(k1):]
        m = c.max() + 1
        return c1, c2, np.bincount(c1, minlength=m), np.bincount(c2, minlength=m)

    n1, n2, nf1, nf2 = codes("n_core")
    a1, a2, af1, af2 = codes("a_toks")
    i1, i2 = cand.i1.values, cand.i2.values
    cand["fq_name_a_s1"] = nf1[n1[i1]].astype(np.float32)
    cand["fq_name_a_s23"] = nf2[n1[i1]].astype(np.float32)
    cand["fq_name_b_s1"] = nf1[n2[i2]].astype(np.float32)
    cand["fq_name_b_s23"] = nf2[n2[i2]].astype(np.float32)
    cand["fq_addr_a_s1"] = af1[a1[i1]].astype(np.float32)
    cand["fq_addr_b_s1"] = af1[a2[i2]].astype(np.float32)
    cand["fq_addr_b_s23"] = af2[a2[i2]].astype(np.float32)
    # number of this S1's candidates that share b's exact core name
    cand["fq_name_b_in_cands"] = pd.DataFrame({"i1": i1, "k": n2[i2]}).groupby(["i1", "k"]).k.transform(
        "size").astype(np.float32).values
    return cand


FREQ_COLS = ["fq_name_a_s1", "fq_name_a_s23", "fq_name_b_s1", "fq_name_b_s23", "fq_addr_a_s1",
             "fq_addr_b_s1", "fq_addr_b_s23", "fq_name_b_in_cands"]
