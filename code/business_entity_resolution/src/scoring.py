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


def pruner_features(cand, P1, P23, name_cos, addr_cos):
    """Cheap, fully vectorised pair signals for the candidate pruner."""
    def eq(col):
        c, _ = pd.factorize(np.concatenate([P1[col].values, P23[col].values]))
        a, b = c[: len(P1)], c[len(P1):]
        return (a[cand.i1.values] == b[cand.i2.values]).astype(np.int8)

    cand["pr_name_cos"] = name_cos
    cand["pr_addr_cos"] = addr_cos
    cand["pr_name_eq"] = eq("n_core")
    cand["pr_addr_eq"] = eq("a_toks")
    first = lambda s: s.str.split(" ", n=1).str[0]  # noqa: E731
    f1, f2 = first(P1.a_nums), first(P23.a_nums)
    c, _ = pd.factorize(np.concatenate([f1.values, f2.values]))
    a, b = c[: len(P1)], c[len(P1):]
    has = (f1.values[cand.i1.values] != "") & (f2.values[cand.i2.values] != "")
    cand["pr_num_eq"] = np.where(has, (a[cand.i1.values] == b[cand.i2.values]).astype(np.int8), -1).astype(np.int8)
    cand["pr_b_addr_empty"] = (P23.a_toks.values[cand.i2.values] == "").astype(np.int8)
    return cand


PRUNER_COLS = CONTEXT_COLS + FREQ_COLS + ["pr_name_cos", "pr_addr_cos", "pr_name_eq", "pr_addr_eq", "pr_num_eq",
                                          "pr_b_addr_empty"]


def _token_counts(P1, P23):
    import collections
    tabs = {}
    for country in pd.unique(P1.country.values):
        m1 = P1.country.values == country
        m2 = P23.country.values == country
        c1, c2 = collections.Counter(), collections.Counter()
        for v in P1.n_core.values[m1]:
            c1.update(set(v.split()))
        for v in P23.n_core.values[m2]:
            c2.update(set(v.split()))
        tabs[country] = (c1, c2, float(np.log((m2.sum() + 1) / (m1.sum() + 1))))
    return tabs


def vocab_features(cand, P1, P23):
    """Name-token vocabulary skew between the reference (S1) pool and the S2/S3 pool,
    per country. Tokens that are common in S2/S3 but rare in S1 (e.g. shop-type words
    or qualifiers used mostly by businesses absent from S1) signal records that likely
    have no S1 match. Also: the strongest-skewed token that the S2/S3 name ADDS on top
    of the S1 name (a qualifier such as 'Holdings' or 'Groupe' added to a near-copy)."""
    import math
    tabs = _token_counts(P1, P23)

    def rec_skew(P, idx_all):
        out = np.zeros((len(P), 2), np.float32)
        for country, (c1, c2, base) in tabs.items():
            idx = np.flatnonzero(P.country.values == country)
            for j, v in zip(idx, P.n_core.values[idx]):
                toks = v.split()
                if toks:
                    r = [math.log((c2.get(t, 0) + 1) / (c1.get(t, 0) + 1)) - base for t in toks]
                    out[j, 0] = max(r)
                    out[j, 1] = sum(r) / len(r)
        return out

    out1, out2 = rec_skew(P1, None), rec_skew(P23, None)
    cand["vc_skew_max_b"] = out2[cand.i2.values, 0]
    cand["vc_skew_mean_b"] = out2[cand.i2.values, 1]
    cand["vc_skew_max_a"] = out1[cand.i1.values, 0]
    added = np.full(len(cand), -9.0, np.float32)
    ctry = P1.country.values[cand.i1.values]
    na, nb = P1.n_core.values[cand.i1.values], P23.n_core.values[cand.i2.values]
    for j in range(len(cand)):
        extra = set(nb[j].split()) - set(na[j].split())
        if extra:
            c1, c2, base = tabs[ctry[j]]
            added[j] = max(math.log((c2.get(t, 0) + 1) / (c1.get(t, 0) + 1)) - base for t in extra)
    cand["vc_added_skew"] = added
    return cand


VOCAB_COLS = ["vc_skew_max_b", "vc_skew_mean_b", "vc_skew_max_a", "vc_added_skew"]


def _digits(s):
    return [int(t) for t in s.split() if t.isdigit() and len(t) <= 6]


def sibling_features(cand, P1, P23, offsets):
    """House-number relation between the S1 record and the candidate, and the size of
    the candidate's number group within this S1's candidate list. Sibling distractors
    carry the S1 number shifted by a learned offset k and come in groups of 1-3 records
    that repeat the SAME shifted number, distinct from the S1's exact-number group."""
    K = set(offsets)
    fa = np.array([(d[0] if d else -1) for d in map(_digits, P1.a_nums.values)], np.int64)
    fb = np.array([(d[0] if d else -1) for d in map(_digits, P23.a_nums.values)], np.int64)
    A, B = fa[cand.i1.values], fb[cand.i2.values]
    both = (A >= 0) & (B >= 0)
    cand["sb_first_diff"] = np.where(both, np.clip(B - A, -100, 100), -999).astype(np.int16)
    cand["sb_exact_first"] = np.where(both, (A == B).astype(np.int8), -1).astype(np.int8)
    na, nb = P1.a_nums.values[cand.i1.values], P23.a_nums.values[cand.i2.values]
    offk = np.zeros(len(cand), np.int8)
    for j in range(len(cand)):
        if not both[j]:
            continue
        da, db = set(_digits(na[j])), set(_digits(nb[j]))
        if any((x - y) in K for x in db - da for y in da):
            offk[j] = 1
    cand["sb_off_k"] = offk
    g = pd.DataFrame({"i1": cand.i1.values, "b": B, "exact": (A == B) & both, "offk": offk.astype(bool)})
    cand["sb_grp_b"] = np.where(B >= 0, g.groupby(["i1", "b"]).b.transform("size").values, -1).astype(np.int16)
    cand["sb_grp_exact"] = g.groupby("i1").exact.transform("sum").values.astype(np.int16)
    cand["sb_grp_offk"] = g.groupby("i1").offk.transform("sum").values.astype(np.int16)
    return cand


SIB_COLS = ["sb_first_diff", "sb_exact_first", "sb_off_k", "sb_grp_b", "sb_grp_exact", "sb_grp_offk"]
