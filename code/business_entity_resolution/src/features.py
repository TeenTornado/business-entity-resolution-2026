"""Pairwise features for (S1, S2/S3) candidate pairs."""
import math
from multiprocessing import Pool

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

FIELDS = ["n_all", "n_core", "n_legal", "n_alt", "a_toks", "a_nums", "a_state", "a_comps", "n_script"]

_G = {}


def _idf(tok, table, n):
    return math.log((n + 1) / (table.get(tok, 0) + 1))


def _soft_overlap(ta, tb):
    """Fraction of tokens of ta that fuzzily appear in tb (handles typos)."""
    if not ta or not tb:
        return 0.0
    hit = 0
    for x in ta:
        if x in tb:
            hit += 1
            continue
        best = max(Levenshtein.normalized_similarity(x, y) for y in tb)
        if best >= 0.75 and len(x) >= 4:
            hit += 0.8
    return hit / len(ta)


def _idf_jacc(sa, sb, table, n):
    if not sa or not sb:
        return 0.0, 0.0
    inter = sa & sb
    wi = sum(_idf(t, table, n) for t in inter)
    wu = sum(_idf(t, table, n) for t in (sa | sb))
    wa = sum(_idf(t, table, n) for t in sa)
    return wi / wu if wu else 0.0, wi / wa if wa else 0.0


def pair_features(rows):
    """rows: list of tuples (a_fields..., b_fields...) -> list of feature lists."""
    ntab, atab, nn = _G["ntab"], _G["atab"], _G["n"]
    out = []
    for a, b in rows:
        a_all, a_core, a_legal, a_alt, a_at, a_num, a_st, a_comps, _ = a
        b_all, b_core, b_legal, b_alt, b_at, b_num, b_st, b_comps, b_scr = b
        ca, cb = a_core.split(), b_core.split()
        sa, sb = set(ca), set(cb)
        jn = "".join(ca), "".join(cb)
        f = []
        # ---- name
        f.append(fuzz.ratio(a_core, b_core))
        f.append(fuzz.token_sort_ratio(a_core, b_core))
        f.append(fuzz.token_set_ratio(a_core, b_core))
        f.append(fuzz.partial_ratio(a_core, b_core) if a_core and b_core else 0)
        f.append(JaroWinkler.similarity(jn[0], jn[1]))
        f.append(fuzz.ratio(jn[0], jn[1]))
        f.append(fuzz.ratio(a_all, b_all))
        f.append(len(sa & sb) / len(sa | sb) if sa or sb else 0)
        wj, wa = _idf_jacc(sa, sb, ntab, nn)
        f.append(wj)
        f.append(wa)
        f.append(_soft_overlap(ca, cb))
        f.append(_soft_overlap(cb, ca))
        f.append(float(bool(ca) and bool(cb) and ca[0] == cb[0]))
        f.append(float(jn[0] in jn[1] or jn[1] in jn[0]) if jn[0] and jn[1] else 0.0)
        best_alt = -1.0
        if b_alt:
            for alt in b_alt.split("|"):
                best_alt = max(best_alt, fuzz.token_set_ratio(a_core, alt), fuzz.ratio(a_core, alt))
        f.append(best_alt)
        la, lb = set(a_legal.split()), set(b_legal.split())
        f.append(float(bool(la) and bool(lb) and not (la & lb)))
        f.append(len(la & lb))
        f.append(len(ca))
        f.append(len(cb))
        f.append(len(jn[0]))
        f.append(float(b_scr))
        f.append(max((_idf(t, ntab, nn) for t in sa & sb), default=0.0))
        f.append(sum(_idf(t, ntab, nn) for t in sb - sa))
        # ---- address
        ta, tb = a_at.split(), b_at.split()
        xa, xb = set(ta), set(tb)
        f.append(float(not tb))
        f.append(fuzz.token_set_ratio(a_at, b_at) if tb else -1)
        f.append(fuzz.token_sort_ratio(a_at, b_at) if tb else -1)
        f.append(fuzz.partial_ratio(a_at, b_at) if tb and ta else -1)
        f.append(len(xa & xb) / len(xa | xb) if xa or xb else 0)
        aj, aa = _idf_jacc(xa, xb, atab, nn)
        f.append(aj)
        f.append(aa)
        f.append(_idf_jacc(xb, xa, atab, nn)[1])
        f.append(_soft_overlap(tb, ta) if tb else -1)
        na, nb = a_num.split(), b_num.split()
        sna, snb = set(na), set(nb)
        f.append(len(na))
        f.append(len(nb))
        f.append(len(sna & snb))
        f.append(len(sna & snb) / len(sna | snb) if sna or snb else -1)
        f.append(float(bool(na) and bool(nb) and na[0] == nb[0]))
        f.append(float(bool(sna) and bool(snb) and not (sna & snb)))
        f.append(float(a_st == b_st) if a_st and b_st else -1)
        pa, pb = set(a_comps.split("|")) - {""}, set(b_comps.split("|")) - {""}
        f.append(len(pa & pb) / min(len(pa), len(pb)) if pa and pb else -1)
        f.append(len(pa & pb))
        f.append(max(_idf(t, atab, nn) for t in xa & xb) if xa & xb else 0.0)
        # ---- combined
        f.append(fuzz.token_set_ratio(a_core + " " + a_at, b_core + " " + b_at))
        # ---- v4: house-number noise vs sibling offsets, street words without numbers, legal add/drop
        da = [x for x in na if x.isdigit()]
        db = [x for x in nb if x.isdigit()]
        if da and db:
            ha, hb = da[0], db[0]
            f.append(Levenshtein.distance(ha, hb))
            f.append(float(ha != hb and (hb.startswith(ha) or ha.startswith(hb) or hb.endswith(ha) or ha.endswith(hb))))
            f.append(math.log1p(abs(int(ha) - int(hb))))
            ia, ib = [int(x) for x in da], [int(x) for x in db]
            rng_b = len(ib) >= 2 and ib[0] < ib[1] <= ib[0] + 20 and ib[0] <= ia[0] <= ib[1]
            rng_a = len(ia) >= 2 and ia[0] < ia[1] <= ia[0] + 20 and ia[0] <= ib[0] <= ia[1]
            f.append(float(rng_a or rng_b))
            f.append(float(any(Levenshtein.distance(x, y) <= 1 for x in da for y in db if len(x) >= 2 and len(y) >= 2)))
        else:
            f.extend([-1, -1, -1, -1, -1])
        aa = " ".join(t for t in ta if not any(ch.isdigit() for ch in t))
        ab = " ".join(t for t in tb if not any(ch.isdigit() for ch in t))
        f.append(fuzz.token_set_ratio(aa, ab) if aa and ab else -1)
        f.append(fuzz.token_sort_ratio(aa, ab) if aa and ab else -1)
        f.append(float((not la) and bool(lb)))
        f.append(float(bool(la) and not lb))
        out.append(f)
    return out


FEATURE_NAMES = [
    "nm_ratio", "nm_tsort", "nm_tset", "nm_partial", "nm_jw_join", "nm_ratio_join", "nm_ratio_all",
    "nm_jacc", "nm_idf_jacc", "nm_idf_cov_a", "nm_soft_ab", "nm_soft_ba", "nm_first_eq", "nm_contain",
    "nm_best_alt", "lg_conflict", "lg_shared", "nm_len_a", "nm_len_b", "nm_chars_a", "nm_script_b",
    "nm_max_idf_shared", "nm_idf_extra_b",
    "ad_empty_b", "ad_tset", "ad_tsort", "ad_partial", "ad_jacc", "ad_idf_jacc", "ad_idf_cov_a",
    "ad_idf_cov_b", "ad_soft_ba", "num_n_a", "num_n_b", "num_shared", "num_jacc", "num_first_eq",
    "num_conflict", "state_eq", "comp_overlap", "comp_shared", "ad_max_idf_shared",
    "all_tset",
    "hn_lev", "hn_prefix", "hn_logdiff", "hn_in_range", "num_any_lev1",
    "ad_alpha_tset", "ad_alpha_tsort", "lg_add", "lg_drop",
]


def build_df_tables(frames, step=1000000):
    """Token document frequencies (S1 + S2/S3 pool) for name-core and address tokens."""
    import collections
    ntab, atab = collections.Counter(), collections.Counter()
    n = 0
    for fr in frames:
        n += len(fr)
        for lo in range(0, len(fr), step):
            part = fr.iloc[lo:lo + step]
            for col, tab in (("n_core", ntab), ("a_toks", atab)):
                for v in part[col].values:
                    tab.update(set(v.split()))
    return dict(ntab), dict(atab), n


def compute(pairs, P1, P23, tables, workers=4, chunk=10000, outer=400000):
    """pairs: DataFrame with integer columns i1 (row in P1), i2 (row in P23)."""
    ntab, atab, n = tables
    _G.update({"ntab": ntab, "atab": atab, "n": n})  # inherited by forked workers
    cols1 = [P1[c].values for c in FIELDS]
    cols2 = [P23[c].values for c in FIELDS]
    i1all, i2all = pairs.i1.values, pairs.i2.values
    outs = []
    with Pool(workers) as pool:
        for lo in range(0, len(pairs), outer):
            i1, i2 = i1all[lo:lo + outer], i2all[lo:lo + outer]
            A = list(zip(*[c[i1] for c in cols1]))
            B = list(zip(*[c[i2] for c in cols2]))
            rows = list(zip(A, B))
            chunks = [rows[i:i + chunk] for i in range(0, len(rows), chunk)]
            res = pool.map(pair_features, chunks, chunksize=1)
            outs.append(np.array([f for part in res for f in part], dtype=np.float32))
    X = np.concatenate(outs) if outs else np.zeros((0, len(FEATURE_NAMES)), np.float32)
    return pd.DataFrame(X, columns=FEATURE_NAMES, index=pairs.index)
