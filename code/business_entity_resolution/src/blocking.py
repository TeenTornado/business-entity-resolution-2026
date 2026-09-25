"""Candidate generation: IDF-weighted sparse token overlap with frequency capping.

Each record becomes a bag of hashed tokens (name words, joined name bigrams,
address words, address word bigrams). Within each country we keep only tokens
whose frequency in the S2/S3 pool is below a cap, weight them by IDF, L2
normalise, and compute S1 x S2/S3 cosine scores with chunked sparse matrix
products. The top-K S2/S3 records per S1 entity form the candidate set.
"""
import numpy as np
import scipy.sparse as sp
from numba import njit
from sklearn.feature_extraction.text import HashingVectorizer

N_FEATURES = 1 << 24


def block_doc(n_core, a_toks, a_comps, a_nums=""):
    toks = []
    nc = n_core.split()
    for t in nc:
        if len(t) >= 2 or t.isdigit():
            toks.append("n:" + t)
    for a, b in zip(nc, nc[1:]):
        toks.append("n:" + a + b)
    joined = "".join(nc)
    if len(nc) > 2:
        toks.append("n:" + joined)
    if len(nc) > 1:
        toks.append("s:" + "".join(sorted(nc)))
    at = a_toks.split()
    for t in at:
        if len(t) >= 2 or t.isdigit():
            toks.append("a:" + t)
    # conjunction keys: selective even when name and address words are common
    if joined:
        for t in dict.fromkeys(at):
            toks.append("j:" + joined + "|" + t)
    nums = [x for x in a_nums.split() if x not in ("0",)][:4]
    for t in dict.fromkeys(x for x in nc if len(x) >= 3):
        for x in nums:
            toks.append("x:" + t + "|" + x)
    for comp in a_comps.split("|"):
        ct = comp.split()
        for a, b in zip(ct, ct[1:]):
            toks.append("b:" + a + "_" + b)
    return " ".join(toks)


def _split(s):
    return s.split()


def vectorize(docs):
    hv = HashingVectorizer(n_features=N_FEATURES, analyzer=_split, binary=True, norm=None, alternate_sign=False, dtype=np.float32)
    return hv.transform(docs).tocsr()


def weight(X1, X2, cap, min_df=1):
    """IDF-weight both matrices using the S2/S3 pool frequency; drop tokens
    with df > cap (too common to be selective)."""
    df2 = np.asarray(X2.sum(axis=0)).ravel()
    df1 = np.asarray(X1.sum(axis=0)).ravel()
    n = X2.shape[0] + X1.shape[0]
    keep = (df2 <= cap) & (df2 >= min_df) & (df1 >= 1)
    idf = np.log((n + 1) / (df1 + df2 + 1)).astype(np.float32)
    idf[~keep] = 0
    D = sp.diags(idf)
    W1 = _l2(X1 @ D)
    W2 = _l2(X2 @ D)
    return W1, W2, keep


def _l2(X):
    X = X.tocsr()
    X.eliminate_zeros()
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    norms[norms == 0] = 1
    return (sp.diags(1 / norms) @ X).astype(np.float32).tocsr()


@njit(cache=True)
def _topk(indptr, indices, data, k):
    n = len(indptr) - 1
    out_r = np.empty(n * k, np.int64)
    out_c = np.empty(n * k, np.int64)
    out_s = np.empty(n * k, np.float32)
    m = 0
    for i in range(n):
        a, b = indptr[i], indptr[i + 1]
        if b == a:
            continue
        d = data[a:b]
        if b - a > k:
            idx = np.argsort(-d)[:k]
        else:
            idx = np.argsort(-d)
        for j in idx:
            out_r[m] = i
            out_c[m] = indices[a + j]
            out_s[m] = d[j]
            m += 1
    return out_r[:m], out_c[:m], out_s[:m]


def topk_rows(C, k):
    """Return (row, col, score) of the top-k entries per row of CSR matrix C."""
    C = C.tocsr()
    return _topk(C.indptr.astype(np.int64), C.indices.astype(np.int64), C.data.astype(np.float32), k)


def candidates(W1, W2T, k, chunk=2000):
    """Yield (row_idx, col_idx, score) arrays chunk by chunk."""
    for i in range(0, W1.shape[0], chunk):
        C = W1[i:i + chunk] @ W2T
        r, c, s = topk_rows(C, k)
        yield r + i, c, s


# ------------------------------------------------------------------ driver
_SHARED = {}


def _work(bounds):
    W1, W2T, k = _SHARED["W1"], _SHARED["W2T"], _SHARED["k"]
    lo, hi = bounds
    rs, cs, ss = [], [], []
    for r, c, s in candidates(W1[lo:hi], W2T, k):
        rs.append(r + lo)
        cs.append(c)
        ss.append(s)
    if not rs:
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float32)
    return np.concatenate(rs), np.concatenate(cs), np.concatenate(ss)


def run(P1, P23, cap=3000, k=30, workers=4, log=print):
    """Generate candidates for every S1 row. Countries are treated as an open
    set: each label present in S1 is blocked against the S2/S3 records with the
    same label (in training data, 100% of true pairs share the country label).
    Returns DataFrame(i1, i2, blk_score, blk_rank) with positional indices."""
    from multiprocessing import Pool
    import pandas as pd

    out = []
    for country in P1.country.unique():
        idx1 = np.flatnonzero(P1.country.values == country)
        idx2 = np.flatnonzero(P23.country.values == country)
        if len(idx2) == 0:
            continue
        docs1 = [block_doc(*r) for r in zip(P1.n_core.values[idx1], P1.a_toks.values[idx1], P1.a_comps.values[idx1], P1.a_nums.values[idx1])]
        docs2 = [block_doc(*r) for r in zip(P23.n_core.values[idx2], P23.a_toks.values[idx2], P23.a_comps.values[idx2], P23.a_nums.values[idx2])]
        X1, X2 = vectorize(docs1), vectorize(docs2)
        del docs1, docs2
        W1, W2, _ = weight(X1, X2, cap)
        del X1, X2
        _SHARED.update(W1=W1, W2T=W2.T.tocsr(), k=k)
        del W2
        step = 20000
        bounds = [(i, min(i + step, len(idx1))) for i in range(0, len(idx1), step)]
        with Pool(workers) as pool:
            parts = pool.map(_work, bounds, chunksize=1)
        r = np.concatenate([p[0] for p in parts])
        c = np.concatenate([p[1] for p in parts])
        s = np.concatenate([p[2] for p in parts])
        df = pd.DataFrame({"i1": idx1[r], "i2": idx2[c], "blk_score": s})
        out.append(df)
        _SHARED.clear()
        log(f"blocking {country}: S1={len(idx1)} S23={len(idx2)} pairs={len(df)}")
    cand = pd.concat(out, ignore_index=True)
    cand = cand.sort_values(["i1", "blk_score"], ascending=[True, False], kind="stable").reset_index(drop=True)
    cand["blk_rank"] = cand.groupby("i1").cumcount().astype(np.int16)
    return cand


def main():
    import argparse
    import time

    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--cap", type=int, default=3000)
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    cols = ["entity_id", "country", "n_core", "a_toks", "a_comps", "a_nums"]
    P1 = pd.read_parquet(f"{a.work}/{a.split}_p1.parquet", columns=cols)
    P23 = pd.concat([pd.read_parquet(f"{a.work}/{a.split}_p{s}.parquet", columns=cols) for s in (2, 3)],
                    ignore_index=True)
    cand = run(P1, P23, cap=a.cap, k=a.k, workers=a.workers,
               log=lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True))
    cand.to_parquet(f"{a.work}/{a.split}_cand.parquet")
    print(f"candidates={len(cand)} for S1={len(P1)}", flush=True)


if __name__ == "__main__":
    main()


def split_cosines(P1, P23, cand, chunk=2000000):
    """Name-only and address-only IDF cosine for each candidate pair (cheap sparse
    dot products; used by the candidate pruner). Returns two float32 arrays."""
    name_cos = np.zeros(len(cand), np.float32)
    addr_cos = np.zeros(len(cand), np.float32)
    c1 = P1.country.values[cand.i1.values]
    for country in np.unique(c1):
        idx1 = np.flatnonzero(P1.country.values == country)
        idx2 = np.flatnonzero(P23.country.values == country)
        rows = np.flatnonzero(c1 == country)
        pos1 = np.full(len(P1), -1, np.int64)
        pos1[idx1] = np.arange(len(idx1))
        pos2 = np.full(len(P23), -1, np.int64)
        pos2[idx2] = np.arange(len(idx2))
        for kind, out in (("name", name_cos), ("addr", addr_cos)):
            def docs(P, idx):
                if kind == "name":
                    res = []
                    for n in P.n_core.values[idx]:
                        nc = n.split()
                        res.append(" ".join(["n:" + t for t in nc] + ["n:" + a + b for a, b in zip(nc, nc[1:])]))
                    return res
                res = []
                for at, comps in zip(P.a_toks.values[idx], P.a_comps.values[idx]):
                    t = ["a:" + x for x in at.split()]
                    for comp in comps.split("|"):
                        ct = comp.split()
                        t += ["b:" + a + "_" + b for a, b in zip(ct, ct[1:])]
                    res.append(" ".join(t))
                return res
            X1, X2 = vectorize(docs(P1, idx1)), vectorize(docs(P23, idx2))
            W1, W2, _ = weight(X1, X2, cap=np.inf)
            del X1, X2
            for lo in range(0, len(rows), chunk):
                r = rows[lo:lo + chunk]
                a = W1[pos1[cand.i1.values[r]]]
                b = W2[pos2[cand.i2.values[r]]]
                out[r] = np.asarray(a.multiply(b).sum(axis=1)).ravel()
            del W1, W2
    return name_cos, addr_cos
