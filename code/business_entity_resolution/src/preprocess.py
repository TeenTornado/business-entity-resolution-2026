"""Read raw TSVs, normalise every record, and cache the parsed fields as parquet."""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from normalize import SCRIPT_RE, Transliterator, addr_parse, name_tokens, state_key

_TL = None


def _init(tl_path):
    global _TL
    d = json.load(open(tl_path, encoding="utf-8")) if tl_path and os.path.exists(tl_path) else {}
    _TL = Transliterator(d.get("word_map"), d.get("comp_map"))


def _proc(chunk):
    names, addrs = chunk
    out = {k: [] for k in ("n_all", "n_core", "n_legal", "n_alt", "a_toks", "a_nums", "a_state", "a_comps", "n_script")}
    for n, a in zip(names, addrs):
        toks, core, legal, alts = name_tokens(n, _TL)
        p = addr_parse(a, _TL)
        out["n_all"].append(" ".join(toks))
        out["n_core"].append(" ".join(core))
        out["n_legal"].append(" ".join(sorted(legal)))
        out["n_alt"].append("|".join(" ".join(x) for x in alts))
        out["a_toks"].append(" ".join(p["toks"]))
        out["a_nums"].append(" ".join(dict.fromkeys(p["nums"])))
        out["a_state"].append(state_key(p["state"]))
        out["a_comps"].append("|".join(p["comps"]))
        out["n_script"].append(bool(SCRIPT_RE.search(n)))
    return out


def read_tsv(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3)


def parse_frame(df, tl_path, workers=4, chunk=20000):
    names = df.business_name.tolist()
    addrs = df.business_address.tolist()
    chunks = [(names[i:i + chunk], addrs[i:i + chunk]) for i in range(0, len(names), chunk)]
    with Pool(workers, initializer=_init, initargs=(tl_path,)) as pool:
        parts = pool.map(_proc, chunks, chunksize=1)
    res = pd.DataFrame({k: [v for p in parts for v in p[k]] for k in parts[0]})
    res.insert(0, "entity_id", df.entity_id.values)
    res["country"] = df.country.values
    res["raw_name"] = df.business_name.values
    res["raw_addr"] = df.business_address.values
    res["n_script"] = res.n_script.astype(np.bool_)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset dir containing train/ and test/")
    ap.add_argument("--work", required=True)
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--translit", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    for s in (1, 2, 3):
        df = read_tsv(f"{a.data}/{a.split}/{a.split}_source{s}.tsv")
        res = parse_frame(df, a.translit, a.workers)
        res.to_parquet(f"{a.work}/{a.split}_p{s}.parquet")
        print(a.split, s, len(res), flush=True)
    if a.split == "train":
        gt = read_tsv(f"{a.data}/train/train_ground_truth.tsv")
        gt.to_parquet(f"{a.work}/train_gt.parquet")


if __name__ == "__main__":
    import multiprocessing as _mp
    _mp.set_start_method("fork", force=True)  # macOS defaults to spawn; workers rely on fork-inherited globals
    main()
