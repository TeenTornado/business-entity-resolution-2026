"""Learn an Indic-script -> Latin word dictionary from the TRAINING ground truth.

Matched (S1, S2/S3) pairs where the S2/S3 name is written in an Indic script and
has the same number of words as the Latin S1 name are aligned position by
position; each script word is mapped to its majority Latin word. Address
components in script (mostly state names) are mapped to the S1 address
component they co-occur with most. Only the provided training data is used.
"""
import argparse
import collections
import json
import re

import pandas as pd

from normalize import SCRIPT_RE, SCRIPT_TOKEN_RE

_PUNCT = re.compile(r"[^\wऀ-෿‌‍]+")


def _words(s):
    return [w for w in (_PUNCT.sub("", t) for t in s.split()) if w]


def learn(s1, s23, gt, min_count=2, min_share=0.5):
    pairs = gt.assign(m=gt.matched_entity_ids.str.split(",")).explode("m")
    pairs = pairs[pairs.m.notna() & (pairs.m != "")]
    s1n = s1.set_index("entity_id")
    s23n = s23.set_index("entity_id")
    a_name = s1n.business_name.reindex(pairs.source1_entity_id).values
    a_addr = s1n.business_address.reindex(pairs.source1_entity_id).values
    b_name = s23n.business_name.reindex(pairs.m).values
    b_addr = s23n.business_address.reindex(pairs.m).values

    wc = collections.defaultdict(collections.Counter)
    for an, bn in zip(a_name, b_name):
        if not isinstance(bn, str) or not SCRIPT_RE.search(bn):
            continue
        aw, bw = _words(an), _words(bn)
        if len(aw) != len(bw):
            continue
        for x, y in zip(bw, aw):
            if SCRIPT_RE.search(x) and not SCRIPT_RE.search(y):
                for xt in SCRIPT_TOKEN_RE.findall(x):
                    wc[xt][y.lower()] += 1
    word_map = {}
    for w, c in wc.items():
        tot = sum(c.values())
        best, n = c.most_common(1)[0]
        if n >= min_count and n / tot >= min_share:
            word_map[w] = best

    cc = collections.defaultdict(collections.Counter)
    comp_tot = collections.Counter()
    for aa, ba in zip(a_addr, b_addr):
        if not isinstance(ba, str) or not SCRIPT_RE.search(ba):
            continue
        acomps = {c.strip().lower() for c in aa.split(",") if c.strip()}
        for bc in {x.strip() for x in ba.split(",")}:
            if SCRIPT_RE.search(bc):
                comp_tot[bc] += 1
                for ac in acomps:
                    cc[bc][ac] += 1
    # a component is mapped only if its best partner co-occurs in most pairs
    comp_map = {}
    for w, c in cc.items():
        best, n = c.most_common(1)[0]
        if n >= min_count and n / comp_tot[w] >= 0.6:
            comp_map[w] = best
    return word_map, comp_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    s1 = pd.read_parquet(f"{a.work}/train_s1.parquet")
    s23 = pd.concat([pd.read_parquet(f"{a.work}/train_s2.parquet"), pd.read_parquet(f"{a.work}/train_s3.parquet")])
    gt = pd.read_parquet(f"{a.work}/train_gt.parquet")
    wm, cm = learn(s1, s23, gt)
    print(f"word_map={len(wm)} comp_map={len(cm)}")
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"word_map": wm, "comp_map": cm}, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
