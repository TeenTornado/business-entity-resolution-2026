"""Augment the TRAINING S2/S3 pool with simulated 'sibling' distractor groups.

Forensics on train showed that about half of the unmatched S2/S3 records are
siblings: near-copies of a real entity with the house number shifted by a
small positive offset k (from a fixed, learnable set), the legal form usually
swapped, and often a qualifier word added. In train, siblings are almost always
single records. In the test pool they are about twice as frequent and come in
groups of 2-3 noisy records, and repeated addresses fool frequency/graph features.

This script reproduces that condition in the training data, using training data
only:
  * the offset set K is learned from S1 vs unmatched-S2/S3 pairs that share a
    name (offsets whose count far exceeds the mirrored negative offset)
  * the qualifier lexicon is learned as name tokens strongly over-represented
    in unmatched vs matched S2/S3 records
  * the legal-form mix is learned from S1 names per country
Each simulated group clones 1-3 of an entity's own S2/S3 records (keeping their
noise), shifts the house number by k, swaps the legal form (p=0.85) and adds a
qualifier (p=0.4). Clones are unmatched, so they are negatives for every S1.
"""
import argparse
import collections
import json
import re

import numpy as np
import pandas as pd

from normalize import SCRIPT_RE
from preprocess import parse_frame

LEGAL_DISPLAY = {"llc": "LLC", "inc": "Inc", "corp": "Corp", "co": "Co", "ltd": "Ltd", "pvt": "Pvt Ltd",
                 "llp": "LLP", "lp": "LP", "plc": "PLC", "pllc": "PLLC", "pc": "PC", "pa": "PA",
                 "public": "Public Limited", "sarl": "SARL", "sas": "SAS", "sasu": "SASU", "eurl": "EURL",
                 "sa": "SA", "snc": "SNC", "sci": "SCI", "ei": "EI", "gmbh": "GmbH"}
LEGAL_RAW_RE = re.compile(
    r"(?i)\b(?:pvt\.?|private|ltd\.?|limited|inc\.?|incorporated|corp\.?|corporation|co\.?|company|"
    r"l\.?l\.?c\.?|llp|l\.p\.|lp|plc|pllc|p\.?c\.?|sarl|s\.a\.s\.?|sas|sasu|eurl|s\.a\.?|snc|sci)(?=\W|$)")


def first_num(s):
    for t in s.split():
        if t.isdigit() and len(t) <= 6:
            return int(t)
    return None


def learn_offsets(P1, P23, owner, max_k=50):
    """Offsets d>0 whose frequency among (S1, unmatched same-name S2/S3) pairs far
    exceeds the mirrored -d (the natural, symmetric background)."""
    a = pd.DataFrame({"key": P1.country + "|" + P1.n_core, "fa": [first_num(x) for x in P1.a_nums]})
    u = P23[owner < 0]
    b = pd.DataFrame({"key": u.country + "|" + u.n_core, "fb": [first_num(x) for x in u.a_nums]})
    a, b = a.dropna(), b.dropna()
    vc = a.key.value_counts()
    a = a[a.key.map(vc) == 1]  # unambiguous S1 name
    m = a.merge(b, on="key")
    d = (m.fb - m.fa).astype(int)
    h = collections.Counter(d[(d.abs() <= max_k) & (d != 0)])
    excess = {k: h.get(k, 0) - h.get(-k, 0) for k in range(1, max_k + 1)}
    noise = np.median(np.abs(list(excess.values()))) + 1
    ks = sorted(k for k, v in excess.items() if v > 10 * noise and v > 50)
    return ks, excess


def learn_qualifiers(P23, owner, min_count=300, ratio=5.0):
    """Latin-script name tokens strongly over-represented in unmatched records."""
    cu, cm = collections.Counter(), collections.Counter()
    latin = ~P23.raw_name.str.contains(SCRIPT_RE, regex=True).values
    for core, o in zip(P23.n_core.values[latin], owner[latin]):
        (cu if o < 0 else cm).update(set(core.split()))
    nu, nm = (owner < 0).sum(), (owner >= 0).sum()
    q = [t for t, c in cu.items()
         if c >= min_count and (c / nu) / ((cm.get(t, 0) + 1) / nm) >= ratio and t.isalpha() and len(t) > 2]
    return sorted(q)


def styled(word, template):
    if template.isupper():
        return word.upper()
    if template.islower():
        return word.lower()
    return word


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--translit", required=True)
    ap.add_argument("--rate", type=float, default=0.4, help="share of eligible S1 entities given a sibling group")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.RandomState(a.seed)
    cols = ["entity_id", "country", "n_core", "n_legal", "a_nums", "raw_name", "raw_addr"]
    P1 = pd.read_parquet(f"{a.work}/train_p1.parquet", columns=cols)
    P23 = pd.concat([pd.read_parquet(f"{a.work}/train_p{s}.parquet", columns=cols) for s in (2, 3)],
                    ignore_index=True)
    gt = pd.read_parquet(f"{a.work}/train_gt.parquet")
    pairs = gt.assign(m=gt.matched_entity_ids.str.split(",")).explode("m")
    pairs = pairs[pairs.m.notna() & (pairs.m != "")]
    owner = pd.Series(pd.Index(P1.entity_id).get_indexer(pairs.source1_entity_id.values), index=pairs.m.values)
    owner = owner.reindex(P23.entity_id.values).fillna(-1).astype(np.int64).values

    ks, excess = learn_offsets(P1, P23, owner)
    quals = learn_qualifiers(P23, owner)
    print(f"learned offsets K={ks}")
    print(f"learned {len(quals)} qualifier tokens, e.g. {quals[:25]}")
    legal_mix = {}
    for c, g in P1.groupby("country"):
        cnt = collections.Counter(t for x in g.n_legal.values for t in x.split() if t in LEGAL_DISPLAY)
        legal_mix[c] = (list(cnt), np.array(list(cnt.values()), float) / sum(cnt.values()))

    # templates: true copies whose raw address contains the S1's first house number
    fa = np.array([first_num(x) for x in P1.a_nums.values], dtype=object)
    t_idx = np.flatnonzero(owner >= 0)
    by_s1 = collections.defaultdict(list)
    for j in t_idx:
        f = fa[owner[j]]
        if f is not None and re.search(rf"(?<!\d)0*{f}(?!\d)", P23.raw_addr.values[j] or ""):
            by_s1[owner[j]].append(j)
    eligible = np.array(sorted(by_s1))
    chosen = eligible[rng.rand(len(eligible)) < a.rate]
    print(f"eligible S1 entities: {len(eligible)}, sibling groups: {len(chosen)}")

    rows = []
    n = 0
    for e in chosen:
        f = fa[e]
        k = ks[rng.randint(len(ks))]
        size = rng.choice([1, 2, 3], p=[0.45, 0.35, 0.20])
        temps = by_s1[e]
        pick = rng.choice(temps, size=size, replace=len(temps) < size)
        forms, probs = legal_mix[P1.country.values[e]]
        cur = set(P1.n_legal.values[e].split())
        new_legal = None
        if rng.rand() < 0.85:
            opts = [(x, p) for x, p in zip(forms, probs) if x not in cur]
            if opts:
                xs, ps = zip(*opts)
                new_legal = xs[rng.choice(len(xs), p=np.array(ps) / sum(ps))]
        qual = quals[rng.randint(len(quals))] if (quals and rng.rand() < 0.4) else None
        for j in pick:
            name, addr = P23.raw_name.values[j], P23.raw_addr.values[j]
            addr = re.sub(rf"(?<!\d)0*{f}(?!\d)", str(f + k), addr, count=1)
            if SCRIPT_RE.search(name):
                pass  # Indic-script name: only the house number is shifted
            elif new_legal is not None:
                name = LEGAL_RAW_RE.sub("", name).strip(" ,.-")
                name = f"{name} {styled(LEGAL_DISPLAY[new_legal], name)}"
            if qual is not None and not SCRIPT_RE.search(name):
                w = name.split()
                pos = len(w) if rng.rand() < 0.5 else 1
                w.insert(min(pos, len(w)), styled(qual.capitalize(), name))
                name = " ".join(w)
            src = P23.entity_id.values[j][:2]
            rows.append((f"{src}-SIM{n}", re.sub(r"\s+", " ", name), addr, P23.country.values[j]))
            n += 1
    sim = pd.DataFrame(rows, columns=["entity_id", "business_name", "business_address", "country"])
    print(f"simulated sibling records: {len(sim)} ({len(sim) / len(P1):.3f} per S1)")
    print(sim.sample(8, random_state=0).to_string())
    parsed = parse_frame(sim, a.translit)
    parsed.to_parquet(f"{a.work}/train_psim.parquet")
    json.dump({"offsets": ks, "qualifiers": quals, "excess": excess}, open(f"{a.work}/siblings.json", "w"))


if __name__ == "__main__":
    main()
