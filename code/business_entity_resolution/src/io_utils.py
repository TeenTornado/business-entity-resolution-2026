"""Shared loaders so every stage sees the S2/S3 pool in the same row order:
S2 records, then S3 records, then (train only) simulated sibling distractors."""
import os

import pandas as pd


def pool_paths(work, split):
    paths = [f"{work}/{split}_p2.parquet", f"{work}/{split}_p3.parquet"]
    sim = f"{work}/{split}_psim.parquet"
    if split == "train" and os.path.exists(sim):
        paths.append(sim)
    return paths


def read_p1(work, split, columns=None):
    return pd.read_parquet(f"{work}/{split}_p1.parquet", columns=columns)


def read_p23(work, split, columns=None):
    return pd.concat([pd.read_parquet(p, columns=columns) for p in pool_paths(work, split)], ignore_index=True)
