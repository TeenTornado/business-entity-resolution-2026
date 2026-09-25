"""Cache the raw TSV files as parquet (used by learn_translit.py)."""
import argparse

from preprocess import read_tsv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--split", default="train")
    a = ap.parse_args()
    for s in (1, 2, 3):
        read_tsv(f"{a.data}/{a.split}/{a.split}_source{s}.tsv").to_parquet(f"{a.work}/{a.split}_s{s}.parquet")
    if a.split == "train":
        read_tsv(f"{a.data}/train/train_ground_truth.tsv").to_parquet(f"{a.work}/train_gt.parquet")


if __name__ == "__main__":
    main()
