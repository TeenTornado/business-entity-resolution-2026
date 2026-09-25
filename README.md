# ML Challenge 2026 — Business Entity Resolution

Cross-source **Entity Resolution**: given business records from 3 independent, noisy
data sources, determine which records refer to the same real-world business. Source 1 is
the deduplicated reference source; the task is to find matching Source 2 / Source 3
records for each Source 1 entity. Evaluated with a precision-heavy **F₀.₅** metric.

See [`student_resource/README.md`](student_resource/README.md) for the full problem
statement, file formats, and evaluation details.

## Dataset

The dataset files are large (individual TSVs are 120–500 MB; ~2.4 GB uncompressed),
so they exceed GitHub's 100 MB per-file limit and are **not** committed to the repo
directly. The complete dataset is published as a **release asset**:

➡️ **[Download the full dataset (.zip)](../../releases/latest)**

After downloading, unzip so you have:

```
student_resource/
├── dataset/
│   ├── train/   train_source1.tsv, train_source2.tsv, train_source3.tsv, train_ground_truth.tsv
│   └── test/    test_source1.tsv, test_source2.tsv, test_source3.tsv
├── utils/validate_submission.py
├── README.md
└── Documentation_template.md
```

### Scale

| File | Rows |
| --- | --- |
| train_source1.tsv | 2,206,822 |
| train_source2.tsv | 5,034,617 |
| train_source3.tsv | 5,285,604 |
| train_ground_truth.tsv | 2,206,822 |
| test_source1.tsv | 1,732,545 |
| test_source2.tsv | 4,887,274 |
| test_source3.tsv | 5,082,317 |

Columns for every source file: `entity_id`, `business_name`, `business_address`,
`country`. Files are **tab-separated** — read with `pd.read_csv(path, sep="\t")`.
