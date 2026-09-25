# Business Entity Resolution: reproducible pipeline

This pipeline goes from raw TSVs to normalisation, blocking, a LightGBM matcher, and
finally `output/matching_results.tsv` + `output/candidate_pairs.tsv`.

It uses only the provided training/test files. There are no external lookups,
APIs, geocoders or pretrained language models. The only model is LightGBM
(MIT licence), which has a few million parameters.

## Environment

* Python 3.11, `pip install -r requirements.txt`
* Tested on 4 vCPU / 15 GB RAM / ~30 GB free disk. Every stage streams or chunks its
  data to stay inside that budget.

## One-command run

```bash
# from this folder; DATA is student_resource/dataset (contains train/ and test/)
bash run_pipeline.sh /path/to/student_resource/dataset ./work /path/to/output
```

The script runs these steps (all under `src/`):

| Step | Script | What it does | Time* |
|---|---|---|---|
| 1 | `cache_raw.py` | caches raw train TSVs as parquet | 1 min |
| 2 | `learn_translit.py` | learns an Indic-script → Latin word dictionary + state-name map from **train** ground-truth pairs | 2 min |
| 3 | `preprocess.py` (train, test) | normalises names/addresses (legal forms, abbreviations, OCR digits, DBA, websites, transliteration, state tags, numbers) | 4 + 4 min |
| 4 | `blocking.py` (train, test) | hashed-token IDF cosine with frequency cap + conjunction keys, top-30 per S1 within the same country label | 12 + 13 min |
| 5 | `train.py` | pairwise + graph-context + frequency features, LightGBM matcher, F0.5 threshold tuned on held-out S1 entities (optional `--stage2` re-scorer evaluated, off by default) | ~60 min |
| 6 | `predict.py` | scores all test candidates in S1 blocks, applies the one-owner rule + threshold, writes both TSVs | ~80 min |
| 7 | `validate_submission.py` | official format check | 1 min |

\*Measured on the 4-vCPU machine above.

## Inference only, with the shipped model

`model/` holds the trained artefacts used for the submitted output:
- `lgb_stage1.txt.gz`: the LightGBM model
- `decision.json`: feature list and threshold 0.713
- `translit.json`: the learned dictionary

To regenerate the test outputs without retraining:

```bash
mkdir -p work/model && cp model/decision.json work/model/ && gunzip -c model/lgb_stage1.txt.gz > work/model/lgb_stage1.txt
cd src
python3 preprocess.py --data $DATA --work ../work --split test --translit ../model/translit.json
python3 blocking.py   --work ../work --split test
python3 predict.py    --work ../work --model-dir ../work/model --out $OUT --split test
```

## Results

- Held-out validation (75,000 train S1 entities never used for fitting, early stopping or
  earlier threshold choices): **macro F0.5 = 0.9782**.
- Blocking keeps 98.14 % of all true train pairs with 30 candidates per S1 entity.

## Source layout

```
src/
  normalize.py       text normalisation (country-agnostic rules, open set of country labels)
  learn_translit.py  learn script->Latin dictionary from train pairs only
  cache_raw.py       raw TSV -> parquet cache
  preprocess.py      parse every record into normalised fields (parallel)
  blocking.py        candidate generation (sparse IDF cosine, numba top-k)
  features.py        43 pairwise string features (rapidfuzz), parallel
  scoring.py         candidate-graph context, frequency features, decision rule, macro F0.5
  train.py           training + validation + threshold tuning
  predict.py         test inference + submission writer
```

## Notes on the decision rule

* The final score for each pair is the stage-2 probability, or stage 1 when stage 2
  does not help on validation.
* A pair is accepted if its probability is at least the tuned threshold.
* Each S2/S3 record is assigned to at most one S1 entity: the one with the highest
  probability. In the training data every S2/S3 record belongs to at most one
  S1 cluster.
* S1 entities with no accepted pair get an empty list, which is the singleton prediction.
