# Business Entity Resolution: reproducible pipeline

The pipeline goes from the raw TSVs through normalisation, blocking and a learned candidate
pruner, then a stage-1 pairwise LightGBM matcher and a stage-2 **collective** LightGBM
matcher, then a multilingual cross-encoder that re-scores uncertain pairs. It ends with
`output/matching_results.tsv` and `output/candidate_pairs.tsv`.

It uses only the provided training/test files. There are no external lookups, APIs or
geocoders.
- **Models:** LightGBM boosters (MIT licence), plus one pretrained encoder,
  `intfloat/multilingual-e5-base` (MIT, 278 M parameters), fine-tuned as a cross-encoder.
- **Hardware:** the cross-encoder step needs a GPU or Apple-silicon MPS; everything else
  runs on CPU.

## Environment

* Python 3.11, `pip install -r requirements.txt` (on macOS also `brew install libomp`)
* Tested on 4 vCPU / 15 GB RAM / about 40 GB free disk, and on Apple-silicon MacBooks.
  Every stage streams or chunks its data to stay inside that budget.

## One-command run

```bash
# from this folder; DATA is student_resource/dataset (contains train/ and test/)
bash run_pipeline.sh /path/to/student_resource/dataset ./work /path/to/output
```

| Step | Script | What it does | Time* |
|---|---|---|---|
| 1 | `cache_raw.py`, `learn_translit.py` | cache train TSVs; learn an Indic-script → Latin word dictionary and state-name map from **train** ground-truth pairs | 3 min |
| 2 | `preprocess.py` (train, test) | normalise names and addresses: legal forms, EN/FR/IN abbreviations, ordinals, OCR digits, DBA, websites, dotted acronyms, transliteration, state tags, house numbers | 8 min |
| 3 | `simulate_siblings.py` | add test-like *sibling* distractor groups to the **train** pool: copies of real entities with the house number shifted by a learned offset, the legal form swapped and a qualifier added | 5 min |
| 4 | `blocking.py` (train, test) | hashed-token IDF cosine with a frequency cap and conjunction keys: top-30 per S1, plus an address-only channel (top-10), within each country label | 25 min |
| 5 | `train.py`, `predict.py` | candidate features, learned candidate **pruner** (this produces `candidate_pairs.tsv`), then the stage-1 pairwise matcher and its test scores/features | ~2.5 h |
| 6 | `collective.py train/predict` | stage-2 collective matcher: out-of-fold stage-1 scores, anchor and competitor group features, F0.5 threshold, one-owner rule, writes both outputs | ~45 min |
| 7 | `collective.py train --reuse-stage2`, `ce_export.py`, `ce_export2.py` | validation pair scores; text pairs for the cross-encoder (hard train pairs, French pseudo-labels, uncertain val/test pairs) | 20 min |
| 8 | `ce.py train/score` (GPU/MPS) | fine-tune e5-base on 750 k pairs, then score 65 k val + 1.37 M test pairs | ~2.5 h + 1 h on an M-series MacBook Pro |
| 9 | `ce_blend.py` | logistic blend fitted on val half A, threshold re-tuned, one-owner rule, writes both outputs | 5 min |
| 10 | `validate_submission.py` | official format check | 1 min |

\*On the 4-vCPU machine above.

## Inference only, with the shipped models

`model/` holds the artefacts used for the submitted output:
- `stage1/`: `lgb_pruner.txt.gz`, `lgb_stage1.txt.gz` and `decision.json` (feature lists,
  prune threshold 0.004, sibling offsets)
- `collective/`: `stage1_fold{0,1}.txt.gz`, `stage2.txt.gz` and `decision.json`
  (threshold 0.7375, anchor threshold 0.5)
- `translit.json`, `siblings.json`

```bash
W=work; mkdir -p $W/stage1 $W/collective
cp model/stage1/decision.json $W/stage1/; cp model/collective/decision.json $W/collective/
for f in lgb_pruner lgb_stage1; do gunzip -c model/stage1/$f.txt.gz > $W/stage1/$f.txt; done
for f in stage1_fold0 stage1_fold1 stage2; do gunzip -c model/collective/$f.txt.gz > $W/collective/$f.txt; done
cd src
python3 preprocess.py --data $DATA --work ../$W --split test --translit ../model/translit.json
python3 blocking.py   --work ../$W --split test
python3 predict.py    --work ../$W --model-dir ../$W/stage1 --out ../$W/stage1_out --split test
python3 collective.py predict --work ../$W --out-dir ../$W/collective --test-feat ../$W/test_pairfeat \
    --out $OUT --candidates ../$W/stage1_out/candidate_pairs.tsv
```

## Results

Validation uses held-out training S1 entities that are never used for fitting. Half A is
used for early stopping and the threshold; half B (37.5 k S1 entities) is reported.

| Model | Validation macro F0.5 (half B) | Public leaderboard |
|---|---|---|
| stage 1 only (v6) | 0.9844 | 0.9764 |
| stage 1 + stage-2 collective (c1) | 0.9877 | 0.9788 |
| + cross-encoder blend, US/India pairs (ce1) | 0.9887 | 0.9819 |
| **+ cross-encoder with French pseudo-labels (ce2, submitted)** | **0.9889** | **0.9832** |

- Candidate set: 6.50 candidates per S1 entity (11.27 M pairs for 1.73 M S1 entities).
  It keeps 98.25 % of all true pairs on validation.
- Matches: 5.79 M pairs, 3.34 per S1 entity; 94.1 % of S1 entities receive at least one.

## Source layout

```
src/
  normalize.py          text normalisation (country-agnostic rules, open set of country labels)
  learn_translit.py     learn script->Latin dictionary from train pairs only
  cache_raw.py          raw TSV -> parquet cache
  preprocess.py         parse every record into normalised fields (parallel)
  io_utils.py           shared loaders (S2, S3, then simulated train siblings)
  simulate_siblings.py  test-like sibling distractors for the train pool
  blocking.py           candidate generation (sparse IDF cosine, numba top-k, address channel)
  features.py           pairwise string features (rapidfuzz), parallel
  scoring.py            candidate-graph, frequency, vocabulary, sibling and name-graph features;
                        decision rule; macro F0.5
  train.py              pruner + stage-1 training, validation, threshold tuning
  predict.py            test candidate stage (per country) + stage-1 scoring
  collective.py         stage-2 collective matcher (train / predict) + submission writer
  ce_export.py          cross-encoder text pairs (hard train pairs, uncertain val/test pairs)
  ce_export2.py         + French pseudo-labelled pairs from a previous blend, wider band
  ce.py                 cross-encoder fine-tuning / scoring (PyTorch; MPS, CUDA or CPU; resumable)
  ce_blend.py           validation-fitted blend of stage-2 and cross-encoder scores + writer
```

## Decision rule

* Pairs with a stage-2 probability in 0.02–0.98 are re-scored:
  logit(p) = 0.449 + 0.718·logit(p_stage2) + 0.368·logit(p_ce).
  All other pairs keep their stage-2 probability.
* A pair is accepted when this probability is at least the tuned threshold (0.6875).
* Each S2/S3 record is assigned to at most one S1 entity: the one with the highest
  probability. In the training data every S2/S3 record belongs to at most one S1 cluster.
* S1 entities with no accepted pair get an empty list, which is the singleton prediction.

## Cross-encoder step on a Mac (what produced the submission)

`mac_ce/RUNBOOK.md`, `mac_ce/run_large.sh` and `mac_ce/run_v2.sh` run steps 8–9 on an Apple-silicon
MacBook Pro. `requirements-ce.txt` lists the extra packages. The blend settings are in
`model/ce_blend.json`.
