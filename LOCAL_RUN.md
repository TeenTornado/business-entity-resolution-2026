# Running the pipeline locally (macOS / Apple Silicon)

## 1. Setup (once, about 10 min)
```bash
brew install libomp python@3.11
git clone -b claude/entity-resolution-ml-6m8f15 https://github.com/TeenTornado/business-entity-resolution-2026.git
cd business-entity-resolution-2026
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r code/business_entity_resolution/requirements.txt
# dataset (2.4 GB unzipped)
curl -L -o ds.zip https://github.com/TeenTornado/business-entity-resolution-2026/releases/download/v1.0-dataset/6ab10eb3b23ba_student_resource.zip
unzip -q ds.zip -d data && rm ds.zip
```

## 2. Full pipeline (about 2-3 h on an M4)
```bash
export D=$PWD/data/student_resource/dataset W=$PWD/work
cd code/business_entity_resolution/src
python3 cache_raw.py --data $D --work $W --split train
python3 learn_translit.py --work $W --out $W/translit.json
python3 preprocess.py --data $D --work $W --split train --translit $W/translit.json
python3 preprocess.py --data $D --work $W --split test  --translit $W/translit.json
python3 simulate_siblings.py --work $W --translit $W/translit.json
python3 blocking.py --work $W --split train
python3 blocking.py --work $W --split test
python3 train.py   --work $W --model-dir $W/m1 --n-train 800000 --prune-thr 0.004     # "VALIDATION" lines = score
python3 predict.py --work $W --model-dir $W/m1 --out $W/out1                          # writes test_pairfeat/ + test_scores.npy
```
If you have 24 GB+ RAM, you can raise `--n-train` (e.g. 1500000).

## 3. Self-training rounds (the lever that moved the leaderboard: v5 0.9758 -> v6 0.9764)
Each round trains on confident test predictions from the previous round (about 20 min).
```bash
cp $W/test_scores.npy $W/scores_r1.npy && mv $W/test_pairfeat $W/feat_r1
python3 train.py --work $W --model-dir $W/m2 --n-train 800000 --prune-thr 0.004 --reuse-pairs \
    --pseudo-dir $W/feat_r1 --pseudo-scores $W/scores_r1.npy --pseudo-weight 1.0
python3 predict.py --work $W --model-dir $W/m2 --out $W/out2 --save-features 0
```

## 4. Validate and send
```bash
python3 ../../../data/student_resource/utils/validate_submission.py \
  --matching $W/out2/matching_results.tsv --candidate $W/out2/candidate_pairs.tsv --test-dir $D/test
```
Paste back only the `VALIDATION ...`, `pruned candidate set` and `matches written` lines (plus any `Traceback`).

## 5. Final submission zip
```bash
mkdir -p pkg/output && cp $W/out2/*.tsv pkg/output/
cp -r code pkg/ && cp Documentation_template.md pkg/
(cd pkg && zip -qr ../Local_Aura_Farmers_submission.zip output code Documentation_template.md -x "*/__pycache__/*")
```
The `candidate_pairs.tsv` in the zip must come from the same run as `matching_results.tsv`.
