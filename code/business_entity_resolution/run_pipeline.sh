#!/usr/bin/env bash
# End-to-end: raw TSVs -> parsed cache -> blocking -> train matcher -> test outputs.
# Usage: bash run_pipeline.sh <dataset_dir> <work_dir> <output_dir>
#   dataset_dir: folder containing train/ and test/ (student_resource/dataset)
set -euo pipefail
DATA=${1:-../../student_resource/dataset}
WORK=${2:-./work}
OUT=${3:-../../output}
SRC="$(cd "$(dirname "$0")/src" && pwd)"
MODEL="$WORK/model"
mkdir -p "$WORK" "$OUT"
cd "$SRC"

# 1. cache raw train files and learn the Indic-script -> Latin word dictionary (train GT only)
python3 cache_raw.py --data "$DATA" --work "$WORK" --split train
python3 learn_translit.py --work "$WORK" --out "$WORK/translit.json"

# 2. normalise all records
python3 preprocess.py --data "$DATA" --work "$WORK" --split train --translit "$WORK/translit.json"
python3 preprocess.py --data "$DATA" --work "$WORK" --split test  --translit "$WORK/translit.json"

# 2b. simulate test-like sibling distractor groups in the TRAIN pool (train data only)
python3 simulate_siblings.py --work "$WORK" --translit "$WORK/translit.json"

# 3. candidate generation
python3 blocking.py --work "$WORK" --split train
python3 blocking.py --work "$WORK" --split test

# 4. train candidate pruner + matcher; tune the F0.5 threshold on held-out train S1 entities
python3 train.py --work "$WORK" --model-dir "$MODEL"

# 5. test inference -> output/candidate_pairs.tsv, output/matching_results.tsv
python3 predict.py --work "$WORK" --model-dir "$MODEL" --out "$OUT" --split test

# 6. validate the submission format
python3 "$(dirname "$DATA")/utils/validate_submission.py" \
    --matching "$OUT/matching_results.tsv" --candidate "$OUT/candidate_pairs.tsv" --test-dir "$DATA/test"
