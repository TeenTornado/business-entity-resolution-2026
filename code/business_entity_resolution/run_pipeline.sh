#!/usr/bin/env bash
# End-to-end: raw TSVs -> parsed cache -> blocking -> stage-1 matcher -> stage-2 collective matcher -> test outputs.
# Usage: bash run_pipeline.sh <dataset_dir> <work_dir> <output_dir>
#   dataset_dir: folder containing train/ and test/ (student_resource/dataset)
set -euo pipefail
DATA=${1:-../../student_resource/dataset}
WORK=$(mkdir -p "${2:-./work}" && cd "${2:-./work}" && pwd)
OUT=$(mkdir -p "${3:-../../output}" && cd "${3:-../../output}" && pwd)
SRC="$(cd "$(dirname "$0")/src" && pwd)"
NTRAIN=800000
PRUNE=0.004
cd "$SRC"

# 1. cache raw train files and learn the Indic-script -> Latin word dictionary (train GT only)
python3 cache_raw.py --data "$DATA" --work "$WORK" --split train
python3 learn_translit.py --work "$WORK" --out "$WORK/translit.json"

# 2. normalise all records
python3 preprocess.py --data "$DATA" --work "$WORK" --split train --translit "$WORK/translit.json"
python3 preprocess.py --data "$DATA" --work "$WORK" --split test  --translit "$WORK/translit.json"

# 3. simulate test-like sibling distractor groups in the TRAIN pool (train data only)
python3 simulate_siblings.py --work "$WORK" --translit "$WORK/translit.json"

# 4. candidate generation (IDF cosine top-30 + address-only top-10, within country label)
python3 blocking.py --work "$WORK" --split train
python3 blocking.py --work "$WORK" --split test

# 5. stage 1: candidate pruner (-> candidate_pairs.tsv) + pairwise LightGBM matcher
python3 train.py --work "$WORK" --model-dir "$WORK/stage1" --n-train $NTRAIN --prune-thr $PRUNE
python3 predict.py --work "$WORK" --model-dir "$WORK/stage1" --out "$WORK/stage1_out" --split test

# 6. stage 2: collective matcher (out-of-fold stage-1 scores + anchor/competitor group features)
python3 collective.py train --work "$WORK" --model-dir "$WORK/stage1" --out-dir "$WORK/collective" \
    --n-train $NTRAIN --prune-thr $PRUNE
python3 collective.py predict --work "$WORK" --out-dir "$WORK/collective" --test-feat "$WORK/test_pairfeat" \
    --out "$OUT" --candidates "$WORK/stage1_out/candidate_pairs.tsv"

# 7. validate the submission format
python3 "$(dirname "$DATA")/utils/validate_submission.py" \
    --matching "$OUT/matching_results.tsv" --candidate "$OUT/candidate_pairs.tsv" --test-dir "$DATA/test"
