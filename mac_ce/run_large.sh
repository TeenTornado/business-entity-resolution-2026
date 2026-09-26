#!/usr/bin/env bash
# One-shot cross-encoder run on the MacBook Pro (Apple silicon, 24 GB).
# Usage (from the repo root):  bash mac_ce/run_large.sh
# Safe to re-run: every step skips work that is already done, training resumes from its
# latest checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
DATA=$ROOT/mac_ce/ce_data
MODEL=$ROOT/mac_ce/ce_model
RES=$ROOT/mac_ce/results
BASE=${BASE:-intfloat/multilingual-e5-large}
MAXPAIRS=${MAXPAIRS:-600000}
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$RES" "$MODEL"

echo "== 1. python env"
if [ ! -d .venv_ce ]; then python3 -m venv .venv_ce; fi
source .venv_ce/bin/activate
pip install -q --upgrade pip
pip install -q torch transformers sentencepiece pandas pyarrow numpy
python3 -c "import torch; assert torch.backends.mps.is_available(), 'MPS not available'; print('torch', torch.__version__, 'MPS ok')"

echo "== 2. data"
cd "$DATA"
for f in ce_train ce_val ce_test; do
  if [ ! -f $f.parquet ]; then cat $f.parquet.part* > $f.parquet; fi
done
cd "$ROOT"
python3 -c "
import pandas as pd
for f in ['ce_train','ce_val','ce_test']:
    d=pd.read_parquet('$DATA/'+f+'.parquet'); print(f, len(d), list(d.columns))
"

echo "== 3. model download (once)"
python3 -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification as M; AutoTokenizer.from_pretrained('$BASE'); M.from_pretrained('$BASE', num_labels=1); print('model ready: $BASE')"

echo "== 4. train (resumable; keep the lid open / mac awake)"
if [ ! -f "$MODEL/final/config.json" ]; then
  caffeinate -dimsu python3 code/business_entity_resolution/src/ce.py train --data "$DATA" --out "$MODEL" \
      --base "$BASE" --max-pairs "$MAXPAIRS" --epochs 1 --bs 16 --lr 2e-5 --max-len 96 --ckpt-every 2000 \
      2>&1 | tee -a "$RES/train.log"
fi

echo "== 5. score uncertain validation + test pairs"
caffeinate -dimsu python3 code/business_entity_resolution/src/ce.py score --data "$DATA" --model "$MODEL" \
    --base "$BASE" --bs 16 --max-len 96 2>&1 | tee -a "$RES/score.log"
cp "$DATA/val_ce.npy" "$DATA/test_ce.npy" "$RES/"
ls -la "$RES"
echo "ALL DONE -> push mac_ce/results/ (see RUNBOOK.md step 5)"
