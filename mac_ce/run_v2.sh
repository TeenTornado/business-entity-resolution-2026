#!/usr/bin/env bash
# v2 cross-encoder: 600k US/India hard pairs + 150k French pseudo-labelled pairs,
# scores the wider uncertain band (1.37M test + 65k val pairs). ~2.5 h train + ~1 h score.
cd "$(dirname "$0")/.."
ROOT=$PWD
DATA=$ROOT/mac_ce/ce_data2 MODEL=$ROOT/mac_ce/ce_model2 RES=$ROOT/mac_ce/results2 \
BASE=${BASE:-intfloat/multilingual-e5-base} MAXPAIRS=0 exec bash mac_ce/run_large.sh
