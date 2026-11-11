#!/bin/bash
# VI-B step 1: parameters, then exact noise draws for PW-Sum.
R="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python3}"
cd $R
mkdir -p logs results/pu
$PY scripts/s10_pu_params.py > logs/s10_pu_params.log 2>&1 || exit 1
$PY scripts/pu_noise.py --reps 10000 --workers 100 > logs/pu_noise.log 2>&1
echo finished > logs/pu_noise.done
