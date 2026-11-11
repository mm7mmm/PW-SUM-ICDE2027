#!/bin/bash
# VI-B step 2: cross-method simulation (waits for the noise files), then complete protocol runs.
R="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python3}"
cd $R
$PY scripts/pu_cross.py --workers 20 > logs/pu_cross.log 2>&1 || exit 1
$PY scripts/pu_e2e.py 90 > logs/pu_e2e.log 2>&1
echo finished > logs/pu_all.done
