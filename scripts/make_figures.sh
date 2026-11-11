#!/bin/bash
# Regenerate the figures of the Experiments section from the stored results (no dataset or long run needed).
set -e
R="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python3}"
cd "$R"
mkdir -p figures
$PY scripts/paper_figures.py
ls figures
