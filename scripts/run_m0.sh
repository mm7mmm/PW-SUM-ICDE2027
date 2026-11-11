set -e
R="$(cd "$(dirname "$0")/.." && pwd)"
D="${PWSUM_DATA:-$R/data}"
PY="${PYTHON:-python3}"
cd $R
mkdir -p logs
chmod a-w $D/*/raw/* 2>/dev/null || true
(cd $D && sha256sum */raw/*) > logs/raw_sha256_before.txt
rm -f $D/Synthetic/processed/*.npy
$PY scripts/m0_real.py sf acs taxi > logs/m0_real.log 2>&1
$PY scripts/m0_synthetic.py > logs/m0_synthetic.log 2>&1
(cd $D && sha256sum */raw/*) > logs/raw_sha256_after.txt
cmp logs/raw_sha256_before.txt logs/raw_sha256_after.txt && echo "raw unchanged"
$PY scripts/m0_summary.py
ls -la $D/*/raw $D/*/processed | head -60
