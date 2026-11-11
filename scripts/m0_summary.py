"""Collect every dataset's summary.json into results/m0_data_summary.{csv,md} (small, committed)."""
import os as _os, sys as _sys  # repository paths: src/pwexp/paths.py
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
from pwexp.paths import PWSUM_ROOT, DATA_DIR, RUST_BIN  # noqa: E402,F401

import csv
import json
import os

ROOT = DATA_DIR
OUT = os.path.join(os.path.dirname(__file__), "..", "results")

rows = []
for name in ("SF", "ACS", "Taxi"):
    s = json.load(open(f"{ROOT}/{name}/processed/summary.json"))
    rows.append(dict(dataset=name, privacy_unit=s["privacy_unit"], N_raw=s["N_raw"], N=s["N"], X_max=s["X_max"],
                     n_truncated=s["n_truncated"], truncated_fraction=s["truncated_fraction"],
                     mean=s["mean_normalised"], var=s["var_normalised"],
                     clipping_bias_relative=s["clipping_bias_relative"], file=s["processed_file"],
                     sha256=s["processed_sha256"]))
syn = json.load(open(f"{ROOT}/Synthetic/processed/summary.json"))
for r in syn["files"]:
    rows.append(dict(dataset=f"Synthetic/{r['dist']}", privacy_unit="one record", N_raw=r["N"], N=r["N"], X_max=1.0,
                     n_truncated=0, truncated_fraction=0.0, mean=r["mean"], var=r["var"],
                     clipping_bias_relative=0.0, file=r["file"], sha256=r["sha256"]))

os.makedirs(OUT, exist_ok=True)
with open(f"{OUT}/m0_data_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
with open(f"{OUT}/m0_data_summary.md", "w") as f:
    f.write("# Data summary\n\nmean/var are of the normalised values in [0,1]. Clipping bias = "
            "(mean after clipping - mean before)/mean before. Synthetic gridmid is generated per K and not stored.\n\n"
            "| dataset | privacy unit | N raw | N | X_max | clipped | clipped % | mean | var | clip bias | sha256 |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        f.write(f"| {r['dataset']} | {r['privacy_unit']} | {r['N_raw']:,} | {r['N']:,} | {r['X_max']:,g} | "
                f"{r['n_truncated']:,} | {r['truncated_fraction']:.4%} | {r['mean']:.6f} | {r['var']:.6f} | "
                f"{r['clipping_bias_relative']:.4%} | `{r['sha256'][:16]}…` |\n")
print(open(f"{OUT}/m0_data_summary.md").read())
