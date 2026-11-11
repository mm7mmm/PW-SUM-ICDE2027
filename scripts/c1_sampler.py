"""V-D step 1: efficiency of the exact rejection sampler of Algorithm pw2 (one valid sample per protocol run).
One single-threaded process per configuration, draws until TL seconds have passed and at least MIN_DRAWS are collected
(at most MAX_DRAWS). Records wall time of every draw and the proposal counts of the three acceptance steps.
Writes results/cost/sampler.csv."""
import os as _os, sys as _sys  # repository paths: src/pwexp/paths.py
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
from pwexp.paths import PWSUM_ROOT, DATA_DIR, RUST_BIN  # noqa: E402,F401

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import csv  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mpc  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from pwexp import pw_core_task as pc  # noqa: E402
from pwexp import pw_sampler as smp  # noqa: E402

SEED, TL, MIN_DRAWS, MAX_DRAWS, BATCH = 20261114, 420.0, 10, 5000, 256


def work(a):
    i, r = a
    rng = np.random.default_rng([SEED, i])
    al, be = pc.weights(r["K"], r["zeta"], "pw", r["eps"])
    t = np.array(r["t"], float)
    ts, props, acc1, acc2 = [], 0, 0, 0
    T0 = time.perf_counter()
    while (time.perf_counter() - T0 < TL or len(ts) < MIN_DRAWS) and len(ts) < MAX_DRAWS:
        t0 = time.perf_counter()
        _, p, a1, a2 = smp.draw_one(al, be, t, rng, 0, BATCH)
        ts.append(time.perf_counter() - t0)
        props += p
        acc1 += a1
        acc2 += a2
    ts = np.array(ts)
    n = len(ts)
    return {"N": r["N"], "eps": r["eps"], "gamma": r["gamma"], "K": r["K"], "zeta": r["zeta"], "M": r["M"], "draws": n, "proposals": props,
            "acceptance": n / props, "step_anchor": acc1 / props, "step_sum": acc2 / acc1, "step_box": n / acc2,
            "time_mean_s": float(ts.mean()), "time_ci_s": float(1.96 * ts.std(ddof=1) / math.sqrt(n)), "time_median_s": float(np.median(ts)), "time_p95_s": float(np.percentile(ts, 95)),
            "time_per_proposal_us": float(ts.sum() / props * 1e6), "total_s": float(ts.sum())}


def main():
    cfg = [r for r in json.load(open(f"{ROOT}/params/pw_params_cost.json")) if "sampler" in r["sweeps"].split("+")]
    os.makedirs(f"{ROOT}/results/cost", exist_ok=True)
    with mpc.Pool(len(cfg)) as p:
        rows = p.map(work, list(enumerate(cfg)), chunksize=1)
    rows.sort(key=lambda r: (r["gamma"], r["N"]))
    with open(f"{ROOT}/results/cost/sampler.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"g={r['gamma']:<5} N={r['N']:<10d} K={r['K']:<6d} n={r['draws']:<5d} acc={r['acceptance']:.3e} steps={r['step_anchor']:.3f}/{r['step_sum']:.4f}/{r['step_box']:.4f} "
              f"t_mean={r['time_mean_s']:.4g}s ±{r['time_ci_s']:.2g} med={r['time_median_s']:.4g}s p95={r['time_p95_s']:.4g}s", flush=True)


if __name__ == "__main__":
    main()
