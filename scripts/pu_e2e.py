"""V-B: complete honest PW-Sum runs (rounding of each user, exact V, integerization, counts, estimator), 2000
repetitions, new parameters; reuses b3_e2e.work. Purpose: check the decomposition used by pu_cross, H_j >= 0, sum U = 0
and S_hat - sum x = floor(s(V)+xi). Encryption is omitted (it does not change the histogram). Writes results/pu/e2e_summary.csv."""
import os as _os, sys as _sys  # repository paths: src/pwexp/paths.py
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
from pwexp.paths import PWSUM_ROOT, DATA_DIR, RUST_BIN  # noqa: E402,F401

import csv
import json
import math
import multiprocessing as mpc
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, f"{ROOT}/scripts")
import numpy as np  # noqa: E402

import b3_e2e as b3  # noqa: E402

b3.SEED = 20261113
REPS, CHUNK = 2000, 50


def plan():
    p = [("Uniform", 10 ** 6, e, "uniform") for e in (0.5, 1.0, 2.0)]
    p += [("Uniform", 10 ** 6, 1.0, d) for d in ("tgauss", "zipf", "bimodal", "gridmid")]
    p += [("SF", 42928, 1.0, "SF"), ("ACS", 2912790, 1.0, "ACS"), ("Taxi", 15167858, 1.0, "Taxi")]
    return p


def main():
    rows = json.load(open(f"{ROOT}/params/pw_params_pu.json"))
    jobs = []
    for ci, (d, N, e, ds) in enumerate(plan()):
        r = next(x for x in rows if x["data"] == d and x["N"] == N and x["eps"] == e and x["gamma"] == 0.1)
        jobs.append((ci, r, ds))
    tasks = [(ci, k, CHUNK, r, ds) for ci, r, ds in jobs for k in range(REPS // CHUNK)]
    tasks.sort(key=lambda t: -t[3]["K"])
    res, bad = {}, {}
    with mpc.Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 40, maxtasksperchild=4) as pool:
        for ci, k, err, qerr, W, b, _ in pool.imap_unordered(b3.work, tasks):
            res.setdefault(ci, {})[k] = (err, qerr, W)
            for x, v in b.items():
                bad.setdefault(ci, {}).setdefault(x, 0)
                bad[ci][x] += v
    out = []
    for ci, r, ds in jobs:
        ks = sorted(res[ci])
        err = np.concatenate([res[ci][k][0] for k in ks])
        qerr = np.concatenate([res[ci][k][1] for k in ks])
        W = np.concatenate([res[ci][k][2] for k in ks]).astype(float)
        s2 = err ** 2
        h = 1.96 * s2.std(ddof=1) / math.sqrt(len(err))
        out.append({"dataset": ds, "N": r["N"], "eps": r["eps"], "gamma": r["gamma"], "K": r["K"], "reps": len(err), "mse": s2.mean(), "ci_lo": s2.mean() - h, "ci_hi": s2.mean() + h,
                    "bias": err.mean(), "bias_ci": 1.96 * err.std(ddof=1) / math.sqrt(len(err)), "quant_emp": np.mean(qerr ** 2), "noise_emp": np.mean(W ** 2) / r["K"] ** 2, "theorem2_bound": r["mse_cert"], **bad[ci]})
        print(out[-1], flush=True)
    with open(f"{ROOT}/results/pu/e2e_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)


if __name__ == "__main__":
    main()
