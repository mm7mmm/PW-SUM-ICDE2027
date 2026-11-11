"""V-E: ablations of PW-Sum under the current parameter rule (eps = 1, delta = 1/N^2). Position vs uniform weights is in V-D.
(b) Shared xi vs independent per-coordinate rounding U_j = floor(V_j + xi_j), from the same exact draws V: integer error E W^2 / K^2,
    rate of sum U != 0, rate of a negative dummy count.
(c) beta = 0 (no sum term; the pairwise budget then allows alpha_j = eps for all j): best achievable real MSE over K, using the
    Gaussian-conditioning value (8/eps^2) sum_j (j - K/2)^2 (labelled approximation; checked at small K in results/m6/ablation_beta0_small_K.csv).
(d) Endpoint vs centre anchor of the rejection sampler: proposals and time per exact draw (30 draws, 300 s cap).
Writes results/ablation/*.csv."""
import os as _os, sys as _sys  # repository paths: src/pwexp/paths.py
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
from pwexp.paths import PWSUM_ROOT, DATA_DIR, RUST_BIN  # noqa: E402,F401

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

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

OUT = f"{ROOT}/results/ablation"
SEED = 20261116
CHUNK = 100


def cfgs():
    return {(r["N"], r["gamma"]): r for r in json.load(open(f"{ROOT}/params/pw_params_cost.json")) if r["eps"] == 1.0}


CF = {}


def xi_chunk(a):
    key, k, n = a
    r = CF[key]
    K = r["K"]
    al, be = pc.weights(K, r["zeta"], "pw", 1.0)
    t = np.array(r["t"], float)
    c = np.r_[0, np.ones(K - 1), 0]
    rng = np.random.default_rng(np.random.SeedSequence([SEED, key[0], int(key[1] * 100), k]))
    _, _, V = smp.sample_s(al, be, t, n, rng, keep_v=n)
    jj = np.arange(K + 1)
    W_sh = np.floor(V @ jj + rng.random(n))
    Uind = np.floor(V + rng.random(V.shape))
    W_ind = Uind @ jj
    return key, W_sh, W_ind, int(np.sum(Uind.sum(axis=1) != 0)), int(np.sum(np.any(t + c + Uind < 0, axis=1)))


def anchor(a):
    key, kind = a
    r = CF[key]
    K = r["K"]
    al, be = pc.weights(K, r["zeta"], "pw", 1.0)
    t = np.array(r["t"], float)
    an = 0 if kind == "endpoint" else K // 2
    rng = np.random.default_rng(np.random.SeedSequence([SEED + 1, key[0], int(key[1] * 100)]))
    ts, props = [], 0
    T0 = time.time()
    while len(ts) < 30 and time.time() - T0 < 300:
        t1 = time.perf_counter()
        _, p, _, _ = smp.draw_one(al, be, t, rng, an, 256 if an == 0 else 4096)
        ts.append(time.perf_counter() - t1)
        props += p
    return {"N": key[0], "gamma": key[1], "K": K, "anchor": kind, "draws": len(ts), "capped": len(ts) < 30, "acceptance": len(ts) / props if props else float("nan"), "mean_s": float(np.mean(ts)) if ts else float("nan"), "median_s": float(np.median(ts)) if ts else float("nan")}


def write(name, rows):
    with open(f"{OUT}/{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    CF.update(cfgs())
    plan = {(N, g): (2000 if N <= 10 ** 6 else 500) for N in (10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7) for g in (0.1, 0.5)}
    tasks = [(key, k, CHUNK) for key, R in plan.items() for k in range(R // CHUNK)]
    res = {}
    with mpc.Pool(90) as pool:
        for key, a, b, c, d in pool.imap_unordered(xi_chunk, tasks):
            res.setdefault(key, []).append((a, b, c, d))
        an = pool.map(anchor, [((N, g), kd) for (N, g) in ((10 ** 4, 0.1), (10 ** 5, 0.1), (10 ** 6, 0.1), (10 ** 5, 0.5), (10 ** 6, 0.5)) for kd in ("endpoint", "centre")], chunksize=1)
    rows = []
    for key in sorted(res):
        K = CF[key]["K"]
        Ws = np.concatenate([x[0] for x in res[key]])
        Wi = np.concatenate([x[1] for x in res[key]])
        R = len(Ws)
        rows.append({"N": key[0], "gamma": key[1], "K": K, "R": R, "shared_noise_mse": float(np.mean(Ws ** 2)) / K ** 2, "indep_noise_mse": float(np.mean(Wi ** 2)) / K ** 2,
                     "ratio_indep_over_shared": float(np.mean(Wi ** 2) / np.mean(Ws ** 2)), "indep_sumU_nonzero_rate": sum(x[2] for x in res[key]) / R, "indep_H_negative_rate": sum(x[3] for x in res[key]) / R,
                     "indep_extra_theory": float(np.sum(np.arange(K + 1) ** 2) / 6) / K ** 2})
    write("shared_xi", rows)
    write("anchor", an)
    rows = []
    for N in (10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7):
        best = min(((8 * (k * (k + 1) * (k + 2) / 12) + 0.25) / k ** 2 + N / (4 * k * k), k) for k in range(1, 20000))
        rows.append({"N": N, "eps": 1.0, "beta0_best_mse_approx": best[0], "beta0_best_K": best[1], "ratio_to_central": best[0] / 2.0})
    write("beta0_best", rows)


if __name__ == "__main__":
    main()
