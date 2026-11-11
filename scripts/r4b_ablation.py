"""Ablation (V-E): separates two choices that r4_ablation.py evaluates together.
(b') From the same exact draws V: (i) the paper's map, cumulative coordinates with one shared xi; (ii) cumulative coordinates with an independent xi_j
per coordinate, same inverse map L; (iii) direct rounding of each coordinate. Reported: noise term E W^2 / K^2, rate of sum U != 0, the largest |U_j - V_j|,
rate of draws with some |U_j - V_j| >= 2 (the margin b_j = t_j + 1 guarantees non-negativity only for smaller errors).
(d') Endpoint vs centre anchor with the SAME batch size (256 and 4096), 30 draws, 300 s cap. Writes results/ablation_v2/*.csv."""
import os as _os, sys as _sys  # repository paths: src/pwexp/paths.py
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
from pwexp.paths import PWSUM_ROOT, DATA_DIR, RUST_BIN  # noqa: E402,F401

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import csv  # noqa: E402
import json  # noqa: E402
import multiprocessing as mpc  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from pwexp import pw_core_task as pc  # noqa: E402
from pwexp import pw_sampler as smp  # noqa: E402

OUT = f"{ROOT}/results/ablation_v2"
SEED, CHUNK = 20261118, 100
CF = {}


def integerize_indep(V, xi):
    """smp.integerize with one offset per cumulative coordinate (xi has shape (n, K))."""
    K = V.shape[1] - 1
    z = np.cumsum(V[:, :0:-1], axis=1)[:, ::-1]
    P = np.cumsum(z, axis=1)
    Q = np.floor(P + xi)
    d = np.diff(np.concatenate([np.zeros((len(V), 1)), Q], axis=1), axis=1)
    U = np.empty_like(V)
    U[:, 0] = -d[:, 0]
    U[:, 1:K] = d[:, :K - 1] - d[:, 1:K]
    U[:, K] = d[:, K - 1]
    return U, Q[:, -1]


def chunk(a):
    key, k, n = a
    r = CF[key]
    K = r["K"]
    al, be = pc.weights(K, r["zeta"], "pw", 1.0)
    t = np.array(r["t"], float)
    rng = np.random.default_rng(np.random.SeedSequence([SEED, key[0], int(key[1] * 100), k]))
    _, _, V = smp.sample_s(al, be, t, n, rng, keep_v=n)
    jj = np.arange(K + 1)
    out = {}
    U1, _ = smp.integerize(V, rng.random(n))
    U2, _ = integerize_indep(V, rng.random((n, K)))
    U3 = np.floor(V + rng.random(V.shape))
    for name, U in (("shared", np.rint(U1)), ("cum_indep", np.rint(U2)), ("direct", U3)):
        e = np.abs(U - V).max(axis=1)
        out[name] = ((U @ jj).astype(float), np.sum(U.sum(axis=1) != 0), float(e.max()), int(np.sum(e >= 2)))
    return key, out


def anchor(a):
    key, kind, batch = a
    r = CF[key]
    K = r["K"]
    al, be = pc.weights(K, r["zeta"], "pw", 1.0)
    t = np.array(r["t"], float)
    an = 0 if kind == "endpoint" else K // 2
    rng = np.random.default_rng(np.random.SeedSequence([SEED + 1, key[0], int(key[1] * 100), batch]))
    ts, props = [], 0
    T0 = time.time()
    while len(ts) < 30 and time.time() - T0 < 300:
        t1 = time.perf_counter()
        _, p, _, _ = smp.draw_one(al, be, t, rng, an, batch)
        ts.append(time.perf_counter() - t1)
        props += p
    return {"N": key[0], "gamma": key[1], "K": K, "anchor": kind, "batch": batch, "draws": len(ts), "capped": len(ts) < 30, "acceptance": len(ts) / props if props else float("nan"), "mean_s": float(np.mean(ts)) if ts else float("nan")}


def write(name, rows):
    with open(f"{OUT}/{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    CF.update({(r["N"], r["gamma"]): r for r in json.load(open(f"{ROOT}/params/pw_params_cost.json")) if r["eps"] == 1.0})
    plan = {(N, g): (2000 if N <= 10 ** 6 else 500) for N in (10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7) for g in (0.1, 0.5)}
    tasks = [(key, k, CHUNK) for key, R in plan.items() for k in range(R // CHUNK)]
    res = {}
    with mpc.Pool(90) as pool:
        for key, o in pool.imap_unordered(chunk, tasks):
            res.setdefault(key, []).append(o)
        an = pool.map(anchor, [((N, g), kd, b) for (N, g) in ((10 ** 4, 0.1), (10 ** 5, 0.1), (10 ** 6, 0.1), (10 ** 5, 0.5), (10 ** 6, 0.5)) for kd in ("endpoint", "centre") for b in (256, 4096)], chunksize=1)
    rows = []
    for key in sorted(res):
        K = CF[key]["K"]
        for name in ("shared", "cum_indep", "direct"):
            W = np.concatenate([o[name][0] for o in res[key]])
            R = len(W)
            rows.append({"N": key[0], "gamma": key[1], "K": K, "method": name, "R": R, "noise_mse": float(np.mean(W ** 2)) / K ** 2, "sumU_nonzero_rate": sum(o[name][1] for o in res[key]) / R,
                         "max_abs_err": max(o[name][2] for o in res[key]), "rate_err_ge_2": sum(o[name][3] for o in res[key]) / R})
    write("rounding", rows)
    write("anchor", an)


if __name__ == "__main__":
    main()
