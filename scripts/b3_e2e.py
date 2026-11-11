"""Phase B3 (quantity E): end-to-end Monte Carlo of honest PW-Sum runs (no encryption; the ciphertext layer does not
change T, Theorem 1 step (5)).

Per repetition: (1) every user quantises y_i to x_i = floor(y_i K) + Bernoulli(frac(y_i K)) (y_i = 1 -> K);
(2) g = histogram of x; (3) Shuffler: exact V ~ q_t (endpoint anchor), xi ~ U[0,1), U = R(V; xi), H = t + c + U;
(4) Server: T = g + H, S_hat = sum_j j T_j - kappa, estimate = S_hat / K (normalised units).
Error = estimate - sum_i y_i. Checks per repetition: H >= 0, sum U = 0, S_hat - sum x = s(U) and
s(U) == floor(s(V) + xi) evaluated in float64 (mismatches counted, not hidden).

Usage: python scripts/b3_e2e.py --reps 2000 [--workers 100] [--only name,...]
Writes results/p1/e2e/<cfg>__<dataset>.npz (per-rep errors) and results/p1/e2e_summary.csv.
"""
import os as _os, sys as _sys  # repository paths: src/pwexp/paths.py
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
from pwexp.paths import PWSUM_ROOT, DATA_DIR, RUST_BIN  # noqa: E402,F401

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
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
from pwexp import synthetic as syn  # noqa: E402

SEED = 20261105
CHUNK = 50
DATA = DATA_DIR
OUT = f"{ROOT}/results/p1/e2e"


def plan():
    """(param set, data name used in params, N, eps, gamma, dataset to run)."""
    p = []
    for N in (10 ** 5, 10 ** 6):
        for e in (0.5, 1.0, 2.0):
            p.append(("main", "synthetic", N, e, 0.1, "uniform"))
    for d in ("tgauss", "zipf", "bimodal", "gridmid"):
        p.append(("main", "synthetic", 10 ** 6, 1.0, 0.1, d))
    for d in ("SF", "ACS", "Taxi"):
        for g in (0.1, 0.5):
            p.append(("real", d, None, 1.0, g, d))
    return p


def load_y(dataset, N, K):
    if dataset == "gridmid":
        return syn.generate("gridmid", N, K=K)
    if dataset in syn.STORED:
        return np.load(f"{DATA}/Synthetic/processed/{dataset}_N{N}.npy")
    return np.load(f"{DATA}/{dataset}/processed/values.npy")


def name(r, dataset):
    return f"{r['set']}_{r['data']}_N{r['N']}_eps{r['eps']}_g{r['gamma']}__{dataset}"


def work(task):
    ci, k, n, r, dataset = task
    rng = np.random.default_rng(np.random.SeedSequence([SEED, ci, k]))
    K, N = r["K"], r["N"]
    y = load_y(dataset, N, K)
    truth = float(y.sum())
    base = np.floor(y * K)
    frac = y * K - base
    base = base.astype(np.int64)
    a, b = pc.weights(K, r["zeta"], "pw", r["eps"])
    t = np.array(r["t"], dtype=np.int64)
    c = np.r_[0, np.ones(K - 1, dtype=np.int64), 0]
    kappa = int(np.sum(np.arange(K + 1) * (t + c)))
    jj = np.arange(K + 1, dtype=np.int64)
    t0 = time.perf_counter()
    _, _, V = smp.sample_s(a, b, t.astype(float), n, rng, keep_v=n)
    xi = rng.random(n)
    U, _ = smp.integerize(V, xi)
    U = np.rint(U).astype(np.int64)
    sV = V @ jj.astype(float)
    err, qerr, W = np.empty(n), np.empty(n), np.empty(n, dtype=np.int64)
    bad = {"H_neg": 0, "sumU_nonzero": 0, "W_ne_floor_sV_xi": 0}
    for i in range(n):
        x = base + (rng.random(N) < frac)
        g = np.bincount(x, minlength=K + 1)
        H = t + c + U[i]
        bad["H_neg"] += int(np.any(H < 0))
        bad["sumU_nonzero"] += int(U[i].sum() != 0)
        T = g + H
        S_hat = int(T @ jj) - kappa
        W[i] = S_hat - int(x.sum())
        bad["W_ne_floor_sV_xi"] += int(W[i] != np.floor(sV[i] + xi[i]))
        err[i] = S_hat / K - truth
        qerr[i] = x.sum() / K - truth
    return ci, k, err, qerr, W, bad, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=100)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    allp = {s: json.load(open(f"{ROOT}/params/pw_search_{s}.json")) for s in ("main", "real")}
    jobs = []
    for ci, (s, data, N, e, g, dataset) in enumerate(plan()):
        r = next(x for x in allp[s] if x["trunc"] == "rule" and x["objective"] == "bytes" and x["data"] == data
                 and (N is None or x["N"] == N) and x["eps"] == e and x["gamma"] == g and x["dlabel"] == "1/N^2")
        nm = name(r, dataset)
        if (args.only and nm not in args.only.split(",")) or os.path.exists(f"{OUT}/{nm}.npz"):
            continue
        jobs.append((ci, r, dataset, nm))
    tasks = [(ci, k, min(CHUNK, args.reps - k * CHUNK), r, d)
             for ci, r, d, _ in jobs for k in range((args.reps + CHUNK - 1) // CHUNK)]
    tasks.sort(key=lambda x: -(x[3]["K"] ** 2 + x[3]["N"] / 100))
    print(f"{len(jobs)} jobs, {len(tasks)} chunks", flush=True)
    acc = {ci: {"parts": {}, "left": sum(1 for x in tasks if x[0] == ci), "bad": {}, "cpu": 0.0} for ci, *_ in jobs}
    meta = {ci: (r, d, nm) for ci, r, d, nm in jobs}
    t0 = time.time()
    with mpc.Pool(args.workers, maxtasksperchild=4) as pool:
        for ci, k, err, qerr, W, bad, el in pool.imap_unordered(work, tasks):
            a = acc[ci]
            a["parts"][k] = (err, qerr, W)
            for x, v in bad.items():
                a["bad"][x] = a["bad"].get(x, 0) + v
            a["cpu"] += el
            a["left"] -= 1
            if a["left"] == 0:
                r, d, nm = meta[ci]
                ks = sorted(a["parts"])
                err = np.concatenate([a["parts"][q][0] for q in ks])
                qerr = np.concatenate([a["parts"][q][1] for q in ks])
                W = np.concatenate([a["parts"][q][2] for q in ks])
                np.savez_compressed(f"{OUT}/{nm}.npz", err=err, qerr=qerr, W=W,
                                    meta=json.dumps({"name": nm, "K": r["K"], "N": r["N"], "eps": r["eps"],
                                                     "gamma": r["gamma"], "dataset": d, "seed": [SEED, ci],
                                                     "chunk": CHUNK, "bad": a["bad"], "cpu_seconds": a["cpu"]}))
                print(f"{time.time() - t0:8.0f}s done {nm} n={len(err)} mse={np.mean(err ** 2):.4f} "
                      f"bias={err.mean():.4f} bad={a['bad']} cpu={a['cpu']:.0f}s", flush=True)
                del acc[ci]
    # summary of everything on disk
    rows = []
    for fn in sorted(os.listdir(OUT)):
        z = np.load(f"{OUT}/{fn}")
        m = json.loads(str(z["meta"]))
        err = z["err"]
        R = len(err)
        se2 = err ** 2
        rows.append({**{k: m[k] for k in ("name", "dataset", "N", "eps", "gamma", "K")}, "reps": R,
                     "mse": se2.mean(), "mse_sd": se2.std(ddof=1),
                     "mse_ci_lo": se2.mean() - 1.96 * se2.std(ddof=1) / np.sqrt(R),
                     "mse_ci_hi": se2.mean() + 1.96 * se2.std(ddof=1) / np.sqrt(R),
                     "bias": err.mean(), "bias_ci_half": 1.96 * err.std(ddof=1) / np.sqrt(R),
                     "err_sd": err.std(ddof=1), "quant_mse_emp": np.mean(z["qerr"] ** 2),
                     **{f"bad_{k}": v for k, v in m["bad"].items()}, "cpu_seconds": m["cpu_seconds"]})
    with open(f"{ROOT}/results/p1/e2e_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
