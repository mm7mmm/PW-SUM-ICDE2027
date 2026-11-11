"""V-B: cross-method MSE on the same inputs. Sweeps come from params/pw_params_pu.json (eps, scale, gamma).

Inputs. Every method reads the same array for a given (dataset, N). Stored file when it exists (Uniform N in 1e4..1e7,
full real datasets); otherwise a without-replacement subsample y[perm[:N]] with a fixed permutation of the dataset
(nested in N). Gridmid is never used here: it depends on the method's own K (see pu_decomp).

PW-Sum. Primary: MSE = E[W^2]/K^2 + sum f_i(1-f_i)/K^2, with W from pu_noise (10^4 exact draws) and the second term
computed exactly from the inputs. 'mse_direct' additionally simulates the rounding of every user with the same W.
Baselines. Direct simulation, REPS repetitions, seeds SeedSequence([SEED, cfg, chunk]) shared by all methods.
Rounding is simulated exactly on distinct input values: sum over users of Ber(f_v) with n_v users equal to v is
Binomial(n_v, f_v); only sums and histograms are needed. The sum of N Laplace(1/eps) (LDP) is the difference of two
Gamma(N, 1/eps) variables, also exact.
Writes results/pu/cross.csv, results/pu/baseline_params.json."""
import os as _os, sys as _sys  # repository paths: src/pwexp/paths.py
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
from pwexp.paths import PWSUM_ROOT, DATA_DIR, RUST_BIN  # noqa: E402,F401

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mpc  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from pwexp import baselines as bl  # noqa: E402

SEED = 20261110
SUB_SEED = 20261112
DATA = DATA_DIR
REPS, CHUNK = 10000, 250
METHODS = ["CDP", "LDP", "GKMPS", "BBGN", "ID-Sum", "Prio3-A", "Prio3-B"]
FULLFILE = {"SF": f"{DATA}/SF/processed/values.npy", "ACS": f"{DATA}/ACS/processed/values.npy", "Taxi": f"{DATA}/Taxi/processed/values.npy"}
D = {}          # (dataset, N) -> compressed input, filled before the pool forks
PARAMS = {}     # (cfg index, method) -> parameters


class SafeGKMPS(bl.GKMPS):
    """Candidates with gamma_G so small that p_1 rounds to 1.0 in float64 have unbounded message counts; they are
    discarded as infinitely expensive instead of raising ZeroDivisionError. All other candidates are unchanged."""

    def _cost(self, eps, delta, N, K, gG):
        try:
            return super()._cost(eps, delta, N, K, gG)
        except ZeroDivisionError:
            es = (1 - gG) * eps
            return {"msgs": math.inf, "bits": 0, "pc": math.exp(-es / (K + 1)), "ell": 0, "bytes": math.inf,
                    "eps_star": es, "eps1": 0.0, "nS": 0, "Gamma": 0}


def method(name):
    return {"CDP": bl.CDP, "LDP": bl.LDP, "GKMPS": SafeGKMPS, "BBGN": bl.BBGN, "ID-Sum": bl.IDSum,
            "Prio3-A": lambda: bl.Prio3DP("A"), "Prio3-B": lambda: bl.Prio3DP("B")}[name]()


def load_input(d, N):
    if d == "Uniform":
        fn = f"{DATA}/Synthetic/processed/uniform_N{N}.npy"
        if os.path.exists(fn):
            return np.load(fn)
        src = np.load(f"{DATA}/Synthetic/processed/uniform_N10000.npy")
    else:
        src = np.load(FULLFILE[d])
        if N == len(src):
            return src
    idx = np.random.default_rng([SUB_SEED, ("Uniform", "SF", "ACS", "Taxi").index(d)]).permutation(len(src))
    return src[idx[:N]]


def compress(y):
    v, n = np.unique(y, return_counts=True)
    return {"v": v, "n": n, "N": int(n.sum()), "truth": float(np.sum(v * n)), "mean": float(y.mean())}


class Q:
    """Exact per-user randomised rounding of a compressed input on grid K."""

    def __init__(self, D0, K):
        self.K, self.n = K, D0["n"]
        z = D0["v"] * K
        self.b = np.floor(z).astype(np.int64)
        self.f = z - self.b
        self.single = self.n == 1
        self.nf = float(np.sum(self.n * self.f))
        self.X0 = int(np.sum(self.n * self.b))
        self.qvar = float(np.sum(self.n * self.f * (1 - self.f))) / K ** 2
        self.sum_x2 = float(np.sum(self.n * (self.b.astype(float) ** 2 + self.f * (2 * self.b + 1))))

    def draw(self, rng):
        c = np.empty_like(self.n)
        s = self.single
        c[s] = rng.random(int(s.sum())) < self.f[s]
        if not s.all():
            c[~s] = rng.binomial(self.n[~s], self.f[~s])
        return c

    def err_int(self, c):
        return float(c.sum()) - self.nf          # sum_i (x_i - K y_i)


def simulate(mn, p, D0, rng, n):
    N = D0["N"]
    if mn == "CDP":
        return rng.laplace(0, 1 / p["eps"], size=n)
    if mn == "LDP":
        sc = 1 / p["eps"]
        return rng.gamma(N, sc, n) - rng.gamma(N, sc, n)
    K = p["K"]
    q = Q(D0, K)
    out = np.empty(n)
    if mn == "ID-Sum":
        m = bl.IDSum()
        jj = np.arange(K + 1, dtype=np.int64)
        b = p["beta"]
        for r in range(n):
            c = q.draw(rng)
            g = (np.bincount(q.b, weights=q.n - c, minlength=K + 1)[:K + 1] + np.bincount(q.b + 1, weights=c, minlength=K + 2)[:K + 1]).astype(np.int64)
            T = rng.binomial(g, b) + m._ageo(rng, p, K + 1)
            out[r] = float(jj @ (T - p["mu"])) / b / K - D0["truth"]
        return out
    if mn == "BBGN":
        q_mod = p["q"]
        noise = bl.dlap(rng, p["alpha"], n)
        for r in range(n):
            c = q.draw(rng)
            X = q.X0 + int(c.sum())
            z = (X + int(noise[r])) % q_mod
            if z > (N * K + q_mod) / 2:
                z -= q_mod
            out[r] = (q.err_int(c) + (z - X)) / K
        return out
    pr, k = {"GKMPS": (p.get("pc"), 1), "Prio3-A": (p.get("p"), 1), "Prio3-B": (p.get("p"), 2)}[mn]
    noise = bl.dlap(rng, pr, (n, k)).sum(axis=1)
    for r in range(n):
        out[r] = (q.err_int(q.draw(rng)) + float(noise[r])) / K
    return out


def work(task):
    i, mn, k, n, key = task
    rng = np.random.default_rng(np.random.SeedSequence([SEED, i, k]))
    return i, mn, k, simulate(mn, PARAMS[(i, mn)], D[key], rng, n)


def select(task):
    i, mn, r = task
    N, eps, g = r["N"], r["eps"], r["gamma"]
    return i, mn, method(mn).select_params(eps, 1 / N ** 2, N, g)


def jsonable(p):
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in p.items()}


def stats(e):
    R = len(e)
    s = e ** 2
    h = 1.96 * s.std(ddof=1) / math.sqrt(R)
    return R, float(s.mean()), float(s.mean() - h), float(s.mean() + h), float(e.mean()), float(1.96 * e.std(ddof=1) / math.sqrt(R))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    os.makedirs(f"{ROOT}/results/pu/raw", exist_ok=True)
    cfgs = json.load(open(f"{ROOT}/params/pw_params_pu.json"))
    if args.only:
        cfgs = [r for r in cfgs if r["data"] in args.only.split(",")]
    t0 = time.time()
    for r in cfgs:
        kk = (r["data"], r["N"])
        if kk not in D:
            D[kk] = compress(load_input(*kk))
            assert D[kk]["N"] == r["N"], kk
    print("inputs ready", {k: (v["N"], len(v["v"])) for k, v in D.items()}, f"{time.time() - t0:.0f}s", flush=True)
    tk = [(i, mn, r) for i, r in enumerate(cfgs) for mn in METHODS]
    with mpc.Pool(args.workers) as pool:
        for i, mn, p in pool.imap_unordered(select, tk):
            PARAMS[(i, mn)] = p
    json.dump([{"data": cfgs[i]["data"], "N": cfgs[i]["N"], "eps": cfgs[i]["eps"], "gamma": cfgs[i]["gamma"], "method": mn, **jsonable(p)}
               for (i, mn), p in sorted(PARAMS.items())], open(f"{ROOT}/results/pu/baseline_params.json", "w"))
    print(f"baseline parameters selected {time.time() - t0:.0f}s", flush=True)
    tasks = [(i, mn, k, min(CHUNK, args.reps - k * CHUNK), (r["data"], r["N"]))
             for i, r in enumerate(cfgs) for mn in METHODS for k in range((args.reps + CHUNK - 1) // CHUNK)]
    tasks.sort(key=lambda t: -len(D[t[4]]["v"]))
    res = {}
    with mpc.Pool(args.workers) as pool:          # forked after PARAMS and D are complete
        for j, (i, mn, k, e) in enumerate(pool.imap_unordered(work, tasks, chunksize=1)):
            res.setdefault((i, mn), {})[k] = e
            if (j + 1) % 1000 == 0:
                print(f"{time.time() - t0:.0f}s {j + 1}/{len(tasks)}", flush=True)
    while not all(os.path.exists(f"{ROOT}/results/pu/noise/N{r['N']}_eps{r['eps']}_g{r['gamma']}.npz") for r in cfgs):
        time.sleep(30)  # the PW-Sum noise draws (pu_noise) may still be running
    rows = []
    for i, r in enumerate(cfgs):
        D0 = D[(r["data"], r["N"])]
        eps, gamma, N = r["eps"], r["gamma"], r["N"]
        base = {"dataset": r["data"], "N": N, "eps": eps, "gamma": gamma, "sweeps": r["sweeps"], "central": 2 / eps ** 2, "target": 2 * (1 + gamma) / eps ** 2}
        z = np.load(f"{ROOT}/results/pu/noise/N{N}_eps{eps}_g{gamma}.npz")
        W = z["W"].astype(float)
        K = r["K"]
        q = Q(D0, K)
        rng = np.random.default_rng(np.random.SeedSequence([SEED, i, 10 ** 6]))
        direct = np.array([(q.err_int(q.draw(rng)) + W[j]) / K for j in range(len(W))])
        Rw, mn2, lo, hi, bias, bh = stats(W / K)
        Rd, md, dlo, dhi, _, _ = stats(direct)
        rows.append({**base, "method": "PW-Sum", "K": K, "feasible": True, "preconditions_ok": True, "reps": Rw, "mse": mn2 + q.qvar, "ci_lo": lo + q.qvar, "ci_hi": hi + q.qvar,
                     "bias": bias, "bias_ci": bh, "noise_term": mn2, "quant_term": q.qvar, "analytic": r["mse_cert"], "mse_direct": md, "direct_lo": dlo, "direct_hi": dhi,
                     "worst_quant": N / (4 * K * K), "bytes_per_user": r["bytes_per_user"]})
        for mn in METHODS:
            e = np.concatenate([res[(i, mn)][k] for k in sorted(res[(i, mn)])])
            np.save(f"{ROOT}/results/pu/raw/{r['data']}_N{N}_eps{eps}_g{gamma}_{mn}.npy", e.astype(np.float32))
            p = PARAMS[(i, mn)]
            m = method(mn)
            KK = p.get("K", "")
            qv = Q(D0, KK).qvar if KK != "" else 0.0
            if mn == "ID-Sum":
                an = m.analytic_mse(p, qvar=qv, sum_x2=Q(D0, KK).sum_x2)
            elif KK != "":
                an = m.analytic_mse(p, qvar=qv)
            else:
                an = m.analytic_mse(p)
            R, ms, lo, hi, bias, bh = stats(e)
            rows.append({**base, "method": mn, "K": KK, "feasible": bool(p.get("feasible", True)), "preconditions_ok": bool(p.get("preconditions_ok", True)), "reps": R, "mse": ms, "ci_lo": lo, "ci_hi": hi,
                         "bias": bias, "bias_ci": bh, "noise_term": ms - qv, "quant_term": qv, "analytic": an, "mse_direct": "", "direct_lo": "", "direct_hi": "", "worst_quant": "", "bytes_per_user": p.get("bytes", "")})
    with open(f"{ROOT}/results/pu/cross.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", len(rows), "rows", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
