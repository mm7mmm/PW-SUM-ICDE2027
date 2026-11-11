"""V-C: poisoning under the current parameter rule.
(a) PW-Sum, in-range attack, full pipeline with common random numbers (logic of m3b_poison.py): per repetition the honest
inputs, honest quantisation, Shuffler noise (V, xi) and the set of controlled users are shared between the clean and the
attacked run. A fraction c of users reports K ("upper") or 0 ("zero"). Measured: shift = attack estimate - clean estimate
(normalised units); Theorem 5 bound = |A| X_max = c N. 200 repetitions per case.
(b) Cross-method shift, by each method's estimator: in-range attack, and the protocol-level attack each message format allows
(LDP: report 1e3; GKMPS: 100 extra messages of +Delta per controlled user; BBGN: the coalition's shares add q/4 mod q).
PW-Sum, ID-Sum, Prio3Sum and CDP decode, reject or clip out-of-range input, so only the in-range attack applies.
Writes results/robust/poison.csv, poison_raw.csv, poison_methods.csv."""
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

import numpy as np  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from pwexp import baselines as bl  # noqa: E402
from pwexp import pw_core_task as pc  # noqa: E402
from pwexp import pw_sampler as smp  # noqa: E402

SEED, REPS = 20261115, 200
FRACS = (0.0, 0.001, 0.01, 0.05, 0.1, 0.3)
DATA = DATA_DIR
CASES = [("Uniform", 10 ** 5, f"{DATA}/Synthetic/processed/uniform_N100000.npy"), ("Uniform", 10 ** 6, f"{DATA}/Synthetic/processed/uniform_N1000000.npy"),
         ("SF", 42928, f"{DATA}/SF/processed/values.npy"), ("ACS", 2912790, f"{DATA}/ACS/processed/values.npy"), ("Taxi", 15167858, f"{DATA}/Taxi/processed/values.npy")]
PR = {}
Y = {}


def work(task):
    ci, k = task
    d, N, _ = CASES[ci]
    r = PR[ci]
    K = r["K"]
    y = Y[ci]
    rng = np.random.default_rng(np.random.SeedSequence([SEED, ci, k]))
    a, b = pc.weights(K, r["zeta"], "pw", 1.0)
    t = np.array(r["t"], np.int64)
    c = np.r_[0, np.ones(K - 1, np.int64), 0]
    kappa = int(np.arange(K + 1) @ (t + c))
    jj = np.arange(K + 1)
    _, _, V = smp.sample_s(a, b, t.astype(float), 1, rng, keep_v=1)
    U, _ = smp.integerize(V, rng.random(1))
    H = t + c + np.rint(U[0]).astype(np.int64)
    x = np.floor(y * K).astype(np.int64) + (rng.random(N) < (y * K - np.floor(y * K)))
    perm = rng.permutation(N)
    est = lambda xs: (int((np.bincount(xs, minlength=K + 1) + H) @ jj) - kappa) / K  # noqa: E731
    clean = est(x)
    truth = float(y.sum())
    out = []
    for f in FRACS:
        m = int(round(f * N))
        for att, val in (("upper", K), ("zero", 0)):
            xa = x.copy()
            xa[perm[:m]] = val
            e = est(xa)
            out.append({"case": f"{d}_N{N}", "dataset": d, "N": N, "K": K, "rep": k, "frac": f, "n_controlled": m, "attack": att, "truth": truth, "clean_est": clean, "attack_est": e,
                        "shift": e - clean, "bound": float(m), "within_bound": abs(e - clean) <= m + 1e-9, "shift_expected": float((val * m - x[perm[:m]].sum()) / K)})
    return out


def methods():
    rows = []
    for N, fn in ((10 ** 5, CASES[0][2]), (10 ** 6, CASES[1][2])):
        y = np.load(fn)
        perm = np.random.default_rng(SEED).permutation(N)
        d = 1 / N ** 2
        b = bl.BBGN().select_params(1.0, d, N, 0.1)
        import c2_comm  # noqa: E402  (installs the guarded GKMPS cost)
        g = bl.GKMPS().select_params(1.0, d, N, 0.1)
        for c in (0.001, 0.01, 0.05, 0.1, 0.3):
            mal = perm[:int(c * N)]
            hon = float(y[mal].sum())
            inrange = len(mal) - hon
            q, K = b["q"], b["K"]
            x_hon = int(np.floor(y * K).sum())
            corr = lambda v: v - q if v > (N * K + q) / 2 else v  # noqa: E731
            bb = (corr((x_hon + q // 4) % q) - corr(x_hon % q)) / K
            rows.append({"N": N, "frac": c, "n_controlled": len(mal), "bound_|A|": float(len(mal)), "in_range": inrange, "PW-Sum": inrange, "ID-Sum": inrange, "Prio3Sum": inrange, "CDP": inrange,
                         "LDP": len(mal) * 1e3 - hon, "GKMPS": len(mal) * 100 * (g["K"] + 1) / g["K"], "BBGN": bb, "BBGN_K": K, "GKMPS_K": g["K"]})
    return rows


def main():
    os.makedirs(f"{ROOT}/results/robust", exist_ok=True)
    params = json.load(open(f"{ROOT}/params/pw_params_pu.json"))
    for ci, (d, N, path) in enumerate(CASES):
        PR[ci] = next(x for x in params if x["data"] == d and x["N"] == N and x["eps"] == 1.0 and x["gamma"] == 0.1)
        Y[ci] = np.load(path)
    tasks = [(ci, k) for ci in range(len(CASES)) for k in range(REPS)]
    with mpc.Pool(40) as pool:
        res = [r for rr in pool.imap_unordered(work, tasks, chunksize=2) for r in rr]
    with open(f"{ROOT}/results/robust/poison_raw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0]))
        w.writeheader()
        w.writerows(res)
    summ = []
    for case in sorted({r["case"] for r in res}):
        for f in FRACS:
            for att in ("upper", "zero"):
                q = [r for r in res if r["case"] == case and r["frac"] == f and r["attack"] == att]
                sh = np.array([r["shift"] for r in q])
                rc = np.array([abs(r["clean_est"] - r["truth"]) / r["truth"] for r in q])
                ra = np.array([abs(r["attack_est"] - r["truth"]) / r["truth"] for r in q])
                summ.append({"case": case, "dataset": q[0]["dataset"], "N": q[0]["N"], "frac": f, "attack": att, "reps": len(q), "n_controlled": q[0]["n_controlled"], "bound": q[0]["bound"],
                             "shift_mean": sh.mean(), "shift_ci": 1.96 * sh.std(ddof=1) / math.sqrt(len(sh)), "max_abs_shift_over_bound": (np.max(np.abs(sh)) / q[0]["bound"]) if q[0]["bound"] else 0.0,
                             "all_within_bound": all(r["within_bound"] for r in q), "shift_equals_expected": all(abs(r["shift"] - r["shift_expected"]) < 1e-6 for r in q),
                             "rel_err_clean": rc.mean(), "rel_err_attack": ra.mean()})
    with open(f"{ROOT}/results/robust/poison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0]))
        w.writeheader()
        w.writerows(summ)
    rows = methods()
    with open(f"{ROOT}/results/robust/poison_methods.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("within bound:", all(s["all_within_bound"] for s in summ), "equals expected:", all(s["shift_equals_expected"] for s in summ), flush=True)
    for s in summ:
        print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in s.items() if k in ("case", "frac", "attack", "shift_mean", "max_abs_shift_over_bound", "rel_err_clean", "rel_err_attack")}, flush=True)


if __name__ == "__main__":
    main()
