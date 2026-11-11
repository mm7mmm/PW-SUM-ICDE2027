"""Self-checks for every baseline (before any comparison):
 (a) unbiasedness: |mean error| / s.e. < 3;  (b) small scale: |empirical MSE / analytic MSE - 1| < 5 %
 (analytic uses the exact quantisation variance of the actual input; ID-Sum also the exact E sum x^2);
 (c) privacy source: the spec section (recorded as text).
Extra: GKMPS right inverse (A~C = I, dominance <= 1) for several Delta; message-level GKMPS simulation and share-level
BBGN simulation versus the shortcuts. Writes results/m5/selfcheck.csv and results/m5/shortcut_checks.csv.
"""
import os as _os, sys as _sys  # repository paths: src/pwexp/paths.py
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
from pwexp.paths import PWSUM_ROOT, DATA_DIR, RUST_BIN  # noqa: E402,F401

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import csv  # noqa: E402
import math  # noqa: E402
import multiprocessing as mpc  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from pwexp import baselines as bl  # noqa: E402

DATA = f"{DATA_DIR}/Synthetic/processed/uniform_N10000.npy"
SEED = 20261109
R = 20000
CHUNK = 1000
PRIV = {"CDP-Laplace": "Laplace mechanism, sensitivity 1 (specs/cdp_ldp_pw.md)",
        "LDP-Laplace": "per-user Laplace, eps-LDP (specs/cdp_ldp_pw.md)",
        "GKMPS": "arXiv:2109.13158 Thm 8 + Thm 2 recipe (Sec 4.5), Thm 10, Cor 11, Thm 12 (specs/gkmps.md)",
        "BBGN": "arXiv:2002.00817 Lemma 5.2 + Cor 6.1 eq.(4) (specs/bbgn.md)",
        "ID-Sum (LNF adaptation)": "arXiv:2504.07362 Thm 2 + Thm 6 (specs/id_sum.md)"}


def methods():
    return [bl.CDP(), bl.LDP(), bl.GKMPS(), bl.BBGN(), bl.IDSum(), bl.Prio3DP("A"), bl.Prio3DP("B")]


_PCACHE = {}


def work(task):
    import time
    mi, k = task
    t0 = time.time()
    m = methods()[mi]
    y = np.load(DATA)
    if mi not in _PCACHE:
        _PCACHE[mi] = m.select_params(1.0, 1 / len(y) ** 2, len(y), 0.1)
    p = _PCACHE[mi]
    t1 = time.time()
    rng = np.random.default_rng(np.random.SeedSequence([SEED, mi, k]))
    e = m.simulate_error(y, p, rng, CHUNK)
    print(f"task {m.name} chunk {k}: select {t1 - t0:.1f}s simulate {time.time() - t1:.1f}s", flush=True)
    return mi, k, e


def analytic(m, p, y):
    if isinstance(m, (bl.CDP, bl.LDP)):
        return m.analytic_mse(p)
    K = p["K"]
    qv = bl.qvar_exact(y, K)
    if isinstance(m, bl.IDSum):
        b = np.floor(y * K)
        f = y * K - b
        return m.analytic_mse(p, qvar=qv, sum_x2=float(np.sum(b * b + f * (2 * b + 1))))
    return m.analytic_mse(p, qvar=qv)


def gkmps_message_level(reps=20000, N=200, K=7, eps=1.0, gamma=0.5):
    m = bl.GKMPS()
    p = m.select_params(eps, 1 / N ** 2, N, gamma)
    K = p["K"]           # use the selected K (the message-count formula was evaluated there)
    D = K + 1
    atoms, G = bl.gkmps_atoms(D)
    e2 = min(1.0, p["gamma_G"] * eps) / 2
    rs = 3 * (1 + math.log(len(atoms) / (1 / N ** 2 / 2)))
    rhat, phat = 3 * (1 + math.log(1 / (1 / N ** 2 / 2))), math.exp(-0.2 * p["eps1"] / D)
    rng = np.random.default_rng(SEED + 77)
    x = rng.integers(0, K + 1, N)
    errs, msgs = np.empty(reps), np.empty(reps)
    for r in range(reps):
        tot = int(np.sum(x + 1))
        cnt = N
        zp = rng.negative_binomial(1 / N, 1 - p["pc"], N)
        zm = rng.negative_binomial(1 / N, 1 - p["pc"], N)
        tot += int(zp.sum() - zm.sum())
        cnt += int(zp.sum() + zm.sum())
        for a, ts, _ in atoms:
            ps = math.exp(-0.2 * e2 / (2 * ts))
            z = rng.negative_binomial(rs / N, 1 - ps, N).sum()
            if a == (-1, 1):
                z += rng.negative_binomial(rhat / N, 1 - phat, N).sum()
            tot += int(z) * sum(a)
            cnt += int(z) * len(a)
        errs[r] = tot - N - x.sum()
        msgs[r] = cnt / N
    return {"check": "GKMPS message-level vs DLap shortcut", "N": N, "K": K, "reps": reps,
            "var_message_level": errs.var(), "var_shortcut_analytic": bl.dlap_var(p["pc"]),
            "ratio": errs.var() / bl.dlap_var(p["pc"]), "mean_err": errs.mean(),
            "msgs_per_user_emp": msgs.mean(), "msgs_per_user_obs6": p["msgs"],
            "msgs_ratio": msgs.mean() / p["msgs"],
            "pass": abs(errs.var() / bl.dlap_var(p["pc"]) - 1) < 0.05 and abs(msgs.mean() / p["msgs"] - 1) < 0.02}


def bbgn_share_level(reps=20000, N=50, eps=1.0, gamma=0.5):
    m = bl.BBGN()
    p = m.select_params(eps, 1 / N ** 2, N, gamma)
    K, q, mm, a = p["K"], p["q"], p["m"], p["alpha"]
    rng = np.random.default_rng(SEED + 88)
    y = rng.random(N)
    errs = np.empty(reps)
    for r in range(reps):
        x = bl.quantise(y, K, rng)
        noise = rng.negative_binomial(1 / N, 1 - a, N) - rng.negative_binomial(1 / N, 1 - a, N)
        yy = (x + noise) % q
        shares = rng.integers(0, q, (N, mm))           # m+1 additive shares: m uniform, last fixes the sum
        last = (yy - shares.sum(axis=1)) % q
        z = int((shares.sum() + last.sum()) % q)
        if z > (N * K + q) / 2:
            z -= q
        errs[r] = z / K - y.sum()
    an = m.analytic_mse(p, qvar=bl.qvar_exact(y, K))
    return {"check": "BBGN share-level vs analytic", "N": N, "K": K, "reps": reps, "m": mm, "q": q,
            "mse_share_level": float(np.mean(errs ** 2)), "mse_analytic": an,
            "ratio": float(np.mean(errs ** 2)) / an, "mean_err": errs.mean(),
            "preconditions_ok(m>=3,N>=19,sigma>=1)": p["preconditions_ok"],
            "pass": abs(np.mean(errs ** 2) / an - 1) < 0.05}


def main():
    os.makedirs(f"{ROOT}/results/m5", exist_ok=True)
    ms = methods()
    tasks = [(mi, k) for mi in range(len(ms)) for k in range(R // CHUNK)]
    res = {}
    with mpc.Pool(40) as pool:
        for mi, k, e in pool.imap_unordered(work, tasks):
            res.setdefault(mi, {})[k] = e
        sc = pool.apply_async(gkmps_message_level)
        sb = pool.apply_async(bbgn_share_level)
        extra = [sc.get(), sb.get()]
    y = np.load(DATA)
    rows = []
    for mi, m in enumerate(ms):
        err = np.concatenate([res[mi][k] for k in sorted(res[mi])])
        p = m.select_params(1.0, 1 / len(y) ** 2, len(y), 0.1)
        an = analytic(m, p, y)
        se2 = err ** 2
        se = err.std(ddof=1) / math.sqrt(len(err))
        rows.append({"method": m.name, "N": len(y), "eps": 1.0, "delta": 1 / len(y) ** 2, "gamma": 0.1,
                     "K": p.get("K", ""), "feasible_at_target": p.get("feasible", True), "reps": len(err),
                     "mean_err": err.mean(), "mean_err_over_se": err.mean() / se, "unbiased_pass": abs(err.mean() / se) < 3,
                     "mse_emp": se2.mean(), "mse_ci_half": 1.96 * se2.std(ddof=1) / math.sqrt(len(se2)),
                     "mse_analytic_exact": an, "ratio": se2.mean() / an, "mse_pass_5pct": abs(se2.mean() / an - 1) < 0.05,
                     "privacy_source": PRIV.get(m.name, "VDAF Prio3Sum + discrete Laplace (specs/prio3sum_dp.md)")})
    with open(f"{ROOT}/results/m5/selfcheck.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    inv = []
    for D in (2, 3, 5, 8, 17, 33, 64):
        ok, dom = bl.gkmps_right_inverse(D)
        inv.append({"check": f"GKMPS right inverse D={D}", "A~C=I": ok, "max_dominance": dom, "pass": ok and dom <= 1 + 1e-12})
    allx = extra + inv
    keys = []
    for r in allx:
        keys += [k for k in r if k not in keys]
    with open(f"{ROOT}/results/m5/shortcut_checks.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(allx)
    for r in rows:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "privacy_source"})
    for r in allx:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
