"""V-D step 2: communication at the same (eps, delta) and the same MSE target. Analytic bytes per user, all links.
PW-Sum (eq:cost), uniform-weight PW-Sum (own best K), GKMPS with the Theorem 2 split and with the optimised split kappa
(s1_gkmps_kappa.work; best-found), BBGN Cor. 6.1, Prio3Sum DAP framing (checked against the measured report sizes), ID-Sum.
GKMPS x 0.2 is the paper's reported best-case reduction, an estimate that is not reproduced.
ID-Sum never reaches the target here, so its achievable trade-off is written separately (lower envelope over K, beta).
Writes results/cost/comm.csv, results/cost/idsum_pareto.csv."""
import os as _os, sys as _sys  # repository paths: src/pwexp/paths.py
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
from pwexp.paths import PWSUM_ROOT, DATA_DIR, RUST_BIN  # noqa: E402,F401

import csv
import json
import math
import multiprocessing as mpc
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from pwexp import baselines as bl  # noqa: E402

_orig = bl.GKMPS._cost


def _safe(self, eps, delta, N, K, gG):
    try:
        return _orig(self, eps, delta, N, K, gG)
    except ZeroDivisionError:
        es = (1 - gG) * eps
        return {"msgs": math.inf, "bits": 0, "pc": math.exp(-es / (K + 1)), "ell": 0, "bytes": math.inf, "eps_star": es, "eps1": 0.0, "nS": 0, "Gamma": 0}


bl.GKMPS._cost = _safe
import s1_gkmps_kappa as s1  # noqa: E402


def row(N, eps, g, method, bytes_, msgs, K, mse, feasible, **kw):
    return {"N": N, "eps": eps, "gamma": g, "method": method, "bytes_per_user": bytes_, "msgs_per_user": msgs, "K": K, "mse_worst": mse, "R_worst": mse * eps ** 2 / 2, "feasible": feasible, **kw}


def work(r):
    N, eps, g = r["N"], r["eps"], r["gamma"]
    d = 1 / N ** 2
    out = [row(N, eps, g, "PW-Sum", r["bytes_per_user"], 1 + r["M"] / N, r["K"], r["mse_cert"], True, M=r["M"]),
           row(N, eps, g, "PW-Uniform", r["uni_bytes"], 1 + r["uni_M"] / N, r["uni_K"], 2 / ((1 - r["uni_zeta"]) ** 2 * eps ** 2) + (N + 1) / (4 * r["uni_K"] ** 2), True, M=r["uni_M"])]
    k = s1.work((N, eps, g))
    out.append(row(N, eps, g, "GKMPS-Thm2", k["eq_bytes_per_user"], k["eq_msgs_per_user"], k["eq_K"], 2 * (1 + g) / eps ** 2, True))
    out.append(row(N, eps, g, "GKMPS", k["kappa_bytes_per_user"], k["kappa_msgs_per_user"], k["kappa_K"], k["kappa_mse_worst"], bool(k["kappa_mse_ok"]), kappa=k["kappa_best"]))
    out.append(row(N, eps, g, "GKMPS x0.2 (estimate)", k["sens_x0.2_bytes_per_user"], k["sens_x0.2_msgs_per_user"], k["kappa_K"], k["kappa_mse_worst"], bool(k["kappa_mse_ok"])))
    for name, m in (("BBGN", bl.BBGN()), ("Prio3-A", bl.Prio3DP("A")), ("Prio3-B", bl.Prio3DP("B")), ("ID-Sum", bl.IDSum())):
        p = m.select_params(eps, d, N, g)
        c = m.comm_cost(p)
        out.append(row(N, eps, g, name, c["bytes_per_user"], c["messages_per_user"], p["K"], p["mse_worst"], bool(p.get("feasible", True))))
    return out


def idsum_env(a):
    N, eps = a
    m = bl.IDSum()
    d = 1 / N ** 2
    b0 = 1 - math.exp(-eps / 2)
    pts = []
    for beta in np.unique(np.r_[b0, 1 - (1 - b0) * np.geomspace(1, 1e-9, 60), 1.0]):
        for K in np.unique(np.r_[np.arange(1, 257), np.geomspace(256, 65536, 60).astype(int)]):
            e = m._eval(eps, d, N, int(K), float(beta))
            pts.append((float(e["bytes"]), float(e["mse_worst"]), int(K), float(beta)))
    pts.sort()
    env, best = [], math.inf
    for b, ms, K, be in pts:
        if ms < best * (1 - 1e-9):
            best = ms
            env.append({"N": N, "eps": eps, "bytes_per_user": b, "mse_worst": ms, "R_worst": ms * eps ** 2 / 2, "K": K, "beta": be})
    return env


def main():
    os.makedirs(f"{ROOT}/results/cost", exist_ok=True)
    cfg = [r for r in json.load(open(f"{ROOT}/params/pw_params_cost.json")) if set(r["sweeps"].split("+")) & {"comm", "eps"}]
    with mpc.Pool(40) as p:
        res = p.map(work, cfg, chunksize=1)
        env = p.map(idsum_env, [(N, 1.0) for N in (10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7)], chunksize=1)
    rows = [x for q in res for x in q]
    keys = []
    for x in rows:
        keys += [k for k in x if k not in keys]
    with open(f"{ROOT}/results/cost/comm.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    flat = [x for q in env for x in q]
    with open(f"{ROOT}/results/cost/idsum_pareto.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0]))
        w.writeheader()
        w.writerows(flat)
    print(len(rows), "rows;", len(flat), "ID-Sum envelope points", flush=True)


if __name__ == "__main__":
    main()
