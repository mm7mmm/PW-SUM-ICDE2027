"""Parameters for the cost section (V-D): current paper rule (s9_params_radius.work), plus the best uniform-weight
variant alpha_j = zeta eps at the same (N, eps, delta, gamma): every K with K^2 > (N+1)eps^2/(8 gamma), zeta as in the rule,
t = eq:radius at alpha = zeta eps, minimise bytes. Stop: alpha <= eps zeta_inf gives M >= (K+1) t(eps zeta_inf) + K - 1, linear in K.
The chosen uniform point is checked with mpmath.iv (256 bits): rho <= delta, Theorem 2 MSE bound <= target.
Grids (delta = 1/N^2): comm: N in 1e3..1e7 x gamma in 0.05..2, eps = 1; eps: N = 1e6, gamma = 0.1, eps in 0.1..2;
sampler: N in 1e3..1e8 x gamma in {0.05, 0.1, 0.5}, eps = 1. Writes params/pw_params_cost.{csv,json}."""
import os as _os, sys as _sys  # repository paths: src/pwexp/paths.py
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
from pwexp.paths import PWSUM_ROOT, DATA_DIR, RUST_BIN  # noqa: E402,F401

import csv
import importlib.util
import json
import math
import multiprocessing as mpc
import os

import numpy as np
from mpmath import iv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sp = importlib.util.spec_from_file_location("s9", f"{ROOT}/scripts/s9_params_radius.py")
s9 = importlib.util.module_from_spec(sp)
sp.loader.exec_module(s9)
iv.prec = 256
NS = (10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7)
GAMMAS = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0)


def plan():
    seen = {}
    for N in NS:
        for g in GAMMAS:
            seen.setdefault((N, 1.0, g), set()).add("comm")
    for e in (0.1, 0.2, 0.5, 1.0, 2.0):
        seen.setdefault((10 ** 6, e, 0.1), set()).add("eps")
    for N in NS + (10 ** 8,):
        for g in (0.05, 0.1, 0.5):
            seen.setdefault((N, 1.0, g), set()).add("sampler")
    return seen


def uniform_best(N, eps, delta, g):
    kmin = math.floor(math.sqrt((N + 1) * eps * eps / (8 * g))) + 1
    zinf = 1 - (1 + g) ** -0.5
    tinf = float(s9.radius(np.array([eps * zinf]), delta)[0])
    best = None
    K = kmin
    while True:
        z = s9.zeta_of(K, N, eps, g)
        if z is not None:
            t = float(s9.radius(np.array([eps * z]), delta)[0])
            M = int((K + 1) * t + K - 1)
            b = (2 * N + M) * s9.ell(K) / N
            if best is None or b < best[0]:
                best = (b, K, z, int(t), M)
        fl = (K + 1) * tinf + K - 1
        if best is not None and (2 * N + fl) * 49 / N > best[0]:
            break
        K += 1
    return best


def verify_uniform(N, eps, gs, K, z, t):
    e, zz = iv.mpf(eps), iv.mpf(repr(z))
    d = iv.mpf(1) / iv.mpf(N) ** 2
    a = e * zz
    rho = (iv.exp(a / 2) - 1) / (iv.exp(a * t / 2) - 1)
    mse = 2 / ((1 - zz) ** 2 * e ** 2) + iv.mpf(N + 1) / (4 * iv.mpf(K) ** 2)
    return bool(rho.b <= d.a) and bool(mse.b <= (2 * (1 + iv.mpf(gs)) / e ** 2).a)


def job(k):
    N, e, g = k
    r = s9.work(("cost", "any", N, e, "1/N^2", g))
    b, K, z, t, M = uniform_best(N, e, 1 / N ** 2, g)
    r.update({"uni_K": K, "uni_zeta": z, "uni_t": t, "uni_M": M, "uni_bytes": b, "uni_ok": verify_uniform(N, e, repr(g), K, z, t)})
    return r


def main():
    seen = plan()
    keys = sorted(seen)
    with mpc.Pool(40) as p:
        rows = p.map(job, keys, chunksize=1)
    for r, k in zip(rows, keys):
        r["sweeps"] = "+".join(sorted(seen[k]))
    cols = [c for c in rows[0] if c != "t"]
    with open(f"{ROOT}/params/pw_params_cost.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    json.dump(rows, open(f"{ROOT}/params/pw_params_cost.json", "w"))
    bad = [r for r in rows if not (r["iv_rho_ok"] and r["iv_mse_ok"] and r["iv_t_minimal"] and r["floor_ok_to_K_safe"] and r["uni_ok"])]
    print(len(rows), "parameter sets; failed checks:", len(bad))
    for r in rows:
        print(f"N={r['N']:<10d} eps={r['eps']:<4} g={r['gamma']:<5} K={r['K']:<6d} M={r['M']:<9d} B/u={r['bytes_per_user']:<8.2f} uni: K={r['uni_K']:<6d} M={r['uni_M']:<11d} B/u={r['uni_bytes']:.2f} {r['sweeps']}", flush=True)


if __name__ == "__main__":
    main()
