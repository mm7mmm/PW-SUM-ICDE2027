"""Parameters of PW-Sum under the parameter rule of the paper (Section V; radius equation and Theorem 1 with
delta_bar = max_{j != j'} (rho_j + rho_j')/2, no p*, L or Z_lb).

For each (N, eps, delta, gamma): enumerate every integer K with K^2 > (N+1)eps^2/(8 gamma); zeta = floor(zeta*(K)*1e6)/1e6
(public grid, step 1e-7, zeta > 0); alpha_j, t_j = ceil((2/alpha_j) ln(1 + (e^{alpha_j/2}-1)/delta)); M = sum t_j + K - 1;
choose K with the smallest C_user = (2 + M/N) ell(K), ell(K) = 48 + ceil(log2(K+1)/8) (also the smallest-M choice).
Enumeration is exhaustive: rho is non-increasing in alpha, alpha_j is non-decreasing in zeta and zeta*(K) < zeta_inf =
1-(1+gamma)^{-1/2}, so M(K) >= floor(K) := M at zeta_inf; beyond K_stop the floor is checked up to K_safe, where the
floor with alpha_j = eps, namely (K+1) t(eps) + K - 1, alone exceeds the best value.
The chosen point is verified with mpmath.iv (256 bits, outward rounding): rho_j <= delta for all j, t_j minimal, Theorem 2's
MSE bound <= 2(1+gamma)/eps^2, delta and gamma taken as exact decimals/rationals.
Writes params/pw_params_radius.{csv,json}."""
import os as _os, sys as _sys  # repository paths: src/pwexp/paths.py
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
from pwexp.paths import PWSUM_ROOT, DATA_DIR, RUST_BIN  # noqa: E402,F401

import csv
import json
import math
import multiprocessing as mpc
import os

import numpy as np
from mpmath import iv

R = PWSUM_ROOT
iv.prec = 256
REAL = {"SF": 42928, "ACS": 2912790, "Taxi": 15167858}
GAMMAS = (0.05, 0.1, 0.25, 0.5, 1.0)
PARETO = (0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
GRID = 1e-7


def ell(K):
    return 48 + -(-int(K).bit_length() // 8)


def weights(K, zeta, eps):
    j = np.arange(K + 1)
    return eps * (zeta + 2 * (1 - zeta) * np.minimum(j, K - j) / K)


def radius(a, delta):
    return np.ceil((2 / a) * np.log1p(np.expm1(a / 2) / delta))


def zeta_of(K, N, eps, g):
    x = 1 + g - eps ** 2 * (N + 1) / (8 * K * K)
    if x <= 1:
        return None
    z = math.floor((1 - x ** -0.5) / GRID) * GRID
    return z if z > 0 else None


def verify(N, eps, gs, dl, K, zeta, t):
    e, z = iv.mpf(eps), iv.mpf(repr(zeta))
    d = iv.mpf(1) / iv.mpf(N) ** (2 if dl == "1/N^2" else 1)
    ok, minimal, rmax = True, True, iv.mpf(0)
    for j in range(K + 1):
        a = e * (z + 2 * (1 - z) * iv.mpf(min(j, K - j)) / K)
        rho = iv.expm1(a / 2) / iv.expm1(a * int(t[j]) / 2) if hasattr(iv, "expm1") else (iv.exp(a / 2) - 1) / (iv.exp(a * int(t[j]) / 2) - 1)
        ok = ok and bool(rho.b <= d.a)
        rmax = rho.b if rho.b > rmax else rmax
        if t[j] > 1:
            r2 = (iv.exp(a / 2) - 1) / (iv.exp(a * (int(t[j]) - 1) / 2) - 1)
            minimal = minimal and bool(r2.a > d.b)
    mse = 2 / ((1 - z) ** 2 * e ** 2) + iv.mpf(N + 1) / (4 * iv.mpf(K) ** 2)
    V = 2 * (1 + iv.mpf(gs)) / e ** 2
    return {"iv_rho_ok": ok, "iv_t_minimal": minimal, "iv_mse_ok": bool(mse.b <= V.a), "rho_max_over_delta": float((rmax / d.a).b),
            "mse_cert": float(mse.b), "vmax": float(V.a), "mse_margin_rel": float(((V.a - mse.b) / V.a).a)}


def work(cfg):
    s, data, N, eps, dl, g = cfg
    delta = 1 / N ** 2 if dl == "1/N^2" else 1 / N
    kmin = math.floor(math.sqrt((N + 1) * eps * eps / (8 * g))) + 1
    zinf = 1 - (1 + g) ** -0.5
    best_b = best_m = None
    K, n_eval = kmin, 0
    while True:
        z = zeta_of(K, N, eps, g)
        if z is not None:
            t = radius(weights(K, z, eps), delta)
            M = int(t.sum()) + K - 1
            b = (2 * N + M) * ell(K) / N
            n_eval += 1
            if best_b is None or b < best_b[0]:
                best_b = (b, K, z, t, M)
            if best_m is None or M < best_m[0]:
                best_m = (M, K, z)
        fl = int(radius(weights(K, zinf, eps), delta).sum()) + K - 1
        if best_b is not None and fl > best_m[0] and (2 * N + fl) * 49 / N > best_b[0]:
            break
        K += 1
    k_stop = K
    te = float(radius(np.array([eps]), delta)[0])
    k_safe = K
    while not ((k_safe + 1) * te + k_safe - 1 > best_m[0] and (2 * N + (k_safe + 1) * te + k_safe - 1) * 49 / N > best_b[0]):
        k_safe += 1
    floor_ok = True
    for k in range(k_stop, k_safe + 1):
        fl = int(radius(weights(k, zinf, eps), delta).sum()) + k - 1
        floor_ok = floor_ok and fl > best_m[0] and (2 * N + fl) * 49 / N > best_b[0]
    b, K, z, t, M = best_b
    c = verify(N, eps, repr(g), dl, K, z, t)
    a_u = np.full(K + 1, eps * z)
    Mu = int(radius(a_u, delta).sum()) + K - 1
    return {"set": s, "data": data, "N": N, "eps": eps, "dlabel": dl, "delta": delta, "gamma": g, "K": K, "zeta": z,
            "t_end": int(t[0]), "t_center": int(t[K // 2]), "M": M, "M_over_N": M / N, "ell": ell(K), "bytes_per_user": b,
            "minM_K": best_m[1], "minM_M": best_m[0], "M_uniform_same_K": Mu, **c, "K_min": kmin, "K_stop": k_stop,
            "K_safe": k_safe, "floor_ok_to_K_safe": floor_ok, "n_K_evaluated": n_eval, "t": [int(x) for x in t]}


def main():
    cfgs = [("main", "synthetic", N, e, "1/N^2", g) for N in (10 ** 5, 10 ** 6) for e in (0.5, 1.0, 2.0) for g in GAMMAS]
    cfgs += [("dn", "synthetic", N, e, "1/N", g) for N in (10 ** 5, 10 ** 6) for e in (0.5, 1.0, 2.0) for g in GAMMAS]
    cfgs += [("real", d, N, e, "1/N^2", g) for d, N in REAL.items() for e in (0.5, 1.0, 2.0) for g in (0.1, 0.5)]
    cfgs += [("pareto", "synthetic", N, 1.0, "1/N^2", g) for N in (10 ** 5, 10 ** 6) for g in PARETO]
    cfgs += [("n1e7", "synthetic", 10 ** 7, 1.0, d, g) for d in ("1/N", "1/N^2") for g in (0.1, 0.5)]
    with mpc.Pool(24) as p:
        rows = p.map(work, cfgs, chunksize=1)
    keys = [k for k in rows[0] if k != "t"]
    with open(f"{R}/params/pw_params_radius.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    json.dump(rows, open(f"{R}/params/pw_params_radius.json", "w"))
    bad = [r for r in rows if not (r["iv_rho_ok"] and r["iv_mse_ok"] and r["floor_ok_to_K_safe"] and r["iv_t_minimal"])]
    print(len(rows), "configs; failed checks:", len(bad))
    for r in bad:
        print("BAD", {k: r[k] for k in keys})
    print("minM != minBytes:", sum(r["minM_K"] != r["K"] for r in rows))
    print("max rho/delta %.6f, min mse margin %.2e" % (max(r["rho_max_over_delta"] for r in rows), min(r["mse_margin_rel"] for r in rows)))
    for r in rows:
        if r["set"] == "main" and r["N"] == 10 ** 6 and r["eps"] == 1.0:
            print(f"g={r['gamma']}: K={r['K']} zeta={r['zeta']:.6f} t0={r['t_end']} tc={r['t_center']} M={r['M']} M/N={r['M_over_N']:.4f} "
                  f"B/u={r['bytes_per_user']:.2f} uni/PW={r['M_uniform_same_K'] / r['M']:.2f} scanned={r['n_K_evaluated']}")


if __name__ == "__main__":
    main()

