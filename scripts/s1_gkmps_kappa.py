"""Supplement 1+2: GKMPS budget split and numerical-search sensitivity (results in NEW files).

Theorem 8 (arXiv:2109.13158) allows any eps* + eps1 + eps2 = eps, delta1 + delta2 = delta. Theorem 2's recipe uses
eps1 = eps2 = min{1, gamma_G eps}/2. Here eps1 = kappa * gamma_G * eps, eps2 = (1 - kappa) * gamma_G * eps with
kappa in {0.01, ..., 0.99}; Theorem 10 / Corollary 11 need eps1, eps2 in (0, 1), so (kappa, gamma_G) with eps_i >= 1
are skipped; delta1 = delta2 = delta/2 (unchanged). D-hat, D-tilde_s, t_s as in Theorem 2 / Theorem 12.
For each config the search is: K in a window [0.5 K0, 2 K0] around the Theorem-2-recipe optimum K0 (coarse stride, then
refinement around the best), gamma_G on 60 log-spaced points up to the MSE-feasible maximum, all 99 kappa; best-found
(not proven optimal).
Sensitivity for the paper's numerical search (SM H.1, Lemma 19; not reproduced): expected messages x 0.2,
"estimate from the reported value, not a reproduction". Writes results/m5/gkmps_kappa.csv.
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

KAPPAS = np.round(np.arange(0.01, 0.995, 0.01), 2)


def msgs_matrix(eps, delta, N, K, gG):
    """Expected messages/user for every kappa (vector), or None where invalid."""
    D = K + 1
    es = (1 - gG) * eps
    pc = math.exp(-es / D)
    e1 = KAPPAS * gG * eps
    e2 = (1 - KAPPAS) * gG * eps
    ok = (e1 < 1) & (e2 < 1) & (e1 > 0) & (e2 > 0)
    G = D * math.ceil(1 + math.log2(D))
    nS = 1 + 2 * (D - 1)
    d1 = d2 = delta / 2
    rhat = 3 * (1 + math.log(1 / d1))
    rs = 3 * (1 + math.log(nS / d2))
    phat = np.exp(-0.2 * e1 / D)
    m = np.arange(2, D + 1)
    ts = -(-G // m)
    ps = np.exp(-0.2 * e2[:, None] / (2 * ts[None, :]))
    p1 = np.exp(-0.2 * e2 / (2 * G))
    tot = 2 * pc / (1 - pc) + 2 * (rs * p1 / (1 - p1) + rhat * phat / (1 - phat)) + 6 * rs * np.sum(ps / (1 - ps), axis=1)
    out = 1 + tot / N
    out[~ok] = np.inf
    return out, pc


def g_max(eps, N, K, V):
    room = (V - N / (4 * K * K)) * K * K
    if room <= 0:
        return 0.0
    pm = (room + 1 - math.sqrt(2 * room + 1)) / room
    return 1 - (-(K + 1) * math.log(pm)) / eps


def best_at_K(eps, delta, N, K, V):
    gm = g_max(eps, N, K, V)
    if gm <= 0:
        return None
    bits = math.ceil(math.log2(2 * (K + 1) + 1))
    ell = bl.ell_bits(bits)
    best = None
    for gG in gm * np.geomspace(1e-3, 1, 60):
        if gG >= 1:
            continue
        mm, pc = msgs_matrix(eps, delta, N, K, gG)
        i = int(np.argmin(mm))
        if np.isfinite(mm[i]) and (best is None or mm[i] < best["msgs"]):
            best = {"K": K, "gamma_G": float(gG), "kappa": float(KAPPAS[i]), "msgs": float(mm[i]), "bytes": 2 * float(mm[i]) * ell,
                    "pc": pc, "ell": ell}
    return best


def work(cfg):
    N, eps, gamma = cfg
    delta = 1 / N ** 2
    V = 2 * (1 + gamma) / eps ** 2
    eq = bl.GKMPS().select_params(eps, delta, N, gamma)          # Theorem 2 equal split
    K0 = eq["K"]
    cand = {}
    Ks = sorted(set(int(k) for k in np.linspace(max(1, K0 // 2), 2 * K0, 40)))
    for K in Ks:
        cand[K] = best_at_K(eps, delta, N, K, V)
    kb = min((k for k in cand if cand[k]), key=lambda k: cand[k]["bytes"])
    step = max(1, (Ks[1] - Ks[0]) if len(Ks) > 1 else 1)
    for K in range(max(1, kb - step), kb + step + 1):
        if K not in cand:
            cand[K] = best_at_K(eps, delta, N, K, V)
    b = min((v for v in cand.values() if v), key=lambda v: v["bytes"])
    mse = bl.dlap_var(b["pc"]) / b["K"] ** 2 + N / (4 * b["K"] ** 2)
    return {"N": N, "eps": eps, "delta": delta, "gamma": gamma, "target_V": V,
            "eq_K": K0, "eq_gamma_G": eq["gamma_G"], "eq_msgs_per_user": eq["msgs"], "eq_bytes_per_user": eq["bytes"],
            "kappa_K": b["K"], "kappa_gamma_G": b["gamma_G"], "kappa_best": b["kappa"], "kappa_msgs_per_user": b["msgs"],
            "kappa_bytes_per_user": b["bytes"], "kappa_mse_worst": mse, "kappa_mse_ok": mse <= V * (1 + 1e-12),
            "bytes_ratio_kappa_over_eq": b["bytes"] / eq["bytes"],
            "sens_x0.2_msgs_per_user": 0.2 * b["msgs"], "sens_x0.2_bytes_per_user": 0.2 * b["bytes"],
            "sens_note": "estimate from the reported value, not a reproduction (SM H.1: up to 40-80% fewer messages, small Δ only)"}


def main():
    cfgs = [(N, e, g) for N in (10 ** 5, 10 ** 6) for e in (0.5, 1.0, 2.0) for g in (0.05, 0.1, 0.25, 0.5, 1.0)]
    with mpc.Pool(30) as pool:
        rows = pool.map(work, cfgs, chunksize=1)
    pw = {}
    for r in csv.DictReader(open(f"{ROOT}/params/pw_params.csv")):
        if r["set"] == "main":
            pw[(int(r["N"]), float(r["eps"]), float(r["gamma"]))] = float(r["bytes_per_user"])
    for r in rows:
        p = pw[(r["N"], r["eps"], r["gamma"])]
        r["pw_bytes_per_user"] = p
        r["gkmps_kappa_over_pw"] = r["kappa_bytes_per_user"] / p
        r["gkmps_sens_x0.2_over_pw"] = r["sens_x0.2_bytes_per_user"] / p
    os.makedirs(f"{ROOT}/results/m5", exist_ok=True)
    with open(f"{ROOT}/results/m5/gkmps_kappa.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"N={r['N']} eps={r['eps']} g={r['gamma']}: eq {r['eq_bytes_per_user']:.4g} B (K={r['eq_K']}) | kappa={r['kappa_best']} "
              f"{r['kappa_bytes_per_user']:.4g} B (K={r['kappa_K']}, gG={r['kappa_gamma_G']:.3g}) ratio {r['bytes_ratio_kappa_over_eq']:.3f} | "
              f"x0.2 {r['sens_x0.2_bytes_per_user']:.4g} B = {r['gkmps_sens_x0.2_over_pw']:.0f}x PW", flush=True)


if __name__ == "__main__":
    main()
