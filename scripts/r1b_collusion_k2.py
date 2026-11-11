"""Full-view privacy under static user-Server collusion, numerical check at K = 2.

Theorem 1 (step 5): given the colluders' state (their inputs, keys, submissions), the Server's view is a randomised
post-processing of T = g + H, and the colluders' contribution to g is a known constant. Hence the divergence between
neighbouring inputs of the target user is the divergence between U and U + (e_x - e_y), independent of how many
users collude. We compute this divergence numerically for the RELEASED integer histogram (not only the continuous
layer), at K = 2 where it is a 2-d integral:
  V = (v0, v1, v2), v0 = -v1 - v2, density ∝ exp(-1/2 sum alpha_j|v_j| - beta|s(v)|) on the box |v_j| <= t_j;
  Lemma 5 for K = 2: P1 = v1 + v2, P2 = v1 + 2 v2 = s(v), Q_a = floor(P_a + xi), U = (-Q1, 2Q1 - Q2, Q2 - Q1).
The xi-integral is exact (piecewise constant in xi); the v-integral is a midpoint rule with step h (reported at h and
h/2). delta_T(eps') = max over ordered pairs x != y of sum_u [P_U(u) - e^{eps'} P_U(u + e_x - e_y)]_+.
The continuous-layer delta_Y(eps') is computed on the same grid for reference.
This is numerical evidence, not a proof; the guarantee is Theorem 1 with the certified delta_bar.
Writes results/m3/collusion_K2.csv and figures/m3/collusion_*.{pdf,png}.
"""
import os as _os, sys as _sys  # repository paths: src/pwexp/paths.py
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
from pwexp.paths import PWSUM_ROOT, DATA_DIR, RUST_BIN  # noqa: E402,F401

import csv
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from pwexp import pw_core_task as pc  # noqa: E402
from pwexp import pw_search as ps  # noqa: E402

OUT, FIG = f"{ROOT}/results/robust_v2", f"{ROOT}/figures/robust_v2"
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
CASES = [(1.0, 0.3, 1e-3), (1.0, 0.1, 1e-3), (0.5, 0.3, 1e-3)]   # (eps, zeta, delta) at K = 2
EPS_GRID = np.round(np.arange(0.0, 1.6001, 0.001), 4)


def hockey(p, q):
    """delta(e) = sum_i [p_i - e^e q_i]_+ for every e in EPS_GRID, via sorting by the log ratio."""
    m = p > 0
    p, q = p[m], q[m]
    with np.errstate(divide="ignore"):
        lr = np.where(q > 0, np.log(p) - np.log(np.where(q > 0, q, 1)), np.inf)
    o = np.argsort(-lr)
    lr, p, q = lr[o], p[o], q[o]
    cp, cq = np.cumsum(p), np.cumsum(q)
    out = np.empty(len(EPS_GRID))
    for i, e in enumerate(EPS_GRID):
        k = np.searchsorted(-lr, -e, side="left")   # entries with lr > e
        out[i] = max(cp[k - 1] - math.exp(e) * cq[k - 1], 0.0) if k > 0 else 0.0
    return out


def compute(eps, zeta, delta, h):
    K = 2
    a, b = pc.weights(K, zeta, "pw", eps)
    t = np.ceil((2 / a) * np.log1p(np.expm1(a / 2) / delta))          # eq:radius (current paper rule)
    rho = np.sort(np.expm1(a / 2) / np.expm1(a * t / 2))              # eq:boundary
    dbar = float(0.5 * (rho[-1] + rho[-2]))                           # Theorem 1
    pair = all(0.5 * (a[i] + a[j]) + b * abs(i - j) <= eps * (1 + 1e-12) for i in range(3) for j in range(3) if i != j)   # eq:pairwise_budget
    c = {"dbar": dbar, "delta_ok": dbar <= delta, "pair_ok": pair}
    v1 = np.arange(-t[1] + h / 2, t[1], h)
    v2 = np.arange(-t[2] + h / 2, t[2], h)
    V1, V2 = np.meshgrid(v1, v2, indexing="ij")
    V0 = -V1 - V2
    S = V1 + 2 * V2
    inbox = np.abs(V0) <= t[0]
    logq = -0.5 * (a[0] * np.abs(V0) + a[1] * np.abs(V1) + a[2] * np.abs(V2)) - b * np.abs(S)
    w = np.where(inbox, np.exp(logq), 0.0)
    w /= w.sum()
    # integer layer
    P1, P2 = V1 + V2, S
    f1, f2 = np.floor(P1), np.floor(P2)
    A, B = 1 - (P1 - f1), 1 - (P2 - f2)
    lo, hi = np.minimum(A, B), np.maximum(A, B)
    qmin = int(min(f1.min(), f2.min())) - 1
    size = int(max(f1.max(), f2.max())) - qmin + 3
    i1, i2 = (f1 - qmin).astype(np.int64), (f2 - qmin).astype(np.int64)
    PQ = (np.bincount((i1 * size + i2).ravel(), (w * lo).ravel(), size * size)
          + np.bincount(((i1 + (A < B)) * size + i2 + (A >= B)).ravel(), (w * (hi - lo)).ravel(), size * size)
          + np.bincount(((i1 + 1) * size + i2 + 1).ravel(), (w * (1 - hi)).ravel(), size * size)).reshape(size, size)
    del i1, i2, A, B, lo, hi, f1, f2, P1, P2
    # delta_T for ordered pairs: shift in Q-space for w = e_x - e_y: (Q1, Q2) -> (Q1 - w0, Q2 + w2 - w0)
    res = {}
    for x in range(3):
        for y in range(3):
            if x == y:
                continue
            wv = np.zeros(3, int)
            wv[x] += 1
            wv[y] -= 1
            d1, d2 = -wv[0], wv[2] - wv[0]
            Sh = np.zeros_like(PQ)
            src = PQ[max(d1, 0):size + min(d1, 0), max(d2, 0):size + min(d2, 0)]
            Sh[max(-d1, 0):size + min(-d1, 0), max(-d2, 0):size + min(-d2, 0)] = src   # Sh[q] = PQ[q + d]
            res[(x, y)] = hockey(PQ.ravel(), Sh.ravel())
    dT = np.max(np.stack(list(res.values())), axis=0)
    # continuous layer: delta_Y(eps') = max over pairs of sum_v [q(v) - e^eps' q(v - Delta)]_+ h^2
    dY = np.zeros_like(EPS_GRID)
    Z = np.exp(logq)[inbox].sum()
    for x in range(3):
        for y in range(3):
            if x == y:
                continue
            D = np.zeros(3)
            D[y], D[x] = 1, -1   # Delta = e_y - e_x
            W0, W1, W2 = V0 - D[0], V1 - D[1], V2 - D[2]
            ok2 = (np.abs(W0) <= t[0]) & (np.abs(W1) <= t[1]) & (np.abs(W2) <= t[2])
            lq2 = -0.5 * (a[0] * np.abs(W0) + a[1] * np.abs(W1) + a[2] * np.abs(W2)) - b * np.abs(W1 + 2 * W2)
            q1 = np.where(inbox, np.exp(logq), 0) / Z
            q2 = np.where(ok2, np.exp(lq2), 0) / Z
            dY = np.maximum(dY, hockey(q1.ravel(), q2.ravel()))
    return dict(t=[int(v) for v in t], dbar=c["dbar"], cert_ok=c["delta_ok"] and c["pair_ok"], dT=dT, dY=dY,
                mass=float(PQ.sum()))


def eps_at(curve, delta):
    ok = np.flatnonzero(curve <= delta)
    return float(EPS_GRID[ok[0]]) if ok.size else float("nan")


rows, curves = [], []
for eps, zeta, delta in CASES:
    r1 = compute(eps, zeta, delta, 0.02)
    r2 = compute(eps, zeta, delta, 0.01)
    eT1, eT2 = eps_at(r1["dT"], delta), eps_at(r2["dT"], delta)
    print(f"eps={eps} zeta={zeta}: t={r2['t']} eps_T(h=.02)={eT1} eps_T(h=.01)={eT2} mass={r2['mass']:.12f}", flush=True)
    eY = eps_at(r2["dY"], delta)
    for frac in (0.0, 0.1, 0.5, 0.9, 0.999):
        rows.append({"K": 2, "eps_target": eps, "zeta": zeta, "delta": delta, "t": r2["t"], "dbar_cert": r2["dbar"],
                     "cert_ok": r2["cert_ok"], "colluding_fraction": frac,
                     "eps_actual_integer_h0.02": eT1, "eps_actual_integer_h0.01": eT2,
                     "eps_actual_continuous_h0.01": eY,
                     "deltaT_at_eps_target_h0.01": float(r2["dT"][np.argmin(np.abs(EPS_GRID - eps))]),
                     "note": "independent of colluding fraction by Theorem 1 step (5); same numbers for every row"})
    curves.append((eps, zeta, delta, r2))
    for e, dt, dy in zip(EPS_GRID, r2["dT"], r2["dY"]):
        pass
with open(f"{OUT}/collusion_K2.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
with open(f"{OUT}/collusion_K2_curves.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["eps_target", "zeta", "delta", "eps_prime", "delta_T_integer", "delta_Y_continuous"])
    for eps, zeta, delta, r in curves:
        for e, dt, dy in zip(EPS_GRID, r["dT"], r["dY"]):
            w.writerow([eps, zeta, delta, e, dt, dy])

fig, axs = plt.subplots(1, 2, figsize=(11, 4))
for eps, zeta, delta, r in curves:
    axs[0].semilogy(EPS_GRID, np.maximum(r["dT"], 1e-16), label=f"δ_T integer, ε={eps}, ζ={zeta}")
    axs[0].semilogy(EPS_GRID, np.maximum(r["dY"], 1e-16), ":", label=f"δ_Y continuous, ε={eps}, ζ={zeta}")
    axs[0].plot([eps], [r["dbar"]], "k*")
axs[0].axhline(1e-3, color="gray", lw=0.8)
axs[0].set_xlabel("ε'")
axs[0].set_ylabel("hockey-stick divergence δ(ε')")
axs[0].set_title("K=2, numerical (★ = certified δ̄ at target ε)")
axs[0].legend(fontsize=6)
fr = [0.0, 0.1, 0.5, 0.9, 0.999]
for eps, zeta, delta, r in curves:
    axs[1].plot(fr, [eps_at(r["dT"], delta)] * len(fr), "o-", label=f"PW-Sum target ε={eps}, ζ={zeta}")
axs[1].set_xlabel("fraction of users colluding with the Server")
axs[1].set_ylabel("numerical ε at δ=1e-3 (integer histogram)")
axs[1].set_title("PW-Sum: unchanged by collusion (Theorem 1)")
axs[1].legend(fontsize=7)
fig.tight_layout()
fig.savefig(f"{FIG}/collusion_K2.pdf")
fig.savefig(f"{FIG}/collusion_K2.png", dpi=150)
for r in rows[::5]:
    print({k: r[k] for k in ("eps_target", "zeta", "t", "dbar_cert", "eps_actual_integer_h0.02",
                             "eps_actual_integer_h0.01", "eps_actual_continuous_h0.01", "deltaT_at_eps_target_h0.01")})
