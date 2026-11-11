"""Collusion and poisoning across methods.

Collusion (static users colluding with the analyzer/Server, fraction c of users, revealing their inputs and noise):
- GKMPS: colluders' noise shares are known, so the analyzer can subtract them; the honest part of the central noise is
  A - B with A, B ~ NB((1-c), p) (infinitely divisible split of NB(1,p)); flooding atoms sum to zero.
- BBGN: honest noise Polya(1-c, α) - Polya(1-c, α) (Polya(r, α) = NB(r, α)).
- The released SUM is a post-processing of the full view, so the exact (ε, δ) of "sum + honest noise" under a shift
  of the target's contribution (|shift| <= K) is a LOWER BOUND on the full-view ε. We report
  ε_sum(δ) = min{ε' : max_{s in {1, K/2, K}} sum_z [P(z) - e^{ε'} P(z - s)]_+ <= δ} on a grid of step 0.005.
- PW-Sum: Theorem 1 (full view, any |C| <= N-1); numerical K = 2 check (r1b_collusion_k2.py) gives ε_actual = ε for every c.
- LDP-Laplace: ε per user, unaffected.
Poisoning: fraction c of controlled users; (i) in-range attack (report the maximum legal value) for every method;
(ii) protocol-level attack allowed by each message format: LDP report A = 1e3; GKMPS 100 extra messages of +Δ per
controlled user; BBGN shares chosen to add q/4 (mod q) per controlled user; PW-Sum / ID-Sum / Prio3Sum / CDP decode,
reject or clip out-of-range input (shift = in-range bound). Shifts are computed by running each analyzer on the
attacked message multiset (one run, N = 1e5 uniform; the clean run uses the same honest randomness).
Writes results/m6/collusion.csv, results/m6/poisoning.csv, figures/m6/fig_collusion.{pdf,png}.
"""
import os as _os, sys as _sys  # repository paths: src/pwexp/paths.py
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
from pwexp.paths import PWSUM_ROOT, DATA_DIR, RUST_BIN  # noqa: E402,F401

import csv
import math
import os
import sys

import numpy as np
from scipy.signal import fftconvolve
from scipy.stats import nbinom

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from pwexp import baselines as bl  # noqa: E402

FRACS = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
EGRID = np.round(np.arange(0, 60.0001, 0.005), 3)


def diff_pmf(r, q):
    """pmf of A - B, A, B iid NB(r, q) (mean r q/(1-q)), on offsets -L..L."""
    L = int(math.ceil(np.log(1e-30) / np.log(q))) + 10
    a = nbinom.pmf(np.arange(L + 1), r, 1 - q)
    d = fftconvolve(a, a[::-1])
    d = np.maximum(d, 0)
    return d / d.sum(), L


def eps_sum(pmf, K, delta):
    best = 0.0
    for s in sorted({1, max(1, K // 2), K}):
        p, qsh = pmf[s:], pmf[:-s]     # P(z) vs P(z - s)
        lr = np.where(qsh > 0, np.log(np.maximum(p, 1e-300)) - np.log(np.maximum(qsh, 1e-300)), np.inf)
        o = np.argsort(-lr)
        lr, p2, q2 = lr[o], p[o], qsh[o]
        cp, cq = np.cumsum(p2), np.cumsum(q2)
        tail = pmf[:s].sum()           # mass of P(z) where z - s is outside the support
        for e in EGRID:
            k = np.searchsorted(-lr, -e, side="left")
            d = (cp[k - 1] - math.exp(e) * cq[k - 1] if k > 0 else 0.0) + tail
            if d <= delta:
                best = max(best, e)
                break
        else:
            return float("inf")
    return best


def collusion():
    rows = []
    for N in (10 ** 5,):
        for eps in (0.5, 1.0, 2.0):
            delta = 1 / N ** 2
            g = bl.GKMPS().select_params(eps, delta, N, 0.1)
            b = bl.BBGN().select_params(eps, delta, N, 0.1)
            for c in FRACS:
                r = 1 - c
                pg, _ = diff_pmf(r, g["pc"])
                pb, _ = diff_pmf(r, b["alpha"])
                rows.append({"N": N, "eps_target": eps, "delta": delta, "colluding_fraction": c,
                             "PW-Sum_eps": eps, "PW-Sum_basis": "Theorem 1 (full view) + numerical K=2 check",
                             "LDP_eps": eps, "GKMPS_K": g["K"], "GKMPS_eps_sum_lower_bound": eps_sum(pg, g["K"], delta),
                             "BBGN_K": b["K"], "BBGN_eps_sum_lower_bound": eps_sum(pb, b["K"], delta)})
                print(rows[-1], flush=True)
    return rows


def poisoning():
    N = 10 ** 5
    y = np.load(f"{DATA_DIR}/Synthetic/processed/uniform_N100000.npy")
    rng = np.random.default_rng(20261112)
    perm = rng.permutation(N)
    rows = []
    g = bl.GKMPS().select_params(1.0, 1 / N ** 2, N, 0.1)
    b = bl.BBGN().select_params(1.0, 1 / N ** 2, N, 0.1)
    for c in (0.01, 0.05, 0.1, 0.3):
        mal = perm[:int(c * N)]
        honest_y_mal = y[mal].sum()
        inrange = len(mal) * 1.0 - honest_y_mal          # every method: report the max legal value (1 after scaling)
        rows.append({"frac": c, "n_controlled": len(mal), "attack": "in-range (report max legal value)",
                     "PW-Sum": inrange, "ID-Sum": inrange, "Prio3Sum": inrange, "CDP": inrange, "LDP": inrange,
                     "GKMPS": inrange, "BBGN": inrange, "bound_|A|": float(len(mal))})
        # protocol-level
        ldp = len(mal) * 1e3 - honest_y_mal               # report 1e3 instead of y (noise unchanged, coupled)
        gk = len(mal) * 100 * (g["K"] + 1) / g["K"]       # 100 extra messages of +Delta, analyzer divides by K
        q, K = b["q"], b["K"]
        x_hon = int(np.floor(y * K).sum())                # deterministic part is enough to show the wrap effect
        z = (x_hon + q // 4) % q        # the coalition's shares add q/4 in total (one controlled user suffices)
        z_c = x_hon % q
        corr = lambda v: v - q if v > (N * K + q) / 2 else v  # noqa: E731
        bb = (corr(z) - corr(z_c)) / K
        rows.append({"frac": c, "n_controlled": len(mal), "attack": "protocol-level (format allows)",
                     "PW-Sum": inrange, "ID-Sum": inrange, "Prio3Sum": inrange, "CDP": inrange, "LDP": ldp,
                     "GKMPS": gk, "BBGN": bb, "bound_|A|": float(len(mal)),
                     "note": "PW/ID-Sum: out-of-range decodes to 0, duplicates dropped (r3b_protocol.py); Prio3Sum: FLP proof "
                             "rejects; CDP: curator clips; LDP: A=1e3 per user; GKMPS: 100 extra +Δ messages per user; "
                             "BBGN: the coalition's shares add q/4 mod q in total (any c > 0)"})
    return rows


def main():
    os.makedirs(f"{ROOT}/results/m6", exist_ok=True)
    os.makedirs(f"{ROOT}/figures/m6", exist_ok=True)
    col = collusion()
    with open(f"{ROOT}/results/m6/collusion.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(col[0]))
        w.writeheader()
        w.writerows(col)
    poi = poisoning()
    keys = []
    for r in poi:
        keys += [k for k in r if k not in keys]
    with open(f"{ROOT}/results/m6/poisoning.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(poi)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, eps in zip(axs, (0.5, 1.0, 2.0)):
        q = [r for r in col if r["eps_target"] == eps]
        c = [r["colluding_fraction"] for r in q]
        ax.plot(c, [r["PW-Sum_eps"] for r in q], "o-", label="PW-Sum (Theorem 1)")
        ax.plot(c, [r["LDP_eps"] for r in q], "x:", label="LDP-Laplace")
        ax.plot(c, [min(r["GKMPS_eps_sum_lower_bound"], 60) for r in q], "s-", label="GKMPS (lower bound, sum only)")
        ax.plot(c, [min(r["BBGN_eps_sum_lower_bound"], 60) for r in q], "^-", label="BBGN (lower bound, sum only)")
        ax.set_yscale("log")
        ax.set_xlabel("fraction of users colluding with the analyzer")
        ax.set_title(f"target ε={eps}, N=1e5, δ=1/N²")
    axs[0].set_ylabel("ε at δ=1/N²")
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{ROOT}/figures/m6/fig_collusion.pdf")
    fig.savefig(f"{ROOT}/figures/m6/fig_collusion.png", dpi=150)
    for r in poi:
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k != "note"})


if __name__ == "__main__":
    main()
