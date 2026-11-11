"""Synthetic data in [0,1] for the PW-Sum utility experiments.

Five distributions, all generated from a fixed seed per (distribution, N):
  uniform    U[0,1]
  tgauss     N(0.5, 0.15^2) truncated to [0,1] (inverse-CDF sampling)
  zipf       bounded Zipf: rank r in {1..R} with P(r) ~ r^-s, value (r-1)/(R-1)
  bimodal    two-ended: with prob 1/2 Beta(a, b) (near 0), else Beta(b, a) (near 1)
  gridmid    worst case for integer-grid quantisation: (floor(u*K) + 1/2)/K, u ~ U[0,1)
gridmid depends on the quantisation grid K of the method being evaluated, so it is
generated on demand for each K instead of being stored. Its u draws depend only on the
seed, so every method sees the same u and only the grid differs.
"""
import numpy as np
from scipy.special import ndtr, ndtri

# order fixes the seeds; new distributions are appended so existing seeds do not change
DISTS = ("uniform", "tgauss", "zipf", "gridmid", "bimodal")
STORED = ("uniform", "tgauss", "zipf", "bimodal")
NS = (10**4, 10**5, 10**6, 10**7)
BASE_SEED = 20261101
TG_MU, TG_SIGMA = 0.5, 0.15
ZIPF_S, ZIPF_R = 1.0, 1000  # s=1.5 gave mean 0.023; s=1.0 gives mean 0.133, median 0.023, skewness 2.08
BI_A, BI_B = 1.0, 19.0      # Beta(1,19): mean 0.05, mode 0; mirrored Beta(19,1) near 1


def zipf_moments():
    """Exact mean, median and skewness of the bounded-Zipf value distribution."""
    r = np.arange(1, ZIPF_R + 1)
    p = r ** -ZIPF_S
    p /= p.sum()
    v = (r - 1) / (ZIPF_R - 1)
    m = (p * v).sum()
    sd = np.sqrt((p * (v - m) ** 2).sum())
    return m, v[np.searchsorted(np.cumsum(p), 0.5)], (p * (v - m) ** 3).sum() / sd ** 3


def seed_for(dist, n):
    """Distinct, fixed seed per (distribution, N)."""
    return BASE_SEED + 100 * DISTS.index(dist) + int(round(np.log10(n)))


def generate(dist, n, K=None, seed=None):
    rng = np.random.default_rng(seed_for(dist, n) if seed is None else seed)
    if dist == "uniform":
        return rng.random(n)
    if dist == "tgauss":
        lo, hi = ndtr((0 - TG_MU) / TG_SIGMA), ndtr((1 - TG_MU) / TG_SIGMA)
        u = lo + (hi - lo) * rng.random(n)
        return np.clip(TG_MU + TG_SIGMA * ndtri(u), 0.0, 1.0)
    if dist == "zipf":
        r = np.arange(1, ZIPF_R + 1)
        p = r ** -ZIPF_S
        p /= p.sum()
        ranks = rng.choice(r, size=n, p=p)
        return (ranks - 1) / (ZIPF_R - 1)
    if dist == "bimodal":
        side = rng.random(n) < 0.5
        b = rng.beta(BI_A, BI_B, size=n)
        return np.where(side, b, 1.0 - b)  # 1 - Beta(a,b) ~ Beta(b,a)
    if dist == "gridmid":
        if K is None or K < 1:
            raise ValueError("gridmid needs the quantisation grid K")
        k = np.minimum(np.floor(rng.random(n) * K), K - 1)
        return (k + 0.5) / K
    raise ValueError(dist)
