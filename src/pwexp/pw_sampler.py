"""Exact rejection sampler of q_t (paper section V, 'exact sampling') with the endpoint anchor j = 0, and the
integerisation map R(v; xi) (Lemma 5).

draw_one implements the three acceptance steps of the exact sampler, and integerize the cumulative rounding.
In ``sample_s``, every accepted
proposal of a batch is kept instead of only the first: proposals are i.i.d., so the accepted ones are i.i.d. draws
of q_t. The target distribution is unchanged (no tuning of the proposal or of the acceptance probabilities).
Laplace density (alpha_j/4) e^{-alpha_j|x|/2}  <=>  numpy laplace with scale 2/alpha_j.
"""
import numpy as np


def draw_one(a, b, t, rng, anchor=0, batch=256):
    """One exact draw. Returns (V, proposals, accepted_step1, accepted_step2)."""
    K = len(a) - 1
    idx = np.r_[0:anchor, anchor + 1:K + 1]
    jj = np.arange(K + 1)
    scale = 2.0 / a[idx]
    props = acc1 = acc2 = 0
    while True:
        X = rng.laplace(0.0, scale, size=(batch, K))
        xa = -X.sum(axis=1)
        k1 = rng.random(batch) < np.exp(-a[anchor] * np.abs(xa) / 2)
        V = np.empty((batch, K + 1))
        V[:, idx], V[:, anchor] = X, xa
        u2 = rng.random(batch)
        k2 = k1 & (u2 < np.exp(-b * np.abs(V @ jj)))
        k3 = k2 & np.all(np.abs(V) <= t, axis=1)
        hit = np.flatnonzero(k3)
        if hit.size:
            f = int(hit[0])
            return V[f], props + f + 1, acc1 + int(k1[:f + 1].sum()), acc2 + int(k2[:f + 1].sum())
        props += batch
        acc1 += int(k1.sum())
        acc2 += int(k2.sum())


def sample_s(a, b, t, n, rng, max_elems=4_000_000, keep_v=0):
    """n exact draws V ~ q_t (endpoint anchor). Returns s(V) for each draw, acceptance counters and up to
    ``keep_v`` full V vectors (for the integerisation checks)."""
    K = len(a) - 1
    jj = np.arange(K + 1, dtype=np.float64)
    scale = 2.0 / a[1:]
    batch = max(64, max_elems // (K + 1))
    s_out = np.empty(n)
    kept = []
    filled = 0
    cnt = {"proposals": 0, "acc_step1": 0, "acc_step2": 0, "acc_box": 0}
    while filled < n:
        X = rng.laplace(0.0, scale, size=(batch, K))
        xa = -X.sum(axis=1)
        k1 = rng.random(batch) < np.exp(-a[0] * np.abs(xa) / 2)
        V = np.empty((int(k1.sum()), K + 1))
        V[:, 1:], V[:, 0] = X[k1], xa[k1]
        s = V @ jj
        k2 = rng.random(len(V)) < np.exp(-b * np.abs(s))
        V, s = V[k2], s[k2]
        k3 = np.all(np.abs(V) <= t, axis=1)
        V, s = V[k3], s[k3]
        cnt["proposals"] += batch
        cnt["acc_step1"] += int(k1.sum())
        cnt["acc_step2"] += int(k2.sum())
        cnt["acc_box"] += int(k3.sum())
        take = min(len(s), n - filled)
        s_out[filled:filled + take] = s[:take]
        if len(kept) < keep_v:
            kept.extend(V[:min(take, keep_v - len(kept))])
        filled += take
    return s_out, cnt, (np.array(kept) if kept else np.empty((0, K + 1)))


def integerize(V, xi):
    """Integerization map R(v; xi), vectorised."""
    K = V.shape[1] - 1
    z = np.cumsum(V[:, :0:-1], axis=1)[:, ::-1]
    P = np.cumsum(z, axis=1)
    Q = np.floor(P + xi[:, None])
    d = np.diff(np.concatenate([np.zeros((len(V), 1)), Q], axis=1), axis=1)
    U = np.empty_like(V)
    U[:, 0] = -d[:, 0]
    U[:, 1:K] = d[:, :K - 1] - d[:, 1:K]
    U[:, K] = d[:, K - 1]
    return U, P[:, -1]
