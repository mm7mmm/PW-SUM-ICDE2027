"""Ablation of the sum term (beta = 0): exact sampling at small K gives E s(V)^2, which checks the Gaussian-conditioning approximation
(8/eps^2) sum_j (j - K/2)^2 used in r4_ablation.py for large K. Writes results/m6/ablation_beta0_small_K.csv."""
import csv
import math
import multiprocessing as mpc
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def work(task):
    K, eps, R = task
    a = np.full(K + 1, eps)
    rng = np.random.default_rng(20261115 + K)
    jj = np.arange(K + 1)
    s_all = []
    while len(s_all) < R:
        X = rng.laplace(0, 2 / a[1:], size=(200000, K))
        xa = -X.sum(axis=1)
        k1 = rng.random(len(xa)) < np.exp(-a[0] * np.abs(xa) / 2)
        V = np.concatenate([xa[k1, None], X[k1]], axis=1)
        s_all.extend(V @ jj)
    s = np.array(s_all[:R])
    approx = 8 / eps ** 2 * float(np.sum((jj - K / 2) ** 2))
    return {"K": K, "eps": eps, "R": R, "E_s2_exact_sampling": float(np.mean(s ** 2)), "ci_half": 1.96 * float(np.std(s ** 2)) / math.sqrt(R),
            "gauss_cond_approx": approx, "ratio_exact_over_approx": float(np.mean(s ** 2)) / approx}


if __name__ == "__main__":
    os.makedirs(f"{ROOT}/results/m6", exist_ok=True)
    with mpc.Pool(5) as p:
        rows = p.map(work, [(K, 1.0, 20000 if K <= 16 else 5000) for K in (4, 8, 16, 24, 32)])
    with open(f"{ROOT}/results/m6/ablation_beta0_small_K.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)
