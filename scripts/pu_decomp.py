"""V-B: PW-Sum error decomposition. MSE = E[W^2]/K^2 (noise term, independent of the data) + sum f_i(1-f_i)/K^2 (rounding
term, exact). eps = 1, delta = 1/N^2, gamma sweep. Synthetic distributions at N = 1e6 share parameters and W with Uniform;
gridmid is built on each K, as the worst case of the rounding term (not used in any cross-method comparison).
Writes results/pu/decomp.csv."""
import os as _os, sys as _sys  # repository paths: src/pwexp/paths.py
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
from pwexp.paths import PWSUM_ROOT, DATA_DIR, RUST_BIN  # noqa: E402,F401

import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, f"{ROOT}/src")
from pwexp import synthetic as syn  # noqa: E402

DATA = DATA_DIR
rows = [r for r in json.load(open(f"{ROOT}/params/pw_params_pu.json")) if "gamma" in r["sweeps"].split("+")]


def qvar(y, K):
    f = y * K - np.floor(y * K)
    return float(np.sum(f * (1 - f))) / K ** 2


out = []
for r in sorted(rows, key=lambda x: (x["data"], x["gamma"])):
    K, N, g = r["K"], r["N"], r["gamma"]
    W = np.load(f"{ROOT}/results/pu/noise/N{N}_eps{r['eps']}_g{g}.npz")["W"].astype(float)
    noise = float(np.mean(W ** 2)) / K ** 2
    h = 1.96 * float((W ** 2).std(ddof=1)) / np.sqrt(len(W)) / K ** 2
    sets = {r["data"]: None}
    if r["data"] == "Uniform":
        sets = {"Uniform": "uniform", "Gaussian": "tgauss", "Zipf": "zipf", "Bimodal": "bimodal", "GridMid": "gridmid"}
    for name, st in sets.items():
        if st == "gridmid":
            y = syn.generate("gridmid", N, K=K)
        elif st:
            y = np.load(f"{DATA}/Synthetic/processed/{st}_N{N}.npy")
        else:
            y = np.load(f"{DATA}/{name}/processed/values.npy")
        q = qvar(y, K)
        out.append({"dataset": name, "N": N, "eps": r["eps"], "gamma": g, "K": K, "zeta": r["zeta"], "noise_term": noise, "noise_ci": h, "quant_term": q,
                    "worst_quant": N / (4 * K * K), "mse": noise + q, "R": (noise + q) / (2 / r["eps"] ** 2), "R_target": 1 + g, "bound": r["mse_cert"]})
        print(name, g, K, f"noise={noise:.4f} quant={q:.4f} worst={N / (4 * K * K):.4f} R={out[-1]['R']:.4f}", flush=True)
with open(f"{ROOT}/results/pu/decomp.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)
