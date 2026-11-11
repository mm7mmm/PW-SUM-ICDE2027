"""Parameter selection check: enumerate every candidate K up to a conservative K_safe instead of pruning with the zeta_inf floor, and check that the
same K and bytes are selected. K_safe uses t(eps) reduced by a margin of 1e-6 before the ceiling is taken, so a floating-point error at an integer boundary cannot cut
the range short. All (N, eps, delta, gamma) of pw_params_{radius,pu,cost}.json. Writes results/params_full_enum.csv."""
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

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sp = importlib.util.spec_from_file_location("s9", f"{ROOT}/scripts/s9_params_radius.py")
s9 = importlib.util.module_from_spec(sp)
sp.loader.exec_module(s9)


def work(r):
    N, eps, dl, g = r["N"], r["eps"], r["dlabel"], r["gamma"]
    delta = 1 / N ** 2 if dl == "1/N^2" else 1 / N
    K = math.floor(math.sqrt((N + 1) * eps * eps / (8 * g))) + 1
    # lower bound on every t_j, valid for all K: alpha_j <= eps, with a margin below the real value
    x = float((2 / eps) * np.log1p(np.expm1(eps / 2) / delta))
    tlow = max(math.ceil(x - 1e-6), 1)
    best = None
    n = 0
    while True:
        z = s9.zeta_of(K, N, eps, g)
        if z is not None:
            t = s9.radius(s9.weights(K, z, eps), delta)
            M = int(t.sum()) + K - 1
            b = (2 * N + M) * s9.ell(K) / N
            n += 1
            if best is None or b < best[0]:
                best = (b, K, z, M)
        lb = (K + 1) * tlow + K - 1
        if best is not None and (2 * N + lb) * 49 / N > best[0]:
            break
        K += 1
    return {"N": N, "eps": eps, "dlabel": dl, "gamma": g, "table_K": r["K"], "table_M": r["M"], "table_bytes": r["bytes_per_user"], "enum_K": best[1], "enum_M": best[3], "enum_bytes": best[0], "K_evaluated": n, "last_K": K, "same": best[1] == r["K"] and best[3] == r["M"]}


def main():
    u = {}
    for f in ("pw_params_radius", "pw_params_pu", "pw_params_cost"):
        for r in json.load(open(f"{ROOT}/params/{f}.json")):
            u.setdefault((r["N"], r["eps"], r["dlabel"], r["gamma"]), {k: v for k, v in r.items() if k != "t"})
    with mpc.Pool(60) as p:
        rows = p.map(work, list(u.values()), chunksize=1)
    with open(f"{ROOT}/results/params_full_enum.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(len(rows), "sets; all same:", all(r["same"] for r in rows), "; mismatches:", [(r["N"], r["eps"], r["dlabel"], r["gamma"]) for r in rows if not r["same"]], "; K evaluated in total:", sum(r["K_evaluated"] for r in rows), flush=True)


if __name__ == "__main__":
    main()
