"""Parameters for the Privacy-Utility section (V-B), current paper rule, same search and interval verification as
s9_params_radius.work. Three sweeps; delta = 1/N^2 throughout.
  eps:   N fixed per dataset, gamma = 0.1, eps in EPS
  scale: eps = 1, gamma = 0.1, N in the per-dataset grid
  gamma: eps = 1, N fixed per dataset, gamma in GAMMAS
Writes params/pw_params_pu.{csv,json}; an entry is unique per (data, N, eps, gamma)."""
import os as _os, sys as _sys  # repository paths: src/pwexp/paths.py
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
from pwexp.paths import PWSUM_ROOT, DATA_DIR, RUST_BIN  # noqa: E402,F401

import csv
import importlib.util
import json
import multiprocessing as mpc
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sp = importlib.util.spec_from_file_location("s9", f"{ROOT}/scripts/s9_params_radius.py")
s9 = importlib.util.module_from_spec(sp)
sp.loader.exec_module(s9)

FULL = {"Uniform": 10 ** 6, "SF": 42928, "ACS": 2912790, "Taxi": 15167858}
EPS = (0.1, 0.2, 0.5, 1.0, 2.0)
GAMMAS = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
SCALE = {"Uniform": (10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7),
         "SF": (10 ** 3, 3000, 10 ** 4, 30000, 42928),
         "ACS": (10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6, 2912790),
         "Taxi": (10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7, 15167858)}


def plan():
    seen = {}
    def add(sweep, d, N, e, g):
        k = (d, N, e, g)
        seen.setdefault(k, set()).add(sweep)
    for d, N in FULL.items():
        for e in EPS:
            add("eps", d, N, e, 0.1)
        for g in GAMMAS:
            add("gamma", d, N, 1.0, g)
    for d, Ns in SCALE.items():
        for N in Ns:
            add("scale", d, N, 1.0, 0.1)
    return seen


def job(k):
    d, N, e, g = k
    return s9.work(("pu", d, N, e, "1/N^2", g))


def main():
    seen = plan()
    keys = sorted(seen, key=lambda k: (k[0], k[1], k[2], k[3]))
    with mpc.Pool(40) as p:
        rows = p.map(job, keys, chunksize=1)
    for r, k in zip(rows, keys):
        r["sweeps"] = "+".join(sorted(seen[k]))
    cols = [c for c in rows[0] if c != "t"]
    with open(f"{ROOT}/params/pw_params_pu.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    json.dump(rows, open(f"{ROOT}/params/pw_params_pu.json", "w"))
    bad = [r for r in rows if not (r["iv_rho_ok"] and r["iv_mse_ok"] and r["iv_t_minimal"] and r["floor_ok_to_K_safe"])]
    print(len(rows), "parameter sets; failed checks:", len(bad))
    for r in rows:
        print(f"{r['data']:8s} N={r['N']:<9d} eps={r['eps']:<4} g={r['gamma']:<5} K={r['K']:<6d} zeta={r['zeta']:.7f} "
              f"t0={r['t_end']:<7d} tc={r['t_center']:<5d} M/N={r['M_over_N']:<8.3f} B/u={r['bytes_per_user']:.2f} {r['sweeps']}")


if __name__ == "__main__":
    main()
