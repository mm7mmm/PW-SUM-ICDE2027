"""V-C: collusion across methods (functions of m6_collusion_poison.py). A fraction c of users reveals its noise to the
Server; eps_sum is the exact eps of the released sum given the honest remaining noise, at delta = 1/N^2, and it is a lower
bound on the full-view eps. PW-Sum: Theorem 1 (the full-view guarantee does not depend on c, for an honest Shuffler that does
not collude with the Server). LDP: eps per user, unaffected. Writes results/robust/collusion.csv."""
import os as _os, sys as _sys  # repository paths: src/pwexp/paths.py
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
from pwexp.paths import PWSUM_ROOT, DATA_DIR, RUST_BIN  # noqa: E402,F401

import csv
import multiprocessing as mpc
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import c2_comm  # noqa: E402,F401  (installs the guarded GKMPS cost)
import m6_collusion_poison as m6  # noqa: E402
from pwexp import baselines as bl  # noqa: E402

FRACS = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)


def work(a):
    N, eps, c, name = a
    d = 1 / N ** 2
    if name == "GKMPS":
        p = bl.GKMPS().select_params(eps, d, N, 0.1)
        q, K = p["pc"], p["K"]
    else:
        p = bl.BBGN().select_params(eps, d, N, 0.1)
        q, K = p["alpha"], p["K"]
    pmf, _ = m6.diff_pmf(1 - c, q)
    return {"N": N, "eps_target": eps, "delta": d, "c": c, "method": name, "K": K, "eps_sum_lower_bound": m6.eps_sum(pmf, K, d)}


def main():
    os.makedirs(f"{ROOT}/results/robust", exist_ok=True)
    tasks = [(N, e, c, m) for N in (10 ** 5, 10 ** 6) for e in (0.5, 1.0, 2.0) for c in FRACS for m in ("GKMPS", "BBGN")]
    with mpc.Pool(40) as p:
        rows = p.map(work, tasks, chunksize=1)
    with open(f"{ROOT}/results/robust/collusion.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        if r["N"] == 10 ** 6 and r["eps_target"] == 1.0:
            print(r, flush=True)


if __name__ == "__main__":
    main()
