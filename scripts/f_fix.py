"""Post-processing steps that need no new sampling at scale.
1. collusion_lb.csv: eps_sum returned the first grid point that meets delta (rounding the threshold up by < 0.005). The last grid point that
   violates delta, returned - 0.005, is the numerical lower bound.
2. comm_msgs.csv: upload / forward / total messages per user, with one convention across methods.
(The finer K = 2 integration is scripts/r1b_collusion_k2.py.)"""
import os as _os, sys as _sys  # repository paths: src/pwexp/paths.py
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
from pwexp.paths import PWSUM_ROOT, DATA_DIR, RUST_BIN  # noqa: E402,F401

import os

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
c = pd.read_csv(f"{ROOT}/results/robust/collusion.csv")
c["grid_first_feasible"] = c["eps_sum_lower_bound"]
c["eps_numerical_lower_bound"] = (c["grid_first_feasible"] - 0.005).clip(lower=0)
c.drop(columns="eps_sum_lower_bound").to_csv(f"{ROOT}/results/robust/collusion_lb.csv", index=False)

m = pd.read_csv(f"{ROOT}/results/cost/comm.csv")


def conv(r):
    n = r.msgs_per_user
    if r.method in ("PW-Sum", "PW-Uniform"):
        return 1.0, n, 1.0 + n                    # msgs_per_user = 1 + M/N is the Shuffler-to-Server count
    if r.method.startswith("GKMPS"):
        return n, n, 2 * n                        # every message crosses both links
    if r.method == "BBGN":
        return n, n - 1, 2 * n - 1                # m shuffled shares on both links, one unshuffled message once
    if r.method == "ID-Sum":
        return 1.0, n - 1, n                      # msgs_per_user already counts both links
    return 1.0, float("nan"), 1.0                 # Prio3: one report; server-to-server bytes are not per-message


m[["upload_msgs", "forward_msgs", "total_msgs"]] = m.apply(lambda r: pd.Series(conv(r)), axis=1)
m.to_csv(f"{ROOT}/results/cost/comm_msgs.csv", index=False)

print("ok", len(c), len(m))
