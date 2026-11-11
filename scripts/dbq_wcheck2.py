"""10,000 exact draws of W for the large-window parameters (independent seeds from the first check); second moment with its standard error."""
import os as _os, sys as _sys  # repository paths: src/pwexp/paths.py
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
from pwexp.paths import PWSUM_ROOT, DATA_DIR, RUST_BIN  # noqa: E402,F401

import json, multiprocessing as mpc, sys
import numpy as np
R = PWSUM_ROOT
sys.path.insert(0, f"{R}/src")
from pwexp import pw_core_task as pc, pw_sampler as smp
r = json.load(open(f"{R}/results/dbq/W2_pw_params.json"))
K, z = r["K"], r["zeta"]


def work(k):
    a, b = pc.weights(K, z, "pw", 1.0)
    rng = np.random.default_rng([20261121, k])
    s, _, _ = smp.sample_s(a, b, np.array(r["t"], float), 125, rng)
    return s, np.floor(s + rng.random(125))


with mpc.Pool(80) as p:
    res = p.map(work, range(80))
s = np.concatenate([x[0] for x in res]) / K
w = np.concatenate([x[1] for x in res]) / K
bound = 2 / (1 - z) ** 2 + 0.25 / K ** 2
for name, v in (("s(V)/K", s), ("W/K", w)):
    m2 = float(np.mean(v ** 2))
    se = float(np.std(v ** 2, ddof=1) / np.sqrt(len(v)))
    print(f"{name}: n={len(v)} E[x^2]={m2:.4f} se={se:.4f} bound={bound:.4f} ratio={m2 / bound:.4f} z={(m2 - bound) / se:.2f} mean={float(np.mean(v)):.3f} max|x|={float(np.max(np.abs(v))):.2f}")
json.dump({"K": K, "zeta": z, "n": len(w), "E_w2": float(np.mean(w ** 2)), "bound": bound}, open(f"{R}/results/dbq/W2_noise_check.json", "w"))
