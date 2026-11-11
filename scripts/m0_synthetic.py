"""Write the stored synthetic datasets, their summary.json and README."""
import os as _os, sys as _sys  # repository paths: src/pwexp/paths.py
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
from pwexp.paths import PWSUM_ROOT, DATA_DIR, RUST_BIN  # noqa: E402,F401

import datetime
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pwexp import synthetic as syn  # noqa: E402

OUT = f"{DATA_DIR}/Synthetic"
os.makedirs(f"{OUT}/raw", exist_ok=True)
os.makedirs(f"{OUT}/processed", exist_ok=True)

rows = []
for dist in syn.STORED:
    for n in syn.NS:
        x = syn.generate(dist, n)
        assert x.shape == (n,) and x.min() >= 0 and x.max() <= 1
        # regenerating with the same seed must give identical data
        assert np.array_equal(x, syn.generate(dist, n))
        path = f"{OUT}/processed/{dist}_N{n}.npy"
        np.save(path, x)
        h = hashlib.sha256(open(path, "rb").read()).hexdigest()
        rows.append(dict(dist=dist, N=n, seed=syn.seed_for(dist, n), mean=float(x.mean()), var=float(x.var()),
                         min=float(x.min()), max=float(x.max()), sum=float(x.sum()),
                         file=f"Synthetic/processed/{dist}_N{n}.npy", sha256=h))
        print(dist, n, x.mean(), x.var(), h[:16], flush=True)

# gridmid: same u for every K, values on grid midpoints
for K in (7, 1000):
    gm = syn.generate("gridmid", 10**4, K=K)
    assert np.allclose((gm * K) % 1, 0.5) and gm.min() > 0 and gm.max() < 1
u = np.random.default_rng(syn.seed_for("gridmid", 10**4)).random(10**4)
assert np.array_equal(syn.generate("gridmid", 10**4, K=1000), (np.minimum(np.floor(u * 1000), 999) + 0.5) / 1000)

zm = syn.zipf_moments()
json.dump(dict(generated=datetime.datetime.now().isoformat(timespec="seconds"), privacy_unit="one record",
               X_max=1.0, n_truncated=0, files=rows,
               params=dict(base_seed=syn.BASE_SEED, dists=syn.DISTS, tgauss=[syn.TG_MU, syn.TG_SIGMA],
                           zipf=dict(s=syn.ZIPF_S, R=syn.ZIPF_R, mean=zm[0], median=zm[1], skew=zm[2]),
                           bimodal=dict(a=syn.BI_A, b=syn.BI_B))),
          open(f"{OUT}/processed/summary.json", "w"), indent=1)

bi_mean = syn.BI_A / (syn.BI_A + syn.BI_B)
with open(f"{OUT}/README.md", "w") as f:
    f.write(f"""# Synthetic

Generated on {datetime.date.today().isoformat()} by `scripts/m0_synthetic.py`
(code `src/pwexp/synthetic.py` in `pw_sum_exp`). No download, so `raw/` is empty.
Values already lie in [0,1] (X_max = 1); nothing is clipped (clipped fraction 0).
Privacy unit: one record.

| distribution | definition |
|---|---|
| uniform | U[0,1] |
| tgauss | N({syn.TG_MU}, {syn.TG_SIGMA}^2) truncated to [0,1], inverse-CDF sampling |
| zipf | rank r in {{1..{syn.ZIPF_R}}}, P(r) proportional to r^-{syn.ZIPF_S}; value (r-1)/{syn.ZIPF_R - 1} |
| bimodal (two-ended) | with prob 1/2 Beta({syn.BI_A:g},{syn.BI_B:g}) (mean {bi_mean:.2f}, mode 0), else Beta({syn.BI_B:g},{syn.BI_A:g}) = 1 - Beta({syn.BI_A:g},{syn.BI_B:g}) (mean {1 - bi_mean:.2f}, mode 1) |
| gridmid | (floor(u K) + 1/2)/K with u ~ U[0,1); worst case for integer-grid unbiased quantisation |

Zipf parameters: exponent s = {syn.ZIPF_S}, R = {syn.ZIPF_R} ranks. Exact distribution: mean {zm[0]:.4f},
median {zm[1]:.4f}, skewness {zm[2]:.2f}. (A first version used s = 1.5, mean 0.023;
changed on request so that the mean lies in 0.1–0.2 while keeping the long tail. The realised means are in the table below.)

Seed for (dist, N): {syn.BASE_SEED} + 100*index(dist) + log10(N), index order {syn.DISTS}.
`gridmid` depends on the quantisation grid K of the method under test, so it is generated
on demand with `generate("gridmid", N, K)` and is not stored. Its u draws depend only on the seed,
so all methods share the same u and differ only in their own grid K.

Same input for every method: every protocol reads the stored file (or, for gridmid, the same seed); protocol
randomness uses separate seeds that are fixed in the experiment code.

Stored files (`processed/<dist>_N<N>.npy`, float64):

| dist | N | seed | mean | variance | min | max | sha256 |
|---|---|---|---|---|---|---|---|
""")
    for r in rows:
        f.write(f"| {r['dist']} | {r['N']} | {r['seed']} | {r['mean']:.6f} | {r['var']:.6f} | {r['min']:.6g} | "
                f"{r['max']:.6g} | `{r['sha256']}` |\n")
print("done")
