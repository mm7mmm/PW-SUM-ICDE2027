"""Quick self-check that needs no dataset (about two minutes): the parameter rule against the stored table, the interval checks,
the sampler and integerization invariants, the baseline estimators, and the Rust binary on small synthetic inputs (if built)."""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np  # noqa: E402

from pwexp import baselines as bl  # noqa: E402
from pwexp import pw_core_task as pc  # noqa: E402
from pwexp import pw_sampler as smp  # noqa: E402
from pwexp.paths import PWSUM_ROOT, RUST_BIN  # noqa: E402

sp = importlib.util.spec_from_file_location("s9", os.path.join(PWSUM_ROOT, "scripts", "s9_params_radius.py"))
s9 = importlib.util.module_from_spec(sp)
sp.loader.exec_module(s9)
fails = []


def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name, flush=True)
    if not ok:
        fails.append(name)


# 1. parameter rule and interval verification reproduce stored parameters
rows = [r for r in json.load(open(os.path.join(PWSUM_ROOT, "params", "pw_params_cost.json"))) if r["N"] <= 10 ** 5 and r["eps"] == 1.0]
for r in rows[:4]:
    q = s9.work(("check", "any", r["N"], r["eps"], r["dlabel"], r["gamma"]))
    check(f"parameters N={r['N']} gamma={r['gamma']}: K={q['K']} M={q['M']} match the table",
          q["K"] == r["K"] and q["M"] == r["M"] and abs(q["zeta"] - r["zeta"]) < 1e-12)
    check(f"interval checks N={r['N']} gamma={r['gamma']}", q["iv_rho_ok"] and q["iv_mse_ok"] and q["iv_t_minimal"] and q["floor_ok_to_K_safe"])

# 2. pairwise budget, exact sampler and integerization (Lemma: zero sum, one rounding of the sum, non-negative dummies)
rng = np.random.default_rng(7)
K, zeta, eps, delta = 40, 0.1, 1.0, 1e-6
a, b = pc.weights(K, zeta, "pw", eps)
check("pairwise budget", all(0.5 * (a[i] + a[j]) + b * abs(i - j) <= eps * (1 + 1e-12) for i in range(K + 1) for j in range(K + 1) if i != j))
t = np.ceil((2 / a) * np.log1p(np.expm1(a / 2) / delta))
_, _, V = smp.sample_s(a, b, t, 500, rng, keep_v=500)
xi = rng.random(500)
U, _ = smp.integerize(V, xi)
U = np.rint(U).astype(np.int64)
jj = np.arange(K + 1)
c = np.r_[0, np.ones(K - 1, np.int64), 0]
check("sampler: zero sum and box", bool(np.all(np.abs(V.sum(axis=1)) < 1e-9) and np.all(np.abs(V) <= t)))
check("integerization: sum U = 0, s(U) = floor(s(V)+xi), H >= 0", bool(np.all(U.sum(axis=1) == 0) and np.all(U @ jj == np.floor(V @ jj + xi)) and np.all(t + c + U >= 0)))

# 3. baselines: unbiased, empirical MSE close to the analytic value
y = np.random.default_rng(3).random(20000)
for m in (bl.BBGN(), bl.Prio3DP("A"), bl.GKMPS()):
    p = m.select_params(1.0, 1 / len(y) ** 2, len(y), 0.1)
    e = m.simulate_error(y, p, np.random.default_rng(5), 3000)
    an = m.analytic_mse(p, qvar=bl.qvar_exact(y, p["K"]))
    check(f"baseline {m.name}: MSE/analytic = {np.mean(e ** 2) / an:.3f}", abs(np.mean(e ** 2) / an - 1) < 0.12)

# 4. Rust binary on small inputs
if os.path.exists(RUST_BIN):
    with tempfile.TemporaryDirectory() as d:
        N = 3000
        q = s9.work(("check", "any", N, 1.0, "1/N^2", 0.5))
        K = q["K"]
        a, b = pc.weights(K, q["zeta"], "pw", 1.0)
        t = np.array(q["t"], np.int64)
        _, _, V = smp.sample_s(a, b, t.astype(float), 1, rng, keep_v=1)
        xi = rng.random(1)
        U, _ = smp.integerize(V, xi)
        U = np.rint(U[0]).astype(np.int64)
        c = np.r_[0, np.ones(K - 1, np.int64), 0]
        H = t + c + U
        jj = np.arange(K + 1, dtype=np.int64)
        hf = os.path.join(d, "H.txt")
        open(hf, "w").write(f"{int(jj @ U)} {int(jj @ (t + c))} {q['M']}\n" + " ".join(map(str, H)) + "\n")
        xs = rng.integers(0, K + 1, N).astype(np.uint32)
        vf = os.path.join(d, "x.u32")
        xs.tofile(vf)
        run = lambda *args: json.loads(subprocess.run([RUST_BIN, *map(str, args)], capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1])  # noqa: E731
        check("rust pw4 (PW-Sum message path)", run("pw4", N, K, hf, 1, 2, 1)["estimate_equals_sum_plus_W"])
        check("rust prio2 (Prio3Sum)", run("prio2", 2000, 100, 1, 2, 1)["aggregate_equals_true_sum"])
        check("rust dbq_pw (database workload, PW-Sum)", run("dbq_pw", vf, K, hf, 2, 0, 1)["estimate_equals_sum_plus_W"])
        np.minimum(xs, 100).astype(np.uint32).tofile(vf)
        check("rust dbq_prio (database workload, Prio3Sum)", run("dbq_prio", vf, 100, 2, 0, 1)["aggregate_equals_true_sum"])
else:
    print(f"SKIP Rust checks: {RUST_BIN} not found (cd rust && cargo build --release)")
print("ALL PASSED" if not fails else f"{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
