"""Cost (V-D): PW-Sum with the real noise path. For every configuration: exact draw V, xi, integerization, H = b + U (Python,
timed); then `pw4` (Rust, HPKE): whole Shuffler step and whole Server step timed per repetition, checking estimate == sum x + W.
Single thread: 7 reps (N = 1e7: 3). Checkpointed in results/c3b_full_runs.jsonl."""
import os as _os, sys as _sys  # repository paths: src/pwexp/paths.py
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
from pwexp.paths import PWSUM_ROOT, DATA_DIR, RUST_BIN  # noqa: E402,F401

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXP = PWSUM_ROOT
sys.path.insert(0, f"{EXP}/src")
from pwexp import pw_core_task as pc  # noqa: E402
from pwexp import pw_sampler as smp  # noqa: E402

BIN = RUST_BIN
OUT = f"{ROOT}/results/c3b_full_runs.jsonl"
HDIR = f"{ROOT}/results/c3b_H"
SEED = 20261117


def prepare(p, idx):
    N, g, K = p["N"], p["gamma"], p["K"]
    a, b = pc.weights(K, p["zeta"], "pw", 1.0)
    t = np.array(p["t"], np.int64)
    rng = np.random.default_rng([SEED, idx])
    t0 = time.perf_counter()
    _, _, V = smp.sample_s(a, b, t.astype(float), 1, rng, keep_v=1)
    sample_s = time.perf_counter() - t0
    xi = rng.random(1)
    ts = []
    for _ in range(20):
        t0 = time.perf_counter()
        U, _ = smp.integerize(V, xi)
        U = np.rint(U).astype(np.int64)
        c = np.r_[0, np.ones(K - 1, np.int64), 0]
        H = t + c + U[0]
        ts.append(time.perf_counter() - t0)
    jj = np.arange(K + 1, dtype=np.int64)
    kappa = int(jj @ (t + c))
    assert np.all(H >= 0) and int(H.sum()) == p["M"]
    W = int(jj @ U[0])
    assert W == int(np.floor(V[0] @ jj.astype(float) + xi[0]))
    fn = f"{HDIR}/N{N}_g{g}.txt"
    with open(fn, "w") as f:
        f.write(f"{W} {kappa} {p['M']}\n" + " ".join(str(int(v)) for v in H) + "\n")
    return fn, {"this_draw_sampler_s": sample_s, "integerize_s": float(np.median(ts)), "W": W}


def run(job):
    key, p, T, reps, fn, extra = job
    t0 = time.time()
    r = subprocess.run(["/usr/bin/time", "-v", BIN, "pw4", str(p["N"]), str(p["K"]), fn, str(reps), str(T), "31"], capture_output=True, text=True)
    if r.returncode != 0:
        print("FAILED", key, r.stderr[-1500:], flush=True)
        return
    res = json.loads(r.stdout.strip().splitlines()[-1])
    rss = next(int(l.split(":")[1]) for l in r.stderr.splitlines() if "Maximum resident set size" in l)
    rec = {"key": key, "N": p["N"], "gamma": p["gamma"], "K": p["K"], "M": p["M"], "threads": T, "reps": reps, **extra, "result": res, "peak_rss_kB": rss, "wall_s": time.time() - t0, "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"done {key} wall={rec['wall_s']:.0f}s ok={res['estimate_equals_sum_plus_W']} rss={rss * 1024 / 1e9:.2f}GB", flush=True)


def main():
    os.makedirs(HDIR, exist_ok=True)
    params = {(r["N"], r["gamma"]): r for r in json.load(open(f"{EXP}/params/pw_params_cost.json")) if r["eps"] == 1.0}
    done = {json.loads(l)["key"] for l in open(OUT)} if os.path.exists(OUT) else set()
    small, big, par = [], [], []
    idx = 0
    for N in (10 ** 5, 10 ** 6, 10 ** 7):
        for g in (0.1, 0.5):
            idx += 1
            p = params[(N, g)]
            fn, extra = prepare(p, idx)
            print("prepared", N, g, extra, flush=True)
            for T in (1, 64):
                reps = 3 if (N == 10 ** 7 and T == 1) else 7
                key = f"pw4_N{N}_g{g}_T{T}"
                if key not in done:
                    (par if T == 64 else (big if N == 10 ** 7 else small)).append((key, p, T, reps, fn, extra))

    def serial(js):
        for j in js:
            run(j)
    with ThreadPoolExecutor(3) as ex:
        fs = [ex.submit(serial, small)] + [ex.submit(run, j) for j in big]
        for f in fs:
            f.result()
    serial(par)
    print("all done", flush=True)


if __name__ == "__main__":
    main()
