"""Same-schedule rerun of PW-Sum (pw4, real H) and Prio3Sum (prio2) with one binary (target) and one job at a time, so that both methods
see the same machine state, core count, concurrency (none) and timing boundaries. PW and Prio jobs alternate per configuration.
N in {1e5, 1e6, 1e7}, gamma in {0.1, 0.5}, eps = 1; threads 1 and 64; 7 repetitions (N = 1e7 single thread: 3).
The load average before each job is recorded. Checkpointed in results/c4_same_schedule.jsonl (new file)."""
import os as _os, sys as _sys  # repository paths: src/pwexp/paths.py
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
from pwexp.paths import PWSUM_ROOT, DATA_DIR, RUST_BIN  # noqa: E402,F401

import json
import math
import os
import subprocess
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXP = PWSUM_ROOT
BIN = RUST_BIN
OUT = f"{ROOT}/results/c4_same_schedule.jsonl"


def prio_K(N, g, eps=1.0):
    v = 2 * (1 + g) / eps ** 2
    K = 1
    while True:
        q = math.exp(-eps / K)
        if (2 * q / (1 - q) ** 2) / K ** 2 + N / (4 * K ** 2) <= v:
            return K
        K += 1


def run(key, argv, extra):
    load = os.getloadavg()
    t0 = time.time()
    r = subprocess.run(["/usr/bin/time", "-v", BIN] + [str(a) for a in argv], capture_output=True, text=True)
    if r.returncode != 0:
        print("FAILED", key, r.stderr[-1500:], flush=True)
        return
    res = json.loads(r.stdout.strip().splitlines()[-1])
    rss = next(int(l.split(":")[1]) for l in r.stderr.splitlines() if "Maximum resident set size" in l)
    rec = {"key": key, **extra, "result": res, "peak_rss_kB": rss, "wall_s": time.time() - t0, "loadavg_before": load, "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"done {key} wall={rec['wall_s']:.0f}s load={load[0]:.1f}", flush=True)


def main():
    params = {(r["N"], r["gamma"]): r for r in json.load(open(f"{EXP}/params/pw_params_cost.json")) if r["eps"] == 1.0}
    prev = {json.loads(l)["key"]: json.loads(l) for l in open(f"{ROOT}/results/c3b_full_runs.jsonl")}
    done = {json.loads(l)["key"] for l in open(OUT)} if os.path.exists(OUT) else set()
    jobs = []
    for T in (64, 1):                      # short 64-thread jobs first
        for N in (10 ** 5, 10 ** 6, 10 ** 7):
            for g in (0.1, 0.5):
                reps = 3 if (N == 10 ** 7 and T == 1) else 7
                p = params[(N, g)]
                fn = f"{ROOT}/results/c3b_H/N{N}_g{g}.txt"
                pv = prev[f"pw4_N{N}_g{g}_T{T}"]
                jobs.append((f"pw4_N{N}_g{g}_T{T}", ["pw4", N, p["K"], fn, reps, T, 31],
                             {"method": "PW-Sum", "N": N, "gamma": g, "K": p["K"], "M": p["M"], "threads": T, "reps": reps, "integerize_s": pv["integerize_s"]}))
                K = prio_K(N, g)
                jobs.append((f"prio2_N{N}_g{g}_T{T}", ["prio2", N, K, reps, T, 41], {"method": "Prio3Sum", "N": N, "gamma": g, "K": K, "threads": T, "reps": reps}))
    for key, argv, extra in jobs:
        if key not in done:
            run(key, argv, extra)
    print("all done", flush=True)


if __name__ == "__main__":
    main()
