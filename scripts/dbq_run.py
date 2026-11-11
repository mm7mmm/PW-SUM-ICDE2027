"""Database workload (next stage): a bounded SUM over a fixed public time window of the Taxi table.

  SELECT SUM(LEAST(trip_distance, 20)) / 20 FROM yellow_trips
  WHERE tpep_pickup_datetime >= :lo AND tpep_pickup_datetime < :hi AND trip_distance BETWEEN 0 AND 200

The raw monthly parquet files are the table; their row groups are split into P partitions scanned in parallel (read-only).
The cleaning rules and the bound are those of the main experiments; N (rows in the window) is treated as public, and the neighbouring
relation is replace-one over the rows of the window. PW-Sum and Prio3Sum-A read the same partitioned input. For every bandwidth the
two methods run one after the other with the same thread count; each run re-scans the partitions. Stages (scan, quantise, PW-Sum
sampling + integerisation in Python; encryption, every network hop, Shuffler, servers in Rust) are timed separately and summed.
Writes results/dbq/runs.jsonl."""
import os as _os, sys as _sys  # repository paths: src/pwexp/paths.py
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
from pwexp.paths import PWSUM_ROOT, DATA_DIR, RUST_BIN  # noqa: E402,F401

import glob
import json
import math
import multiprocessing as mpc
import os
import subprocess
import sys
import time

import numpy as np
import pyarrow.parquet as pq

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, f"{ROOT}/src")
sys.path.insert(0, f"{ROOT}/scripts")
from pwexp import pw_core_task as pc  # noqa: E402
from pwexp import pw_sampler as smp  # noqa: E402
import importlib.util  # noqa: E402

_sp = importlib.util.spec_from_file_location("s9", f"{ROOT}/scripts/s9_params_radius.py")
s9 = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(s9)

FILES = sorted(glob.glob(f"{DATA_DIR}/Taxi/raw/yellow_tripdata_2025-0*.parquet"))
BIN = RUST_BIN
OUT = f"{ROOT}/results/dbq"
P, THREADS, EPS, GAMMA = 16, 16, 1.0, 0.1
WINDOWS = {"W1": ("2025-01-01", "2025-01-10"), "W2": ("2025-01-01", "2025-04-01")}
BWS = (100, 1000, 10000)
REPS = {"W1": 3, "W2": 2}
SEED = 20261119


def units():
    u = []
    for f in FILES:
        md = pq.ParquetFile(f).metadata
        u += [(f, g, md.row_group(g).num_rows) for g in range(md.num_row_groups)]
    parts = [[] for _ in range(P)]
    load = [0] * P
    for f, g, n in sorted(u, key=lambda x: -x[2]):
        i = int(np.argmin(load))
        parts[i].append((f, g))
        load[i] += n
    return parts


def scan(args):
    part, lo, hi = args
    lo, hi = np.datetime64(lo), np.datetime64(hi)
    out = []
    for f, g in part:
        t = pq.ParquetFile(f).read_row_group(g, columns=["tpep_pickup_datetime", "trip_distance"])
        ts = t.column(0).to_numpy(zero_copy_only=False).astype("datetime64[us]")
        d = np.asarray(t.column(1).to_numpy(zero_copy_only=False), dtype=np.float64)
        m = (ts >= lo) & (ts < hi) & np.isfinite(d) & (d >= 0) & (d <= 200)
        out.append(np.minimum(d[m], 20.0) / 20.0)
    return np.concatenate(out) if out else np.empty(0)


def scan_all(parts, lo, hi, pool):
    t0 = time.perf_counter()
    ys = pool.map(scan, [(p, lo, hi) for p in parts])
    y = np.concatenate(ys)
    return y, time.perf_counter() - t0


def quantise(y, K, seed, path):
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    z = y * K
    b = np.floor(z)
    x = (b + (rng.random(len(y)) < z - b)).astype(np.uint32)
    x.tofile(path)
    return int(x.sum()), time.perf_counter() - t0


def prio_K(N, g, eps=1.0):
    v = 2 * (1 + g) / eps ** 2
    K = 1
    while True:
        q = math.exp(-eps / K)
        if (2 * q / (1 - q) ** 2) / K ** 2 + N / (4 * K ** 2) <= v:
            return K
        K += 1


def pw_noise(r, rng, path):
    K = r["K"]
    a, b = pc.weights(K, r["zeta"], "pw", EPS)
    t = np.array(r["t"], np.int64)
    t0 = time.perf_counter()
    _, _, V = smp.sample_s(a, b, t.astype(float), 1, rng, keep_v=1)
    t_samp = time.perf_counter() - t0
    xi = rng.random(1)
    t0 = time.perf_counter()
    U, _ = smp.integerize(V, xi)
    U = np.rint(U[0]).astype(np.int64)
    c = np.r_[0, np.ones(K - 1, np.int64), 0]
    H = t + c + U
    t_int = time.perf_counter() - t0
    jj = np.arange(K + 1, dtype=np.int64)
    assert np.all(H >= 0) and int(H.sum()) == r["M"]
    W = int(jj @ U)
    with open(path, "w") as f:
        f.write(f"{W} {int(jj @ (t + c))} {r['M']}\n" + " ".join(str(int(v)) for v in H) + "\n")
    return t_samp, t_int, W


def rust(args):
    r = subprocess.run(["/usr/bin/time", "-v", BIN] + [str(a) for a in args], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-2000:])
    res = json.loads(r.stdout.strip().splitlines()[-1])
    res["peak_rss_kB"] = next(int(l.split(":")[1]) for l in r.stderr.splitlines() if "Maximum resident set size" in l)
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    fn = f"{OUT}/runs.jsonl"
    done = {json.loads(l)["key"] for l in open(fn)} if os.path.exists(fn) else set()
    parts = units()
    with mpc.Pool(P) as pool:
        for wname, (lo, hi) in WINDOWS.items():
            y, _ = scan_all(parts, lo, hi, pool)
            N = len(y)
            truth = float(y.sum())
            r = s9.work(("dbq", "Taxi", N, EPS, "1/N^2", GAMMA))
            assert r["iv_rho_ok"] and r["iv_mse_ok"] and r["iv_t_minimal"] and r["floor_ok_to_K_safe"]
            Kp = prio_K(N, GAMMA)
            print(f"{wname}: N={N} truth={truth:.3f} PW K={r['K']} M={r['M']} zeta={r['zeta']} | Prio K={Kp}", flush=True)
            json.dump({k: v for k, v in r.items()}, open(f"{OUT}/{wname}_pw_params.json", "w"))
            for rep in range(REPS[wname]):
                for bw in BWS:
                    for meth in ("PW-Sum", "Prio3Sum"):
                        key = f"{wname}_{meth}_bw{bw}_rep{rep}"
                        if key in done:
                            continue
                        y2, t_scan = scan_all(parts, lo, hi, pool)
                        assert len(y2) == N
                        vpath = f"{OUT}/{wname}_{meth}.u32"
                        seed = SEED + 1000 * rep + bw
                        rec = {"key": key, "window": wname, "lo": lo, "hi": hi, "N": N, "method": meth, "bw_mbps": bw, "rep": rep, "threads": THREADS, "partitions": P,
                               "t_scan": t_scan, "truth_normalised": truth}
                        if meth == "PW-Sum":
                            xsum, t_q = quantise(y2, r["K"], seed, vpath)
                            hpath = f"{OUT}/{wname}_H_{rep}_{bw}.txt"
                            t_s, t_i, W = pw_noise(r, np.random.default_rng([SEED, rep, bw]), hpath)
                            res = rust(["dbq_pw", vpath, r["K"], hpath, THREADS, bw, seed])
                            rec.update({"K": r["K"], "M": r["M"], "t_quantise": t_q, "t_sample": t_s, "t_integerize": t_i, **res})
                            rec["stage_total"] = t_scan + t_q + res["t_encrypt"] + res["t_upload"] + t_s + t_i + res["t_shuffler"] + res["t_forward"] + res["t_server"]
                            rec["estimate_normalised"] = res["estimate"] / r["K"]
                            rec["ok"] = res["estimate_equals_sum_plus_W"] and res["true_sum"] == xsum
                        else:
                            xsum, t_q = quantise(y2, Kp, seed, vpath)
                            res = rust(["dbq_prio", vpath, Kp, THREADS, bw, seed])
                            rec.update({"K": Kp, "t_quantise": t_q, **res})
                            rec["stage_total"] = (t_scan + t_q + res["t_encrypt"] + res["t_upload"] + res["t_leader1"] + res["t_leader_to_helper"] + res["t_helper"]
                                                  + res["t_helper_to_leader"] + res["t_leader2"] + res["t_collector"])
                            rec["estimate_normalised"] = res["estimate"] / Kp
                            rec["ok"] = res["aggregate_equals_true_sum"] and res["true_sum"] == xsum
                        rec["abs_error_normalised"] = abs(rec["estimate_normalised"] - truth)
                        with open(fn, "a") as f:
                            f.write(json.dumps(rec) + "\n")
                        print(f"done {key} total={rec['stage_total']:.1f}s ok={rec['ok']} err={rec['abs_error_normalised']:.3f}", flush=True)
                        os.remove(vpath)
    print("all done", flush=True)


if __name__ == "__main__":
    main()
