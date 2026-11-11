"""V-B, PW-Sum noise term: exact draws V ~ q_t (Algorithm pw2), xi ~ U[0,1), W = floor(s(V) + xi) for every distinct
(N, eps, gamma) of params/pw_params_pu.json. In integer units the estimator error is W + (rounding error), so
MSE = E[W^2]/K^2 + sum f_i(1-f_i)/K^2 exactly. Stored: results/pu/noise/<key>.npz (W, s, meta).
Chunk k of config i uses SeedSequence([SEED, i, k]).
Usage: pu_noise.py --reps 10000 --workers 90 | pu_noise.py --probe 30   (timing of the hardest configs, nothing stored)"""
import os as _os, sys as _sys  # repository paths: src/pwexp/paths.py
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
from pwexp.paths import PWSUM_ROOT, DATA_DIR, RUST_BIN  # noqa: E402,F401

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mpc  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from pwexp import pw_core_task as pc  # noqa: E402
from pwexp import pw_sampler as smp  # noqa: E402

SEED = 20261109
CHUNK = 250
OUT = f"{ROOT}/results/pu/noise"


def key(r):
    return f"N{r['N']}_eps{r['eps']}_g{r['gamma']}"


def configs():
    u = {}
    for r in json.load(open(f"{ROOT}/params/pw_params_pu.json")):
        u.setdefault(key(r), r)
    return [u[k] for k in sorted(u)]


def work(task):
    i, k, n, r = task
    rng = np.random.default_rng(np.random.SeedSequence([SEED, i, k]))
    a, b = pc.weights(r["K"], r["zeta"], "pw", r["eps"])
    t0 = time.perf_counter()
    s, cnt, _ = smp.sample_s(a, b, np.array(r["t"], float), n, rng)
    return i, k, s, np.floor(s + rng.random(n)).astype(np.int64), cnt, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=90)
    ap.add_argument("--probe", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cfgs = configs()
    if args.probe:
        hard = sorted(enumerate(cfgs), key=lambda x: -x[1]["K"] / x[1]["zeta"])[:6]
        for i, r in hard:
            _, _, s, W, cnt, el = work((i, 10 ** 6, args.probe, r))
            print(key(r), "K", r["K"], "zeta", round(r["zeta"], 5), "acc", cnt["acc_box"] / cnt["proposals"],
                  "sec/draw", el / args.probe, "est CPU-h for 1e4", el / args.probe * 1e4 / 3600, flush=True)
        return
    todo = [(i, r) for i, r in enumerate(cfgs) if not os.path.exists(f"{OUT}/{key(r)}.npz")]
    tasks = [(i, k, min(CHUNK, args.reps - k * CHUNK), r) for i, r in todo for k in range((args.reps + CHUNK - 1) // CHUNK)]
    tasks.sort(key=lambda x: -x[3]["K"] / x[3]["zeta"])
    print(len(cfgs), "distinct configs,", len(todo), "to do,", len(tasks), "chunks", flush=True)
    acc = {i: {"s": {}, "W": {}, "left": sum(1 for x in tasks if x[0] == i), "cpu": 0.0, "acc": 0, "prop": 0} for i, _ in todo}
    t0 = time.time()
    with mpc.Pool(args.workers) as pool:
        for i, k, s, W, cnt, el in pool.imap_unordered(work, tasks):
            a = acc[i]
            a["s"][k], a["W"][k] = s, W
            a["cpu"] += el
            a["acc"] += cnt["acc_box"]
            a["prop"] += cnt["proposals"]
            a["left"] -= 1
            if a["left"] == 0:
                r = cfgs[i]
                ks = sorted(a["W"])
                W_all = np.concatenate([a["W"][q] for q in ks])
                s_all = np.concatenate([a["s"][q] for q in ks])
                np.savez_compressed(f"{OUT}/{key(r)}.npz", W=W_all, s=s_all,
                                    meta=json.dumps({"key": key(r), "K": r["K"], "zeta": r["zeta"], "seed": [SEED, i], "chunk": CHUNK,
                                                     "acceptance": a["acc"] / a["prop"], "cpu_seconds": a["cpu"]}))
                print(f"{time.time() - t0:8.0f}s {key(r)} K={r['K']} n={len(W_all)} E[W]={W_all.mean():.2f} "
                      f"E[W^2]/K^2={np.mean(W_all.astype(float) ** 2) / r['K'] ** 2:.4f} cpu={a['cpu']:.0f}s acc={a['acc'] / a['prop']:.2e}", flush=True)
                del acc[i]


if __name__ == "__main__":
    main()
