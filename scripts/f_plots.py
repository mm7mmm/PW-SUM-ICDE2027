"""Robustness and ablation figures. Collusion uses the numerical lower bound; ablation separates rounding coordinates from the shared
offset and uses the same batch size for both anchors; run cost uses the real noise path with whole-step timing."""
import os as _os, sys as _sys  # repository paths: src/pwexp/paths.py
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
from pwexp.paths import PWSUM_ROOT, DATA_DIR, RUST_BIN  # noqa: E402,F401

import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
S = f"{ROOT}/scripts"
COST2 = PWSUM_ROOT


def load(name):
    sp = importlib.util.spec_from_file_location(name, f"{S}/{name}.py")
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


rp, cp = load("r_plots"), load("c_plots")
plt, BLUE, OR, GR, RD, PU = rp.plt, rp.BLUE, rp.OR, rp.GR, rp.RD, rp.PU
from matplotlib.ticker import NullLocator  # noqa: E402


def collusion():
    orig = rp.pd.read_csv
    def rd(p, *a, **k):
        d = orig(p.replace("robust/collusion.csv", "robust/collusion_lb.csv"), *a, **k)
        return d.assign(eps_sum_lower_bound=d.eps_numerical_lower_bound) if "eps_numerical_lower_bound" in d else d
    rp.pd.read_csv = rd
    sv = rp.save
    rp.save = lambda f, n: sv(f, n + "_v2")
    rp.fig_collusion()
    rp.pd.read_csv = orig
    rp.save = sv


def poison():
    sv = rp.save
    def s2(f, n):
        f.axes[0].set_title(r"Example attacks, $N=10^6$", pad=3, fontsize=9)
        sv(f, n + "_v2")
    rp.save = s2
    rp.fig_poison()
    rp.save = sv


def ablation():
    x = pd.read_csv(f"{ROOT}/results/ablation_v2/rounding.csv")
    a = pd.read_csv(f"{ROOT}/results/ablation_v2/anchor.csv")
    b = pd.read_csv(f"{ROOT}/results/ablation/beta0_best.csv").sort_values("N")
    fig, axs = plt.subplots(1, 4, figsize=(9.2, 2.45))
    N10 = (1e3, 1e4, 1e5, 1e6, 1e7)
    def p10(v, _=None):
        return rf"$10^{{{int(round(np.log10(v)))}}}$"
    x1 = x[x.gamma == 0.1]
    ST = {"cum_indep": (GR, "v", "Cumulative, independent $\\xi_j$"), "direct": (OR, "^", "Direct rounding")}
    for ax, col, yl, ttl in ((axs[0], "noise_mse", "Noise term of MSE", "Rounding: noise term"), (axs[1], "max_abs_err", r"max $|U_j-V_j|$", "Rounding: coordinate error")):
        for m, (c, mk, lab) in ST.items():
            t = x1[x1.method == m].sort_values("N")
            ax.plot(t.N, t[col], color=c, lw=1.9, marker=mk, ms=6.5, mfc="white", mew=1.2, label=lab, zorder=4)
        t = x1[x1.method == "shared"].sort_values("N")
        rp.pw(ax, t.N, t[col], label="Cumulative, shared $\\xi$")
        ax.set_xscale("log")
        rp.fixed(ax.xaxis, N10, p10)
        ax.set_xlabel(r"$N$", labelpad=1)
        ax.set_ylabel(yl)
        ax.set_title(ttl, pad=3, fontsize=9)
    axs[0].set_yscale("log")
    axs[0].yaxis.set_minor_locator(NullLocator())
    axs[0].set_ylim(1.5, 6000)
    axs[0].legend(frameon=False, fontsize=6.6, loc="upper left", handlelength=2.1, labelspacing=0.2, borderaxespad=0.1)
    ax = axs[2]
    ax.plot(b.N, b.ratio_to_central, color=RD, lw=1.9, marker="s", ms=6, label=r"No sum term ($\beta=0$)")
    rp.pw(ax, b.N, [1.1] * len(b), label=r"PW-Sum target ($\gamma=0.1$)")
    ax.set_xscale("log")
    rp.fixed(ax.xaxis, N10, p10)
    ax.set_yscale("log")
    rp.fixed(ax.yaxis, (1, 2, 5, 10, 20, 50, 100))
    ax.set_ylim(0.9, 400)
    ax.set_xlabel(r"$N$", labelpad=1)
    ax.set_ylabel(r"$R_{\rm MSE}$")
    ax.set_title("Sum term", pad=3, fontsize=9)
    ax.legend(frameon=False, fontsize=6.8, loc="upper left", handlelength=2.1)
    ax = axs[3]
    q = a[(a.gamma == 0.1) & (a.batch == 4096)].sort_values("K")
    t = q[q.anchor == "centre"]
    ax.plot(t.K, t.mean_s, color=OR, lw=1.9, marker="^", ms=6.5, mfc="white", mew=1.2, label="Centre coordinate")
    t = q[q.anchor == "endpoint"]
    rp.pw(ax, t.K, t.mean_s, label="Endpoint coordinate")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel(r"$K$", labelpad=1)
    ax.set_ylim(0.01, 300)
    ax.set_ylabel("Time per sample (s)")
    ax.set_title(r"Sampler (batch 4096)", pad=3, fontsize=9)
    ax.legend(frameon=False, fontsize=6.8, loc="upper left", handlelength=2.1)
    fig.tight_layout(w_pad=0.5)
    rp.save(fig, "abl_design_v2")


def run_table():
    s = pd.read_csv(f"{ROOT}/results/cost/sampler.csv").set_index(["N", "gamma"]).time_mean_s
    rows = []
    for l in open(f"{COST2}/results/c3b_full_runs.jsonl"):
        r = json.loads(l)
        x = r["result"]
        sm = float(s[(r["N"], r["gamma"])])
        sh, sv = np.array(x["shuffler_total_s"]), np.array(x["server_total_s"])
        rows.append({"method": "PW-Sum", "N": r["N"], "gamma": r["gamma"], "threads": r["threads"], "K": r["K"], "M": r["M"], "reps": r["reps"], "user_us": 1e6 * np.median(x["user_enc_per_msg_s"]), "sampler_s": sm, "integerize_s": r["integerize_s"],
                     "shuffler_step_s": float(np.median(sh)), "server_step_s": float(np.median(sv)), "dummy_enc_s": float(np.median(x["shuffler_dummy_enc_s"])), "server_side_s": sm + r["integerize_s"] + float(np.median(sh + sv)), "peak_gb": r["peak_rss_kB"] * 1024 / 1e9, "ok": x["estimate_equals_sum_plus_W"], "ct_len": x["ct_len"], "bytes": (2 * r["N"] + r["M"]) * x["ct_len"] / r["N"]})
    for l in open(f"{COST2}/results/m4_runs.jsonl"):
        r = json.loads(l)
        if r.get("method") != "Prio3Sum":
            continue
        x = r["result"]
        le, he = np.array([a["total_s"] for a in x["leader"]]), np.array([a["total_s"] for a in x["helper"]])
        rows.append({"method": "Prio3Sum", "N": r["N"], "gamma": r["gamma"], "threads": r["threads"], "K": r["K"], "M": "", "reps": r["reps"], "user_us": 1e6 * np.median(x["client_per_report_s"]), "shuffler_step_s": float(np.median(le)), "server_step_s": float(np.median(he)), "server_side_s": float(np.median(le + he)), "peak_gb": r["peak_rss_kB"] * 1024 / 1e9, "ok": x["aggregate_equals_true_sum"], "bytes": x["bytes"]["upload_report_bytes"] + x["bytes"]["server_to_server_per_report"]})
    d = pd.DataFrame(rows)
    d.to_csv(f"{ROOT}/results/cost/run_summary_v2.csv", index=False)
    return d


def run_fig():
    d = run_table()
    orig, sv = cp.run_tables, cp.plt.close
    cp.run_tables = lambda: d
    from matplotlib.figure import Figure
    osave = Figure.savefig
    Figure.savefig = lambda self, fn, *a, **k: osave(self, str(fn).replace("cost_run.", "cost_run_v2."), *a, **k)
    cp.fig_run()
    Figure.savefig = osave
    cp.run_tables = orig
    print(d.round(4).to_string())


if __name__ == "__main__":
    collusion()
    poison()
    ablation()
    print("ok")
