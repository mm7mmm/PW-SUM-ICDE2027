"""V-D figures (line charts; PW-Sum is the thick blue solid line). Writes figures/cost_*.{pdf,png}."""
import os as _os, sys as _sys  # repository paths: src/pwexp/paths.py
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
from pwexp.paths import PWSUM_ROOT, DATA_DIR, RUST_BIN  # noqa: E402,F401

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = f"{ROOT}/figures"
COST2 = PWSUM_ROOT
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9.5,
                     "legend.fontsize": 8.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e4e4", "grid.linewidth": 0.6, "axes.axisbelow": True})
BLUE = "#1f5fd6"
ST = {"PW-Uniform": ("#0f9d9d", "-", "s", 6.5), "GKMPS": ("#e8890c", "-", "^", 7), "BBGN": ("#2ca02c", "-", "v", 7),
      "Prio3-A": ("#8c564b", "-", "P", 7), "Prio3-B": ("#e377c2", "none", "X", 8), "ID-Sum": ("#9467bd", "-", "D", 4.5)}


def nice(v, _=None):
    return f"{v:g}"


def pow10(v, _=None):
    e = np.log10(v)
    return rf"$10^{{{int(round(e))}}}$" if abs(e - round(e)) < 1e-9 else f"{v:g}"


def setx(ax, ticks, fmt=nice):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(fmt))


def pw(ax, x, y, **kw):
    ax.plot(x, y, color=BLUE, lw=3.0, ls="-", marker="o", ms=5, mec="white", mew=0.7, zorder=10, **kw)


def line(ax, x, y, name, **kw):
    c, ls, m, ms = ST[name]
    ax.plot(x, y, color=c, ls=ls, lw=1.9, marker=m, ms=ms, mfc="white" if name in ("GKMPS", "BBGN", "Prio3-A") else c, mew=1.2, zorder=4, **kw)


def fig_sampler():
    s = pd.read_csv(f"{ROOT}/results/cost/sampler.csv")
    fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.45))
    G = {0.1: None, 0.05: ("#e8890c", "^"), 0.5: ("#2ca02c", "v")}
    for ax, col, yl in ((axs[0], "acceptance", "Acceptance rate"), (axs[1], "time_mean_s", "Time per sample (s)")):
        for g, st in G.items():
            q = s[s.gamma == g].sort_values("K")
            if st is None:
                pw(ax, q.K, q[col], label=rf"$\gamma={g}$")
            else:
                ax.plot(q.K, q[col], color=st[0], lw=1.9, marker=st[1], ms=6, label=rf"$\gamma={g}$", zorder=4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$K$", labelpad=1)
        ax.set_ylabel(yl)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
    axs[0].legend(frameon=False, fontsize=7.5, loc="lower left", handlelength=2.4, labelspacing=0.25)
    fig.tight_layout(w_pad=0.8)
    fig.savefig(f"{OUT}/cost_sampler.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/cost_sampler.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_comm():
    c = pd.read_csv(f"{ROOT}/results/cost/comm.csv")
    env = pd.read_csv(f"{ROOT}/results/cost/idsum_pareto.csv")
    c = c[c.feasible.astype(bool) | c.method.eq("ID-Sum")]
    M = ["PW-Uniform", "GKMPS", "BBGN", "Prio3-A", "Prio3-B"]
    fig, axs = plt.subplots(1, 4, figsize=(9.2, 2.55))
    panels = [(axs[0], c[(c.N == 10 ** 6) & (c.eps == 1.0)], "gamma", r"$\gamma$", (0.05, 0.1, 0.2, 0.5, 1, 2), nice, r"$N=10^6$, $\epsilon=1$"),
              (axs[1], c[(c.gamma == 0.1) & (c.eps == 1.0)], "N", r"$N$", (1e3, 1e4, 1e5, 1e6, 1e7), pow10, r"$\gamma=0.1$, $\epsilon=1$"),
              (axs[2], c[(c.N == 10 ** 6) & (c.gamma == 0.1)], "eps", r"$\epsilon$", (0.1, 0.2, 0.5, 1, 2), nice, r"$N=10^6$, $\gamma=0.1$")]
    for ax, q, x, xl, ticks, fmt, title in panels:
        for m in M:
            t = q[q.method == m].sort_values(x)
            if len(t):
                line(ax, t[x], t.bytes_per_user, m)
        t = q[q.method == "PW-Sum"].sort_values(x)
        pw(ax, t[x], t.bytes_per_user)
        setx(ax, ticks, fmt)
        ax.set_yscale("log")
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_ylim(60, 3e8)
        ax.set_xlabel(xl, labelpad=1)
        ax.set_title(title, pad=3, fontsize=9)
    axs[0].set_ylabel("Bytes per user")
    ax = axs[3]
    q = c[(c.N == 10 ** 6) & (c.eps == 1.0)]
    for m in ("PW-Uniform", "BBGN", "Prio3-A"):
        t = q[q.method == m].sort_values("gamma")
        line(ax, t.bytes_per_user, t.R_worst, m)
    t = env[env.N == 10 ** 6].sort_values("bytes_per_user")
    ax.plot(t.bytes_per_user, t.R_worst, color=ST["ID-Sum"][0], lw=1.9, zorder=4)
    t = q[q.method == "PW-Sum"].sort_values("gamma")
    pw(ax, t.bytes_per_user, t.R_worst)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(85, 1300)
    ax.set_ylim(0.8, 500)
    ax.xaxis.set_major_locator(FixedLocator((100, 200, 500, 1000)))
    ax.xaxis.set_major_formatter(FuncFormatter(nice))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Bytes per user", labelpad=1)
    ax.set_ylabel(r"$R_{\rm MSE}$ (worst case)", labelpad=1)
    ax.set_title(r"Trade-off, $N=10^6$", pad=3, fontsize=9)
    h = [plt.Line2D([], [], color=BLUE, lw=3.0, marker="o", ms=5, mec="white", mew=0.7, label="PW-Sum")]
    for m in ("PW-Uniform", "GKMPS", "BBGN", "Prio3-A", "Prio3-B", "ID-Sum"):
        cc, ls, mk, ms = ST[m]
        h.append(plt.Line2D([], [], color=cc, ls=ls, lw=1.9, marker=mk, ms=ms * 0.85, mfc="white" if m in ("GKMPS", "BBGN", "Prio3-A") else cc, mew=1.1, label=m))
    fig.legend(handles=h, loc="upper center", ncol=len(h), frameon=False, bbox_to_anchor=(0.5, 1.0), columnspacing=1.1, handlelength=2.5, handletextpad=0.45)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=0.5)
    fig.savefig(f"{OUT}/cost_comm_v2.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/cost_comm_v2.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_ablation():
    c = pd.read_csv(f"{ROOT}/results/cost/comm.csv")
    c = c[c.eps == 1.0]
    p = c[c.method == "PW-Sum"].set_index(["N", "gamma"]).M
    u = c[c.method == "PW-Uniform"].set_index(["N", "gamma"]).M
    fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.45))
    G = (0.05, 0.1, 0.2, 0.5, 1, 2)
    ax = axs[0]
    x = list(G)
    ax.plot(x, [u[(10 ** 6, g)] / 1e6 for g in G], color=ST["PW-Uniform"][0], lw=1.9, marker="s", ms=6, label="Uniform weights", zorder=4)
    pw(ax, x, [p[(10 ** 6, g)] / 1e6 for g in G], label="Position weights")
    setx(ax, (0.05, 0.2, 0.5, 2))
    ax.set_yscale("log")
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel(r"$\gamma$", labelpad=1)
    ax.set_ylabel(r"Dummies per user $M/N$")
    ax.set_title(r"$N=10^6$", pad=3, fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right", handlelength=2.4)
    ax = axs[1]
    Ns = (10 ** 3, 10 ** 5, 10 ** 7)
    S = {10 ** 3: ("#e8890c", "^"), 10 ** 5: (BLUE, "o"), 10 ** 7: ("#2ca02c", "v")}
    for N in Ns:
        y = [u[(N, g)] / p[(N, g)] for g in G]
        ax.plot(x, y, color=S[N][0], lw=3.0 if N == 10 ** 5 else 1.9, marker=S[N][1], ms=6, label=rf"$N=10^{int(np.log10(N))}$")
    setx(ax, (0.05, 0.2, 0.5, 2))
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator((1, 2, 5, 10, 20)))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(nice))
    ax.set_xlabel(r"$\gamma$", labelpad=1)
    ax.set_ylabel("Uniform / position dummies")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right", handlelength=2.4)
    fig.tight_layout(w_pad=0.8)
    fig.savefig(f"{OUT}/cost_ablation.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/cost_ablation.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def run_tables():
    s = pd.read_csv(f"{ROOT}/results/cost/sampler.csv").set_index(["N", "gamma"]).time_mean_s
    rows = []
    for l in open(f"{COST2}/results/c3_hpke_runs.jsonl"):
        r = json.loads(l)
        x = r["result"]
        sm = float(s[(r["N"], r["gamma"])])
        d, sh, sv = np.median(x["shuffler_dummy_enc_s"]), np.median(x["shuffler_shuffle_s"]), np.median(x["server_decrypt_count_s"])
        rows.append({"method": "PW-Sum", "N": r["N"], "gamma": r["gamma"], "threads": r["threads"], "K": r["K"], "M": r["M"], "reps": r["reps"], "user_us": 1e6 * np.median(x["user_enc_per_msg_s"]), "sampler_s": sm,
                     "dummy_enc_s": d, "shuffle_s": sh, "server_s": sv, "shuffler_s": sm + d + sh, "server_side_s": sm + d + sh + sv, "peak_gb": r["peak_rss_kB"] * 1024 / 1e9, "ok": x["estimate_ok"], "ct_len": x["ct_len"], "bytes": (2 * r["N"] + r["M"]) * x["ct_len"] / r["N"]})
    for l in open(f"{COST2}/results/m4_runs.jsonl"):
        r = json.loads(l)
        if r.get("method") != "Prio3Sum":
            continue
        x = r["result"]
        le, he = np.median([a["total_s"] for a in x["leader"]]), np.median([a["total_s"] for a in x["helper"]])
        rows.append({"method": "Prio3Sum", "N": r["N"], "gamma": r["gamma"], "threads": r["threads"], "K": r["K"], "M": "", "reps": r["reps"], "user_us": 1e6 * np.median(x["client_per_report_s"]), "sampler_s": "",
                     "dummy_enc_s": "", "shuffle_s": "", "server_s": "", "shuffler_s": le, "server_side_s": le + he, "helper_s": he, "peak_gb": r["peak_rss_kB"] * 1024 / 1e9, "ok": x["aggregate_equals_true_sum"], "ct_len": "", "bytes": x["bytes"]["upload_report_bytes"] + x["bytes"]["server_to_server_per_report"]})
    d = pd.DataFrame(rows)
    d.to_csv(f"{ROOT}/results/cost/run_summary.csv", index=False)
    return d


def fig_run():
    d = run_tables()
    d["N"] = d["N"].astype(float)
    fig, axs = plt.subplots(1, 3, figsize=(8.0, 2.5))
    def curves(ax, col, T):
        for g, ls in ((0.1, "-"), (0.5, "--")):
            q = d[(d.method == "Prio3Sum") & (d.gamma == g) & (d.threads == T)].sort_values("N")
            ax.plot(q.N, q[col], color=ST["Prio3-A"][0], ls=ls, lw=1.9, marker="P", ms=7, mfc="white", mew=1.2, zorder=4)
            q = d[(d.method == "PW-Sum") & (d.gamma == g) & (d.threads == T)].sort_values("N")
            ax.plot(q.N, q[col], color=BLUE, ls=ls, lw=3.0, marker="o", ms=5, mec="white", mew=0.7, zorder=10)
    for ax, col, T, yl, title in ((axs[0], "server_side_s", 1, "Server-side time (s)", "1 thread"), (axs[1], "server_side_s", 64, "Server-side time (s)", "64 threads"), (axs[2], "peak_gb", 1, "Peak memory (GB)", "1 thread")):
        curves(ax, col, T)
        setx(ax, (1e5, 1e6, 1e7), pow10)
        ax.set_yscale("log")
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_xlabel(r"$N$", labelpad=1)
        ax.set_ylabel(yl)
        ax.set_title(title, pad=3, fontsize=9)
    h = [plt.Line2D([], [], color=BLUE, lw=3.0, marker="o", ms=5, mec="white", mew=0.7, label=r"PW-Sum, $\gamma=0.1$"),
         plt.Line2D([], [], color=BLUE, lw=3.0, ls="--", label=r"PW-Sum, $\gamma=0.5$"),
         plt.Line2D([], [], color=ST["Prio3-A"][0], lw=1.9, marker="P", ms=6, mfc="white", label=r"Prio3Sum, $\gamma=0.1$"),
         plt.Line2D([], [], color=ST["Prio3-A"][0], lw=1.9, ls="--", marker="P", ms=6, mfc="white", label=r"Prio3Sum, $\gamma=0.5$")]
    fig.legend(handles=h, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0), columnspacing=1.5, handlelength=2.8)
    fig.tight_layout(rect=(0, 0, 1, 0.89), w_pad=0.8)
    fig.savefig(f"{OUT}/cost_run.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/cost_run.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    return d


if __name__ == "__main__":
    fig_sampler()
    fig_comm()
    fig_ablation()
    print("ok")
