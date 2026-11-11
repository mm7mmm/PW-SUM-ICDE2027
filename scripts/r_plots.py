"""V-C and V-E figures (line charts; PW-Sum is the thick blue solid line). Writes figures/rob_*, abl_*."""
import os as _os, sys as _sys  # repository paths: src/pwexp/paths.py
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
from pwexp.paths import PWSUM_ROOT, DATA_DIR, RUST_BIN  # noqa: E402,F401

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = f"{ROOT}/figures"
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9.5,
                     "legend.fontsize": 8.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e4e4", "grid.linewidth": 0.6, "axes.axisbelow": True})
BLUE, GREY = "#1f5fd6", "#555555"
OR, GR, RD, PU, BR = "#e8890c", "#2ca02c", "#d62728", "#9467bd", "#8c564b"


def nice(v, _=None):
    return f"{v:g}"


def pct(v, _=None):
    return f"{100 * v:g}%"


def pw(ax, x, y, **kw):
    ax.plot(x, y, color=BLUE, lw=3.0, ls=kw.pop("ls", "-"), marker="o", ms=5, mec="white", mew=0.7, zorder=10, **kw)


def fixed(axis, ticks, fmt=nice):
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_minor_locator(NullLocator())
    axis.set_major_formatter(FuncFormatter(fmt))


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_collusion():
    c = pd.read_csv(f"{ROOT}/results/robust/collusion.csv")
    c = c[c.N == 10 ** 6]
    fig, axs = plt.subplots(1, 3, figsize=(8.0, 2.45))
    for ax, e in zip(axs, (0.5, 1.0, 2.0)):
        q = c[c.eps_target == e]
        for m, col, mk in (("GKMPS", OR, "^"), ("BBGN", GR, "v")):
            t = q[q.method == m].sort_values("c")
            ax.plot(t.c, t.eps_sum_lower_bound, color=col, lw=1.9, marker=mk, ms=7, mfc="white", mew=1.3, zorder=4)
        t = q[q.method == "GKMPS"].sort_values("c")
        pw(ax, t.c, [e] * len(t))
        ax.set_yscale("log")
        fixed(ax.yaxis, [v for v in (0.5, 1, 2, 5, 10, 20) if v >= e * 0.9])
        ax.set_xticks((0, 0.25, 0.5, 0.75, 1))
        ax.set_xlabel("Fraction of colluding users", labelpad=1)
        ax.set_title(rf"Target $\epsilon={e:g}$", pad=3, fontsize=9)
    axs[0].set_ylabel(r"Privacy loss $\epsilon$")
    h = [plt.Line2D([], [], color=BLUE, lw=3.0, marker="o", ms=5, mec="white", mew=0.7, label="PW-Sum (Theorem 1)"), plt.Line2D([], [], color=OR, lw=1.9, marker="^", ms=6.5, mfc="white", label="GKMPS (lower bound)"), plt.Line2D([], [], color=GR, lw=1.9, marker="v", ms=6.5, mfc="white", label="BBGN (lower bound)")]
    fig.legend(handles=h, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0), columnspacing=1.5, handlelength=2.8)
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=0.8)
    save(fig, "rob_collusion")


def fig_poison():
    p = pd.read_csv(f"{ROOT}/results/robust/poison.csv")
    m = pd.read_csv(f"{ROOT}/results/robust/poison_methods.csv")
    fig, axs = plt.subplots(1, 3, figsize=(8.0, 2.55))
    F = (0.001, 0.01, 0.05, 0.1, 0.3)
    ax = axs[0]
    q = m[m.N == 10 ** 6].sort_values("frac")
    ax.plot(q.frac, q.LDP, color=RD, lw=1.9, marker="s", ms=6, label="LDP")
    ax.plot(q.frac, q.GKMPS, color=OR, lw=1.9, marker="^", ms=7, mfc="white", mew=1.3, label="GKMPS")
    ax.plot(q.frac, q.BBGN.abs(), color=GR, lw=1.9, marker="v", ms=7, mfc="white", mew=1.3, label="BBGN")
    ax.plot(q.frac, q["bound_|A|"], color="#000000", ls=":", lw=1.3, zorder=12, label=r"Bound $|\mathcal{A}|$")
    pw(ax, q.frac, q["PW-Sum"], label="PW-Sum")
    ax.set_xscale("log")
    fixed(ax.xaxis, F, pct)
    ax.tick_params(axis="x", labelsize=7.2)
    ax.set_yscale("log")
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Controlled users", labelpad=1)
    ax.set_ylabel("Shift of the estimate")
    ax.set_ylim(2e2, 3e12)
    ax.set_title(r"Largest attack, $N=10^6$", pad=3, fontsize=9)
    ax.legend(frameon=False, fontsize=6.8, loc="upper left", ncol=1, handlelength=2.2, labelspacing=0.2, borderaxespad=0.1)
    S = {"Uniform_N1000000": (BLUE, "o", "Uniform"), "SF_N42928": ("#111111", "s", "SF"), "ACS_N2912790": (GR, "v", "ACS"), "Taxi_N15167858": (PU, "D", "Taxi")}
    for ax, col, yl, title in ((axs[1], "max_abs_shift_over_bound", r"max $|$shift$|$ / bound", "PW-Sum vs. Theorem 5"), (axs[2], "rel_err_attack", "Relative error", "PW-Sum under attack")):
        for case, (cc, mk, lab) in S.items():
            t = p[(p.case == case) & (p.attack == "upper") & (p.frac > 0)].sort_values("frac")
            if case.startswith("Uniform"):
                pw(ax, t.frac, t[col], label=lab)
            else:
                ax.plot(t.frac, t[col], color=cc, lw=1.9, marker=mk, ms=6, label=lab, zorder=4)
        ax.set_xscale("log")
        fixed(ax.xaxis, F, pct)
        ax.tick_params(axis="x", labelsize=7.2)
        ax.set_xlabel("Controlled users", labelpad=1)
        ax.set_ylabel(yl)
        ax.set_title(title, pad=3, fontsize=9)
    axs[1].set_ylim(0, 1.05)
    axs[1].legend(frameon=False, fontsize=7, loc="lower left", handlelength=2.2, labelspacing=0.2, borderaxespad=0.1)
    axs[2].set_yscale("log")
    axs[2].yaxis.set_minor_locator(NullLocator())
    fig.tight_layout(w_pad=0.7)
    save(fig, "rob_poison")


def fig_ablation():
    x = pd.read_csv(f"{ROOT}/results/ablation/shared_xi.csv")
    b = pd.read_csv(f"{ROOT}/results/ablation/beta0_best.csv")
    a = pd.read_csv(f"{ROOT}/results/ablation/anchor.csv")
    fig, axs = plt.subplots(1, 3, figsize=(8.0, 2.5))
    N10 = (1e3, 1e4, 1e5, 1e6, 1e7)
    def p10(v, _=None):
        return rf"$10^{{{int(round(np.log10(v)))}}}$"
    ax = axs[0]
    for g, ls in ((0.1, "-"), (0.5, "--")):
        t = x[x.gamma == g].sort_values("N")
        ax.plot(t.N, t.indep_noise_mse, color=OR, ls=ls, lw=1.9, marker="^", ms=6.5, mfc="white", mew=1.2)
        pw(ax, t.N, t.shared_noise_mse, ls=ls)
    ax.set_xscale("log")
    fixed(ax.xaxis, N10, p10)
    ax.set_yscale("log")
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel(r"$N$", labelpad=1)
    ax.set_ylabel(r"Noise term of MSE")
    ax.set_title("Integerization", pad=3, fontsize=9)
    ax.legend(handles=[plt.Line2D([], [], color=BLUE, lw=3, marker="o", ms=5, mec="white", label="Shared $\\xi$"), plt.Line2D([], [], color=OR, lw=1.9, marker="^", ms=6, mfc="white", label="Independent"), plt.Line2D([], [], color=GREY, lw=1.6, label=r"$\gamma=0.1$"), plt.Line2D([], [], color=GREY, lw=1.6, ls="--", label=r"$\gamma=0.5$")], frameon=False, fontsize=7, loc="upper left", handlelength=2.2, labelspacing=0.2, borderaxespad=0.1)
    ax = axs[1]
    b = b.sort_values("N")
    ax.plot(b.N, b.ratio_to_central, color=RD, lw=1.9, marker="s", ms=6, label=r"No sum term ($\beta=0$)")
    pw(ax, b.N, [1.1] * len(b), label=r"PW-Sum ($\gamma=0.1$ target)")
    ax.set_xscale("log")
    fixed(ax.xaxis, N10, p10)
    ax.set_yscale("log")
    fixed(ax.yaxis, (1, 2, 5, 10, 20, 50, 100))
    ax.set_xlabel(r"$N$", labelpad=1)
    ax.set_ylabel(r"$R_{\rm MSE}$")
    ax.set_ylim(0.9, 400)
    ax.set_title("Sum term", pad=3, fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="upper left", handlelength=2.2)
    ax = axs[2]
    q = a[a.gamma == 0.1].sort_values("K")
    t = q[q.anchor == "centre"]
    ax.plot(t.K, t.mean_s, color=OR, lw=1.9, marker="^", ms=6.5, mfc="white", mew=1.2, label="Centre anchor")
    t = q[q.anchor == "endpoint"]
    pw(ax, t.K, t.mean_s, label="Endpoint anchor")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel(r"$K$", labelpad=1)
    ax.set_ylabel("Time per sample (s)")
    ax.set_title(r"Sampler ($\gamma=0.1$)", pad=3, fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="upper left", handlelength=2.2)
    fig.tight_layout(w_pad=0.7)
    save(fig, "abl_design")


if __name__ == "__main__":
    fig_collusion()
    fig_poison()
    fig_ablation()
    print("ok")
