"""V-B figures (line charts; PW-Sum is the thick blue solid line). Reads results/pu/{cross,decomp}.csv.
Ring markers on the near-central baselines only keep overlapping curves visible. Writes figures/pu_*.{pdf,png}."""
import os as _os, sys as _sys  # repository paths: src/pwexp/paths.py
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
from pwexp.paths import PWSUM_ROOT, DATA_DIR, RUST_BIN  # noqa: E402,F401

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = f"{ROOT}/figures"
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9.5,
                     "legend.fontsize": 8.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e4e4", "grid.linewidth": 0.6, "axes.axisbelow": True})
BLUE = "#1f5fd6"
GREY = "#555555"
ST = {"CDP": ("#111111", "--", "o"), "LDP": ("#d62728", "-", "s"), "GKMPS": ("#e8890c", "-", "^"), "BBGN": ("#2ca02c", "-", "v"),
      "ID-Sum": ("#9467bd", "-", "D"), "Prio3-A": ("#8c564b", "-", "P"), "Prio3-B": ("#e377c2", "-", "X")}
ORDER = ["LDP", "ID-Sum", "Prio3-B", "GKMPS", "BBGN", "Prio3-A", "CDP"]
LEG = ("CDP", "LDP", "GKMPS", "BBGN", "ID-Sum", "Prio3-A", "Prio3-B")
DSETS = ["Uniform", "SF", "ACS", "Taxi"]
BASE_MS = {"GKMPS": 10.5, "BBGN": 10.5, "Prio3-A": 8.5, "CDP": 7.5}
df = pd.read_csv(f"{ROOT}/results/pu/cross.csv")
df["R"] = df["mse"] / df["central"]


def nice(v, _=None):
    return f"{v:g}"


def base_line(ax, q, x, y, name):
    c, ls, m = ST[name]
    q = q.sort_values(x)
    ms = BASE_MS.get(name, 6.5)
    ax.plot(q[x], q[y], color=c, ls=ls, lw=1.9, zorder=3)
    ax.plot(q[x], q[y], ls="none", marker=m, ms=ms, mec=c, mfc=c if name not in BASE_MS else "white", mew=1.4, zorder=4)


def pw_line(ax, q, x, y):
    q = q.sort_values(x)
    ax.plot(q[x], q[y], color=BLUE, lw=3.0, ls="-", marker="o", ms=5, mec="white", mew=0.7, zorder=10)


def legend(fig, extra, skip=(), y=1.0):
    h = [plt.Line2D([], [], color=BLUE, lw=3.0, marker="o", ms=5, mec="white", mew=0.7, label="PW-Sum")]
    for n in LEG:
        if n not in skip:
            h.append(plt.Line2D([], [], color=ST[n][0], ls=ST[n][1], lw=1.9, marker=ST[n][2], ms=BASE_MS.get(n, 6.5) * 0.8, mfc="white" if n in BASE_MS else ST[n][0], mew=1.2, label=n))
    h += list(extra)
    fig.legend(handles=h, loc="upper center", ncol=len(h), frameon=False, bbox_to_anchor=(0.5, y), columnspacing=1.2, handlelength=2.6, handletextpad=0.5)


def panels(sweep, x, y, xlabel, ylabel, fname, xticks=None, ylim=None, yticks=None, ref=None, extra=(), skip=(), xfmt=None):
    fig, axs = plt.subplots(1, 4, figsize=(9.2, 2.5))
    for ax, d in zip(axs, DSETS):
        q = df[(df.dataset == d) & df.sweeps.str.contains(sweep)]
        for name in ORDER:
            if name not in skip:
                base_line(ax, q[q.method == name], x, y, name)
        p = q[q.method == "PW-Sum"]
        pw_line(ax, p, x, y)
        if ref:
            ref(ax, p)
        ax.set_xscale("log")
        ax.set_yscale("log")
        if xticks:
            xs = [v for v in xticks if q[x].min() <= v <= q[x].max()]
            ax.xaxis.set_major_locator(FixedLocator(xs))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.xaxis.set_major_formatter(FuncFormatter(xfmt or nice))
        else:
            ax.xaxis.set_minor_locator(NullLocator())
        if ylim:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(q[y].min() * 0.35, q[y].max() * 3)
        if yticks:
            ax.yaxis.set_major_locator(FixedLocator(yticks))
            ax.yaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_major_formatter(FuncFormatter(nice))
        else:
            ax.yaxis.set_minor_locator(NullLocator())
        ax.set_title(d, pad=3)
        ax.set_xlabel(xlabel, labelpad=1)
    axs[0].set_ylabel(ylabel)
    legend(fig, extra, skip)
    fig.tight_layout(rect=(0, 0, 1, 0.89), w_pad=0.5)
    fig.savefig(f"{OUT}/{fname}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/{fname}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def bound(x):
    def f(ax, p):
        p = p.sort_values(x)
        ax.plot(p[x], p["analytic"], color="#000000", ls=":", lw=1.3, zorder=12)
    return f


def tgt(ax, p):
    p = p.sort_values("gamma")
    ax.plot(p["gamma"], 1 + p["gamma"], color=GREY, ls=":", lw=1.8, zorder=12)


BOUND = plt.Line2D([], [], color="#000000", ls=":", lw=1.3, label="Thm. 2 bound")
EPS_T = (0.1, 0.2, 0.5, 1, 2)
panels("eps", "eps", "mse", r"$\epsilon$", "MSE", "pu_eps", xticks=EPS_T, ref=bound("eps"), extra=[BOUND])


def nfmt(v, _=None):
    import math
    e = int(round(math.log10(v)))
    return rf"$10^{e}$" if abs(v - 10 ** e) < 1e-6 * v else f"{v:g}"


panels("scale", "N", "mse", r"$N$", "MSE", "pu_scale", xticks=(1e3, 1e4, 1e5, 1e6, 1e7), xfmt=nfmt, ref=bound("N"), extra=[BOUND])
TGT = plt.Line2D([], [], color=GREY, ls=":", lw=1.8, label=r"$1+\gamma$")
panels("gamma", "gamma", "R", r"$\gamma$", r"$R_{\rm MSE}$", "pu_gamma", xticks=(0.05, 0.1, 0.2, 0.5, 1, 2), ylim=(0.85, 3.6), yticks=(1, 1.5, 2, 3), ref=tgt, extra=[TGT], skip=("LDP", "ID-Sum"))

dc = pd.read_csv(f"{ROOT}/results/pu/decomp.csv")
dc["n_r"] = dc["noise_term"] * dc["eps"] ** 2 / 2
dc["q_r"] = dc["quant_term"] * dc["eps"] ** 2 / 2
C = {"GridMid": ("#d62728", "s"), "Uniform": ("#e8890c", "^"), "Zipf": ("#8c564b", "P"), "SF": ("#111111", "o"), "ACS": ("#2ca02c", "v"), "Taxi": ("#9467bd", "D")}
GT = (0.05, 0.1, 0.2, 0.5, 1, 2)
fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.45))
ax = axs[0]
u = dc[dc.dataset == "Uniform"].sort_values("gamma")
for n in ("GridMid", "Uniform", "Zipf"):
    q = dc[dc.dataset == n].sort_values("gamma")
    ax.plot(q["gamma"], q["q_r"], color=C[n][0], lw=1.9, marker=C[n][1], ms=6, label=n)
ax.plot(u["gamma"], u["n_r"], color=BLUE, lw=3.0, marker="o", ms=5, mec="white", mew=0.7, label="Noise", zorder=10)
ax.set_xscale("log")
ax.set_yscale("log")
ax.xaxis.set_major_locator(FixedLocator((0.05, 0.2, 0.5, 2)))
ax.xaxis.set_minor_locator(NullLocator())
ax.xaxis.set_major_formatter(FuncFormatter(nice))
ax.yaxis.set_minor_locator(NullLocator())
ax.set_xlabel(r"$\gamma$", labelpad=1)
ax.set_ylabel(r"term $/\ (2/\epsilon^2)$")
ax.set_title(r"Noise vs. quantization ($N=10^6$)", pad=3, fontsize=9)
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.8), fontsize=6.8, handlelength=2.2, labelspacing=0.25, borderaxespad=0.1)
ax = axs[1]
for n in ("GridMid", "Uniform", "Zipf", "SF", "ACS", "Taxi"):
    q = dc[dc.dataset == n].sort_values("gamma")
    ax.plot(q["gamma"], q["R"], color=C[n][0], lw=1.9, marker=C[n][1], ms=5.5, label=n)
ax.plot(u["gamma"], 1 + u["gamma"], color=GREY, ls=":", lw=1.8, label=r"$1+\gamma$", zorder=9)
ax.set_xscale("log")
ax.set_yscale("log")
ax.xaxis.set_major_locator(FixedLocator((0.05, 0.2, 0.5, 2)))
ax.xaxis.set_minor_locator(NullLocator())
ax.xaxis.set_major_formatter(FuncFormatter(nice))
ax.yaxis.set_major_locator(FixedLocator((1, 1.5, 2, 3)))
ax.yaxis.set_minor_locator(NullLocator())
ax.yaxis.set_major_formatter(FuncFormatter(nice))
ax.set_xlabel(r"$\gamma$", labelpad=1)
ax.set_ylabel(r"PW-Sum $R_{\rm MSE}$")
ax.set_title("By dataset", pad=3, fontsize=9)
ax.legend(frameon=False, ncol=1, fontsize=6.8, loc="upper left", handlelength=2.2, labelspacing=0.25, borderaxespad=0.1)
fig.tight_layout(w_pad=0.8)
fig.savefig(f"{OUT}/pu_decomp.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/pu_decomp.png", dpi=170, bbox_inches="tight")
print("ok")
