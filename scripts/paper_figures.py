"""Figures of the Experiments section, in the form used in the paper.
Runs the plotting scripts in this directory and restyles each figure before it is saved: legend and title text as in the
paper, R_MSE in the utility figures (LDP and ID-Sum lie above the plotted range), at least 8 pt text (7 pt legends) at the
printed width, and panel letters (a), (b), ... Intermediate figures and tables go to a temporary directory."""
import importlib.util
import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
S, OUT, TMP = f"{ROOT}/scripts", f"{ROOT}/figures", tempfile.mkdtemp()
os.makedirs(TMP, exist_ok=True)

_tocsv = pd.DataFrame.to_csv
def _safe_csv(self, path=None, *a, **k):  # never rewrite result tables
    if isinstance(path, str) and "/results/" in path:
        path = os.path.join(TMP, os.path.basename(path))
    return _tocsv(self, path, *a, **k)
pd.DataFrame.to_csv = _safe_csv

TW, CW = 7.14, 3.49  # IEEEtran \textwidth and \columnwidth (in)
MAP = {"pu_eps": ("utility_eps", TW), "pu_scale": ("utility_scale", TW),
       "rob_collusion_v2": ("collusion", 0.82 * TW), "rob_poison_v2": ("poisoning", 0.82 * TW),
       "cost_comm_v2": ("communication", TW), "cost_ablation": ("position_weights", CW), "cost_run_v4": ("running_time", 0.82 * TW),
       "abl_design_v2": ("ablation", TW), "dbq_bw_v2": ("database", 0.82 * TW)}
REP = [("Prio3-A", "Prio3Sum-A"), ("Prio3-B", "Prio3Sum-B"), ("PW-Uniform", "Uniform"), ("PW-Sum vs. Theorem 5", "PW-Sum vs. Proposition 1"),
       (r"Bound $|\mathcal{A}|$", "Bound (Prop. 1)"), ("Server-side time (s)", "Shuffler + Server (s)"),
       (r"PW-Sum target ($\gamma=0.1$)", "PW-Sum target\n" + r"($1+\gamma=1.1$)"), (r"No sum term ($\beta=0$)", "No sum term\n(Gaussian approx.)"),
       ("Centre coordinate", "Middle coordinate"),
       ("Sum of stage times, N=", "N = "), ("Network part (largest window)", "Transfer, N = 11,197,553"), ("Time (s)", "Sum of stage times (s)")]
MIN_TXT, MIN_LEG = 8.0, 7.0


def finish(fig, width, stem):
    s = width / fig.get_figwidth()
    legs = list(fig.legends) + [ax.get_legend() for ax in fig.axes if ax.get_legend()]
    leg_texts = {t for L in legs for t in L.get_texts()}
    for ax in fig.axes:
        for axis, which in ((ax.xaxis, "x"), (ax.yaxis, "y")):
            tl = axis.get_ticklabels()
            cur = tl[0].get_fontsize() if tl else plt.rcParams["xtick.labelsize"]
            ax.tick_params(axis=which, which="both", labelsize=max(cur, MIN_TXT / s))
    for t in fig.findobj(Text):
        txt = t.get_text()
        for a, b in REP:
            txt = txt.replace(a, b)
        t.set_text(txt)
        mn = MIN_LEG if t in leg_texts else MIN_TXT
        if txt and t.get_fontsize() * s < mn:
            t.set_fontsize(mn / s)
    if len(fig.axes) > 1:
        for i, ax in enumerate(fig.axes):
            ax.set_title(f"({chr(97 + i)}) {ax.get_title()}".rstrip(), fontsize=max(ax.title.get_fontsize(), MIN_TXT / s))
    if stem == "cost_ablation":  # the upper-right legend covers the uniform curve
        L = fig.axes[0].get_legend()
        L.set_loc("upper right")
        for t in L.get_texts():
            t.set_text({"Uniform weights": "Uniform", "Position weights": "PW-Sum"}.get(t.get_text(), t.get_text()))
    if stem == "abl_design_v2":  # (d): label the three tested K instead of a single 10^3 tick
        fig.axes[2].set_ylim(0.9, 4000)  # room for the two-line legend above the curves
        fig.axes[2].yaxis.set_major_locator(FixedLocator([1, 10, 100]))
        ax = fig.axes[3]
        ax.xaxis.set_major_locator(FixedLocator([139, 423, 1329]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _=None: f"{v:.0f}"))
        ax.xaxis.set_minor_locator(NullLocator())
    if stem == "rob_poison_v2":  # 5% and 10% labels collide at 8 pt
        for ax in fig.axes:
            ax.xaxis.set_major_locator(FixedLocator([0.001, 0.01, 0.1]))
    fig.canvas.draw()
    top = 1.0
    if fig.legends:
        r = fig.canvas.get_renderer()
        top = min(L.get_window_extent(r).transformed(fig.transFigure.inverted()).y0 for L in fig.legends) - 0.01
    fig.tight_layout(rect=(0, 0, 1, top), w_pad=1.0)


_osave = Figure.savefig
def _hook(self, fn, *a, **k):
    fn = str(fn)
    stem, ext = os.path.splitext(os.path.basename(fn))
    if stem in MAP:
        new, w = MAP[stem]
        if not getattr(self, "_styled", False):
            finish(self, w, stem)
            self._styled = True
        fn = os.path.join(OUT, new + ext)
        print("wrote", fn)
    else:
        fn = os.path.join(TMP, os.path.basename(fn))
    return _osave(self, fn, *a, **k)
Figure.savefig = _hook


def load(name, src_fix=None):
    path = f"{S}/{name}.py"
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp)
    if src_fix is None:
        sp.loader.exec_module(m)
    else:
        src = open(path, encoding="utf-8").read()
        for a, b in src_fix:
            assert src.count(a) == 1, a[:60]
            src = src.replace(a, b)
        m.__file__ = path
        exec(compile(src, path, "exec"), m.__dict__)
    return m


# Utility figures: R_MSE on a narrow axis around 1, CDP at 1, bound of Theorem 2 as a ratio
R_ARGS = 'ylim=(0.85, 2.35), yticks=(1, 1.5, 2), skip=("LDP", "ID-Sum"), ref=boundR'
load("pu_plots", [
    ('BOUND = plt.Line2D(', 'def boundR(x):\n    def f(ax, p):\n        p = p.sort_values(x)\n        ax.plot(p[x], p["analytic"] / p["central"], color="#000000", ls=":", lw=1.3, zorder=12)\n    return f\n\n\nBOUND = plt.Line2D('),
    ('panels("eps", "eps", "mse", r"$\\epsilon$", "MSE", "pu_eps", xticks=EPS_T, ref=bound("eps"), extra=[BOUND])',
     'panels("eps", "eps", "R", r"$\\epsilon$", r"$R_{\\rm MSE}$", "pu_eps", xticks=EPS_T, ' + R_ARGS + '("eps"), extra=[BOUND])'),
    ('panels("scale", "N", "mse", r"$N$", "MSE", "pu_scale", xticks=(1e3, 1e4, 1e5, 1e6, 1e7), xfmt=nfmt, ref=bound("N"), extra=[BOUND])',
     'panels("scale", "N", "R", r"$N$", r"$R_{\\rm MSE}$", "pu_scale", xticks=(1e3, 1e4, 1e5, 1e6, 1e7), xfmt=nfmt, ' + R_ARGS + '("N"), extra=[BOUND])'),
])
fp = load("f_plots")
fp.collusion()
fp.poison()
fp.ablation()
fp.cp.fig_comm()
fp.cp.fig_ablation()
load("g_plots").main()
load("dbq_plot")
print("done")
