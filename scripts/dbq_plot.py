"""Summary and line chart of the database workload (results/dbq/runs.jsonl). Writes results/dbq/summary_v2.csv and figure dbq_bw."""
import os as _os, sys as _sys  # repository paths: src/pwexp/paths.py
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
from pwexp.paths import PWSUM_ROOT, DATA_DIR, RUST_BIN  # noqa: E402,F401

import importlib.util
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sp = importlib.util.spec_from_file_location("r_plots", f"{ROOT}/scripts/r_plots.py")
rp = importlib.util.module_from_spec(sp)
sp.loader.exec_module(rp)
plt = rp.plt
rows = [json.loads(l) for l in open(f"{ROOT}/results/dbq/runs.jsonl")]
d = pd.DataFrame(rows)
d["t_network"] = d[["t_upload", "t_forward"]].sum(axis=1, min_count=1).fillna(0) + d.get("t_leader_to_helper", 0).fillna(0) + d.get("t_helper_to_leader", 0).fillna(0)
d.loc[d.method == "PW-Sum", "t_network"] = d.t_upload + d.t_forward
d["t_compute"] = d.stage_total - d.t_network
d["bytes_total"] = d[["bytes_upload", "bytes_forward", "bytes_leader_to_helper", "bytes_helper_to_leader"]].sum(axis=1, min_count=1)
d["bytes_per_record"] = d.bytes_total / d.N
g = d.groupby(["window", "N", "method", "bw_mbps"]).agg(reps=("rep", "count"), total_med=("stage_total", "median"), total_min=("stage_total", "min"), total_max=("stage_total", "max"),
                                                       network=("t_network", "median"), compute=("t_compute", "median"), scan=("t_scan", "median"), bytes_per_record=("bytes_per_record", "median"),
                                                       ok=("ok", "all"), peak_gb=("peak_rss_kB", lambda x: x.max() * 1024 / 1e9)).reset_index()
g.to_csv(f"{ROOT}/results/dbq/summary_v2.csv", index=False)
pd.set_option("display.width", 220, "display.max_columns", 30)
print(g.round(3).to_string())
pv = g.pivot_table(index=["window", "bw_mbps"], columns="method", values="total_med")
pv["prio_over_pw"] = pv["Prio3Sum"] / pv["PW-Sum"]
print(pv.round(2).to_string())
e = d.groupby(["window", "method"]).agg(mse_norm=("abs_error_normalised", lambda x: float(np.mean(np.asarray(x) ** 2))), n=("abs_error_normalised", "count"))
print(e.round(3).to_string())

fig, axs = plt.subplots(1, 3, figsize=(8.0, 2.5))
BR = "#8c564b"
W = sorted(g.window.unique())
for ax, w in zip(axs[:2], W):
    for m in ("Prio3Sum", "PW-Sum"):
        t = g[(g.window == w) & (g.method == m)].sort_values("bw_mbps")
        if m == "PW-Sum":
            rp.pw(ax, t.bw_mbps, t.total_med)
        else:
            ax.plot(t.bw_mbps, t.total_med, color=BR, lw=1.9, marker="P", ms=7, mfc="white", mew=1.2, zorder=4)
    n = int(g[g.window == w].N.iloc[0])
    ax.set_title(f"Sum of stage times, N={n:,}", pad=3, fontsize=9)
    ax.set_ylabel("Time (s)")
ax = axs[2]
w = W[-1]
for m, ls in (("Prio3Sum", "-"), ("PW-Sum", "-")):
    t = g[(g.window == w) & (g.method == m)].sort_values("bw_mbps")
    if m == "PW-Sum":
        rp.pw(ax, t.bw_mbps, t.network)
    else:
        ax.plot(t.bw_mbps, t.network, color=BR, lw=1.9, marker="P", ms=7, mfc="white", mew=1.2, zorder=4)
ax.set_title("Network part (largest window)", pad=3, fontsize=9)
ax.set_ylabel("Transfer time (s)")
for ax in axs:
    ax.set_xscale("log")
    ax.set_yscale("log")
    rp.fixed(ax.xaxis, (100, 1000, 10000), lambda v, _=None: {100: "0.1", 1000: "1", 10000: "10"}[int(v)])
    ax.yaxis.set_minor_locator(rp.NullLocator() if hasattr(rp, "NullLocator") else None)
    ax.set_xlabel("Link bandwidth (Gbit/s)", labelpad=1)
for ax in axs[:2]:
    ax.set_yscale("linear")
    ax.set_ylim(bottom=0)
h = [plt.Line2D([], [], color=rp.BLUE, lw=3.0, marker="o", ms=5, mec="white", mew=0.7, label="PW-Sum"), plt.Line2D([], [], color=BR, lw=1.9, marker="P", ms=6, mfc="white", label="Prio3Sum")]
fig.legend(handles=h, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=0.8)
rp.save(fig, "dbq_bw_v2")
print("ok")
