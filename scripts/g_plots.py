"""Same-schedule run summary (results/c4_same_schedule.jsonl): writes results/cost/run_summary_v4.csv and figure cost_run_v4."""
import os as _os, sys as _sys  # repository paths: src/pwexp/paths.py
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
from pwexp.paths import PWSUM_ROOT, DATA_DIR, RUST_BIN  # noqa: E402,F401

import importlib.util
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
COST2 = PWSUM_ROOT
sp = importlib.util.spec_from_file_location("c_plots", f"{ROOT}/scripts/c_plots.py")
cp = importlib.util.module_from_spec(sp)
sp.loader.exec_module(cp)


def table():
    s = pd.read_csv(f"{ROOT}/results/cost/sampler.csv").set_index(["N", "gamma"]).time_mean_s
    rows = []
    for l in open(f"{COST2}/results/c4_same_schedule.jsonl"):
        r = json.loads(l)
        x = r["result"]
        base = {"N": r["N"], "gamma": r["gamma"], "threads": r["threads"], "K": r["K"], "reps": r["reps"], "peak_gb": r["peak_rss_kB"] * 1024 / 1e9, "load_before": r["loadavg_before"][0]}
        if r["method"] == "PW-Sum":
            sm = float(s[(r["N"], r["gamma"])])
            tot = np.array(x["shuffler_total_s"]) + np.array(x["server_total_s"])
            rows.append({"method": "PW-Sum", **base, "M": r["M"], "user_us": 1e6 * np.median(x["user_enc_per_msg_s"]), "sampler_s": sm, "integerize_s": r["integerize_s"],
                         "shuffler_step_s": float(np.median(x["shuffler_total_s"])), "server_step_s": float(np.median(x["server_total_s"])),
                         "server_side_s": sm + r["integerize_s"] + float(np.median(tot)), "ok": x["estimate_equals_sum_plus_W"], "bytes": (2 * r["N"] + r["M"]) * x["ct_len"] / r["N"]})
        else:
            le, he = np.array([a["total_s"] for a in x["leader"]]), np.array([a["total_s"] for a in x["helper"]])
            rows.append({"method": "Prio3Sum", **base, "M": "", "user_us": 1e6 * np.median(x["client_per_report_s"]), "shuffler_step_s": float(np.median(le)), "server_step_s": float(np.median(he)),
                         "server_side_s": float(np.median(le + he)), "ok": x["aggregate_equals_true_sum"], "bytes": x["bytes"]["upload_report_bytes"] + x["bytes"]["server_to_server_per_report"]})
    d = pd.DataFrame(rows)
    d.to_csv(f"{ROOT}/results/cost/run_summary_v4.csv", index=False)
    return d


def main():
    d = table()
    from matplotlib.figure import Figure
    osave = Figure.savefig
    Figure.savefig = lambda self, fn, *a, **k: osave(self, str(fn).replace("cost_run.", "cost_run_v4."), *a, **k)
    cp.run_tables = lambda: d
    cp.fig_run()
    Figure.savefig = osave
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(d.round(4).to_string())
    pv = d.pivot_table(index=["N", "gamma", "threads"], columns="method", values="server_side_s")
    pv["prio_over_pw"] = pv["Prio3Sum"] / pv["PW-Sum"]
    print(pv.round(3).to_string())
    m = d.pivot_table(index=["N", "gamma", "threads"], columns="method", values="peak_gb")
    print((m["Prio3Sum"] / m["PW-Sum"]).round(2).to_string())


if __name__ == "__main__":
    main()
