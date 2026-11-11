"""Check and clean the real datasets (SF, ACS, Taxi), clip to the public bound X_max, normalise to [0,1].

Usage: python scripts/m0_real.py {sf,acs,taxi} [...]
Raw files are only read. Writes $PWSUM_DATA/<name>/processed/values.npy (float64 in [0,1]),
processed/summary.json and $PWSUM_DATA/<name>/README.md.
"""
import os as _os, sys as _sys  # repository paths: src/pwexp/paths.py
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
from pwexp.paths import PWSUM_ROOT, DATA_DIR, RUST_BIN  # noqa: E402,F401

import datetime
import glob
import hashlib
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = DATA_DIR
DOWNLOAD_NOTE = ("Downloaded on 2026-10-30 from the URLs listed in DATA.md. Raw files are never modified.")
COMMON_NOTE = ("All methods are evaluated against the sum of the *clipped, normalised* values in "
               "`processed/values.npy`; clipping is applied identically before every method, so the clipping bias "
               "below is the same for all methods and is not part of any method's error.")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def finish(name, x, xmax, unit, n_raw, audit, extra):
    """x: cleaned values (>= 0, finite) in original units. Clip at xmax, normalise, save, summarise."""
    x = np.asarray(x, dtype=np.float64)
    assert np.isfinite(x).all() and (x >= 0).all()
    n_trunc = int((x > xmax).sum())
    y = np.minimum(x, xmax) / xmax
    out = f"{ROOT}/{name}/processed/values.npy"
    np.save(out, y)
    s = dict(name=name, privacy_unit=unit, N_raw=int(n_raw), N=int(len(y)), X_max=xmax,
             n_truncated=n_trunc, truncated_fraction=n_trunc / len(y),
             mean_raw_units=float(x.mean()), mean_clipped_units=float(y.mean() * xmax),
             clipping_bias_units=float(y.mean() * xmax - x.mean()),
             clipping_bias_relative=float((y.mean() * xmax - x.mean()) / x.mean()),
             median_raw_units=float(np.median(x)), max_raw_units=float(x.max()),
             mean_normalised=float(y.mean()), var_normalised=float(y.var()), sum_normalised=float(y.sum()),
             audit=audit, processed_file=f"{name}/processed/values.npy", processed_sha256=sha256(out),
             generated=datetime.datetime.now().isoformat(timespec="seconds"), **extra)
    json.dump(s, open(f"{ROOT}/{name}/processed/summary.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(s, indent=1, ensure_ascii=False), flush=True)
    return s


def rows_md(rows):
    return "\n".join(f"| {k} | {v} |" for k, v in rows)


def write_readme(name, title, src, raw_hashes, rules, s, unit_text, notes, units="USD"):
    a = s["audit"]
    with open(f"{ROOT}/{name}/README.md", "w") as f:
        f.write(f"# {title}\n\n## Source\n\n| item | value |\n|---|---|\n{rows_md(src)}\n\n{DOWNLOAD_NOTE}\n\n")
        f.write("## Raw files (sha256)\n\n| file | bytes | sha256 |\n|---|---|---|\n")
        for fn, (b, h) in raw_hashes.items():
            f.write(f"| raw/{fn} | {b} | `{h}` |\n")
        f.write(f"\n## Privacy unit\n\n{unit_text}\n\n")
        f.write("## Cleaning rules (fixed before cleaning, applied in this order)\n\n")
        f.write("\n".join(f"{i + 1}. {r}" for i, r in enumerate(rules)))
        f.write("\n\n## Audit (counts before cleaning)\n\n| check | count |\n|---|---|\n")
        f.write(rows_md(a.items()) + "\n\n## Result\n\n| quantity | value |\n|---|---|\n")
        f.write(rows_md([
            ("N raw (records in scope)", f"{s['N_raw']:,}"),
            ("N after cleaning", f"{s['N']:,}"),
            ("X_max (public, fixed a priori)", f"{s['X_max']:,} {units}"),
            ("records clipped to X_max", f"{s['n_truncated']:,}"),
            ("clipped fraction", f"{s['truncated_fraction']:.4%}"),
            (f"mean before clipping ({units})", f"{s['mean_raw_units']:,.4f}"),
            (f"mean after clipping ({units})", f"{s['mean_clipped_units']:,.4f}"),
            ("clipping bias (after - before)", f"{s['clipping_bias_units']:,.4f} {units} "
                                               f"({s['clipping_bias_relative']:.4%})"),
            (f"median before clipping ({units})", f"{s['median_raw_units']:,.4f}"),
            (f"max before clipping ({units})", f"{s['max_raw_units']:,.4f}"),
            ("mean of normalised values", f"{s['mean_normalised']:.8f}"),
            ("variance of normalised values", f"{s['var_normalised']:.8f}"),
            ("sum of normalised values (ground truth)", f"{s['sum_normalised']:.6f}"),
            ("processed/values.npy sha256", f"`{s['processed_sha256']}`"),
        ]) + "\n\n")
        f.write("`processed/values.npy`: float64 array, one entry per privacy unit, min(value, X_max)/X_max in [0,1]. "
                "`processed/summary.json` holds the same numbers. Produced by `scripts/m0_real.py` in "
                f"this repository.\n\n{COMMON_NOTE}\n\n## Notes\n\n" + "\n".join(f"- {n}" for n in notes) + "\n")


# ---------------------------------------------------------------- SF
def sf():
    raw = f"{ROOT}/SF/raw/export.csv"
    hashes = {"export.csv": (os.path.getsize(raw), sha256(raw))}
    df = pd.read_csv(raw, dtype=str, keep_default_na=False, na_values=[""])
    n_file = len(df)
    asof = sorted(df["data_as_of"].dropna().unique().tolist())
    f = df[(df["Year Type"] == "Fiscal") & (df["Year"] == "2026")]
    tc_str = f["Total Compensation"]
    tc = pd.to_numeric(tc_str.str.replace(",", "", regex=False), errors="coerce")
    xmax = 500_000.0
    audit = {
        "rows in export (all years)": f"{n_file:,}",
        "rows with Year Type = Fiscal, Year = 2026": f"{len(f):,}",
        "Total Compensation empty": int(tc_str.isna().sum()),
        "Total Compensation not parseable as a number": int((tc_str.notna() & tc.isna()).sum()),
        "Total Compensation < 0": int((tc < 0).sum()),
        "Total Compensation == 0": int((tc == 0).sum()),
        "0 < Total Compensation <= 1,000": int(((tc > 0) & (tc <= 1000)).sum()),
        "Total Compensation > 500,000 (clipped)": int((tc > xmax).sum()),
        "Total Compensation > 1,000,000": int((tc > 1e6).sum()),
        "rows identical in every column to another FY2026 row (kept)": int(f.duplicated(keep="first").sum()),
    }
    keep = tc.notna() & (tc > 0)
    s = finish("SF", tc[keep].to_numpy(), xmax, "one FY2026 compensation record (record-level DP)",
               len(f), audit, dict(data_as_of=asof, fiscal_year="FY2026 (2025-07-01..2026-06-30)"))
    write_readme(
        "SF", "San Francisco Employee Compensation, FY2026 (Total Compensation)",
        [("dataset", "DataSF Employee Compensation (88g8-5mnd), full export"),
         ("URL", "https://data.sfgov.org/City-Management-and-Ethics/Employee-Compensation/88g8-5mnd"),
         ("download date", "2026-10-30"),
         ("data_as_of in file", ", ".join(asof)),
         ("scope", "Year Type = Fiscal, Year = 2026 (latest complete fiscal year: July 2025 – June 2026)"),
         ("field", "Total Compensation (salaries + benefits, USD)")],
        hashes,
        ["Keep rows with Year Type = Fiscal and Year = 2026.",
         "Parse Total Compensation (thousands separators removed).",
         "Remove rows whose Total Compensation is missing or not a number.",
         "Remove rows with Total Compensation <= 0 (no compensation to aggregate; negative values are "
         "accounting reversals).",
         "Clip at X_max = 500,000 USD and divide by X_max."],
        s,
        "**Record-level.** Each FY2026 row (one employee in one job/department) is one contribution. The export "
        "has no `Employee Identifier` column, and `Employee Name` cannot reliably identify a person (homonyms), so "
        "rows are **not** merged and no surrogate identity is created. An employee with several rows contributes "
        "several values; the DP guarantee of every method protects one row, not one employee, and no "
        "employee-level DP claim is made for SF.",
        ["X_max = 500,000 USD was fixed before looking at the data (public prior on city pay: "
         "it is above the salary + benefits of almost all city positions, while a few department heads, doctors "
         "and overtime-heavy roles exceed it).",
         "Rows identical in every column are kept: with no identifier there is no basis to call them duplicates.",
         "Employee names are not copied into processed files or this README."])


# ---------------------------------------------------------------- ACS
def acs():
    raw = f"{ROOT}/ACS/raw/csv_pus.zip"
    hashes = {"csv_pus.zip": (os.path.getsize(raw), sha256(raw))}
    parts = []
    with zipfile.ZipFile(raw) as z:
        members = sorted(m for m in z.namelist() if m.lower().endswith(".csv"))
        for m in members:
            with z.open(m) as fh:
                parts.append(pd.read_csv(fh, usecols=["SERIALNO", "STATE", "AGEP", "PINCP"],
                                         dtype={"SERIALNO": str, "STATE": "Int64", "AGEP": "Int64",
                                                "PINCP": "float64"}))
    df = pd.concat(parts, ignore_index=True)
    years = sorted(df["SERIALNO"].str[:4].unique().tolist())
    n_states = int(df["STATE"].nunique())
    miss = df["PINCP"].isna()
    young = df["AGEP"] < 15
    p = df["PINCP"]
    xmax = 100_000.0
    audit = {
        "person records in files": f"{len(df):,}",
        "AGEP missing": int(df["AGEP"].isna().sum()),
        "AGEP < 15": f"{int(young.sum()):,}",
        "PINCP blank with AGEP < 15 (not applicable by design, removed)": f"{int((miss & young).sum()):,}",
        "PINCP blank with AGEP >= 15 (truly missing, removed)": int((miss & ~young).sum()),
        "PINCP present with AGEP < 15": int((~miss & young).sum()),
        "PINCP < 0 (negative income, set to 0)": f"{int((p < 0).sum()):,}",
        "PINCP == 0": f"{int((p == 0).sum()):,}",
        "PINCP > 100,000 (clipped)": f"{int((p > xmax).sum()):,}",
        "PINCP min / max": f"{p.min():,.0f} / {p.max():,.0f}",
    }
    x = np.maximum(p[~miss].to_numpy(), 0.0)
    s = finish("ACS", x, xmax, "one person record", len(df), audit,
               dict(members=members, serialno_years=years, n_state_codes=n_states))
    write_readme(
        "ACS", "ACS PUMS 2024 1-year, US person file (PINCP)",
        [("dataset", "American Community Survey 2024 1-Year PUMS, person records, 50 states + DC "
                     "(psam_pusa + psam_pusb)"),
         ("URL", "https://www2.census.gov/programs-surveys/acs/data/pums/2024/1-Year/csv_pus.zip"),
         ("download date", "2026-10-30"),
         ("field", "PINCP (total person's income in the past 12 months, as published)"),
         ("members used", ", ".join(members)),
         ("survey year (SERIALNO prefix)", ", ".join(years)),
         ("distinct STATE codes", n_states)],
        hashes,
        ["Read AGEP and PINCP from every person CSV in the zip.",
         "Remove rows with blank PINCP (N/A for persons under 15 by design; any blank at AGEP >= 15 is counted "
         "separately as truly missing).",
         "Set negative PINCP to 0.",
         "Clip at X_max = 100,000 USD and divide by X_max: x = clip(PINCP, 0, 100000)/100000."],
        s,
        "**Person-level.** One PUMS person record is one contribution; person weights (PWGTP) are not used, so "
        "the sum is over sample records, not a population estimate.",
        ["PINCP is used as published, **without the ADJINC adjustment factor**: values are in the respondent's "
         "reporting-period dollars, not constant 2024 dollars.",
         "X_max changed from 1,000,000 to 100,000 USD on the user's instruction (2026-10-30); the clipped "
         "fraction and clipping bias above refer to 100,000.",
         "Latest year as provided: 2024 (zip contains ACS2024_PUMS_README.pdf). Whether a 2025 1-year file "
         "existed on 2026-10-30 could not be checked at download time.",
         "Puerto Rico (separate psam_p72 file) is not part of csv_pus.zip and is not included."])


# ---------------------------------------------------------------- Taxi
def taxi():
    files = sorted(glob.glob(f"{ROOT}/Taxi/raw/yellow_tripdata_*.parquet"))
    hashes = {os.path.basename(f): (os.path.getsize(f), sha256(f)) for f in files}
    xmax, invalid = 20.0, 200.0
    xs, audit = [], {}
    tot = dict(rows=0, nonfinite=0, neg=0, over200=0, zero=0, kept=0, clipped=0)
    for f in files:
        d = pq.read_table(f, columns=["trip_distance"]).column("trip_distance").to_numpy(zero_copy_only=False)
        d = np.asarray(d, dtype=np.float64)  # nulls become NaN
        mon = os.path.basename(f)[len("yellow_tripdata_"):-len(".parquet")]
        fin = np.isfinite(d)
        neg = fin & (d < 0)
        over = fin & (d > invalid)
        ok = fin & (d >= 0) & (d <= invalid)
        c = dict(rows=len(d), nonfinite=int((~fin).sum()), neg=int(neg.sum()), over200=int(over.sum()),
                 zero=int((ok & (d == 0)).sum()), kept=int(ok.sum()), clipped=int((ok & (d > xmax)).sum()))
        for k in tot:
            tot[k] += c[k]
        audit[f"{mon}: rows / missing-or-nonfinite / <0 / >200 / ==0 (kept) / kept / >20 (clipped)"] = \
            " / ".join(f"{c[k]:,}" for k in ("rows", "nonfinite", "neg", "over200", "zero", "kept", "clipped"))
        xs.append(d[ok])
    audit["total: rows / missing-or-nonfinite / <0 / >200 / ==0 (kept) / kept / >20 (clipped)"] = \
        " / ".join(f"{tot[k]:,}" for k in ("rows", "nonfinite", "neg", "over200", "zero", "kept", "clipped"))
    x = np.concatenate(xs)
    assert len(x) == tot["kept"] and len(x) >= 10**7
    s = finish("Taxi", x, xmax, "one trip record", tot["rows"], audit,
               dict(files=[os.path.basename(f) for f in files], counts_total=tot))
    write_readme(
        "Taxi", "NYC TLC yellow taxi trips 2025-01..2025-04 (trip_distance)",
        [("dataset", "NYC Taxi & Limousine Commission, Yellow Taxi Trip Records"),
         ("URL", "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-{01,02,03,04}.parquet"),
         ("landing page", "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page"),
         ("download date", "2026-10-30"),
         ("field", "trip_distance (miles, taximeter)"),
         ("months", "2025-01, 2025-02, 2025-03, 2025-04 (consecutive)")],
        hashes,
        ["Read trip_distance from the four monthly files; no filter on any other field (fare_amount is not used).",
         "Remove records whose trip_distance is missing or not finite.",
         "Remove records with trip_distance < 0.",
         "Remove records with trip_distance > 200 miles (treated as clearly invalid).",
         "Keep 0 <= trip_distance <= 200 (zero-distance trips are kept), clip at X_max = 20 miles and divide by "
         "X_max: x = clip(trip_distance, 0, 20)/20."],
        s,
        "**Trip-level.** One trip record is one contribution (records carry no rider or driver identity).",
        ["The statistic is the sum of trip_distance.",
         "Rules were fixed before cleaning. Records are kept as they appear in each "
         "monthly file; pickup timestamps are not filtered.",
         "N >= 1e7 is met with four months."], units="miles")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        {"sf": sf, "acs": acs, "taxi": taxi}[arg]()
