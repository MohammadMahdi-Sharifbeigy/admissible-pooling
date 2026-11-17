"""
Aggregate finished runs over seeds.

    python scripts/collect.py --draft A
    python scripts/collect.py --draft C --out papers/C_graph_ibp/results

Reads runs/<draft>/<dataset>/<name>/seed*/test_results.csv (only runs with done.flag) and
writes <out>/test_all.csv (every seed) and <out>/test_summary.csv with mean, std,
95% t-interval half-width and number of seeds per (dataset, name, eps).
The per-draft table writers in the notebooks read test_summary.csv.
"""
import argparse
import glob
import os
import sys

import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from interval_gnn.experiments import ROOT  # noqa: E402

PAPER_DIRS = {"A": "A_admissible_pooling", "B": "B_envelope_convolution", "C": "C_graph_ibp"}


def collect(draft: str, root: str = os.path.join(ROOT, "runs"), pattern: str = "test_results.csv") -> pd.DataFrame:
    frames = []
    for f in glob.glob(os.path.join(root, draft, "*", "*", "seed*", pattern)):
        run = os.path.dirname(f)
        if not os.path.exists(os.path.join(run, "done.flag")):
            continue
        df = pd.read_csv(f)
        parts = os.path.normpath(run).split(os.sep)
        df["dataset"], df["name"], df["seed"] = parts[-3], parts[-2], int(parts[-1][4:])
        df["run"] = run
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize(df: pd.DataFrame, keys=("dataset", "name", "eps"),
              values=("clean_acc", "verified_acc")) -> pd.DataFrame:
    keys = [k for k in keys if k in df.columns]
    rows = []
    for k, g in df.groupby(keys):
        row = dict(zip(keys, k if isinstance(k, tuple) else (k,)), n_seeds=g["seed"].nunique())
        for v in values:
            if v not in g:
                continue
            x = g[v].dropna()
            n = len(x)
            row[f"{v}_mean"] = x.mean()
            row[f"{v}_std"] = x.std(ddof=1) if n > 1 else 0.0
            row[f"{v}_ci95"] = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / n ** 0.5 if n > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", required=True, choices=list(PAPER_DIRS))
    ap.add_argument("--root", default=os.path.join(ROOT, "runs"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    out = a.out or os.path.join(ROOT, "papers", PAPER_DIRS[a.draft], "results")
    os.makedirs(out, exist_ok=True)
    df = collect(a.draft, a.root)
    if df.empty:
        print(f"no finished runs under {os.path.join(a.root, a.draft)}")
        return
    df.to_csv(os.path.join(out, "test_all.csv"), index=False)
    s = summarize(df)
    s.to_csv(os.path.join(out, "test_summary.csv"), index=False)
    print(s.to_string(index=False))
    print(f"\nwrote {out}/test_all.csv and test_summary.csv")


if __name__ == "__main__":
    main()
