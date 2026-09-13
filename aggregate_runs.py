from __future__ import annotations


import argparse
from pathlib import Path

import pandas as pd

from config import PROJECT_ROOT
from src.evaluation.exporters import build_aggregate, build_trust_summary, export_excel
from src.evaluation.plots import generate_all_figures
from src.utils import ensure_relative

TABLE_NAMES = ["summary", "round_metrics", "trust", "class_metrics", "confusion", "partitions", "dataset_stats"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="Run directories containing partial/*.parquet")
    ap.add_argument("--out", required=True, help="Output directory for the combined results")
    ap.add_argument("--excel-name", default="results_summary.xlsx")
    ap.add_argument("--dpi", type=int, default=220)
    args = ap.parse_args()

    out_dir = ensure_relative(PROJECT_ROOT, args.out)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    dfs = {}
    for name in TABLE_NAMES:
        parts = []
        for run in args.runs:
            run_dir = ensure_relative(PROJECT_ROOT, run)
            for f in sorted((run_dir / "partial").glob(f"*_{name}.parquet")):
                parts.append(pd.read_parquet(f))
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if not df.empty and {"dataset", "seed"}.issubset(df.columns):
            dedup_keys = [c for c in ["dataset", "seed", "block", "task", "scenario", "method",
                                      "alpha", "round", "client_id", "class", "true", "pred", "key"]
                          if c in df.columns]
            df = df.drop_duplicates(subset=dedup_keys, keep="last")
        dfs[name] = df

    dfs["aggregate"] = build_aggregate(dfs["summary"])
    dfs["trust_summary"] = build_trust_summary(dfs["trust"])
    generate_all_figures(dfs, fig_dir, args.dpi)
    export_excel(out_dir / args.excel_name, dfs)
    n_seeds = dfs["summary"]["seed"].nunique() if not dfs["summary"].empty else 0
    print(f"Combined {len(args.runs)} runs, {n_seeds} seeds -> {out_dir}")


if __name__ == "__main__":
    main()
