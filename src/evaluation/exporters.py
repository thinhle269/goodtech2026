from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

METRICS = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted", "mcc", "roc_auc"]


def build_aggregate(summary_df: pd.DataFrame) -> pd.DataFrame:

    if summary_df.empty:
        return pd.DataFrame()
    keys = ["dataset", "block", "task", "scenario", "method", "alpha", "malicious_fraction"]
    present = [m for m in METRICS if m in summary_df.columns]
    agg = summary_df.groupby(keys, dropna=False).agg(
        n_seeds=("seed", "nunique"),
        **{f"{m}_mean": (m, "mean") for m in present},
        **{f"{m}_std": (m, "std") for m in present},
    ).reset_index()
    for m in present:
        agg[f"{m}_fmt"] = agg.apply(
            lambda r: "" if pd.isna(r[f"{m}_mean"]) else (
                f"{r[f'{m}_mean']:.4f}" if pd.isna(r[f"{m}_std"]) else
                f"{r[f'{m}_mean']:.4f} ± {r[f'{m}_std']:.4f}"),
            axis=1)
    return agg


def build_trust_summary(trust_df: pd.DataFrame) -> pd.DataFrame:

    if trust_df.empty or "trust" not in trust_df.columns:
        return pd.DataFrame()
    df = trust_df[trust_df["method"].str.startswith("tafed")]
    if df.empty:
        return pd.DataFrame()
    rows = []
    keys = ["dataset", "block", "task", "scenario", "method", "seed"]
    for key_vals, g in df.groupby(keys):
        honest = g.loc[~g["malicious"], "trust"]
        mal = g.loc[g["malicious"], "trust"]
        row = dict(zip(keys, key_vals))
        row["trust_honest_mean"] = float(honest.mean()) if len(honest) else np.nan
        row["trust_malicious_mean"] = float(mal.mean()) if len(mal) else np.nan
        row["trust_separation"] = (row["trust_honest_mean"] - row["trust_malicious_mean"]
                                   if len(mal) and len(honest) else np.nan)
        row["gated_fraction_malicious"] = float(g.loc[g["malicious"], "gated"].mean()) if len(mal) else np.nan
        rows.append(row)
    per_seed = pd.DataFrame(rows)
    agg = per_seed.groupby(["dataset", "block", "task", "scenario", "method"], dropna=False).agg(
        n_seeds=("seed", "nunique"),
        trust_honest_mean=("trust_honest_mean", "mean"),
        trust_malicious_mean=("trust_malicious_mean", "mean"),
        trust_separation_mean=("trust_separation", "mean"),
        trust_separation_std=("trust_separation", "std"),
        gated_fraction_malicious=("gated_fraction_malicious", "mean"),
    ).reset_index()
    return agg


def export_excel(path: Path, tables: Dict[str, List[dict] | pd.DataFrame]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, data in tables.items():
            df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
            if df.empty:
                df = pd.DataFrame({"empty": []})
            df.to_excel(writer, sheet_name=name[:31], index=False)
