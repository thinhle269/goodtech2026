from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHOD_ORDER = ["centralized", "fedavg", "fedprox", "tafed"]
METHOD_LABEL = {"centralized": "Centralized", "fedavg": "FedAvg", "fedprox": "FedProx", "tafed": "TAFed (ours)"}
METHOD_COLOR = {"centralized": "#7f7f7f", "fedavg": "#1f77b4", "fedprox": "#2ca02c", "tafed": "#d62728"}


def _mlabel(m: str) -> str:
    return METHOD_LABEL.get(m, m)


def _save(fig, path: Path, dpi: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_learning_curves(round_df: pd.DataFrame, out: Path, dpi: int = 220):
    if round_df.empty:
        return
    df = round_df[(round_df.get("block", "main") == "main") & (round_df["scenario"] == "clean")]
    for (dataset, task), group in df.groupby(["dataset", "task"]):
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for method in [m for m in METHOD_ORDER if m in group["method"].unique()]:
            g = group[group["method"] == method].groupby("round")["f1_macro"].agg(["mean", "std"]).reset_index()
            ax.plot(g["round"], g["mean"], label=_mlabel(method), color=METHOD_COLOR.get(method))
            if g["std"].notna().any():
                ax.fill_between(g["round"], g["mean"] - g["std"].fillna(0), g["mean"] + g["std"].fillna(0),
                                alpha=0.15, color=METHOD_COLOR.get(method))
        ax.set_xlabel("Communication round")
        ax.set_ylabel("Validation Macro-F1")
        ax.set_title(f"{dataset} | {task} | clean")
        ax.grid(alpha=0.25)
        ax.legend()
        _save(fig, out / f"learning_{dataset}_{task}_clean.png", dpi)


def plot_poison_robustness(summary_df: pd.DataFrame, out: Path, dpi: int = 220):
    if summary_df.empty:
        return
    df = summary_df[(summary_df["block"] == "main") & (summary_df["method"] != "centralized")]
    for (dataset, task), group in df.groupby(["dataset", "task"]):
        if group["malicious_fraction"].nunique() < 2:
            continue
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for method in [m for m in METHOD_ORDER if m in group["method"].unique()]:
            g = (group[group["method"] == method]
                 .groupby("malicious_fraction")["f1_macro"].agg(["mean", "std"]).reset_index()
                 .sort_values("malicious_fraction"))
            ax.errorbar(g["malicious_fraction"] * 100, g["mean"], yerr=g["std"].fillna(0),
                        marker="o", capsize=3, label=_mlabel(method), color=METHOD_COLOR.get(method))
        ax.set_xlabel("Malicious clients (%)")
        ax.set_ylabel("Test Macro-F1")
        ax.set_title(f"{dataset} | {task} | sign-flip robustness")
        ax.grid(alpha=0.25)
        ax.legend()
        _save(fig, out / f"poison_{dataset}_{task}.png", dpi)


def plot_alpha_sensitivity(summary_df: pd.DataFrame, out: Path, dpi: int = 220):
    if summary_df.empty:
        return
    df = summary_df[(summary_df["scenario"] == "clean") &
                    (summary_df["block"].isin(["main", "alpha_sweep"])) &
                    (summary_df["method"] != "centralized")]
    for (dataset, task), group in df.groupby(["dataset", "task"]):
        if group["alpha"].nunique() < 2:
            continue
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for method in [m for m in METHOD_ORDER if m in group["method"].unique()]:
            g = (group[group["method"] == method]
                 .groupby("alpha")["f1_macro"].agg(["mean", "std"]).reset_index().sort_values("alpha"))
            if len(g) < 2:
                continue
            ax.errorbar(g["alpha"], g["mean"], yerr=g["std"].fillna(0), marker="s", capsize=3,
                        label=_mlabel(method), color=METHOD_COLOR.get(method))
        ax.set_xscale("log")
        ax.set_xticks(sorted(group["alpha"].unique()))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel(r"Dirichlet $\alpha$ (log scale)")
        ax.set_ylabel("Test Macro-F1")
        ax.set_title(f"{dataset} | {task} | Non-IID sensitivity")
        ax.grid(alpha=0.25)
        ax.legend()
        _save(fig, out / f"alpha_{dataset}_{task}.png", dpi)


def plot_trust(trust_df: pd.DataFrame, out: Path, dpi: int = 220, scenario: str = "signflip20"):
    if trust_df.empty or "trust" not in trust_df.columns:
        return
    df = trust_df[(trust_df["method"] == "tafed") & (trust_df["scenario"] == scenario) &
                  (trust_df.get("block", "main") == "main")]
    for (dataset, task), group in df.groupby(["dataset", "task"]):
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for cid, g in group.groupby(["seed", "client_id"]):
            g = g.sort_values("round")
            color = "#d62728" if g["malicious"].iloc[0] else "#1f77b4"
            ax.plot(g["round"], g["trust"], alpha=0.18, color=color, linewidth=0.8)
        for is_mal, color, label in [(False, "#1f77b4", "Honest (mean)"), (True, "#d62728", "Malicious (mean)")]:
            g = group[group["malicious"] == is_mal].groupby("round")["trust"].mean().reset_index()
            if len(g):
                ax.plot(g["round"], g["trust"], color=color, linewidth=2.5, label=label)
        ax.set_xlabel("Communication round")
        ax.set_ylabel("EMA trust score")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{dataset} | {task} | client trust under {scenario}")
        ax.grid(alpha=0.25)
        ax.legend()
        _save(fig, out / f"trust_{dataset}_{task}_{scenario}.png", dpi)


def plot_confusions(confusion_df: pd.DataFrame, out: Path, dpi: int = 220):
    if confusion_df.empty:
        return
    df = confusion_df[(confusion_df["block"] == "main") & (confusion_df["method"] == "tafed") &
                      (confusion_df["task"] == "end_to_end") &
                      (confusion_df["scenario"].isin(["clean", "signflip20"]))]
    for (dataset, scenario), group in df.groupby(["dataset", "scenario"]):
        pivot = group.pivot_table(index="true", columns="pred", values="count", aggfunc="sum", fill_value=0)
        names = sorted(pivot.index.tolist(), key=lambda x: (x != "Benign", x))
        pivot = pivot.reindex(index=names, columns=names, fill_value=0)
        cm = pivot.to_numpy(dtype=float)
        row_sum = cm.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1
        cmn = cm / row_sum
        fig, ax = plt.subplots(figsize=(1.1 * len(names) + 2.2, 1.0 * len(names) + 1.8))
        im = ax.imshow(cmn, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set(xticks=np.arange(len(names)), yticks=np.arange(len(names)),
               xticklabels=names, yticklabels=names, xlabel="Predicted", ylabel="True")
        ax.set_title(f"{dataset} | TAFed end-to-end | {scenario}")
        ax.tick_params(axis="x", rotation=45)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{cmn[i, j]*100:.1f}", ha="center", va="center", fontsize=7,
                        color="white" if cmn[i, j] > 0.5 else "black")
        _save(fig, out / f"cm_{dataset}_end_to_end_{scenario}_tafed.png", dpi)


def plot_ablation(summary_df: pd.DataFrame, out: Path, dpi: int = 220, scenario: str = "signflip20"):
    if summary_df.empty:
        return
    base = summary_df[(summary_df["block"] == "main") & (summary_df["method"] == "tafed") &
                      (summary_df["scenario"] == scenario) & (summary_df["task"] == "end_to_end")]
    abl = summary_df[(summary_df["block"] == "ablation") & (summary_df["task"] == "end_to_end")]
    if abl.empty:
        return
    for dataset, _ in abl.groupby("dataset"):
        rows = []
        b = base[base["dataset"] == dataset]
        if len(b):
            rows.append(("TAFed (full)", b["f1_macro"].mean(), b["f1_macro"].std()))
        for method, g in abl[abl["dataset"] == dataset].groupby("method"):
            nice = method.replace("tafed_", "w/o ").replace("no_", "").replace("ema", "EMA")
            rows.append((nice, g["f1_macro"].mean(), g["f1_macro"].std()))
        labels = [r[0] for r in rows]
        means = [r[1] for r in rows]
        stds = [0 if pd.isna(r[2]) else r[2] for r in rows]
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        colors = ["#d62728"] + ["#9edae5"] * (len(rows) - 1)
        ax.bar(labels, means, yerr=stds, capsize=3, color=colors)
        ax.set_ylabel("End-to-end Test Macro-F1")
        ax.set_title(f"{dataset} | trust-component ablation ({scenario})")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
        _save(fig, out / f"ablation_{dataset}_{scenario}.png", dpi)


def plot_summary(summary_df: pd.DataFrame, out: Path, dpi: int = 220):
    if summary_df.empty:
        return
    df = summary_df[(summary_df["block"] == "main") & (summary_df["scenario"] == "clean")]
    for (dataset, task), group in df.groupby(["dataset", "task"]):
        agg = group.groupby("method")["f1_macro"].agg(["mean", "std"]).reset_index()
        agg["order"] = agg["method"].map({m: i for i, m in enumerate(METHOD_ORDER)}).fillna(99)
        agg = agg.sort_values("order")
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.bar([_mlabel(m) for m in agg["method"]], agg["mean"], yerr=agg["std"].fillna(0),
               capsize=3, color=[METHOD_COLOR.get(m, "#999") for m in agg["method"]])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Test Macro-F1")
        ax.set_title(f"{dataset} | {task} | clean")
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.25)
        _save(fig, out / f"summary_{dataset}_{task}_clean.png", dpi)


def generate_all_figures(dfs: dict, fig_dir: Path, dpi: int):
    plot_learning_curves(dfs["round_metrics"], fig_dir, dpi)
    plot_summary(dfs["summary"], fig_dir, dpi)
    plot_poison_robustness(dfs["summary"], fig_dir, dpi)
    plot_alpha_sensitivity(dfs["summary"], fig_dir, dpi)
    plot_trust(dfs["trust"], fig_dir, dpi)
    plot_confusions(dfs["confusion"], fig_dir, dpi)
    plot_ablation(dfs["summary"], fig_dir, dpi)
