from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from config import PROJECT_ROOT, RUNTIME_CONFIG
from src.evaluation.exporters import build_aggregate, build_trust_summary, export_excel
from src.evaluation.plots import generate_all_figures
from src.experiments import run_dataset
from src.utils import choose_device, ensure_relative, load_json, make_run_dir, save_json, set_seed

TABLE_NAMES = ["summary", "round_metrics", "trust", "class_metrics", "confusion", "partitions", "dataset_stats"]


def parse_args():
    p = argparse.ArgumentParser(description="TAFed-MSID full experimental pipeline")
    p.add_argument("--config", type=str, default=None, help="Path to a runtime config JSON")
    p.add_argument("--seeds", type=int, nargs="*", default=None, help="Override project.seeds")
    p.add_argument("--datasets", type=str, nargs="*", default=None, help="Override data.datasets")
    p.add_argument("--tag", type=str, default=None, help="Suffix for the run directory name")
    return p.parse_args()


def main():
    args = parse_args()
    cfg_path = Path(args.config) if args.config else RUNTIME_CONFIG
    if not cfg_path.exists():
        raise FileNotFoundError(f"Runtime config not found: {cfg_path}. Run `python config.py` once first.")
    cfg = load_json(cfg_path)
    if args.seeds:
        cfg["project"]["seeds"] = list(args.seeds)
    if args.datasets:
        cfg["data"]["datasets"] = list(args.datasets)

    seeds = [int(s) for s in cfg["project"].get("seeds", [cfg["project"].get("seed", 42)])]
    device = choose_device(cfg["project"].get("device", "auto"))
    print(f"[TAFed-MSID] device={device} seeds={seeds} datasets={cfg['data']['datasets']}")

    out_root = ensure_relative(PROJECT_ROOT, cfg["output"]["root"])
    run_name = cfg["project"]["name"] + (f"_{args.tag}" if args.tag else "")
    run_dir = make_run_dir(out_root, run_name)
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    partial_dir = run_dir / "partial"
    partial_dir.mkdir(parents=True, exist_ok=True)
    save_json(cfg, run_dir / "config_snapshot.json")

    tables = {k: [] for k in TABLE_NAMES}
    t_start = time.time()
    for seed in seeds:
        set_seed(seed, bool(cfg["project"].get("deterministic", True)))
        for dataset_name in cfg["data"]["datasets"]:
            print(f"\n=== Dataset: {dataset_name} | seed {seed} ===", flush=True)
            t0 = time.time()
            result = run_dataset(dataset_name, cfg, PROJECT_ROOT, run_dir, device, seed)
            for k in TABLE_NAMES:
                tables[k].extend(result[k])
                if result[k]:
                    pd.DataFrame(result[k]).to_parquet(
                        partial_dir / f"{dataset_name}_seed{seed}_{k}.parquet", index=False)
            print(f"=== {dataset_name} seed {seed} done in {(time.time()-t0)/60:.1f} min ===", flush=True)

    dfs = {k: pd.DataFrame(v) for k, v in tables.items()}
    dfs["aggregate"] = build_aggregate(dfs["summary"])
    dfs["trust_summary"] = build_trust_summary(dfs["trust"])

    config_rows = []
    def flatten(prefix, value):
        if isinstance(value, dict):
            for k, v in value.items():
                flatten(f"{prefix}.{k}" if prefix else k, v)
        else:
            config_rows.append({"parameter": prefix, "value": str(value)})
    flatten("", cfg)
    dfs["config"] = pd.DataFrame(config_rows)

    generate_all_figures(dfs, fig_dir, int(cfg["output"]["figures_dpi"]))

    excel_path = run_dir / cfg["output"]["excel_name"]
    export_excel(excel_path, dfs)
    (PROJECT_ROOT / "outputs" / "latest_run.txt").write_text(str(run_dir), encoding="utf-8")
    print(f"\nCompleted in {(time.time()-t_start)/60:.1f} min. Results: {run_dir}")
    print(f"Excel: {excel_path}")
    print(f"Figures: {fig_dir}")


if __name__ == "__main__":
    main()
