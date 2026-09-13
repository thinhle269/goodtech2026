from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from src.data.loader import load_dataset
from src.data.preprocess import prepare_multistage
from src.evaluation.metrics import predict, metric_dict, detailed_metrics
from src.fl.aggregators import resolve_trust_variant
from src.fl.server import train_federated
from src.training import train_centralized


def _stable_offset(*parts: str) -> int:
    key = "|".join(parts).encode("utf-8")
    return int(hashlib.sha256(key).hexdigest()[:8], 16) % 100000


def _clean_scenario(cfg) -> dict:
    for s in cfg["scenarios"]:
        if s.get("attack", "none") == "none":
            return s
    return {"name": "clean", "attack": "none", "malicious_fraction": 0.0, "attack_scale": 3.0}


class DatasetRunner:
    def __init__(self, dataset_name: str, cfg: dict, project_root: Path, run_dir: Path,
                 device: torch.device, seed: int):
        self.name = dataset_name
        self.cfg = cfg
        self.run_dir = run_dir
        self.device = device
        self.seed = seed
        self.eval_bs = int(cfg["training"].get("eval_batch_size", 8192))

        X, labels, self.dataset_stats = load_dataset(dataset_name, cfg["data"][dataset_name], cfg["data"], project_root)
        self.bundle = prepare_multistage(X, labels, cfg["data"]["split"], cfg["data"]["clip_quantiles"],
                                         seed, int(cfg["data"]["min_class_samples"]))
        del X, labels

        self.tables: Dict[str, List[dict]] = {k: [] for k in
                                              ["summary", "round_metrics", "trust", "class_metrics",
                                               "confusion", "partitions", "dataset_stats"]}
        self.model_registry = {}


    def _task_data(self, task: str):
        return {"binary": self.bundle.binary, "family": self.bundle.family, "flat": self.bundle.flat}[task]

    def _task_seed(self, task: str) -> int:
        return self.seed + _stable_offset(self.name, task)

    def _tag(self, row: dict, **extra) -> dict:
        return {"dataset": self.name, "seed": self.seed, **extra, **row}

    def _record_eval(self, model, task: str, task_data, scenario_name: str, method: str,
                     block: str, alpha: float, malicious_fraction: float, extra_summary=None):
        pred, prob = predict(model, task_data.X_test, self.eval_bs, self.device)
        metrics = metric_dict(task_data.y_test, pred, prob)
        self.tables["summary"].append(self._tag(
            {**metrics, "n_test": len(task_data.y_test), "n_classes": len(task_data.class_names),
             "n_features": task_data.X_test.shape[1], **(extra_summary or {})},
            task=task, scenario=scenario_name, method=method, block=block,
            alpha=alpha, malicious_fraction=malicious_fraction))
        cls, cm = detailed_metrics(task_data.y_test, pred, task_data.class_names)
        for r in cls:
            self.tables["class_metrics"].append(self._tag(r, task=task, scenario=scenario_name,
                                                          method=method, block=block))
        for i, tname in enumerate(task_data.class_names):
            for j, pname in enumerate(task_data.class_names):
                self.tables["confusion"].append(self._tag(
                    {"true": tname, "pred": pname, "count": int(cm[i, j])},
                    task=task, scenario=scenario_name, method=method, block=block))
        return metrics

    def _run_federated(self, task: str, scenario: dict, agg: str, method_label: str,
                       block: str, alpha: float | None = None, trust_cfg: dict | None = None):
        task_data = self._task_data(task)
        t0 = time.time()
        model, rounds, trusts, malicious, partition = train_federated(
            task_data, agg, self.cfg, scenario, self.device, self._task_seed(task),
            method_label=method_label, alpha=alpha, trust_cfg=trust_cfg)
        eff_alpha = alpha if alpha is not None else float(self.cfg["federated"]["partition"].get("alpha", 0.5))
        mf = float(scenario.get("malicious_fraction", 0.0))
        metrics = self._record_eval(model, task, task_data, scenario["name"], method_label, block,
                                    eff_alpha, mf, {"malicious_clients": ",".join(map(str, malicious))})
        for r in rounds:
            self.tables["round_metrics"].append(self._tag(r, task=task, scenario=scenario["name"], block=block))
        for r in trusts:
            self.tables["trust"].append(self._tag(r, task=task, scenario=scenario["name"], block=block))
        if block == "main":
            for cid, idx in partition.items():
                values, counts = np.unique(task_data.y_train[idx], return_counts=True)
                dist = {task_data.class_names[v]: int(c) for v, c in zip(values, counts)}
                self.tables["partitions"].append(self._tag(
                    {"client_id": cid, "n": int(len(idx)), "class_distribution": str(dist)},
                    task=task, scenario=scenario["name"], method=method_label))
        self.model_registry[(task, scenario["name"], method_label, block, eff_alpha)] = model
        print(f"    [{self.name} s{self.seed}] {task:6s} {scenario['name']:10s} {method_label:16s} "
              f"({block}) f1M={metrics['f1_macro']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        return model

    def _compose_end_to_end(self, scenario_name: str, method: str, block: str, alpha: float,
                            malicious_fraction: float):
        b_key = ("binary", scenario_name, method, block, alpha)
        f_key = ("family", scenario_name, method, block, alpha)
        if b_key not in self.model_registry or f_key not in self.model_registry:
            return
        bundle = self.bundle
        bpred, _ = predict(self.model_registry[b_key], bundle.X_test_common, self.eval_bs, self.device)
        fpred, _ = predict(self.model_registry[f_key], bundle.X_test_common, self.eval_bs, self.device)
        end_pred = np.zeros(len(bpred), dtype=np.int64)
        attack_mask = bpred == 1
        end_pred[attack_mask] = fpred[attack_mask] + 1
        metrics = metric_dict(bundle.y_test_end, end_pred, None)
        self.tables["summary"].append(self._tag(
            {**metrics, "n_test": len(bundle.y_test_end), "n_classes": len(bundle.end_class_names),
             "n_features": bundle.X_test_common.shape[1]},
            task="end_to_end", scenario=scenario_name, method=method, block=block,
            alpha=alpha, malicious_fraction=malicious_fraction))
        cls, cm = detailed_metrics(bundle.y_test_end, end_pred, bundle.end_class_names)
        for r in cls:
            self.tables["class_metrics"].append(self._tag(r, task="end_to_end", scenario=scenario_name,
                                                          method=method, block=block))
        for i, tname in enumerate(bundle.end_class_names):
            for j, pname in enumerate(bundle.end_class_names):
                self.tables["confusion"].append(self._tag(
                    {"true": tname, "pred": pname, "count": int(cm[i, j])},
                    task="end_to_end", scenario=scenario_name, method=method, block=block))
        print(f"    [{self.name} s{self.seed}] e2e    {scenario_name:10s} {method:16s} "
              f"({block}) f1M={metrics['f1_macro']:.4f}", flush=True)


    def run_main(self):
        exp = self.cfg["experiments"]
        tasks = []
        if exp.get("run_binary_stage", True):
            tasks.append("binary")
        if exp.get("run_family_stage", True):
            tasks.append("family")
        if exp.get("run_flat_stage", True):
            tasks.append("flat")
        flat_scenarios = set(exp.get("flat_scenarios") or [s["name"] for s in self.cfg["scenarios"]])
        main_alpha = float(self.cfg["federated"]["partition"].get("alpha", 0.5))
        clean = _clean_scenario(self.cfg)

        for task in tasks:
            task_data = self._task_data(task)
            if exp.get("run_centralized", True):
                t0 = time.time()
                torch.manual_seed(self._task_seed(task))
                model, hist = train_centralized(task_data, self.cfg, self.device, self._task_seed(task))
                m = self._record_eval(model, task, task_data, clean["name"], "centralized", "main",
                                      main_alpha, 0.0)
                for h in hist:
                    self.tables["round_metrics"].append(self._tag(h, task=task, scenario=clean["name"], block="main"))
                self.model_registry[(task, clean["name"], "centralized", "main", main_alpha)] = model
                print(f"    [{self.name} s{self.seed}] {task:6s} {clean['name']:10s} {'centralized':16s} "
                      f"(main) f1M={m['f1_macro']:.4f} ({time.time()-t0:.0f}s)", flush=True)

            for scenario in self.cfg["scenarios"]:
                if task == "flat" and scenario["name"] not in flat_scenarios:
                    continue
                for method in self.cfg["federated"]["methods"]:
                    self._run_federated(task, scenario, method, method, "main")

        if "binary" in tasks and "family" in tasks:
            pairs = sorted({(k[1], k[2]) for k in self.model_registry
                            if k[0] == "binary" and k[3] == "main"} &
                           {(k[1], k[2]) for k in self.model_registry
                            if k[0] == "family" and k[3] == "main"})
            scen_frac = {s["name"]: float(s.get("malicious_fraction", 0.0)) for s in self.cfg["scenarios"]}
            for scenario_name, method in pairs:
                self._compose_end_to_end(scenario_name, method, "main", main_alpha,
                                         scen_frac.get(scenario_name, 0.0))

    def run_alpha_sweep(self):
        sweep = self.cfg["experiments"].get("alpha_sweep", {})
        if not sweep.get("enabled", False):
            return
        clean = _clean_scenario(self.cfg)
        tasks = [t for t in sweep.get("tasks", ["binary", "family"])]
        for a in sweep.get("values", []):
            for task in tasks:
                for method in sweep.get("methods", self.cfg["federated"]["methods"]):
                    self._run_federated(task, clean, method, method, "alpha_sweep", alpha=float(a))
            if "binary" in tasks and "family" in tasks:
                for method in sweep.get("methods", self.cfg["federated"]["methods"]):
                    self._compose_end_to_end(clean["name"], method, "alpha_sweep", float(a), 0.0)

    def run_ablation(self):
        ab = self.cfg["experiments"].get("ablation", {})
        if not ab.get("enabled", False):
            return
        scen_name = ab.get("scenario", "signflip20")
        scenario = next((s for s in self.cfg["scenarios"] if s["name"] == scen_name), None)
        if scenario is None:
            print(f"  ablation scenario '{scen_name}' not found; skipping")
            return
        base_trust = self.cfg["federated"]["trust"]
        main_alpha = float(self.cfg["federated"]["partition"].get("alpha", 0.5))
        mf = float(scenario.get("malicious_fraction", 0.0))
        for variant in ab.get("variants", []):
            tcfg = resolve_trust_variant(base_trust, variant)
            label = f"tafed_{variant}"
            for task in ["binary", "family"]:
                self._run_federated(task, scenario, "tafed", label, "ablation", trust_cfg=tcfg)
            self._compose_end_to_end(scenario["name"], label, "ablation", main_alpha, mf)

    def save_models(self):
        if not self.cfg["experiments"].get("save_models", True):
            return
        for (task, scen, method, block, _alpha), model in self.model_registry.items():
            if block != "main":
                continue
            mdir = self.run_dir / "models" / self.name / task
            mdir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), mdir / f"{scen}_{method}_seed{self.seed}.pt")

    def finish(self):
        for k, v in self.dataset_stats.items():
            if k in {"raw_label_counts", "family_counts", "binary_counts"}:
                for name, cnt in v.items():
                    self.tables["dataset_stats"].append(self._tag(
                        {"key": f"{k}.{name}", "value": str(cnt)}))
            else:
                self.tables["dataset_stats"].append(self._tag({"key": k, "value": str(v)}))
        return self.tables


def run_dataset(dataset_name: str, cfg: dict, project_root: Path, run_dir: Path,
                device: torch.device, seed: int):
    runner = DatasetRunner(dataset_name, cfg, project_root, run_dir, device, seed)
    runner.run_main()
    runner.run_alpha_sweep()
    runner.run_ablation()
    runner.save_models()
    return runner.finish()
