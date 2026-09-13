from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RUNTIME_CONFIG = PROJECT_ROOT / "configs" / "runtime_config.json"


EDGE_FILE = r"D:\Chi Van\Edge-IIoTset 2022\Edge-IIoTset 2022\Edge-IIoTset dataset\Selected dataset for ML and DL\DNN-EdgeIIoT-dataset.csv"
CIC_FILES = [
    r"D:\Chi Van\CIC_IoT_Attack_2023\Merged01.csv",
    r"D:\Chi Van\CIC_IoT_Attack_2023\Merged03.csv",
    r"D:\Chi Van\CIC_IoT_Attack_2023\Merged06.csv",
    r"D:\Chi Van\CIC_IoT_Attack_2023\Merged07.csv",
]

DEFAULT_CONFIG = {
    "project": {
        "name": "TAFed-MSID",
        "seeds": [42],
        "device": "auto",
        "deterministic": True
    },
    "data": {
        "datasets": ["edge_iiotset", "ciciot2023"],
        "cache_dir": "data/processed",
        "edge_iiotset": {
            "files": [EDGE_FILE],
            "label_column": "Attack_type",


            "drop_columns": [
                "frame.time", "ip.src_host", "ip.dst_host",
                "arp.src.proto_ipv4", "arp.dst.proto_ipv4",
                "http.file_data", "http.request.full_uri", "http.request.uri.query",
                "icmp.transmit_timestamp",
                "tcp.options", "tcp.payload", "tcp.srcport", "tcp.dstport",
                "udp.port", "mqtt.msg", "Attack_label"
            ],
            "benign_cap": 40000,
            "attack_cap": 10000,
            "data_seed": 1337,
            "categorical_max_cardinality": 16
        },
        "ciciot2023": {
            "files": CIC_FILES,
            "label_column": "Label",
            "drop_columns": [],
            "benign_cap": 40000,
            "attack_cap": 6000,
            "data_seed": 1337,
            "categorical_max_cardinality": 0
        },
        "csv_chunksize": 500000,
        "drop_constant_features": True,
        "clip_quantiles": [0.001, 0.999],
        "split": {"train": 0.70, "val": 0.15, "test": 0.15},
        "min_class_samples": 20
    },
    "model": {
        "hidden_dims": [128, 64],
        "dropout": 0.20
    },
    "training": {
        "batch_size": 128,
        "local_epochs": 2,
        "centralized_epochs": 15,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "class_weighting": True,
        "early_stopping_patience": 5,
        "eval_batch_size": 8192
    },
    "federated": {
        "num_clients": 10,
        "rounds": 25,
        "client_fraction": 1.0,
        "partition": {"mode": "dirichlet", "alpha": 0.5, "min_client_samples": 50},
        "methods": ["fedavg", "fedprox", "tafed"],
        "fedprox_mu": 0.01,
        "trust": {
            "ema_beta": 0.7,
            "min_trust": 0.05,
            "cosine_gate": True,
            "gate_threshold": 0.0,
            "weights": {"quality": 0.4, "cosine": 0.4, "norm": 0.2}
        }
    },
    "scenarios": [
        {"name": "clean", "attack": "none", "malicious_fraction": 0.0, "attack_scale": 3.0},
        {"name": "signflip10", "attack": "sign_flip", "malicious_fraction": 0.10, "attack_scale": 3.0},
        {"name": "signflip20", "attack": "sign_flip", "malicious_fraction": 0.20, "attack_scale": 3.0},
        {"name": "signflip30", "attack": "sign_flip", "malicious_fraction": 0.30, "attack_scale": 3.0},
        {"name": "signflip40", "attack": "sign_flip", "malicious_fraction": 0.40, "attack_scale": 3.0}
    ],
    "experiments": {
        "run_centralized": True,
        "run_binary_stage": True,
        "run_family_stage": True,
        "run_flat_stage": True,
        "flat_scenarios": ["clean", "signflip20"],
        "alpha_sweep": {
            "enabled": True,
            "values": [0.1, 0.3, 1.0],
            "methods": ["fedavg", "fedprox", "tafed"],
            "tasks": ["binary", "family"]
        },
        "ablation": {
            "enabled": True,
            "scenario": "signflip20",
            "variants": ["no_quality", "no_cosine", "no_norm", "no_ema"]
        },
        "save_models": True
    },
    "output": {
        "root": "outputs/runs",
        "excel_name": "results_summary.xlsx",
        "figures_dpi": 220
    }
}


def initialize_config(overwrite: bool = False) -> Path:
    RUNTIME_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    for p in ["data/processed", "outputs/runs"]:
        (PROJECT_ROOT / p).mkdir(parents=True, exist_ok=True)
    if RUNTIME_CONFIG.exists() and not overwrite:
        print(f"Config already exists: {RUNTIME_CONFIG}")
        print("Edit that JSON directly, or call initialize_config(overwrite=True).")
        return RUNTIME_CONFIG
    with RUNTIME_CONFIG.open("w", encoding="utf-8") as f:
        json.dump(DEFAULT_CONFIG, f, indent=2)
    print(f"Created runtime configuration: {RUNTIME_CONFIG}")
    return RUNTIME_CONFIG


if __name__ == "__main__":
    initialize_config(overwrite=False)
