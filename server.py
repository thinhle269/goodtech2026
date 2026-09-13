from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from src.data.partition import make_partition
from src.evaluation.metrics import predict, metric_dict
from src.fl.aggregators import (coordinate_median, fedavg, fltrust, krum,
                                tafed, trimmed_mean)
from src.fl.client import local_train
from src.models.mlp import build_model
from src.utils import apply_delta, flatten_state_delta, state_dict_to_cpu


def _fltrust_root_indices(y: np.ndarray, size: int, seed: int) -> np.ndarray:

    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    per_class = np.maximum(1, np.round(size * counts / counts.sum()).astype(int))
    picks = []
    for c, k in zip(classes, per_class):
        idx_c = np.where(y == c)[0]
        picks.append(rng.choice(idx_c, size=min(k, len(idx_c)), replace=False))
    return np.concatenate(picks)


def train_federated(task_data, agg_method: str, cfg: dict, scenario: dict, device: torch.device,
                    seed: int, method_label: str | None = None, alpha: float | None = None,
                    trust_cfg: dict | None = None):

    method_label = method_label or agg_method
    fed = cfg["federated"]
    train_cfg = dict(cfg["training"])
    K = int(fed["num_clients"])
    rounds = int(fed["rounds"])
    C = len(task_data.class_names)
    input_dim = task_data.X_train.shape[1]

    part_cfg = dict(fed["partition"])
    if alpha is not None:
        part_cfg["alpha"] = float(alpha)
    part_seed = seed if alpha is None else seed + int(round(alpha * 1000))
    partition = make_partition(task_data.y_train, part_cfg, K, part_seed)

    torch.manual_seed(seed + 7)
    model = build_model(input_dim, C, cfg["model"]).to(device)
    local_model = build_model(input_dim, C, cfg["model"]).to(device)
    global_state = state_dict_to_cpu(model.state_dict())

    X_tr = torch.from_numpy(task_data.X_train).to(device)
    y_tr = torch.from_numpy(task_data.y_train).to(device)


    splits: Dict[int, tuple] = {}
    for cid, idx in partition.items():
        rng = np.random.default_rng(part_seed + 31 * cid + 11)
        p = rng.permutation(idx)
        nv = max(1, int(round(0.10 * len(p)))) if len(p) >= 10 else 1
        va, tr = p[:nv], p[nv:]
        splits[cid] = (
            torch.as_tensor(tr, dtype=torch.long, device=device),
            torch.as_tensor(va, dtype=torch.long, device=device),
        )

    frac_mal = float(scenario.get("malicious_fraction", 0.0))
    mal_order = np.random.default_rng(seed + 4242).permutation(K)
    n_mal = int(round(K * frac_mal))
    malicious = set(int(c) for c in mal_order[:n_mal])
    attack = scenario.get("attack", "none")
    attack_scale = float(scenario.get("attack_scale", 3.0))

    sel_rng = np.random.default_rng(seed + 777)
    frac_part = float(fed.get("client_fraction", 1.0))
    prox_mu = float(fed.get("fedprox_mu", 0.0)) if agg_method == "fedprox" else 0.0
    tcfg = trust_cfg if trust_cfg is not None else fed["trust"]

    root_split = None
    if agg_method == "fltrust":
        root_idx = _fltrust_root_indices(task_data.y_train,
                                         int(fed.get("fltrust_root_size", 200)),
                                         seed + 2024)
        root_t = torch.as_tensor(root_idx, dtype=torch.long, device=device)
        root_split = (root_t, root_t[:1])

    previous_trust: Dict[int, float] = {}
    round_rows, trust_rows = [], []

    for rnd in range(1, rounds + 1):
        if frac_part >= 1.0:
            selected = list(range(K))
        else:
            m = max(1, int(np.ceil(K * frac_part)))
            selected = sorted(sel_rng.choice(K, size=m, replace=False).tolist())

        local_states, metas, actual_ids = [], [], []
        for cid in selected:
            tr_idx, va_idx = splits[cid]
            if tr_idx.numel() < 2 or va_idx.numel() < 1:
                continue
            is_mal = cid in malicious
            local_state, meta = local_train(
                local_model, global_state, X_tr, y_tr, tr_idx, va_idx,
                train_cfg, C, device, seed + 1000 * rnd + cid,
                prox_mu=prox_mu, label_flip=(attack == "label_flip" and is_mal),
            )
            if attack == "sign_flip" and is_mal:
                delta = flatten_state_delta(local_state, global_state)
                local_state = apply_delta(global_state, delta, -attack_scale)
            elif attack == "model_noise" and is_mal:
                g = torch.Generator().manual_seed(seed + 5000 * rnd + cid)
                local_state = {k: v + torch.randn(v.shape, generator=g) * attack_scale
                               for k, v in local_state.items()}
            local_states.append(local_state)
            metas.append(meta)
            actual_ids.append(cid)

        if not local_states:
            raise RuntimeError("No client produced an update this round.")

        if agg_method == "tafed":
            global_state, logs = tafed(local_states, metas, global_state, tcfg, previous_trust, actual_ids)
        elif agg_method == "krum":
            global_state, logs = krum(local_states, metas, n_byzantine=n_mal)
        elif agg_method == "median":
            global_state, logs = coordinate_median(local_states, metas)
        elif agg_method == "trimmed_mean":
            trim = max(frac_mal, float(fed.get("trimmed_mean_default", 0.1)))
            global_state, logs = trimmed_mean(local_states, metas, trim_fraction=trim)
        elif agg_method == "fltrust":
            server_state, _ = local_train(
                local_model, global_state, X_tr, y_tr, root_split[0], root_split[1],
                train_cfg, C, device, seed + 9000 * rnd, prox_mu=0.0, label_flip=False)
            global_state, logs = fltrust(local_states, metas, global_state, server_state)
        else:
            global_state, logs = fedavg(local_states, metas)

        model.load_state_dict(global_state)
        y_pred, probs = predict(model, task_data.X_val, int(train_cfg.get("eval_batch_size", 8192)), device)
        metrics = metric_dict(task_data.y_val, y_pred, probs)
        round_rows.append({"round": rnd, "method": method_label, **metrics})
        for cid, lg, meta in zip(actual_ids, logs, metas):
            trust_rows.append({
                "round": rnd, "client_id": cid, "method": method_label,
                "malicious": cid in malicious, "val_loss": meta["val_loss"],
                "val_acc": meta["val_acc"], "n": meta["n"], **lg
            })

    model.load_state_dict(global_state)
    return model, round_rows, trust_rows, sorted(malicious), partition
