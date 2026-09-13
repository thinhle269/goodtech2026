from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.evaluation.metrics import predict, metric_dict
from src.models.mlp import build_model


def train_centralized(task_data, cfg: dict, device: torch.device, seed: int):
    torch.manual_seed(seed + 7)
    model = build_model(task_data.X_train.shape[1], len(task_data.class_names), cfg["model"]).to(device)

    counts = np.bincount(task_data.y_train, minlength=len(task_data.class_names)).astype(float)
    counts[counts == 0] = 1.0
    w = counts.sum() / (len(counts) * counts)
    weight = torch.tensor(w, dtype=torch.float32, device=device) if cfg["training"].get("class_weighting", True) else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["training"]["learning_rate"]),
                           weight_decay=float(cfg["training"]["weight_decay"]))

    X = torch.from_numpy(task_data.X_train).to(device)
    y = torch.from_numpy(task_data.y_train).to(device)
    bs = int(cfg["training"]["batch_size"])
    eval_bs = int(cfg["training"].get("eval_batch_size", 8192))
    n = X.shape[0]
    g = torch.Generator().manual_seed(seed)

    best_state, best_f1, patience = None, -1.0, 0
    history = []
    for epoch in range(1, int(cfg["training"]["centralized_epochs"]) + 1):
        model.train()
        perm = torch.randperm(n, generator=g).to(device)
        for i in range(0, n, bs):
            sl = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(X[sl]), y[sl])
            loss.backward()
            opt.step()
        pred, prob = predict(model, task_data.X_val, eval_bs, device)
        m = metric_dict(task_data.y_val, pred, prob)
        history.append({"round": epoch, "method": "centralized", **m})
        if m["f1_macro"] > best_f1 + 1e-6:
            best_f1 = m["f1_macro"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= int(cfg["training"].get("early_stopping_patience", 5)):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history
