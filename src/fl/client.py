from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn

from src.utils import state_dict_to_cpu


def _class_weight_tensor(y: torch.Tensor, n_classes: int, enabled: bool):
    if not enabled:
        return None
    counts = torch.bincount(y, minlength=n_classes).double()
    counts = torch.clamp(counts, min=1.0)
    w = counts.sum() / (n_classes * counts)
    return w.float().to(y.device)


def local_train(model: nn.Module, global_state: Dict[str, torch.Tensor],
                X_full: torch.Tensor, y_full: torch.Tensor,
                tr_idx: torch.Tensor, va_idx: torch.Tensor,
                train_cfg: dict, num_classes: int, device: torch.device, seed: int,
                prox_mu: float = 0.0, label_flip: bool = False) -> Tuple[dict, dict]:

    model.load_state_dict(global_state)
    Xl = X_full[tr_idx]
    yl = y_full[tr_idx].clone()
    if label_flip and num_classes > 1:
        yl = (yl + 1) % num_classes

    weights = _class_weight_tensor(yl, num_classes, train_cfg.get("class_weighting", True))
    criterion = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(model.parameters(), lr=float(train_cfg["learning_rate"]),
                           weight_decay=float(train_cfg["weight_decay"]))
    if prox_mu > 0:
        global_params = {k: v.detach().to(device) for k, v in global_state.items()}

    bs = int(train_cfg["batch_size"])
    n = Xl.shape[0]
    g = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(int(train_cfg["local_epochs"])):
        perm = torch.randperm(n, generator=g).to(device)
        for i in range(0, n, bs):
            sl = perm[i:i + bs]
            xb, yb = Xl[sl], yl[sl]
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            if prox_mu > 0:
                prox = torch.zeros((), device=device)
                for name, p in model.named_parameters():
                    prox = prox + torch.sum((p - global_params[name]) ** 2)
                loss = loss + 0.5 * prox_mu * prox
            loss.backward()
            opt.step()


    val_loss, val_acc = evaluate_loss(model, X_full[va_idx], y_full[va_idx],
                                      int(train_cfg.get("eval_batch_size", 8192)))
    return state_dict_to_cpu(model.state_dict()), {
        "val_loss": float(val_loss), "val_acc": float(val_acc), "n": int(n)
    }


@torch.no_grad()
def evaluate_loss(model: nn.Module, X: torch.Tensor, y: torch.Tensor, batch_size: int):
    if y.numel() == 0:
        return 1e9, 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    model.eval()
    loss_sum, correct = 0.0, 0
    for i in range(0, X.shape[0], batch_size):
        xb, yb = X[i:i + batch_size], y[i:i + batch_size]
        logits = model(xb)
        loss_sum += float(criterion(logits, yb).item())
        correct += int((logits.argmax(1) == yb).sum().item())
    n = int(y.numel())
    return loss_sum / n, correct / n
