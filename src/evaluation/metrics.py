from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef, confusion_matrix, classification_report, roc_auc_score


def predict(model, X: np.ndarray, batch_size: int, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    preds, probs = [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i + batch_size]).to(device)
            p = torch.softmax(model(xb), dim=1)
            probs.append(p.cpu().numpy())
            preds.append(p.argmax(1).cpu().numpy())
    return np.concatenate(preds), np.concatenate(probs)


def metric_dict(y_true, y_pred, probs=None) -> Dict[str, float]:
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    _, _, f1w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p),
        "recall_macro": float(r),
        "f1_macro": float(f1),
        "f1_weighted": float(f1w),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0,
    }
    if probs is not None and probs.ndim == 2 and probs.shape[1] == 2 and len(np.unique(y_true)) == 2:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, probs[:, 1]))
        except Exception:
            out["roc_auc"] = float("nan")
    return out


def detailed_metrics(y_true, y_pred, class_names):
    report = classification_report(y_true, y_pred, labels=list(range(len(class_names))), target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    rows = []
    for name in class_names:
        d = report.get(name, {})
        rows.append({"class": name, "precision": d.get("precision", 0.0), "recall": d.get("recall", 0.0), "f1": d.get("f1-score", 0.0), "support": d.get("support", 0.0)})
    return rows, cm
