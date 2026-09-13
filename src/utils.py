from __future__ import annotations

import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def choose_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def make_run_dir(root: Path, project_name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = project_name.replace(" ", "_")
    run_dir = root / f"{stamp}_{safe}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def state_dict_to_cpu(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in state.items()}


def flatten_state_delta(local_state, global_state) -> torch.Tensor:
    chunks = []
    for k in global_state:
        a = local_state[k].detach().float().cpu().reshape(-1)
        b = global_state[k].detach().float().cpu().reshape(-1)
        chunks.append(a - b)
    return torch.cat(chunks) if chunks else torch.tensor([], dtype=torch.float32)


def apply_delta(global_state, delta: torch.Tensor, scale: float):
    out = {}
    pos = 0
    for k, tensor in global_state.items():
        n = tensor.numel()
        d = delta[pos:pos+n].reshape(tensor.shape).to(tensor.dtype)
        out[k] = tensor.detach().cpu() + scale * d
        pos += n
    return out


def ensure_relative(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p
