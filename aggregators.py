from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

from src.utils import flatten_state_delta


def weighted_average(states: List[dict], weights: np.ndarray) -> dict:
    weights = np.asarray(weights, dtype=np.float64)
    if weights.sum() <= 0:
        weights = np.ones(len(states), dtype=np.float64)
    weights = weights / weights.sum()
    out = {}
    for k in states[0]:
        acc = torch.zeros_like(states[0][k], dtype=torch.float32)
        for s, w in zip(states, weights):
            acc = acc + s[k].detach().cpu().float() * float(w)
        out[k] = acc.to(states[0][k].dtype)
    return out


def fedavg(states: List[dict], metas: List[dict]) -> Tuple[dict, List[dict]]:
    n = np.array([m["n"] for m in metas], dtype=float)
    state = weighted_average(states, n)
    logs = [{"trust": 1.0, "quality": 1.0, "cosine": 1.0, "norm": 1.0, "gated": False} for _ in n]
    return state, logs


def tafed(states: List[dict], metas: List[dict], global_state: dict, trust_cfg: dict,
          previous_trust: Dict[int, float], client_ids: List[int]) -> Tuple[dict, List[dict]]:

    deltas = [flatten_state_delta(s, global_state) for s in states]
    D = torch.stack(deltas, dim=0)
    reference = D.median(dim=0).values

    sample_n = np.array([m["n"] for m in metas], dtype=float)
    norms = torch.linalg.vector_norm(D, dim=1).numpy().astype(float)
    median_norm = float(np.median(norms)) + 1e-12

    val_losses = np.array([float(m["val_loss"]) for m in metas], dtype=float)
    quality = np.exp(-np.clip(val_losses, 0, 50))
    quality = quality / max(float(quality.max()), 1e-12)

    ref_norm = float(torch.linalg.vector_norm(reference).item())
    cos_raw = np.zeros(len(deltas), dtype=float)
    for i, d in enumerate(deltas):
        denom = float(torch.linalg.vector_norm(d).item()) * ref_norm
        cos_raw[i] = float(torch.dot(d, reference).item()) / denom if denom > 1e-12 else 0.0
    cosine = (1.0 + cos_raw) / 2.0

    norm_score = np.exp(-np.abs(np.log((norms + 1e-12) / median_norm)))

    w = trust_cfg["weights"]
    total_w = max(float(w["quality"] + w["cosine"] + w["norm"]), 1e-12)
    raw = (w["quality"] * quality + w["cosine"] * cosine + w["norm"] * norm_score) / total_w

    min_trust = float(trust_cfg.get("min_trust", 0.05))
    gate_on = bool(trust_cfg.get("cosine_gate", True)) and float(w["cosine"]) > 0
    gate_thr = float(trust_cfg.get("gate_threshold", 0.0))
    gated = np.zeros(len(raw), dtype=bool)
    if gate_on:
        gated = cos_raw < gate_thr
        raw = np.where(gated, min_trust, raw)

    beta = float(trust_cfg["ema_beta"])
    trust, logs = [], []
    for i, cid in enumerate(client_ids):
        prev = previous_trust.get(cid, float(raw[i]))
        t = beta * prev + (1.0 - beta) * float(raw[i])
        t = max(min_trust, min(1.0, t))
        previous_trust[cid] = t
        trust.append(t)
        logs.append({
            "trust": t, "quality": float(quality[i]), "cosine": float(cosine[i]),
            "norm": float(norm_score[i]), "gated": bool(gated[i])
        })
    agg_weights = sample_n * np.array(trust, dtype=float)
    return weighted_average(states, agg_weights), logs


def _stack_states(states: List[dict]):
    keys = list(states[0].keys())
    flat = torch.stack([torch.cat([s[k].detach().cpu().float().reshape(-1) for k in keys])
                        for s in states], dim=0)
    return flat, keys


def _unflatten(vec: torch.Tensor, template: dict) -> dict:
    out, pos = {}, 0
    for k, t in template.items():
        n = t.numel()
        out[k] = vec[pos:pos + n].reshape(t.shape).to(t.dtype)
        pos += n
    return out


def krum(states: List[dict], metas: List[dict], n_byzantine: int) -> Tuple[dict, List[dict]]:

    flat, _ = _stack_states(states)
    n = flat.shape[0]
    f = min(int(n_byzantine), max(0, (n - 3) // 2))
    m = max(1, n - f - 2)
    dists = torch.cdist(flat, flat, p=2.0) ** 2
    scores = []
    for i in range(n):
        d = torch.cat([dists[i, :i], dists[i, i + 1:]])
        scores.append(torch.sort(d).values[:m].sum().item())
    winner = int(np.argmin(scores))
    logs = [{"trust": 1.0 if i == winner else 0.0, "quality": 0.0, "cosine": 0.0,
             "norm": 0.0, "gated": False} for i in range(n)]
    return {k: v.detach().cpu().clone() for k, v in states[winner].items()}, logs


def coordinate_median(states: List[dict], metas: List[dict]) -> Tuple[dict, List[dict]]:

    flat, _ = _stack_states(states)
    med = flat.median(dim=0).values
    logs = [{"trust": 1.0, "quality": 0.0, "cosine": 0.0, "norm": 0.0, "gated": False}
            for _ in states]
    return _unflatten(med, states[0]), logs


def trimmed_mean(states: List[dict], metas: List[dict], trim_fraction: float) -> Tuple[dict, List[dict]]:

    flat, _ = _stack_states(states)
    n = flat.shape[0]
    k = int(np.ceil(trim_fraction * n))
    k = min(k, (n - 1) // 2)
    sorted_flat, _ = torch.sort(flat, dim=0)
    kept = sorted_flat[k:n - k] if k > 0 else sorted_flat
    mean = kept.mean(dim=0)
    logs = [{"trust": 1.0, "quality": 0.0, "cosine": 0.0, "norm": 0.0, "gated": False}
            for _ in states]
    return _unflatten(mean, states[0]), logs


def fltrust(states: List[dict], metas: List[dict], global_state: dict,
            server_state: dict) -> Tuple[dict, List[dict]]:

    g0 = flatten_state_delta(server_state, global_state)
    g0_norm = float(torch.linalg.vector_norm(g0).item())
    deltas = [flatten_state_delta(s, global_state) for s in states]
    trusts, scaled = [], []
    for d in deltas:
        d_norm = float(torch.linalg.vector_norm(d).item())
        denom = d_norm * g0_norm
        cos = float(torch.dot(d, g0).item()) / denom if denom > 1e-12 else 0.0
        t = max(0.0, cos)
        trusts.append(t)
        scaled.append(d * (g0_norm / d_norm) if d_norm > 1e-12 else d)
    total = sum(trusts)
    if total <= 1e-12:
        agg_delta = torch.zeros_like(g0)
    else:
        agg_delta = sum(t * d for t, d in zip(trusts, scaled)) / total
    new_flat = torch.cat([global_state[k].detach().cpu().float().reshape(-1)
                          for k in global_state]) + agg_delta
    logs = [{"trust": float(t), "quality": 0.0, "cosine": float(t), "norm": 0.0,
             "gated": t <= 0.0} for t in trusts]
    return _unflatten(new_flat, global_state), logs


def resolve_trust_variant(base_trust: dict, variant: str | None) -> dict:

    cfg = json_copy(base_trust)
    if not variant:
        return cfg
    w = cfg["weights"]
    if variant == "no_quality":
        w["quality"] = 0.0
    elif variant == "no_cosine":
        w["cosine"] = 0.0
        cfg["cosine_gate"] = False
    elif variant == "no_norm":
        w["norm"] = 0.0
    elif variant == "no_ema":
        cfg["ema_beta"] = 0.0
    else:
        raise ValueError(f"Unknown ablation variant: {variant}")
    s = w["quality"] + w["cosine"] + w["norm"]
    if s <= 0:
        raise ValueError("Ablation removed all trust components")
    for k in w:
        w[k] = w[k] / s
    return cfg


def json_copy(obj):
    import json
    return json.loads(json.dumps(obj))
