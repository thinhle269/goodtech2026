from __future__ import annotations

from typing import Dict
import numpy as np


def iid_partition(y: np.ndarray, num_clients: int, seed: int) -> Dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    parts = np.array_split(idx, num_clients)
    return {i: p.astype(int) for i, p in enumerate(parts)}


def dirichlet_partition(y: np.ndarray, num_clients: int, alpha: float, min_samples: int, seed: int, max_tries: int = 100) -> Dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    for _ in range(max_tries):
        buckets = [[] for _ in range(num_clients)]
        for c in classes:
            idx_c = np.where(y == c)[0]
            rng.shuffle(idx_c)
            props = rng.dirichlet(np.repeat(alpha, num_clients))
            cuts = (np.cumsum(props) * len(idx_c)).astype(int)[:-1]
            splits = np.split(idx_c, cuts)
            for k, split in enumerate(splits):
                buckets[k].extend(split.tolist())
        sizes = [len(b) for b in buckets]
        if min(sizes) >= min_samples:
            return {i: np.array(rng.permutation(b), dtype=int) for i, b in enumerate(buckets)}
    print(f"  [partition] dirichlet(alpha={alpha}) failed to satisfy min_samples={min_samples} "
          f"after {max_tries} tries; falling back to IID")
    return iid_partition(y, num_clients, seed)


def make_partition(y: np.ndarray, cfg: dict, num_clients: int, seed: int) -> Dict[int, np.ndarray]:
    mode = cfg.get("mode", "iid").lower()
    if mode == "iid":
        return iid_partition(y, num_clients, seed)
    if mode == "dirichlet":
        return dirichlet_partition(y, num_clients, float(cfg.get("alpha", 0.5)), int(cfg.get("min_client_samples", 20)), seed)
    raise ValueError(f"Unknown partition mode: {mode}")
