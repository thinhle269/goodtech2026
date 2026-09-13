from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from .labels import derive_labels

RESERVED = ["__raw_label", "__binary", "__family"]

import re as _re


def _sanitize_col(name: str) -> str:
    out = _re.sub(r"[^0-9A-Za-z_.\-/:]+", "_", name).strip("_")
    return out[:64] if out else "col"


def _cache_key(dataset_name: str, ds_cfg: dict, common_cfg: dict) -> str:
    files = [str(Path(f)) for f in ds_cfg["files"]]
    sizes = []
    for f in files:
        p = Path(f)
        sizes.append(p.stat().st_size if p.exists() else -1)
    payload = {
        "version": 2,
        "dataset": dataset_name,
        "files": files,
        "sizes": sizes,
        "label_column": ds_cfg["label_column"],
        "drop_columns": sorted(ds_cfg.get("drop_columns", [])),
        "benign_cap": ds_cfg.get("benign_cap"),
        "attack_cap": ds_cfg.get("attack_cap"),
        "data_seed": ds_cfg.get("data_seed", 0),
        "cat_card": ds_cfg.get("categorical_max_cardinality", 0),
        "drop_constant": common_cfg.get("drop_constant_features", True),
    }
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _read_files(files: List[str], chunksize: int) -> pd.DataFrame:
    frames = []
    for f in files:
        p = Path(f)
        if not p.exists():
            raise FileNotFoundError(f"Dataset file not found: {p}")
        if p.suffix.lower() in {".parquet", ".pq"}:
            frames.append(pd.read_parquet(p))
            continue
        print(f"  reading {p.name} ...", flush=True)
        for chunk in pd.read_csv(p, low_memory=False, chunksize=chunksize):
            frames.append(chunk)
    return pd.concat(frames, ignore_index=True)


def _class_cap_sample(df: pd.DataFrame, family: pd.Series, label_col: str,
                      benign_cap: int | None, attack_cap: int | None, seed: int) -> pd.DataFrame:

    if benign_cap is None and attack_cap is None:
        return df
    rng_base = np.int64(seed)
    keep_parts = []
    for raw, group in df.groupby(label_col, sort=True):
        fam = family.loc[group.index].iloc[0]
        cap = benign_cap if fam == "Benign" else attack_cap
        if cap is None or len(group) <= cap:
            keep_parts.append(group)
        else:
            h = int(hashlib.md5(str(raw).encode()).hexdigest()[:8], 16)
            keep_parts.append(group.sample(n=cap, random_state=int((rng_base + h) % (2**31 - 1))))
    out = pd.concat(keep_parts).sort_index().reset_index(drop=True)
    return out


def load_dataset(dataset_name: str, ds_cfg: dict, common_cfg: dict, project_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    cache_dir = project_root / common_cfg.get("cache_dir", "data/processed")
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(dataset_name, ds_cfg, common_cfg)
    cache_file = cache_dir / f"{dataset_name}_{key}.parquet"
    stats_file = cache_dir / f"{dataset_name}_{key}_stats.json"

    if cache_file.exists() and stats_file.exists():
        print(f"[{dataset_name}] using cached processed data: {cache_file.name}")
        full = pd.read_parquet(cache_file)
        labels = pd.DataFrame({
            "raw_label": full["__raw_label"], "binary": full["__binary"], "family": full["__family"]
        })
        X = full.drop(columns=RESERVED)
        stats = json.loads(stats_file.read_text(encoding="utf-8"))
        return X, labels, stats

    label_col = ds_cfg["label_column"]
    df = _read_files(ds_cfg["files"], int(common_cfg.get("csv_chunksize", 500000)))
    rows_raw = len(df)
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not in {dataset_name}; columns: {list(df.columns)[:40]}")


    drop_cols = [c for c in ds_cfg.get("drop_columns", []) if c in df.columns]
    df = df.drop(columns=drop_cols)


    df = df.replace([np.inf, -np.inf], np.nan)
    df = df[df[label_col].notna()]
    df = df[df[label_col].astype(str) != label_col]
    rows_no_na_label = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    rows_dedup = len(df)


    labels_full = derive_labels(dataset_name, df[label_col])
    df = _class_cap_sample(df, labels_full["family"], label_col,
                           ds_cfg.get("benign_cap"), ds_cfg.get("attack_cap"),
                           int(ds_cfg.get("data_seed", 0)))
    labels = derive_labels(dataset_name, df[label_col]).reset_index(drop=True)


    X = df.drop(columns=[label_col])
    for c in list(X.columns):
        lc = c.lower()
        if lc in {"attack_label", "attack_type", "label", "class", "category"} or lc.endswith("_label"):
            X = X.drop(columns=[c])

    cat_card = int(ds_cfg.get("categorical_max_cardinality", 0))
    obj_cols = [c for c in X.columns if X[c].dtype == object]
    encoded, dropped_obj = [], []
    for c in obj_cols:
        num = pd.to_numeric(X[c], errors="coerce")
        if float(num.notna().mean()) >= 0.99:
            X[c] = num
            continue
        if cat_card > 0:


            s = X[c].where(X[c].notna(), "absent").astype(str).str.strip()
            s = s.replace({"0": "absent", "0.0": "absent", "nan": "absent", "": "absent"})
            if s.nunique(dropna=False) <= cat_card:
                X[c] = s
                encoded.append(c)
                continue
        dropped_obj.append(c)
    if dropped_obj:
        X = X.drop(columns=dropped_obj)
    if encoded:
        X = pd.get_dummies(X, columns=encoded, dummy_na=False)
        X.columns = [_sanitize_col(str(c)) for c in X.columns]
        cols = pd.Index(X.columns)
        if cols.duplicated().any():
            merged = {}
            for name in cols.unique():
                sub = X.loc[:, cols == name]
                merged[name] = sub.iloc[:, 0] if sub.shape[1] == 1 else sub.max(axis=1)
            X = pd.DataFrame(merged, index=X.index)

    X = X.apply(pd.to_numeric, errors="coerce").astype(np.float32)
    if common_cfg.get("drop_constant_features", True):
        nunique = X.nunique(dropna=False)
        X = X.loc[:, nunique > 1]
    if X.empty:
        raise ValueError("No usable features remained after preprocessing.")
    X = X.reset_index(drop=True)

    stats = {
        "dataset": dataset_name,
        "rows_raw": int(rows_raw),
        "rows_after_label_filter": int(rows_no_na_label),
        "rows_after_dedup": int(rows_dedup),
        "rows_final": int(len(X)),
        "n_features": int(X.shape[1]),
        "label_column": label_col,
        "dropped_columns": drop_cols,
        "onehot_columns": encoded,
        "dropped_object_columns": dropped_obj,
        "raw_label_counts": labels["raw_label"].value_counts().to_dict(),
        "family_counts": labels["family"].value_counts().to_dict(),
        "binary_counts": {str(k): int(v) for k, v in labels["binary"].value_counts().items()},
        "source_files": [str(f) for f in ds_cfg["files"]],
    }

    full = X.copy()
    full["__raw_label"] = labels["raw_label"].to_numpy()
    full["__binary"] = labels["binary"].to_numpy()
    full["__family"] = labels["family"].to_numpy()
    full.to_parquet(cache_file, index=False)
    stats_file.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"[{dataset_name}] processed {len(X)} rows x {X.shape[1]} features (cached -> {cache_file.name})")
    return X, labels, stats
