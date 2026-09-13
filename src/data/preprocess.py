from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class PreparedTask:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    class_names: list[str]
    feature_names: list[str]


@dataclass
class PreparedBundle:
    binary: PreparedTask
    family: PreparedTask
    flat: PreparedTask
    X_test_common: np.ndarray
    y_test_end: np.ndarray
    end_class_names: list[str]


def _safe_split(idx, strat_y, test_size, seed):
    series = pd.Series(strat_y)
    strat = strat_y if series.value_counts().min() >= 2 else None
    return train_test_split(idx, test_size=test_size, random_state=seed, stratify=strat)


def prepare_multistage(X: pd.DataFrame, labels: pd.DataFrame, split_cfg: dict, clip_q: list[float],
                       seed: int, min_class_samples: int) -> PreparedBundle:
    family_all = labels["family"].astype(str).reset_index(drop=True)
    binary_all = labels["binary"].astype(int).reset_index(drop=True)

    family_counts = family_all.value_counts()
    rare_attack = set(family_counts[(family_counts < min_class_samples) & (family_counts.index != "Benign")].index)
    if rare_attack:
        print(f"  dropping rare families (<{min_class_samples} samples): {sorted(rare_attack)}")
    keep = ~family_all.isin(rare_attack)
    X2 = X.loc[keep].reset_index(drop=True)
    family_all = family_all.loc[keep].reset_index(drop=True)
    binary_all = binary_all.loc[keep].reset_index(drop=True)

    idx = np.arange(len(X2))
    strat_base = family_all.to_numpy() if family_all.value_counts().min() >= 2 else binary_all.to_numpy()
    trainval_idx, test_idx = _safe_split(idx, strat_base, split_cfg["test"], seed)
    relative_val = split_cfg["val"] / (split_cfg["train"] + split_cfg["val"])
    strat_trainval = family_all.iloc[trainval_idx].to_numpy()
    if pd.Series(strat_trainval).value_counts().min() < 2:
        strat_trainval = binary_all.iloc[trainval_idx].to_numpy()
    train_idx, val_idx = _safe_split(trainval_idx, strat_trainval, relative_val, seed + 1)

    X_train_df = X2.iloc[train_idx].copy()
    X_val_df = X2.iloc[val_idx].copy()
    X_test_df = X2.iloc[test_idx].copy()

    med = X_train_df.median(numeric_only=True)
    X_train_df = X_train_df.fillna(med).fillna(0.0)
    X_val_df = X_val_df.fillna(med).fillna(0.0)
    X_test_df = X_test_df.fillna(med).fillna(0.0)

    q_low, q_high = clip_q
    lo = X_train_df.quantile(q_low)
    hi = X_train_df.quantile(q_high)
    X_train_df = X_train_df.clip(lo, hi, axis=1)
    X_val_df = X_val_df.clip(lo, hi, axis=1)
    X_test_df = X_test_df.clip(lo, hi, axis=1)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df).astype(np.float32)
    X_val = scaler.transform(X_val_df).astype(np.float32)
    X_test = scaler.transform(X_test_df).astype(np.float32)
    X_train = np.ascontiguousarray(X_train)
    X_val = np.ascontiguousarray(X_val)
    X_test = np.ascontiguousarray(X_test)

    feature_names = list(X2.columns)


    yb_train = binary_all.iloc[train_idx].to_numpy(dtype=np.int64)
    yb_val = binary_all.iloc[val_idx].to_numpy(dtype=np.int64)
    yb_test = binary_all.iloc[test_idx].to_numpy(dtype=np.int64)
    binary_task = PreparedTask(X_train, yb_train, X_val, yb_val, X_test, yb_test,
                               ["Benign", "Attack"], feature_names)


    attack_families = sorted([x for x in family_all.unique().tolist() if x != "Benign"])
    if len(attack_families) < 2:
        raise ValueError(f"Family task needs at least two attack families; found {attack_families}")
    end_class_names = ["Benign"] + attack_families
    end_map = {name: i for i, name in enumerate(end_class_names)}

    fam_train = family_all.iloc[train_idx].reset_index(drop=True)
    fam_val = family_all.iloc[val_idx].reset_index(drop=True)
    fam_test = family_all.iloc[test_idx].reset_index(drop=True)


    def attack_subset(fam: pd.Series, Xarr: np.ndarray):
        mask = fam.ne("Benign").to_numpy()
        y = np.array([end_map[f] - 1 for f in fam[mask]], dtype=np.int64)
        return np.ascontiguousarray(Xarr[mask]), y

    Xf_train, yf_train = attack_subset(fam_train, X_train)
    Xf_val, yf_val = attack_subset(fam_val, X_val)
    Xf_test, yf_test = attack_subset(fam_test, X_test)
    family_task = PreparedTask(Xf_train, yf_train, Xf_val, yf_val, Xf_test, yf_test,
                               attack_families, feature_names)


    yflat_train = np.array([end_map[f] for f in fam_train], dtype=np.int64)
    yflat_val = np.array([end_map[f] for f in fam_val], dtype=np.int64)
    yflat_test = np.array([end_map[f] for f in fam_test], dtype=np.int64)
    flat_task = PreparedTask(X_train, yflat_train, X_val, yflat_val, X_test, yflat_test,
                             end_class_names, feature_names)

    return PreparedBundle(
        binary=binary_task,
        family=family_task,
        flat=flat_task,
        X_test_common=X_test,
        y_test_end=yflat_test,
        end_class_names=end_class_names,
    )
