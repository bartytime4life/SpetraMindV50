import json
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(data: Dict[str, Any], path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(df: pd.DataFrame, path: str, index: bool = False) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    df.to_csv(path, index=index)


def load_numpy(path: str) -> np.ndarray:
    """Loads .npy or .npz (expects array at key 'arr_0' if .npz) or CSV/TSV."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path)
    if ext == ".npz":
        data = np.load(path)
        if "arr_0" in data:
            return data["arr_0"]
        for k in data.files:
            return data[k]
        raise ValueError(f"No arrays found in npz: {path}")
    if ext in (".csv", ".tsv"):
        df = pd.read_csv(path, sep="," if ext == ".csv" else "\t", header=None)
        return df.values
    raise ValueError(f"Unsupported array file: {path}")


def save_numpy(arr: np.ndarray, path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    np.save(path, arr)


def try_load(path: Optional[str]) -> Optional[np.ndarray]:
    if path is None or not os.path.exists(path):
        return None
    return load_numpy(path)


def load_preds_truth(
    pred_dir: str,
    y_true_path: Optional[str] = None,
    mu_name: str = "mu.npy",
    sigma_name: str = "sigma.npy",
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Load μ, σ from a directory and optional y_true."""
    mu = load_numpy(os.path.join(pred_dir, mu_name))
    sigma = load_numpy(os.path.join(pred_dir, sigma_name))
    y_true = try_load(y_true_path)
    if sigma.shape != mu.shape:
        raise ValueError(f"Shape mismatch: mu {mu.shape} vs sigma {sigma.shape}")
    if y_true is not None and y_true.shape != mu.shape:
        raise ValueError(f"Shape mismatch: mu {mu.shape} vs y_true {y_true.shape}")
    return mu, sigma, y_true


def maybe_load_bin_meta(meta_path: Optional[str]) -> Optional[pd.DataFrame]:
    if not meta_path or not os.path.exists(meta_path):
        return None
    df = pd.read_csv(meta_path)
    return df


def maybe_load_planet_ids(ids_path: Optional[str], n: int) -> pd.Series:
    if ids_path and os.path.exists(ids_path):
        df = pd.read_csv(ids_path)
        if "planet_id" in df.columns:
            ids = df["planet_id"].astype(str)
        else:
            ids = df.iloc[:, 0].astype(str)
        if len(ids) != n:
            if len(ids) < n:
                pad = pd.Series([f"obj_{i}" for i in range(len(ids), n)])
                ids = pd.concat([ids, pad], ignore_index=True)
            else:
                ids = ids.iloc[:n]
        return ids.reset_index(drop=True)
    return pd.Series([f"obj_{i}" for i in range(n)])
