"""Regression metrics and error-analysis helpers for RUL models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def nasa_score(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    reduction: str = "mean",
) -> float:
    """NASA/PHM asymmetric RUL score (lower is better).

    Over-estimating RUL (a late prediction) is penalised more heavily than
    under-estimating it. ``d = y_pred - y_true``.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    d = y_pred - y_true
    penalties = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)

    if reduction == "sum":
        return float(penalties.sum())
    if reduction == "mean":
        return float(penalties.mean())
    raise ValueError("reduction must be 'mean' or 'sum'")


def regression_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> dict[str, float | int]:
    """Compute RMSE, MAE, R2, NASA score, and late-prediction statistics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if len(y_true) == 0:
        raise ValueError("Cannot evaluate empty arrays")

    late = y_pred > y_true
    return {
        "n": int(len(y_true)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "nasa_mean": nasa_score(y_true, y_pred, reduction="mean"),
        "nasa_sum": nasa_score(y_true, y_pred, reduction="sum"),
        "late_count": int(late.sum()),
        "late_rate": float(late.mean()),
        "mean_error": float(np.mean(y_pred - y_true)),
    }


def predictions_frame(
    units: np.ndarray | pd.Series,
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> pd.DataFrame:
    """Build a per-engine prediction/error table."""
    df = pd.DataFrame(
        {
            "unit": np.asarray(units),
            "y_true": np.asarray(y_true, dtype=float),
            "y_pred": np.asarray(y_pred, dtype=float),
        }
    )
    df["error"] = df["y_pred"] - df["y_true"]
    df["abs_error"] = df["error"].abs()
    df["late"] = df["error"] > 0
    return df


def error_by_rul_bin(
    pred_df: pd.DataFrame,
    bins: tuple[float, ...] = (0, 30, 60, 90, 125, np.inf),
    labels: tuple[str, ...] = ("0-30", "30-60", "60-90", "90-125", ">125"),
) -> pd.DataFrame:
    """Summarise error by true-RUL range."""
    df = pred_df.copy()
    df["rul_bin"] = pd.cut(df["y_true"], bins=bins, labels=labels, include_lowest=True)
    out = (
        df.groupby("rul_bin", observed=True)
        .agg(
            n=("unit", "count"),
            mae=("abs_error", "mean"),
            rmse=("error", lambda x: float(np.sqrt(np.mean(np.square(x))))),
            late_rate=("late", "mean"),
        )
        .reset_index()
    )
    return out


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    """Write JSON with NumPy-friendly conversion."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    path.write_text(json.dumps(payload, indent=2, default=_default), encoding="utf-8")
