"""Reusable inference helpers for the CLI and Streamlit app."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.features import add_engineered_features


def prepare_input(df: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    """Validate and normalise an inference frame.

    The input must contain op1..op3 and the sensor columns the model was trained
    on. ``unit`` and ``cycle`` are optional; if absent, a single unit and a
    sequential cycle index are assumed.
    """
    required = ["op1", "op2", "op3"] + list(bundle["sensor_columns"])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    if "unit" not in out.columns:
        out["unit"] = 1
    if "cycle" not in out.columns:
        out["cycle"] = out.groupby("unit").cumcount() + 1

    out["unit"] = out["unit"].astype(int)
    out["cycle"] = out["cycle"].astype(int)
    return out.sort_values(["unit", "cycle"]).reset_index(drop=True)


def predict_from_history(df: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    """Predict RUL from the last observed cycle of each unit in ``df``."""
    prepared = prepare_input(df, bundle)
    featured = add_engineered_features(
        prepared,
        sensors=bundle["sensor_columns"],
        windows=tuple(bundle["windows"]),
        include_cycle=bool(bundle["include_cycle"]),
    )

    last = (
        featured.sort_values(["unit", "cycle"])
        .groupby("unit")
        .tail(1)
        .sort_values("unit")
        .reset_index(drop=True)
    )

    missing = [c for c in bundle["features"] if c not in last.columns]
    if missing:
        raise ValueError(f"Feature engineering did not produce: {missing}")

    pred = bundle["model"].predict(last[bundle["features"]])
    pred = np.clip(pred, 0.0, float(bundle["max_rul"]))

    out = last[["unit", "cycle"]].copy()
    out["predicted_RUL"] = pred
    out["priority"] = pd.cut(
        out["predicted_RUL"],
        bins=[-1, 20, 50, np.inf],
        labels=["High", "Medium", "Low"],
    ).astype(str)
    out["recommended_action"] = out["predicted_RUL"].apply(
        lambda r: (
            "Inspect immediately"
            if r <= 20
            else "Schedule maintenance soon"
            if r <= 50
            else "Continue monitoring"
        )
    )
    return out
