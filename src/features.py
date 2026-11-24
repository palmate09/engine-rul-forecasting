"""RUL target construction and time-series feature engineering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass
class FeatureBuildResult:
    """Container for the prepared train/test frames and feature metadata."""

    train: pd.DataFrame
    test: pd.DataFrame
    feature_columns: list[str]
    sensor_columns: list[str]
    dropped_sensors: list[str]


def add_rul_target(train: pd.DataFrame, max_rul: int = 125) -> pd.DataFrame:
    """Add a capped RUL target to a run-to-failure training frame.

    For an engine whose last observed cycle is ``T``, the RUL at cycle ``t``
    is ``T - t``. Capping at ``max_rul`` creates the standard piecewise-linear
    target used in C-MAPSS benchmarks.
    """
    if max_rul <= 0:
        raise ValueError("max_rul must be positive")

    df = train.copy()
    failure_cycle = df.groupby("unit")["cycle"].transform("max")
    df["RUL"] = (failure_cycle - df["cycle"]).clip(upper=max_rul).astype(int)
    return df


def drop_constant_sensors(
    train: pd.DataFrame,
    test: pd.DataFrame,
    tolerance: float = 1e-8,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Drop sensors that do not vary in the training data.

    The same columns are removed from test, which prevents train/test feature
    mismatch. A sensor that is constant in train but not test is still dropped:
    it carries no training signal and cannot be trusted at inference time.
    """
    sensors = [c for c in train.columns if c.startswith("s") and c[1:].isdigit()]
    if not sensors:
        raise ValueError("No sensor columns found.")

    stds = train[sensors].std()
    dropped = stds[stds < tolerance].index.tolist()
    kept = [c for c in sensors if c not in dropped]

    train_out = train.drop(columns=dropped)
    test_out = test.drop(columns=[c for c in dropped if c in test.columns])
    return train_out, test_out, kept, dropped


def add_engineered_features(
    df: pd.DataFrame,
    sensors: Sequence[str],
    windows: Sequence[int] = (10, 30, 50),
    include_cycle: bool = False,
) -> pd.DataFrame:
    """Add past-only rolling statistics and first differences.

    All rolling features end at the current cycle. No future values are used,
    which is essential for a valid RUL model.
    """
    base = df.sort_values(["unit", "cycle"]).copy()
    new_features: dict[str, pd.Series] = {}

    for sensor in sensors:
        new_features[f"{sensor}_diff"] = (
            base.groupby("unit", sort=False)[sensor].diff().fillna(0.0)
        )

    for window in windows:
        if window <= 0:
            raise ValueError("windows must contain positive integers")
        for sensor in sensors:
            grouped = base.groupby("unit", sort=False)[sensor]
            new_features[f"{sensor}_mean_{window}"] = grouped.transform(
                lambda x, w=window: x.rolling(w, min_periods=1).mean()
            )
            new_features[f"{sensor}_std_{window}"] = grouped.transform(
                lambda x, w=window: x.rolling(w, min_periods=1).std()
            )
            new_features[f"{sensor}_min_{window}"] = grouped.transform(
                lambda x, w=window: x.rolling(w, min_periods=1).min()
            )
            new_features[f"{sensor}_max_{window}"] = grouped.transform(
                lambda x, w=window: x.rolling(w, min_periods=1).max()
            )

    if include_cycle:
        # Use the raw cycle index, not a per-engine normalization: dividing by
        # each engine's maximum cycle would use future information and would
        # differ between full training trajectories and truncated test ones.
        new_features["cycle_index"] = base["cycle"].astype(float)

    return pd.concat(
        [base, pd.DataFrame(new_features, index=base.index)],
        axis=1,
    )


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return model feature columns, excluding identifiers and the target."""
    return [c for c in df.columns if c not in {"unit", "cycle", "RUL"}]


def build_feature_frames(
    train: pd.DataFrame,
    test: pd.DataFrame,
    max_rul: int = 125,
    windows: Sequence[int] = (10, 30, 50),
    include_cycle: bool = False,
) -> FeatureBuildResult:
    """Run the full feature pipeline for train and test frames."""
    train = add_rul_target(train, max_rul=max_rul)
    train, test, sensors, dropped = drop_constant_sensors(train, test)

    train_f = add_engineered_features(
        train, sensors, windows=windows, include_cycle=include_cycle
    )
    test_f = add_engineered_features(
        test, sensors, windows=windows, include_cycle=include_cycle
    )

    cols = feature_columns(train_f)
    missing = [c for c in cols if c not in test_f.columns]
    if missing:
        raise ValueError(f"Test frame is missing engineered columns: {missing}")

    return FeatureBuildResult(
        train=train_f,
        test=test_f,
        feature_columns=cols,
        sensor_columns=list(sensors),
        dropped_sensors=list(dropped),
    )
