"""Report figures for the C-MAPSS RUL project."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid")
FIG_DPI = 150


def _save(fig: plt.Figure, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_rul_distribution(train: pd.DataFrame, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(train["RUL"], bins=30, kde=True, ax=ax, color="#4C72B0")
    ax.set_title("Training RUL distribution (capped)")
    ax.set_xlabel("Remaining useful life (cycles)")
    ax.set_ylabel("Count")
    _save(fig, path)


def plot_sensor_trends(
    train: pd.DataFrame,
    sensors: Sequence[str],
    path: str | Path,
    n_engines: int = 3,
) -> None:
    sensors = list(sensors)[:4]
    if not sensors:
        return

    engines = sorted(train["unit"].unique())[:n_engines]
    ncols = 2
    nrows = int(np.ceil(len(sensors) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.2 * nrows), squeeze=False)
    axes = axes.flatten()

    for ax, sensor in zip(axes, sensors):
        for unit in engines:
            d = train[train["unit"] == unit].sort_values("cycle")
            ax.plot(d["cycle"], d[sensor], label=f"Engine {unit}", linewidth=1.2)
        ax.set_title(sensor)
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Sensor value")

    for ax in axes[len(sensors) :]:
        ax.set_visible(False)
    axes[0].legend(fontsize=8)
    _save(fig, path)


def plot_correlation_heatmap(
    train: pd.DataFrame,
    sensors: Sequence[str],
    path: str | Path,
) -> None:
    sensors = list(sensors)
    if len(sensors) < 2:
        return
    corr = train[sensors].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.2,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Sensor correlation matrix")
    _save(fig, path)


def plot_model_comparison(comparison: pd.DataFrame, path: str | Path) -> None:
    df = comparison.sort_values("rmse", ascending=True)
    models = df[df["model"] != "Mean baseline"]
    baseline = df[df["model"] == "Mean baseline"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    sns.barplot(data=df, x="rmse", y="model", ax=axes[0], color="#4C72B0")
    axes[0].set_title("Validation RMSE (lower is better)")
    axes[0].set_xlabel("RMSE (cycles)")
    axes[0].set_ylabel("")

    sns.barplot(data=models, x="nasa_mean", y="model", ax=axes[1], color="#DD8452")
    axes[1].set_title("Validation NASA score (lower is better)")
    axes[1].set_xlabel("NASA mean score")
    axes[1].set_ylabel("")
    if not baseline.empty:
        axes[1].text(
            0.98,
            0.03,
            f"Mean baseline NASA = {baseline.iloc[0]['nasa_mean']:.0f} (off scale)",
            transform=axes[1].transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color="#555555",
        )

    _save(fig, path)


def plot_predictions(pred_df: pd.DataFrame, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    colors = np.where(pred_df["late"], "#C44E52", "#4C72B0")
    ax.scatter(
        pred_df["y_true"],
        pred_df["y_pred"],
        c=colors,
        alpha=0.75,
        edgecolor="white",
        linewidth=0.4,
    )
    upper = max(float(pred_df["y_true"].max()), float(pred_df["y_pred"].max())) * 1.05
    ax.plot([0, upper], [0, upper], "k--", linewidth=1, label="Perfect prediction")
    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ax.set_xlabel("True RUL (cycles)")
    ax.set_ylabel("Predicted RUL (cycles)")
    ax.set_title("Test predictions (red = late prediction)")
    ax.legend(loc="upper left")
    _save(fig, path)


def plot_error_distribution(pred_df: pd.DataFrame, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(pred_df["error"], bins=25, kde=True, ax=ax, color="#55A868")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Test error distribution (prediction - truth)")
    ax.set_xlabel("Error (cycles); positive = late prediction")
    ax.set_ylabel("Count")
    _save(fig, path)


def plot_error_by_bin(bin_df: pd.DataFrame, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=bin_df, x="rul_bin", y="mae", ax=ax, color="#8172B2")
    ax.set_title("MAE by true-RUL range")
    ax.set_xlabel("True RUL range (cycles)")
    ax.set_ylabel("MAE (cycles)")
    _save(fig, path)


def plot_ablation(ablation: pd.DataFrame, path: str | Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.barplot(
        data=ablation,
        x="test_rmse",
        y="model",
        hue="feature_set",
        ax=axes[0],
    )
    axes[0].set_title("Test RMSE by feature set")
    axes[0].set_xlabel("RMSE (cycles)")
    axes[0].set_ylabel("")

    sns.barplot(
        data=ablation,
        x="test_nasa",
        y="model",
        hue="feature_set",
        ax=axes[1],
    )
    axes[1].set_title("Test NASA score by feature set")
    axes[1].set_xlabel("NASA mean score")
    axes[1].set_ylabel("")
    axes[1].legend(title="")
    axes[0].legend(title="")
    _save(fig, path)


def plot_feature_importance(
    importance: pd.DataFrame,
    path: str | Path,
    top_n: int = 20,
) -> None:
    if importance.empty:
        return
    df = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(df) + 2))
    sns.barplot(data=df, x="importance", y="feature", ax=ax, color="#4C72B0")
    ax.set_title(f"Top {len(df)} features")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    _save(fig, path)
