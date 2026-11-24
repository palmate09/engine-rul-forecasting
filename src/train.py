"""Train, select, evaluate, and save a C-MAPSS RUL model.

Usage:
    python -m src.train --subset FD001
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import plots
from src.data import DEFAULT_DATA_DIR, PROJECT_ROOT, SUBSETS, load_cmapss
from src.evaluation import (
    error_by_rul_bin,
    predictions_frame,
    regression_metrics,
    save_json,
)
from src.features import build_feature_frames
from src.modeling import make_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="FD001", choices=SUBSETS)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--max-rul", type=int, default=125)
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=[10, 30, 50],
        help="Rolling window sizes in cycles.",
    )
    parser.add_argument(
        "--include-cycle",
        action="store_true",
        help="Add the raw cycle index as a feature (off by default to avoid shortcut learning).",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--val-repeats",
        type=int,
        default=5,
        help="Number of random truncation points per validation engine.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument(
        "--figure-dir", type=Path, default=PROJECT_ROOT / "reports" / "figures"
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--permutation-repeats",
        type=int,
        default=5,
        help="Repeats for permutation importance on validation.",
    )
    return parser.parse_args()


def make_truncated_validation(
    df: pd.DataFrame,
    units: np.ndarray,
    rng: np.random.Generator,
    repeats: int = 5,
) -> pd.DataFrame:
    """Truncate each validation engine at several random cycles.

    Taking the last cycle of a full run-to-failure engine would always give
    RUL=0, which is not representative of the official test set. Multiple
    cutoffs per engine reduce the variance of model selection.
    """
    if repeats < 1:
        raise ValueError("val repeats must be at least 1")

    rows: list[pd.Series] = []
    for unit in units:
        d = df[df["unit"] == unit].sort_values("cycle")
        failure = int(d["cycle"].max())
        low = max(1, int(0.25 * failure))
        high = max(low + 1, failure)
        cutoffs: set[int] = set()
        attempts = 0
        while len(cutoffs) < repeats and attempts < repeats * 20:
            cutoffs.add(int(rng.integers(low, high)))
            attempts += 1
        for cutoff in sorted(cutoffs):
            rows.append(d[d["cycle"] == cutoff].iloc[0])
    return pd.DataFrame(rows).sort_values(["unit", "cycle"]).reset_index(drop=True)


def select_best_model(comparison: pd.DataFrame) -> str:
    """Select by average rank of validation RMSE and NASA score."""
    candidates = comparison[comparison["model"] != "Mean baseline"].copy()
    if candidates.empty:
        return comparison.iloc[0]["model"]

    candidates["rank_rmse"] = candidates["rmse"].rank(method="min")
    candidates["rank_nasa"] = candidates["nasa_mean"].rank(method="min")
    candidates["rank_avg"] = (candidates["rank_rmse"] + candidates["rank_nasa"]) / 2.0
    candidates = candidates.sort_values(["rank_avg", "rmse"])
    return str(candidates.iloc[0]["model"])


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.subset} from {args.data_dir} ...")
    train_raw, test_raw, rul = load_cmapss(args.subset, args.data_dir)
    print(f"  train: {train_raw.shape}, test: {test_raw.shape}, rul: {rul.shape}")

    print("Building features ...")
    built = build_feature_frames(
        train_raw,
        test_raw,
        max_rul=args.max_rul,
        windows=tuple(args.windows),
        include_cycle=args.include_cycle,
    )
    train_f, test_f = built.train, built.test
    feature_cols = built.feature_columns
    print(
        f"  features: {len(feature_cols)} | sensors kept: {len(built.sensor_columns)} "
        f"| dropped: {built.dropped_sensors}"
    )

    # Engine-level split. Test engines are never used for training or selection.
    units = np.array(sorted(train_f["unit"].unique()))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(units)
    n_val = max(1, int(round(len(units) * args.val_fraction)))
    val_units = units[:n_val]
    train_units = units[n_val:]
    if len(train_units) == 0:
        raise ValueError("No training engines remain; reduce --val-fraction.")

    tr = train_f[train_f["unit"].isin(train_units)].copy()
    va_last = make_truncated_validation(
        train_f, val_units, rng, repeats=args.val_repeats
    )

    X_tr, y_tr = tr[feature_cols], tr["RUL"]
    X_va, y_va = va_last[feature_cols], va_last["RUL"]
    print(f"  train engines: {len(train_units)} | validation engines: {len(val_units)}")
    print(f"  train rows: {len(tr)} | validation rows: {len(va_last)}")

    models = make_models(random_state=args.seed)
    comparison_rows: list[dict[str, object]] = []
    fitted: dict[str, object] = {}

    print("\nTraining models ...")
    for name, model in models.items():
        started = time.perf_counter()
        model.fit(X_tr, y_tr)
        fit_time = time.perf_counter() - started
        pred = np.clip(model.predict(X_va), 0.0, args.max_rul)
        metrics = regression_metrics(y_va, pred)
        comparison_rows.append(
            {
                "model": name,
                "fit_time_s": fit_time,
                **metrics,
            }
        )
        fitted[name] = model
        print(
            f"  {name:22s} RMSE={metrics['rmse']:7.3f}  "
            f"MAE={metrics['mae']:7.3f}  NASA={metrics['nasa_mean']:8.3f}  "
            f"late={metrics['late_count']:3d}/{metrics['n']}"
        )

    comparison = pd.DataFrame(comparison_rows)
    best_name = select_best_model(comparison)
    comparison["selected"] = comparison["model"] == best_name
    comparison = comparison.sort_values(
        ["selected", "rmse"], ascending=[False, True]
    ).reset_index(drop=True)
    comparison.to_csv(args.output_dir / "model_comparison.csv", index=False)
    print(f"\nSelected model: {best_name}")

    # Validation predictions from the train-only model (valid for analysis).
    best_val_model = fitted[best_name]
    val_pred = np.clip(best_val_model.predict(X_va), 0.0, args.max_rul)
    val_pred_df = predictions_frame(va_last["unit"].values, y_va.values, val_pred)
    val_pred_df.to_csv(args.output_dir / "validation_predictions.csv", index=False)

    # Refit the selected model on all non-test engines for the final artifact.
    print("Refitting selected model on all non-test engines ...")
    final_model = clone(best_val_model)
    final_model.fit(train_f[feature_cols], train_f["RUL"])

    test_last = (
        test_f.sort_values(["unit", "cycle"])
        .groupby("unit")
        .tail(1)
        .sort_values("unit")
        .reset_index(drop=True)
    )
    if len(test_last) != len(rul):
        raise ValueError(
            f"Test engines ({len(test_last)}) do not match RUL rows ({len(rul)})."
        )

    y_test_raw = rul["RUL"].to_numpy(dtype=float)
    y_test_capped = np.minimum(y_test_raw, args.max_rul)
    pred_test = np.clip(final_model.predict(test_last[feature_cols]), 0.0, args.max_rul)

    test_metrics_raw = regression_metrics(y_test_raw, pred_test)
    test_metrics_capped = regression_metrics(y_test_capped, pred_test)
    test_pred_df = predictions_frame(test_last["unit"].values, y_test_raw, pred_test)
    test_pred_df["y_true_capped"] = y_test_capped
    test_pred_df["error_capped"] = test_pred_df["y_pred"] - test_pred_df["y_true_capped"]
    test_pred_df.to_csv(args.output_dir / "test_predictions.csv", index=False)

    bins = error_by_rul_bin(test_pred_df)
    bins.to_csv(args.output_dir / "error_by_rul_bin.csv", index=False)

    print("\nFinal test results (primary target = capped at "
          f"{args.max_rul}):")
    for key in ("rmse", "mae", "r2", "nasa_mean", "late_count", "late_rate"):
        value = test_metrics_capped[key]
        print(f"  {key:12s}: {value:.4f}" if isinstance(value, float) else f"  {key:12s}: {value}")
    print("Final test results (raw official RUL target):")
    for key in ("rmse", "mae", "r2", "nasa_mean"):
        print(f"  {key:12s}: {test_metrics_raw[key]:.4f}")

    # ------------------------- interpretation -------------------------
    importance_path = args.output_dir / "feature_importance.csv"
    perm_path = args.output_dir / "feature_importance_permutation.csv"

    rf_name = "Random Forest"
    if rf_name in fitted:
        rf = fitted[rf_name].named_steps["randomforestregressor"]
        rf_importance = pd.DataFrame(
            {"feature": feature_cols, "importance": rf.feature_importances_}
        ).sort_values("importance", ascending=False)
        rf_importance.to_csv(importance_path, index=False)

    if args.permutation_repeats > 0:
        try:
            perm = permutation_importance(
                best_val_model,
                X_va,
                y_va,
                n_repeats=args.permutation_repeats,
                random_state=args.seed,
                n_jobs=-1,
                scoring="neg_root_mean_squared_error",
            )
            perm_df = pd.DataFrame(
                {
                    "feature": feature_cols,
                    "importance": perm.importances_mean,
                    "importance_std": perm.importances_std,
                }
            ).sort_values("importance", ascending=False)
            perm_df.to_csv(perm_path, index=False)
        except Exception as exc:  # pragma: no cover - optional analysis
            print(f"  permutation importance skipped: {exc}")
            perm_df = pd.DataFrame()

    # ----------------------------- plots ------------------------------
    if not args.no_plots:
        print("Saving figures ...")
        plots.plot_rul_distribution(
            train_f, args.figure_dir / "rul_distribution.png"
        )
        plot_sensors = [
            s
            for s in ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"]
            if s in train_raw.columns
        ][:4]
        plots.plot_sensor_trends(
            train_raw, plot_sensors, args.figure_dir / "sensor_trends.png"
        )
        plots.plot_correlation_heatmap(
            train_raw, built.sensor_columns, args.figure_dir / "sensor_correlation.png"
        )
        plots.plot_model_comparison(
            comparison, args.figure_dir / "model_comparison.png"
        )
        plots.plot_predictions(
            test_pred_df, args.figure_dir / "test_predictions.png"
        )
        plots.plot_error_distribution(
            test_pred_df, args.figure_dir / "error_distribution.png"
        )
        plots.plot_error_by_bin(bins, args.figure_dir / "error_by_rul_bin.png")

        if perm_path.exists():
            perm_plot = pd.read_csv(perm_path)
            plots.plot_feature_importance(
                perm_plot.rename(columns={"importance": "importance"}),
                args.figure_dir / "feature_importance.png",
            )
        elif importance_path.exists():
            plots.plot_feature_importance(
                pd.read_csv(importance_path),
                args.figure_dir / "feature_importance.png",
            )

    # --------------------------- save artifacts ------------------------
    metrics_payload = {
        "subset": args.subset,
        "best_model": best_name,
        "selection_rule": "average rank of validation RMSE and NASA mean score",
        "config": {
            "max_rul": args.max_rul,
            "windows": list(args.windows),
            "include_cycle": args.include_cycle,
            "val_fraction": args.val_fraction,
            "val_repeats": args.val_repeats,
            "seed": args.seed,
            "n_features": len(feature_cols),
            "dropped_sensors": built.dropped_sensors,
        },
        "data": {
            "n_train_engines": int(train_raw["unit"].nunique()),
            "n_test_engines": int(test_raw["unit"].nunique()),
            "n_train_rows": int(len(train_raw)),
            "n_test_rows": int(len(test_raw)),
            "n_train_engines_used_for_selection": int(len(train_units)),
            "n_validation_engines": int(len(val_units)),
            "n_validation_rows": int(len(va_last)),
        },
        "validation": comparison.to_dict(orient="records"),
        "test": {
            "primary_target": f"capped_at_{args.max_rul}",
            "capped": test_metrics_capped,
            "raw": test_metrics_raw,
        },
    }
    save_json(metrics_payload, args.output_dir / "metrics.json")

    bundle = {
        "model": final_model,
        "features": feature_cols,
        "sensor_columns": built.sensor_columns,
        "dropped_sensors": built.dropped_sensors,
        "max_rul": args.max_rul,
        "windows": list(args.windows),
        "include_cycle": args.include_cycle,
        "subset": args.subset,
        "best_model": best_name,
        "metrics": metrics_payload,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    model_path = args.model_dir / "rul_model.joblib"
    joblib.dump(bundle, model_path)
    print(f"\nSaved model -> {model_path}")
    print(f"Saved results -> {args.output_dir}")
    print(f"Saved figures -> {args.figure_dir}")


if __name__ == "__main__":
    main()
