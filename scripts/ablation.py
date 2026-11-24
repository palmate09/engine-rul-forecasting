"""Compare raw sensor features with engineered rolling-window features.

Usage:
    python scripts/ablation.py --subset FD001
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import plots
from src.data import DEFAULT_DATA_DIR, SUBSETS, load_cmapss
from src.evaluation import regression_metrics
from src.features import build_feature_frames
from src.modeling import make_models
from src.train import make_truncated_validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="FD001", choices=SUBSETS)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--max-rul", type=int, default=125)
    parser.add_argument("--windows", type=int, nargs="+", default=[10, 30, 50])
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--val-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "results" / "ablation.csv"
    )
    args = parser.parse_args()

    train_raw, test_raw, rul = load_cmapss(args.subset, args.data_dir)
    built = build_feature_frames(
        train_raw,
        test_raw,
        max_rul=args.max_rul,
        windows=tuple(args.windows),
    )
    train_f, test_f = built.train, built.test

    feature_sets = {
        "raw sensors": ["op1", "op2", "op3"] + built.sensor_columns,
        "engineered": built.feature_columns,
    }

    units = np.array(sorted(train_f["unit"].unique()))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(units)
    n_val = max(1, int(round(len(units) * args.val_fraction)))
    val_units = units[:n_val]
    train_units = units[n_val:]

    tr = train_f[train_f["unit"].isin(train_units)].copy()
    va = make_truncated_validation(
        train_f, val_units, rng, repeats=args.val_repeats
    )

    test_last = (
        test_f.sort_values(["unit", "cycle"])
        .groupby("unit")
        .tail(1)
        .sort_values("unit")
        .reset_index(drop=True)
    )
    y_test = np.minimum(
        rul["RUL"].to_numpy(dtype=float), args.max_rul
    )

    model_zoo = make_models(random_state=args.seed)
    rows: list[dict[str, object]] = []

    for feature_set_name, columns in feature_sets.items():
        for model_name in ("Ridge", "HistGradientBoosting"):
            model = model_zoo[model_name]
            model.fit(tr[columns], tr["RUL"])
            val_pred = np.clip(model.predict(va[columns]), 0, args.max_rul)
            val_metrics = regression_metrics(va["RUL"], val_pred)

            final_model = clone(model)
            final_model.fit(train_f[columns], train_f["RUL"])
            test_pred = np.clip(
                final_model.predict(test_last[columns]), 0, args.max_rul
            )
            test_metrics = regression_metrics(y_test, test_pred)

            rows.append(
                {
                    "feature_set": feature_set_name,
                    "model": model_name,
                    "n_features": len(columns),
                    "val_rmse": val_metrics["rmse"],
                    "val_mae": val_metrics["mae"],
                    "val_nasa": val_metrics["nasa_mean"],
                    "test_rmse": test_metrics["rmse"],
                    "test_mae": test_metrics["mae"],
                    "test_nasa": test_metrics["nasa_mean"],
                }
            )
            print(
                f"{feature_set_name:12s} {model_name:22s} "
                f"val_rmse={val_metrics['rmse']:6.2f} "
                f"test_rmse={test_metrics['rmse']:6.2f} "
                f"test_nasa={test_metrics['nasa_mean']:6.2f}"
            )

    out = pd.DataFrame(rows).sort_values(["model", "feature_set"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    plots.plot_ablation(out, PROJECT_ROOT / "reports" / "figures" / "ablation.png")
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
