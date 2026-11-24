"""Reproduce the Ridge alpha sweep used for model selection.

Usage:
    python scripts/tune_ridge.py --subset FD001
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import DEFAULT_DATA_DIR, SUBSETS, load_cmapss
from src.evaluation import regression_metrics
from src.features import build_feature_frames
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
        "--alphas",
        type=float,
        nargs="+",
        default=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "ridge_alpha_sweep.csv",
    )
    args = parser.parse_args()

    train_raw, test_raw, _ = load_cmapss(args.subset, args.data_dir)
    built = build_feature_frames(
        train_raw,
        test_raw,
        max_rul=args.max_rul,
        windows=tuple(args.windows),
    )
    train_f = built.train

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

    X_tr, y_tr = tr[built.feature_columns], tr["RUL"]
    X_va, y_va = va[built.feature_columns], va["RUL"]

    rows: list[dict[str, float]] = []
    for alpha in args.alphas:
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=float(alpha)),
        )
        model.fit(X_tr, y_tr)
        pred = np.clip(model.predict(X_va), 0.0, args.max_rul)
        metrics = regression_metrics(y_va, pred)
        rows.append(
            {
                "alpha": float(alpha),
                "val_rmse": metrics["rmse"],
                "val_mae": metrics["mae"],
                "val_r2": metrics["r2"],
                "val_nasa": metrics["nasa_mean"],
                "late_count": metrics["late_count"],
            }
        )
        print(
            f"alpha={alpha:8g}  RMSE={metrics['rmse']:7.3f}  "
            f"MAE={metrics['mae']:7.3f}  NASA={metrics['nasa_mean']:8.3f}"
        )

    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
