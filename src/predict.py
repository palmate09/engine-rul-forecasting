"""Predict RUL from a CSV of sensor history.

Usage:
    python -m src.predict --input data/sample_input.csv --output predictions.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import PROJECT_ROOT
from src.inference import predict_from_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "predictions.csv")
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models" / "rul_model.joblib",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(
            f"Model not found: {args.model}\nRun `python -m src.train` first."
        )

    bundle = joblib.load(args.model)
    df = pd.read_csv(args.input)
    predictions = predict_from_history(df, bundle)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)

    print(predictions.to_string(index=False))
    print(f"\nSaved predictions -> {args.output}")


if __name__ == "__main__":
    main()
