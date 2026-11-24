"""Create a header-bearing sample CSV for the Streamlit upload demo."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import ALL_COLUMNS, load_cmapss


def main() -> None:
    _, test, _ = load_cmapss("FD001")
    # Use the longest test trajectory so the sample has enough rolling history.
    unit = int(test.groupby("unit")["cycle"].max().idxmax())
    sample = test[test["unit"] == unit].sort_values("cycle").tail(60).copy()
    out = PROJECT_ROOT / "data" / "sample_input.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out, index=False, columns=ALL_COLUMNS)
    print(f"Wrote {out} ({len(sample)} rows)")


if __name__ == "__main__":
    main()
