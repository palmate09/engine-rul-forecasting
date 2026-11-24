"""Data loading utilities for the NASA C-MAPSS turbofan dataset.

The dataset is distributed as space-separated text files with 26 columns:
unit number, cycle, three operational settings, and 21 sensor measurements.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "CMAPSS"

SUBSETS = ("FD001", "FD002", "FD003", "FD004")

BASE_COLUMNS = ["unit", "cycle", "op1", "op2", "op3"]
SENSOR_COLUMNS = [f"s{i}" for i in range(1, 22)]
ALL_COLUMNS = BASE_COLUMNS + SENSOR_COLUMNS


def _paths_for_subset(data_dir: Path, subset: str) -> dict[str, Path]:
    return {
        "train": data_dir / f"train_{subset}.txt",
        "test": data_dir / f"test_{subset}.txt",
        "rul": data_dir / f"RUL_{subset}.txt",
    }


def load_cmapss(
    subset: str = "FD001",
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the train, test, and ground-truth RUL files for one subset.

    Returns
    -------
    train : DataFrame
        Run-to-failure trajectories with columns unit, cycle, op1..op3, s1..s21.
    test : DataFrame
        Truncated trajectories with the same columns.
    rul : DataFrame
        One column, ``RUL``, with one true remaining-life value per test unit.
    """
    subset = subset.upper().strip()
    if subset not in SUBSETS:
        raise ValueError(f"subset must be one of {SUBSETS}, got {subset!r}")

    data_dir = Path(data_dir)
    paths = _paths_for_subset(data_dir, subset)
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing C-MAPSS files:\n  "
            + "\n  ".join(missing)
            + "\nRun `python scripts/download_data.py` from the project root."
        )

    train = pd.read_csv(paths["train"], sep=r"\s+", header=None, names=ALL_COLUMNS)
    test = pd.read_csv(paths["test"], sep=r"\s+", header=None, names=ALL_COLUMNS)
    rul = pd.read_csv(paths["rul"], sep=r"\s+", header=None, names=["RUL"])

    if train.empty or test.empty or rul.empty:
        raise ValueError(f"One or more {subset} files are empty.")

    for name, frame in (("train", train), ("test", test)):
        if frame["unit"].nunique() == 0:
            raise ValueError(f"{name} has no units.")

    return train, test, rul
