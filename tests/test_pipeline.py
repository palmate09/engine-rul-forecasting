"""Basic correctness tests for the data and feature pipeline."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_cmapss
from src.evaluation import nasa_score, predictions_frame
from src.features import (
    add_engineered_features,
    add_rul_target,
    build_feature_frames,
    drop_constant_sensors,
)


class TestCmapssPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.train, cls.test, cls.rul = load_cmapss("FD001")

    def test_rul_target_is_capped_and_ends_at_zero(self) -> None:
        df = add_rul_target(self.train, max_rul=125)
        self.assertLessEqual(int(df["RUL"].max()), 125)
        last = df.sort_values(["unit", "cycle"]).groupby("unit").tail(1)
        self.assertTrue((last["RUL"] == 0).all())
        for _, group in df.groupby("unit"):
            values = group.sort_values("cycle")["RUL"].to_numpy()
            self.assertTrue(np.all(np.diff(values) <= 0))

    def test_constant_sensors_are_dropped(self) -> None:
        train, test, kept, dropped = drop_constant_sensors(self.train, self.test)
        self.assertIn("s1", dropped)
        self.assertNotIn("s1", train.columns)
        self.assertNotIn("s1", test.columns)
        self.assertTrue(kept)
        self.assertEqual(
            [c for c in train.columns if c.startswith("s")],
            kept,
        )

    def test_engineered_features_match_between_train_and_test(self) -> None:
        built = build_feature_frames(
            self.train, self.test, max_rul=125, windows=(5, 10)
        )
        self.assertTrue(built.feature_columns)
        missing = [c for c in built.feature_columns if c not in built.test.columns]
        self.assertEqual(missing, [])
        self.assertNotIn("unit", built.feature_columns)
        self.assertNotIn("cycle", built.feature_columns)
        self.assertNotIn("RUL", built.feature_columns)

    def test_rolling_features_do_not_use_future_values(self) -> None:
        df = add_engineered_features(self.train, ["s2"], windows=(5,))
        unit = int(df["unit"].iloc[0])
        d = df[df["unit"] == unit].sort_values("cycle").reset_index(drop=True)
        # The first row's rolling mean must equal its own value, not a future one.
        self.assertAlmostEqual(
            float(d.loc[0, "s2_mean_5"]), float(d.loc[0, "s2"]), places=6
        )

    def test_optional_cycle_feature_is_not_future_normalized(self) -> None:
        df = add_engineered_features(
            self.train, ["s2"], windows=(5,), include_cycle=True
        )
        self.assertIn("cycle_index", df.columns)
        d = df[df["unit"] == int(df["unit"].iloc[0])]
        self.assertTrue((d["cycle_index"] == d["cycle"]).all())
        # A normalized feature would end at 1.0; the raw index must not.
        self.assertGreater(float(d["cycle_index"].max()), 1.0)

    def test_nasa_score_penalises_late_predictions_more(self) -> None:
        true = np.array([50.0, 50.0])
        early = np.array([40.0, 50.0])
        late = np.array([60.0, 50.0])
        self.assertEqual(nasa_score(true, true), 0.0)
        self.assertGreater(nasa_score(true, late), nasa_score(true, early))

    def test_predictions_frame_shapes(self) -> None:
        frame = predictions_frame([1, 2], [10, 20], [12, 18])
        self.assertEqual(len(frame), 2)
        self.assertTrue(frame.loc[0, "late"])
        self.assertFalse(frame.loc[1, "late"])


if __name__ == "__main__":
    unittest.main()
