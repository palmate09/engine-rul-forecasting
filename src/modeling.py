"""Model zoo for tabular RUL regression."""

from __future__ import annotations

from typing import Dict

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


def make_models(random_state: int = 42, n_jobs: int = -1) -> Dict[str, Pipeline]:
    """Return sensible baseline and ensemble models.

    All models are pipelines so imputation is fitted only on training data.
    XGBoost is included when the optional dependency is installed.
    """
    models: Dict[str, Pipeline] = {
        "Mean baseline": make_pipeline(
            SimpleImputer(strategy="median"),
            DummyRegressor(strategy="mean"),
        ),
        "Ridge": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=1.0),
        ),
        "Random Forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=random_state,
                n_jobs=n_jobs,
            ),
        ),
        "HistGradientBoosting": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=300,
                max_leaf_nodes=31,
                l2_regularization=0.1,
                random_state=random_state,
            ),
        ),
    }

    try:
        from xgboost import XGBRegressor

        models["XGBoost"] = make_pipeline(
            SimpleImputer(strategy="median"),
            XGBRegressor(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                objective="reg:squarederror",
                tree_method="hist",
                random_state=random_state,
                n_jobs=n_jobs,
            ),
        )
    except ImportError:
        # XGBoost is optional; the project remains runnable without it.
        pass

    return models
