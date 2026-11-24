"""Streamlit prototype for C-MAPSS remaining-useful-life estimation.

Run from the project root:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_cmapss
from src.inference import predict_from_history

st.set_page_config(
    page_title="C-MAPSS RUL Estimator",
    page_icon="🛠️",
    layout="wide",
)

MODEL_PATH = PROJECT_ROOT / "models" / "rul_model.joblib"
RESULTS_DIR = PROJECT_ROOT / "results"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
SAMPLE_PATH = PROJECT_ROOT / "data" / "sample_input.csv"


@st.cache_resource
def load_bundle(path: str, mtime: float) -> dict:
    return joblib.load(path)


@st.cache_data
def load_test_data(subset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, test, rul = load_cmapss(subset)
    return test, rul


@st.cache_data
def load_metrics(path: str, mtime: float) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def actual_rul_map(test: pd.DataFrame, rul: pd.DataFrame) -> dict[int, int]:
    return dict(zip(sorted(test["unit"].unique()), rul["RUL"].astype(int).tolist()))


def priority_from_rul(rul: float) -> str:
    if rul <= 20:
        return "High"
    if rul <= 50:
        return "Medium"
    return "Low"


def main() -> None:
    st.title("🛠️ Predictive Maintenance: Remaining Useful Life")
    st.caption(
        "C-MAPSS turbofan engine demo. Predicts remaining life from sensor history. "
        "Decision-support only; trained on simulated benchmark data."
    )

    if not MODEL_PATH.exists():
        st.error(
            "Model not found. Run `python -m src.train --subset FD001` from the "
            "project root, then refresh this page."
        )
        st.stop()

    bundle = load_bundle(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
    test, rul = load_test_data(bundle["subset"])
    rul_map = actual_rul_map(test, rul)

    st.sidebar.header("Model")
    st.sidebar.write(f"**Subset:** {bundle['subset']}")
    st.sidebar.write(f"**Selected model:** {bundle['best_model']}")
    st.sidebar.write(f"**Features:** {len(bundle['features'])}")
    st.sidebar.write(f"**RUL cap:** {bundle['max_rul']} cycles")
    st.sidebar.divider()
    st.sidebar.caption(
        "Dataset: NASA C-MAPSS. Results are for a simulated benchmark and "
        "must not be used as a real maintenance or airworthiness decision."
    )

    page = st.sidebar.radio(
        "View",
        ["Engine demo", "Batch predictions", "Upload sensor history", "Model & metrics"],
    )

    if page == "Engine demo":
        engine_demo(bundle, test, rul_map)
    elif page == "Batch predictions":
        batch_predictions(bundle, test, rul_map)
    elif page == "Upload sensor history":
        upload_page(bundle)
    else:
        model_metrics(bundle)


def engine_demo(bundle: dict, test: pd.DataFrame, rul_map: dict[int, int]) -> None:
    st.subheader("Single-engine prediction")
    units = sorted(test["unit"].unique().tolist())
    unit = st.selectbox("Select a test engine", units, index=0)
    history = test[test["unit"] == unit].sort_values("cycle")

    pred = predict_from_history(history, bundle).iloc[0]
    actual = rul_map.get(int(unit))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted RUL", f"{pred['predicted_RUL']:.0f} cycles")
    c2.metric("Priority", pred["priority"])
    c3.metric("Last observed cycle", f"{int(history['cycle'].max())}")
    if actual is not None:
        c4.metric("Actual RUL (evaluation only)", f"{actual} cycles")

    if actual is not None:
        error = float(pred["predicted_RUL"]) - actual
        if error > 0:
            st.warning(
                f"Late prediction: the model overestimates RUL by {error:.0f} cycles. "
                "This is the costly direction in maintenance planning."
            )
        else:
            st.info(f"Early/safe prediction by {abs(error):.0f} cycles.")

    st.write(f"**Recommended action:** {pred['recommended_action']}")
    buffer = max(2.0, 0.10 * float(pred["predicted_RUL"]))
    planning_rul = max(0.0, float(pred["predicted_RUL"]) - buffer)
    st.caption(
        f"Conservative planning window: about {planning_rul:.0f} cycles "
        f"(10% safety buffer; tune this with real maintenance costs)."
    )

    preferred = [
        s
        for s in ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"]
        if s in history.columns
    ]
    selected = st.multiselect(
        "Sensors to plot", preferred, default=preferred[:4]
    )
    if selected:
        st.line_chart(history.set_index("cycle")[selected])

    with st.expander("Raw sensor history"):
        st.dataframe(history, use_container_width=True)


def batch_predictions(
    bundle: dict, test: pd.DataFrame, rul_map: dict[int, int]
) -> None:
    st.subheader("Batch predictions for all test engines")
    with st.spinner("Scoring all test engines ..."):
        preds = predict_from_history(test, bundle)

    preds["actual_RUL"] = preds["unit"].map(rul_map)
    preds["error"] = preds["predicted_RUL"] - preds["actual_RUL"]
    preds["abs_error"] = preds["error"].abs()
    preds["late"] = preds["error"] > 0
    preds["planning_RUL"] = (
        preds["predicted_RUL"]
        - preds["predicted_RUL"].mul(0.10).clip(lower=2.0)
    ).clip(lower=0.0)
    preds = preds.sort_values("predicted_RUL", ascending=True).reset_index(drop=True)

    high = int((preds["priority"] == "High").sum())
    medium = int((preds["priority"] == "Medium").sum())
    low = int((preds["priority"] == "Low").sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("High priority", high)
    c2.metric("Medium priority", medium)
    c3.metric("Low priority", low)

    st.dataframe(
        preds[
            [
                "unit",
                "cycle",
                "predicted_RUL",
                "planning_RUL",
                "priority",
                "recommended_action",
                "actual_RUL",
                "error",
                "late",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download batch predictions as CSV",
        preds.to_csv(index=False).encode("utf-8"),
        file_name=f"cmapss_{bundle['subset']}_predictions.csv",
        mime="text/csv",
    )

    st.caption(
        "Actual RUL is shown only for evaluation on the public benchmark; "
        "a deployed system would not have this column."
    )


def upload_page(bundle: dict) -> None:
    st.subheader("Predict from your own sensor history")
    st.write(
        "Upload a CSV with one row per cycle. Required columns: "
        "`unit`, `cycle`, `op1`, `op2`, `op3`, and the sensor columns "
        "the model was trained on."
    )
    st.caption(
        "For best results, provide at least 50 cycles of history per engine. "
        "Shorter histories are scored with partial rolling windows."
    )

    if SAMPLE_PATH.exists():
        st.download_button(
            "Download a sample input CSV",
            SAMPLE_PATH.read_bytes(),
            file_name="sample_input.csv",
            mime="text/csv",
        )

    uploaded = st.file_uploader("Upload sensor history CSV", type=["csv"])
    if uploaded is None:
        with st.expander("Expected columns"):
            st.code(
                "unit, cycle, op1, op2, op3, "
                + ", ".join(bundle["sensor_columns"]),
                language="text",
            )
        return

    try:
        df = pd.read_csv(uploaded)
        preds = predict_from_history(df, bundle)
    except Exception as exc:
        st.error(f"Could not score this file: {exc}")
        return

    if len(preds) == 1:
        row = preds.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted RUL", f"{row['predicted_RUL']:.0f} cycles")
        c2.metric("Priority", row["priority"])
        c3.metric("Last cycle", int(row["cycle"]))
        st.write(f"**Recommended action:** {row['recommended_action']}")
        planning = max(0.0, float(row["predicted_RUL"]) - max(2.0, 0.10 * float(row["predicted_RUL"])))
        st.caption(f"Conservative planning window: about {planning:.0f} cycles.")
    else:
        st.dataframe(preds, use_container_width=True, hide_index=True)

    st.download_button(
        "Download predictions",
        preds.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv",
    )


def model_metrics(bundle: dict) -> None:
    st.subheader("Model and evaluation summary")

    metrics_path = RESULTS_DIR / "metrics.json"
    if metrics_path.exists():
        metrics = load_metrics(str(metrics_path), metrics_path.stat().st_mtime)
        test = metrics["test"]["capped"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test RMSE", f"{test['rmse']:.1f} cycles")
        c2.metric("Test MAE", f"{test['mae']:.1f} cycles")
        c3.metric("Test R²", f"{test['r2']:.3f}")
        c4.metric("NASA score", f"{test['nasa_mean']:.3f}")
        st.caption(
            f"Primary target: official test RUL capped at {bundle['max_rul']} cycles. "
            f"Late predictions: {test['late_count']} of {test['n']}."
        )
    else:
        st.info("Run training to generate `results/metrics.json`.")

    comparison_path = RESULTS_DIR / "model_comparison.csv"
    if comparison_path.exists():
        st.write("**Validation model comparison**")
        comparison = pd.read_csv(comparison_path)
        st.dataframe(comparison, use_container_width=True, hide_index=True)

    st.write("**Figures**")
    figure_specs = [
        ("model_comparison.png", "Model comparison"),
        ("test_predictions.png", "Predicted vs true RUL"),
        ("error_distribution.png", "Error distribution"),
        ("error_by_rul_bin.png", "Error by RUL range"),
        ("feature_importance.png", "Top features"),
        ("sensor_trends.png", "Sensor trends"),
    ]
    available = [(FIG_DIR / name, caption) for name, caption in figure_specs if (FIG_DIR / name).exists()]
    if not available:
        st.info("No figures found. Run training without `--no-plots`.")
        return

    for i in range(0, len(available), 2):
        cols = st.columns(2)
        for col, (path, caption) in zip(cols, available[i : i + 2]):
            with col:
                st.image(str(path), caption=caption, use_container_width=True)


if __name__ == "__main__":
    main()
