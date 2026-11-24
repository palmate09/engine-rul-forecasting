# Notebooks

This project uses a script-based workflow rather than a notebook, so the same
code path is used for training, testing, and the Streamlit prototype:

- `src/train.py` — end-to-end training, selection, test evaluation, and figures
- `src/features.py` — target construction and rolling features
- `reports/PROJECT_REPORT.md` — full methodology, results, and error analysis
- `scripts/ablation.py` — raw-sensor vs engineered-feature comparison
- `scripts/tune_ridge.py` — Ridge alpha sweep

The report and `results/` contain the executed outputs. A notebook is optional;
if you want one, import the modules from `src/` and load the saved artifacts in
`results/` rather than duplicating the pipeline.
