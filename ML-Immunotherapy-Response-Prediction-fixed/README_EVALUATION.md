# Unified immunotherapy response model evaluation

Run from this directory:
```bash
python -m pip install pandas numpy scikit-learn openpyxl
python build_sample_info.py   # if you need to rebuild metadata
python run_all_models.py
```
`run_all_models.py` is the unified evaluation runner. It filters pretreatment samples and supported response labels, groups patient variants such as Pt27A/Pt27B, creates shared patient-grouped stratified 5-fold CV splits (10 repeats), checks for group leakage, and fits preprocessing within each model pipeline. All three classifiers and the training-fold majority benchmark use the same splits.

Outputs are written under `analysis_output/`: shared split assignments, per-fold metrics, repeated patient predictions, mean/SD summary, repeated-fold confusion totals, error-frequency analysis, and fold-level feature importance plus summary.

**Caution:** The feature-importance export is exploratory. Feature rankings from small folds may be unstable; do not interpret as biological/clinical evidence. This runner treats Pt27A/B as samples within one patient group for splitting, not averaged into one row. Confirm the scientific exclusion/label choices with the project supervisor.
