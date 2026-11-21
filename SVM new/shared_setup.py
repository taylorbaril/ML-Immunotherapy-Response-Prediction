"""
Shared setup for the GSE78220 immunotherapy project.

Every model (logistic regression, SVM, random forest) imports this file so that
all three use the SAME samples, labels, features and cross-validation folds.

Usage in a notebook:
    from shared_setup import load_data, get_folds, evaluate_model
    X, y, groups, info = load_data()
    folds = get_folds(info)

Needs these files next to it:
    data/GSE78220_PatientFPKM.xlsx   (processed expression table from GEO)
    sample_info.csv                  (sample table from the inspection notebook)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (balanced_accuracy_score, roc_auc_score,
                             recall_score, confusion_matrix)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

EXPR_PATH = "data/GSE78220_PatientFPKM.xlsx"
INFO_PATH = "sample_info.csv"
FOLDS_PATH = "cv_folds.csv"

# --- Choices to confirm with Dr. Puthiyedth --------------------------------
# Responder = Complete or Partial Response, Non-responder = Progressive Disease
# (Hugo et al., 2016). Stable disease does not appear in GSE78220.
RESPONDER_LABELS = ["Complete Response", "Partial Response"]
NON_RESPONDER_LABELS = ["Progressive Disease"]

# Pt16 was biopsied on treatment, so it is left out of a pre-treatment study.
PRETREATMENT_ONLY = True

N_FOLDS = 5
RANDOM_STATE = 42
# ---------------------------------------------------------------------------


def load_data():
    """Return X (samples x genes, log2(FPKM + 1)), y (1 = responder), groups (patient id), info."""
    expr = pd.read_excel(EXPR_PATH).set_index("Gene")
    info = pd.read_csv(INFO_PATH)

    if PRETREATMENT_ONLY:
        info = info[info["biopsy_time"] == "pre-treatment"]
    info = info[info["response_original"].isin(RESPONDER_LABELS + NON_RESPONDER_LABELS)]
    info = info.reset_index(drop=True)

    X = np.log2(expr[info["expression_column"]].T + 1)   # log transform uses no labels, so it is safe here
    X.index = info["sample_title"]
    y = info["response_original"].isin(RESPONDER_LABELS).astype(int).to_numpy()
    groups = info["patient_id"].to_numpy()
    return X, y, groups, info


def make_folds(info):
    """Create the fold assignment once and save it, so every teammate uses the same splits."""
    y = info["response_original"].isin(RESPONDER_LABELS).astype(int)
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold = np.empty(len(info), dtype=int)
    for k, (_, test_idx) in enumerate(cv.split(info, y, groups=info["patient_id"])):
        fold[test_idx] = k + 1
    out = pd.DataFrame({"sample_title": info["sample_title"],
                        "patient_id": info["patient_id"],
                        "fold": fold})
    out.to_csv(FOLDS_PATH, index=False)
    return out


def get_folds(info):
    """Return a list of (train_idx, test_idx) pairs read from cv_folds.csv."""
    try:
        folds = pd.read_csv(FOLDS_PATH)
    except FileNotFoundError:
        folds = make_folds(info)
    fold_of = dict(zip(folds["sample_title"], folds["fold"]))
    fold = info["sample_title"].map(fold_of).to_numpy()
    if np.isnan(fold.astype(float)).any():
        raise ValueError("cv_folds.csv does not match the current samples; delete it and rerun.")
    return [(np.where(fold != k)[0], np.where(fold == k)[0]) for k in sorted(set(fold))]


N_REPEATS = 10
REPEATED_FOLDS_PATH = "cv_folds_repeated.csv"


def make_repeated_folds(info):
    """10 repeats of 5-fold CV (seeds 0-9), saved so every model uses the same 50 splits.

    Folds are made at the patient level with StratifiedKFold, then mapped back to samples.
    This keeps Pt27A and Pt27B together and guarantees both classes in every test fold.
    """
    y = info["response_original"].isin(RESPONDER_LABELS).astype(int)
    patients = info.assign(label=y).groupby("patient_id")["label"].first()
    rows = []
    for r in range(N_REPEATS):
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=r)
        fold_of_patient = {}
        for k, (_, test_idx) in enumerate(cv.split(patients.index, patients.values)):
            for p in patients.index[test_idx]:
                fold_of_patient[p] = k + 1
        fold = info["patient_id"].map(fold_of_patient).to_numpy()
        rows.append(pd.DataFrame({"repeat": r + 1,
                                  "sample_title": info["sample_title"],
                                  "patient_id": info["patient_id"],
                                  "fold": fold}))
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(REPEATED_FOLDS_PATH, index=False)
    return out


def get_repeated_folds(info):
    """Return {repeat: [(train_idx, test_idx), ...]} read from cv_folds_repeated.csv."""
    try:
        folds = pd.read_csv(REPEATED_FOLDS_PATH)
    except FileNotFoundError:
        folds = make_repeated_folds(info)
    splits = {}
    for r, grp in folds.groupby("repeat"):
        fold_of = dict(zip(grp["sample_title"], grp["fold"]))
        fold = info["sample_title"].map(fold_of).to_numpy()
        splits[r] = [(np.where(fold != k)[0], np.where(fold == k)[0]) for k in sorted(set(fold))]
    return splits


def evaluate_repeated(model, X, y, repeated_folds, name="model"):
    """Run evaluate_model for every repeat; returns 50 rows (10 repeats x 5 folds) and the summed confusion matrix."""
    all_rows, total_cm = [], np.zeros((2, 2), dtype=int)
    for r, folds in repeated_folds.items():
        rows, cm = evaluate_model(model, X, y, folds, name=name)
        rows.insert(1, "repeat", r)
        all_rows.append(rows)
        total_cm += cm
    return pd.concat(all_rows, ignore_index=True), total_cm


def evaluate_model(model, X, y, folds, name="model"):
    """Fit on each training fold, test on the held-out fold, return one row of metrics per fold."""
    rows, total_cm = [], np.zeros((2, 2), dtype=int)
    for k, (train_idx, test_idx) in enumerate(folds, start=1):
        model.fit(X.iloc[train_idx], y[train_idx])
        pred = model.predict(X.iloc[test_idx])
        if hasattr(model, "decision_function"):
            score = model.decision_function(X.iloc[test_idx])
        else:
            score = model.predict_proba(X.iloc[test_idx])[:, 1]
        cm = confusion_matrix(y[test_idx], pred, labels=[0, 1])
        total_cm += cm
        rows.append({
            "model": name,
            "fold": k,
            "n_test": len(test_idx),
            "balanced_accuracy": balanced_accuracy_score(y[test_idx], pred),
            "roc_auc": roc_auc_score(y[test_idx], score),
            "sensitivity": recall_score(y[test_idx], pred, pos_label=1, zero_division=0),
            "specificity": recall_score(y[test_idx], pred, pos_label=0, zero_division=0),
            "TN": cm[0, 0], "FP": cm[0, 1], "FN": cm[1, 0], "TP": cm[1, 1],
        })
    return pd.DataFrame(rows), total_cm
