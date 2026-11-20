# Random forest workflow for GSE78220 (COMP 3710 group project, Model 3)
# Repeated cross-validation (10 x 5 folds). Saves results in the SAME CSV layout as the
# logistic regression file: Model,Repeat,Fold,Balanced Accuracy,ROC-AUC,Sensitivity,Specificity,TN,FP,FN,TP
# Each "# %%" block can be pasted into its own Jupyter / Colab cell.

# %% 0. Imports and settings
import json
import os

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

# Folder holding GSE78220_PatientFPKM.xlsx and sample_info.csv (from the Step 3 notebook).
DATA_DIR = os.environ.get(
    "DATA_DIR", "/Users/dteen/Desktop/All Folders/School/Applied AI/Group Project/Data"
)
EXPR_FILE = "GSE78220_PatientFPKM.xlsx"
INFO_FILE = "sample_info.csv"
SPLITS_FILE = "cv_splits_repeated.json"      # SHARE this file so all three models use identical splits
RESULTS_FILE = "random_forest_results.csv"   # main results, same layout as the logistic regression CSV
SUMMARY_FILE = "rf_final_results.csv"        # one-row-per-model summary

RANDOM_STATE = 42
N_SPLITS = 5       # folds per repeat
N_REPEATS = 10     # the whole 5-fold cross-validation is repeated this many times (new shuffle each time)

# --- Decisions to CONFIRM with the supervisor and teammates before modelling ---------
RESPONDER_LABELS = ["Complete Response", "Partial Response"]   # -> 1 (positive class)
NON_RESPONDER_LABELS = ["Progressive Disease"]                 # -> 0
EXCLUDE_TIMING = ["OnTx"]          # drop the on-treatment sample (Pt16.OnTx)?
DROP_SAMPLES = []                  # e.g. ["Pt27B.baseline"] to keep only ONE of Pt27A/Pt27B
USE_FEATURE_SELECTION = False      # data-based gene filter, done INSIDE each fold
K_GENES = 1000                     # only used if USE_FEATURE_SELECTION is True
# -------------------------------------------------------------------------------------

# Fixed starting settings (tuning comes later, after the workflow works)
RF_PARAMS = dict(n_estimators=200, max_depth=3, min_samples_leaf=2, random_state=RANDOM_STATE)

MODEL_NAME = "Random Forest"          # model names follow the logistic regression CSV style
BASELINE_NAME = "Majority Benchmark"


# %% 1. Load the expression table and the sample information table
expr = pd.read_excel(os.path.join(DATA_DIR, EXPR_FILE))
expr = expr.set_index(expr.columns[0])          # genes in rows, samples in columns
info = pd.read_csv(os.path.join(DATA_DIR, INFO_FILE))
print("Expression table (genes x samples):", expr.shape)
print("Sample info rows:", len(info))

# Guard: sample_info.csv must have exactly one row per expression column.
if (len(info) != expr.shape[1] or info["sample_id"].isna().any()
        or info["sample_id"].duplicated().any()):
    raise ValueError(
        f"sample_info.csv has {len(info)} rows but the expression table has {expr.shape[1]} samples "
        "(or it has blank/duplicate sample_id values). Re-run build_sample_info.py to rebuild it."
    )


# %% 2. Pick the eligible samples and build the label y
# Patient ID without the A/B suffix, so Pt27A and Pt27B share one patient (group).
info["patient_base"] = info["patient_id"].str.replace(r"(?<=\d)[A-Za-z]$", "", regex=True)

unknown = set(DROP_SAMPLES) - set(info["sample_id"])
assert not unknown, f"DROP_SAMPLES names not found in sample_info.csv: {unknown}"

is_dropped = info["timing_from_name"].isin(EXCLUDE_TIMING) | info["sample_id"].isin(DROP_SAMPLES)
print("Left out:", info.loc[is_dropped, "sample_id"].tolist() or "none")
eligible = info[~is_dropped].copy()

label_map = {**{lab: 1 for lab in RESPONDER_LABELS}, **{lab: 0 for lab in NON_RESPONDER_LABELS}}
eligible["y"] = eligible["response_original"].map(label_map)
bad = eligible[eligible["y"].isna()]
if len(bad):                                     # report, never silently drop or guess
    raise ValueError(f"Unmapped or missing response labels:\n{bad[['sample_id', 'response_original']]}")
eligible["y"] = eligible["y"].astype(int)
eligible = eligible.reset_index(drop=True)

missing_cols = set(eligible["sample_id"]) - set(expr.columns)
assert not missing_cols, f"Sample IDs not found in expression table: {missing_cols}"

print(f"\nEligible samples: {len(eligible)} from {eligible['patient_base'].nunique()} patients")
print(eligible["response_original"].value_counts())
print("Class balance (1 = responder):", eligible["y"].value_counts().to_dict())


# %% 3. Build the feature matrix X (rows = samples, columns = genes)
sample_ids = eligible["sample_id"].tolist()
gene_names = expr.index.to_numpy()
X = expr[sample_ids].T.to_numpy(dtype=float)     # transpose: samples as rows
y = eligible["y"].to_numpy()
groups = eligible["patient_base"].to_numpy()

print("\nMissing values in X:", int(np.isnan(X).sum()))   # report only, do not fill
X = np.log2(X + 1)   # fixed, not data-driven, so it is safe to apply before CV.
# (Trees ignore monotonic transforms, but use the SAME X for all three models.)
print("X shape:", X.shape)


# %% 4. Repeated cross-validation splits: patient-grouped, stratified, SAVED so everyone shares them
def make_or_load_splits():
    """Return a list of (repeat, fold, train_rows, test_rows)."""
    path = os.path.join(DATA_DIR, SPLITS_FILE)
    pos = {sid: i for i, sid in enumerate(sample_ids)}
    if os.path.exists(path):
        with open(path) as f:
            saved = json.load(f)
        for r in {s["repeat"] for s in saved}:      # each repeat must cover exactly our samples
            covered = {x for s in saved if s["repeat"] == r for x in s["test"]}
            if covered != set(sample_ids):
                raise ValueError(f"{SPLITS_FILE} was made for a different set of samples. Delete it "
                                 "and re-run, then share the new file so all models use the same folds.")
        print("Loaded shared splits from", path)
        return [(s["repeat"], s["fold"], [pos[x] for x in s["train"]], [pos[x] for x in s["test"]])
                for s in saved]
    splits, saved = [], []
    for r in range(1, N_REPEATS + 1):               # a new shuffle for every repeat
        cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE + r)
        for k, (tr, te) in enumerate(cv.split(X, y, groups), 1):
            splits.append((r, k, tr.tolist(), te.tolist()))
            saved.append({"repeat": r, "fold": k,
                          "train": [sample_ids[i] for i in tr], "test": [sample_ids[i] for i in te]})
    with open(path, "w") as f:
        json.dump(saved, f, indent=1)
    print("Created and saved splits to", path)
    return splits


splits = make_or_load_splits()
for r, k, tr, te in splits:
    shared = set(groups[tr]) & set(groups[te])
    assert not shared, f"Repeat {r} fold {k}: patient(s) {shared} appear in both train and test"
sizes = [len(te) for _, _, _, te in splits]
print(f"{len(splits)} folds ({N_REPEATS} repeats x {N_SPLITS}), test size {min(sizes)}-{max(sizes)} samples, "
      f"folds with only one class: {sum(len(set(y[te])) < 2 for _, _, _, te in splits)}")
for r, k, tr, te in splits[:N_SPLITS]:             # show repeat 1 as an example
    print(f"Repeat {r} fold {k}: train={len(tr)}, test={len(te)}, test responders={int(y[te].sum())}")


# %% 5. Define the models
def make_random_forest():
    steps = []
    if USE_FEATURE_SELECTION:       # fitted on the training fold only, because it is in the Pipeline
        steps.append(("select", SelectKBest(f_classif, k=K_GENES)))
    # No StandardScaler: trees split on thresholds, so feature scale does not matter.
    steps.append(("rf", RandomForestClassifier(**RF_PARAMS)))
    return Pipeline(steps)


def make_baseline():
    return DummyClassifier(strategy="most_frequent")   # always predicts the training majority class


# %% 6. Evaluate one model across all repeats and folds
def evaluate(model, name):
    rows = []
    for r, k, tr, te in splits:
        m = clone(model).fit(X[tr], y[tr])             # fit(): learn from training fold only
        pred = m.predict(X[te])                        # predict(): class labels (0/1)
        proba = m.predict_proba(X[te])[:, 1]           # probability of responder, used for ROC-AUC
        tn, fp, fn, tp = confusion_matrix(y[te], pred, labels=[0, 1]).ravel()
        rows.append({
            "Model": name, "Repeat": r, "Fold": k,
            "Balanced Accuracy": balanced_accuracy_score(y[te], pred),
            "ROC-AUC": roc_auc_score(y[te], proba) if len(set(y[te])) == 2 else np.nan,
            "Sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,   # responders found
            "Specificity": tn / (tn + fp) if (tn + fp) else np.nan,   # non-responders found
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        })
    return pd.DataFrame(rows)


# %% 7. Run the random forest and the benchmark, then save the CSV
rf_results = evaluate(make_random_forest(), MODEL_NAME)
base_results = evaluate(make_baseline(), BASELINE_NAME)
results = pd.concat([rf_results, base_results], ignore_index=True)

# Same row order as the logistic regression CSV: for each repeat and fold, model row then benchmark row.
results["_order"] = results["Model"].map({MODEL_NAME: 0, BASELINE_NAME: 1})
results = results.sort_values(["Repeat", "Fold", "_order"]).drop(columns="_order").reset_index(drop=True)

results_path = os.path.join(DATA_DIR, RESULTS_FILE)
results.to_csv(results_path, index=False)
print(f"\nSaved results to {results_path} ({len(results)} rows + header)")

metrics = ["Balanced Accuracy", "ROC-AUC", "Sensitivity", "Specificity"]
pd.set_option("display.width", 200)
print("\nFirst rows of the results CSV (full table is in the file):")
print(results.head(10).round(3).to_string(index=False))

print(f"\nMean and standard deviation across all {N_REPEATS * N_SPLITS} folds:")
print(results.groupby("Model")[metrics].agg(["mean", "std"]).round(3).T)

print("\nBalanced accuracy averaged within each repeat:")
print(results.groupby(["Repeat", "Model"])["Balanced Accuracy"].mean().unstack().round(3))

cm_totals = results.groupby("Model")[["TN", "FP", "FN", "TP"]].sum()
for name in [MODEL_NAME, BASELINE_NAME]:
    tn, fp, fn, tp = cm_totals.loc[name]
    print(f"\nConfusion matrix summed over all folds and repeats: {name}")
    print(pd.DataFrame([[tn, fp], [fn, tp]], index=["actual non-responder", "actual responder"],
                       columns=["pred non-responder", "pred responder"]))
print(f"(Each sample is tested once per repeat, so these counts are {N_REPEATS}x the number of samples.)")


# %% 7b. Final summary CSV (one row per model: averages, spread and total confusion counts)
summary = results.groupby("Model")[metrics].agg(["mean", "std"])
summary.columns = [f"{m.lower().replace(' ', '_').replace('-', '_')}_{s}" for m, s in summary.columns]
summary = summary.reset_index()
summary = summary.merge(cm_totals.add_suffix("_total").reset_index(), on="Model")
summary["settings"] = summary["Model"].map({
    MODEL_NAME: str(RF_PARAMS) + (f"; top {K_GENES} genes" if USE_FEATURE_SELECTION else ""),
    BASELINE_NAME: "strategy=most_frequent",
})
summary["n_samples"] = len(y)
summary["n_patients"] = len(set(groups))
summary["n_splits"] = N_SPLITS
summary["n_repeats"] = N_REPEATS
num_cols = [c for c in summary.columns if c.endswith(("_mean", "_std"))]
summary[num_cols] = summary[num_cols].round(4)

summary_path = os.path.join(DATA_DIR, SUMMARY_FILE)
summary.to_csv(summary_path, index=False)
print("\nSaved summary to", summary_path)
print(summary.to_string(index=False))


# %% 8. Optional: which genes does the forest rely on? (interpretation only, not a score)
final = make_random_forest().fit(X, y)
rf = final.named_steps["rf"]
if USE_FEATURE_SELECTION:
    kept = gene_names[final.named_steps["select"].get_support()]
else:
    kept = gene_names
top = pd.Series(rf.feature_importances_, index=kept).sort_values(ascending=False).head(20)
print("\nTop 20 genes by importance (fit on all eligible samples):")
print(top.round(4))
# Caution: with ~27 samples these rankings are unstable. Do not treat them as findings.
