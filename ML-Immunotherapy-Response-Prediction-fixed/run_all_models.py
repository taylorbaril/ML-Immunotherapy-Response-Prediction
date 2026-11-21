#!/usr/bin/env python3
"""Unified, patient-grouped repeated-CV comparison of LR, RF, and linear SVM."""
import os, json
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, confusion_matrix, recall_score

DATA_DIR=os.environ.get("DATA_DIR", os.path.dirname(os.path.abspath(__file__)))
EXPR=os.path.join(DATA_DIR,"GSE78220_PatientFPKM.xlsx")
INFO=os.path.join(DATA_DIR,"sample_info.csv")
OUT=os.path.join(DATA_DIR,"analysis_output")
N_SPLITS=5; N_REPEATS=10; SEED=42
RESP=["Complete Response","Partial Response"]; NONRESP=["Progressive Disease"]

os.makedirs(OUT,exist_ok=True)
expr=pd.read_excel(EXPR).set_index("Gene")
info=pd.read_csv(INFO)
# Retain only pre-treatment and clearly mapped response classes.
info=info[info["biopsy_time"].astype(str).str.lower().str.contains("pre")].copy()
info=info[info["response_original"].isin(RESP+NONRESP)].copy()
info["patient_group"]=info["patient_id"].astype(str).str.replace(r"(?<=\d)[A-Za-z]$","",regex=True)
info["y"]=info["response_original"].isin(RESP).astype(int)
# Do not duplicate patient-level response across multiple baseline samples? Grouping in CV keeps them together.
missing=set(info["expression_column"])-set(expr.columns)
if missing: raise ValueError(f"Expression columns missing: {missing}")
X=np.log2(expr[info["expression_column"]].T.astype(float)+1)
X.index=info["sample_title"].astype(str).values
X.columns=expr.index.astype(str)
y=info["y"].to_numpy(); groups=info["patient_group"].to_numpy()
if pd.Series(groups).nunique()<N_SPLITS: raise ValueError("Too few groups for CV.")
# All repeats' splits are generated and persisted once for re-use.
split_rows=[]; splitsets=[]
for rep in range(1,N_REPEATS+1):
    cv=StratifiedGroupKFold(n_splits=N_SPLITS,shuffle=True,random_state=SEED+rep)
    rep_splits=list(cv.split(X,y,groups))
    for fold,(tr,te) in enumerate(rep_splits,1):
        if set(groups[tr])&set(groups[te]): raise AssertionError("Patient leakage across split")
        if len(set(y[te]))<2: raise ValueError(f"Repeat {rep}, fold {fold} test set has one class")
        splitsets.append((rep,fold,tr,te))
        for ix in range(len(info)):
            split_rows.append({"repeat":rep,"fold":fold,"sample":info.iloc[ix]["sample_title"],
                              "patient_group":groups[ix],"set":"test" if ix in set(te) else "train"})
pd.DataFrame(split_rows).to_csv(os.path.join(OUT,"shared_cv_splits.csv"),index=False)

models={
 "Logistic Regression":Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler()),
  ("model",LogisticRegression(C=0.1,class_weight="balanced",max_iter=5000,random_state=SEED))]),
 "Random Forest":Pipeline([("impute",SimpleImputer(strategy="median")),
  ("model",RandomForestClassifier(n_estimators=200,max_depth=3,min_samples_leaf=2,
    class_weight="balanced",random_state=SEED))]),
 "Linear SVM":Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler()),
  ("model",SVC(kernel="linear",C=1.0,class_weight="balanced"))]),
 "Majority Benchmark":Pipeline([("impute",SimpleImputer(strategy="median")),("model",DummyClassifier(strategy="most_frequent"))])
}
results=[]; predictions=[]; importance=[]
for name,model in models.items():
 for rep,fold,tr,te in splitsets:
  m=Pipeline(model.steps) if isinstance(model,Pipeline) else model
  import sklearn.base
  m=sklearn.base.clone(model)
  m.fit(X.iloc[tr],y[tr]); pred=m.predict(X.iloc[te])
  estimator=m.named_steps["model"]
  if hasattr(estimator,"predict_proba"): score=m.predict_proba(X.iloc[te])[:,1]
  elif hasattr(estimator,"decision_function"): score=m.decision_function(X.iloc[te])
  else: score=pred.astype(float)
  tn,fp,fn,tp=confusion_matrix(y[te],pred,labels=[0,1]).ravel()
  results.append({"Model":name,"Repeat":rep,"Fold":fold,"Balanced Accuracy":balanced_accuracy_score(y[te],pred),
   "ROC-AUC":roc_auc_score(y[te],score) if len(set(y[te]))==2 else np.nan,
   "Sensitivity":tp/(tp+fn) if tp+fn else np.nan,"Specificity":tn/(tn+fp) if tn+fp else np.nan,
   "TN":tn,"FP":fp,"FN":fn,"TP":tp})
  for ix,a,p,s in zip(te,y[te],pred,score):
   predictions.append({"Model":name,"Repeat":rep,"Fold":fold,"Sample":info.iloc[ix]["sample_title"],
    "Patient group":groups[ix],"Actual":int(a),"Predicted":int(p),"Score":float(s),"Correct":bool(a==p)})
  if name!="Majority Benchmark":
   feat=X.columns
   if "model" in m.named_steps:
    est=m.named_steps["model"]
    if hasattr(est,"feature_importances_"):
     vals=est.feature_importances_; method="RF importance"
    elif hasattr(est,"coef_"):
     vals=np.abs(est.coef_.ravel()); method="absolute coefficient"
    else: vals=None; method=""
    if vals is not None:
     for g,v in zip(feat,vals): importance.append({"Model":name,"Repeat":rep,"Fold":fold,"Gene":g,"Importance":float(v),"Method":method})
res=pd.DataFrame(results); pred_df=pd.DataFrame(predictions); imp=pd.DataFrame(importance)
res.to_csv(os.path.join(OUT,"fold_metrics.csv"),index=False)
pred_df.to_csv(os.path.join(OUT,"patient_predictions_all_folds.csv"),index=False)
metrics=["Balanced Accuracy","ROC-AUC","Sensitivity","Specificity"]
summary=res.groupby("Model")[metrics].agg(["mean","std"]).round(4)
summary.to_csv(os.path.join(OUT,"model_summary_mean_sd.csv"))
res.groupby("Model")[["TN","FP","FN","TP"]].sum().to_csv(os.path.join(OUT,"confusion_totals_repeated.csv"))
# misclassification frequency by patient/model
wrong=pred_df[pred_df["Correct"]==False].groupby(["Model","Patient group","Sample"]).size().rename("Misclassified folds").reset_index()
wrong.to_csv(os.path.join(OUT,"misclassification_frequency.csv"),index=False)
if not imp.empty:
 imp.to_csv(os.path.join(OUT,"fold_feature_importance.csv"),index=False)
 stable=imp.groupby(["Model","Gene"])["Importance"].agg(["mean","std","count"]).reset_index()
 stable.sort_values(["Model","mean"],ascending=[True,False]).to_csv(os.path.join(OUT,"feature_stability_summary.csv"),index=False)
# descriptive audit details
with open(os.path.join(OUT,"dataset_audit.txt"),"w") as f:
 f.write(f"Samples: {len(info)}\\nPatients/group IDs: {pd.Series(groups).nunique()}\\nResponders: {int(y.sum())}\\nNonresponders: {int((y==0).sum())}\\n")
 f.write("Excluded rows: on-treatment samples and rows without Complete/Partial/Progressive response label.\\n")
 f.write("Patient-grouped stratified 5-fold CV, 10 repeats; splits shared across models.\\n")
 f.write("All data-driven transforms (imputation, scaling) are fit inside each fold pipeline.\\n")
print(summary)
print("Outputs written to:",OUT)
