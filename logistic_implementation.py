import pandas as pandas
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, confusion_matrix
#patient reponse labels
# 1 = responder 
# 0 = non-responder 
responses = {
    "Pt1": 0, "Pt2": 1, "Pt4": 1, "Pt5": 1,
    "Pt6": 1, "Pt7": 0, "Pt8": 1, "Pt9": 1,
    "Pt10": 0, "Pt12": 0, "Pt13": 1, "Pt14": 0,
    "Pt15": 1, "Pt19": 1, "Pt20": 0, "Pt22": 0,
    "Pt23": 0, "Pt25": 0, "Pt27A": 1, "Pt27B": 1,
    "Pt28": 1, "Pt29": 0, "Pt31": 0, "Pt32": 0,
    "Pt35": 1, "Pt37": 1, "Pt38": 1
}
#load dataset
# Load gene expression data.
data_file = pandas.read_excel("GSE78220_PatientFPKM.xlsx")

# Make genes the rows.
data_file = data_file.set_index("Gene") # sets gene column as index column

# Select patient samples and transpose the data.
# Rows = patients, columns = genes.

X = data_file[[patient + ".baseline" for patient in responses]].T
X.index = X.index.str.replace(".baseline", "", regex=False)
# Convert values to numbers and apply log transformation.
X = X.apply(pandas.to_numeric, errors="coerce") # pandas sometimes reads numbers as texxt
X = np.log1p(X.clip(lower=0)) # makes sure no values are negative, and log reduces large values

#prepare labels

#create the response labels.
response_labels = pandas.Series(responses)

# combine Pt27A and Pt27B because they are the same patient.
# search for pt27a / b and remove last letter
X.index = [
    name[:-1] if name in ["Pt27A", "Pt27B"] else name
    for name in X.index
]

response_labels.index = [
    name[:-1] if name in ["Pt27A", "Pt27B"] else name
    for name in response_labels.index
]

#average the two Pt27 gene expression samples.
X = X.groupby(X.index).mean()

#keep one response label per patient.
# groupby ensures pt27 only gets 1 response, and first is just selecting the first response given.
response_labels = response_labels.groupby(response_labels.index).first()

# Make sure X and y have matching patient order, dont want patients having mislabelled reponse
X = X.loc[response_labels.index]


#create models
# Logistic Regression with L2 regularization.
# The pipeline handles missing values and scaling.

model = make_pipeline(
    SimpleImputer(strategy="median"), # handles missing values and replaces them with median
    StandardScaler(), # makes it so values are closer together, big numbers can dominate
    LogisticRegression(
        penalty="l2", # penalizes large model coefficents
        C=0.1, # strong regularization, smaller c = strong, large = weak
        class_weight="balanced", # adjusts importance of eahc response class
        max_iter=5000 # max iterations to find coefficents
    )
)
#benchmark that always predicts the most common class / response
benchmark = DummyClassifier(strategy="most_frequent")
#cross validation , Divide patients into 5 groups , Repeat the process 10 times.
cross_validation = RepeatedStratifiedKFold(
    n_splits=5, # split patients into groups of 5, 4 groups used for training, 1 for testing
    n_repeats=10, # repeat entire process 10 times using different patient splits
    random_state=42 # just a random seed, doesn't really matter.
)

results = []
fold_assignments = []
patient_predictions = []

# Go through each training/testing split. enumerate adds counter to each split
# 
for fold, (train, test) in enumerate(cross_validation.split(X, response_labels)):

    X_train = X.iloc[train] #iloc uses row pos. to retrieve data and labels
    X_test = X.iloc[test]

    y_train = response_labels.iloc[train] # contains responses for training
    y_test = response_labels.iloc[test] # contains responses for testing

    # Save the patient assignments .
    repeat = fold // 5 + 1
    fold_number = fold % 5 + 1
 # record which patients are assigned to training and testing in each fold
    for patient in X.index:
        fold_assignments.append({
            "repeat": repeat,
            "fold": fold_number,
            "patient": patient,
            "set": "test" if patient in X_test.index else "train"
        })

    #evaluate Logistic Regression and the benchmark
    # using exactly the same training/testing patients.

    for name, classifier in [
        ("Logistic Regression", model),
        ("Majority Benchmark", benchmark)
    ]:
        #train model
        classifier.fit(X_train, y_train) # fit tells model to learn from data
        # LR, learns coefficents that help estimate probability of patient

        #predict patient response
        predictions = classifier.predict(X_test)
        #give it the testing patients
        if name == "Logistic Regression":
            for patient, actual, predicted in zip(X_test.index, y_test, predictions):
                patient_predictions.append({
                "Patient": patient,
                "Actual Response": "Responder" if actual == 1 else "Non-responder",
                "Predicted Response": "Responder" if predicted == 1 else "Non-responder",
                "Correct": actual == predicted})

        #get predicted probabilities, need this for ROC-AUC predict () gives either 0,1. predict_proba gives percent
        # ROC-AUC = Receiver Operating Characteristic - Area Under the Curve
        
        probabilities = classifier.predict_proba(X_test)[:, 1] # predict probability of class 1

        # Calculate confusion matrix. how many predictions were wrong or right,
        # tn = true negative, fp = false positive, fn = false negative, tp = true positive
        tn, fp, fn, tp = confusion_matrix( y_test, predictions, labels=[0, 1]).ravel()
        # .ravel seperates the values into seperate variables
        
        # Calculate performance metrics.
        # sens = how many actual responders did the model identify
        # spec = how manny actual non-responders did it identify
        sensitivity = tp / (tp + fn) 
        specificity = tn / (tn + fp)

        results.append({
            "Model": name,
            "Repeat": repeat,
            "Fold": fold_number,
            "Balanced Accuracy": balanced_accuracy_score(
                y_test, predictions
            ),
            "ROC-AUC": roc_auc_score(y_test, probabilities),
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp
        })
#display result
patient_predictions = pandas.DataFrame(patient_predictions)

print("\npatient predications")
print(patient_predictions.to_string(index=False))
results = pandas.DataFrame(results)

# Calculate average performance and standard deviation.
summary = results.groupby("Model")[ ["Balanced Accuracy", "ROC-AUC", "Sensitivity", "Specificity"]].agg(["mean", "std"])
print("\nModel Performance")
print(summary.round(3))
# save results
results.to_csv("logistic_regression_results.csv", index=False)
summary.to_csv("model_summary.csv")

pandas.DataFrame(fold_assignments).to_csv("shared_cv_splits.csv", index=False)
print("\nResults saved successfully")
