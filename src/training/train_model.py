"""
train_model.py

This script trains a baseline machine learning model and saves:
1. The trained model
2. Baseline feature statistics (used later for data drift detection)
3. Baseline performance metrics (used later for model monitoring)

This file represents the "ground truth" state of the model at training time.
"""

# =========================
# 1. IMPORT REQUIRED LIBRARIES
# =========================

import json                     # used to save metrics and statistics as JSON files
import os                       # used to create directories if they do not exist

import numpy as np              # numerical operations
import pandas as pd             # tabular data handling

from sklearn.datasets import load_breast_cancer     # example dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score
)

import joblib                   # used to save and load trained ML models


# =========================
# 2. CREATE OUTPUT DIRECTORIES IF THEY DON'T EXIST
# =========================

# path where the trained model will be saved
MODEL_DIR = "models/baseline"

# path where baseline reports will be saved
REPORT_DIR = "reports/baseline"

# create directories safely (no error if they already exist)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# =========================
# 3. LOAD DATASET
# =========================

# load the breast cancer dataset from sklearn
dataset = load_breast_cancer()

# convert features to a pandas DataFrame
X = pd.DataFrame(dataset.data, columns=dataset.feature_names)

# target variable (0 or 1)
y = pd.Series(dataset.target, name="target")

# Explanation:
# X -> input features
# y -> labels (what the model tries to predict)


# =========================
# 4. TRAIN-TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,        # 20% data for testing
    random_state=42,      # ensures reproducibility
    stratify=y            # keeps class distribution balanced
)

# Why this matters:
# - Training set: used to learn patterns
# - Test set: simulates unseen production data


# =========================
# 5. TRAIN BASELINE MODEL
# =========================

# initialize logistic regression model
model = LogisticRegression(
    max_iter=1000,        # ensures convergence
    solver="lbfgs"        # stable solver for small/medium datasets
)

# train the model using training data
model.fit(X_train, y_train)


# =========================
# 6. MODEL EVALUATION
# =========================

# generate predictions on test data
y_pred = model.predict(X_test)

# generate probability predictions (needed for ROC-AUC)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# store metrics in a dictionary
performance_metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "roc_auc": roc_auc
}


# =========================
# 7. COMPUTE BASELINE FEATURE STATISTICS
# =========================

# these statistics will be used later to detect data drift

feature_statistics = {}

for column in X_train.columns:
    feature_statistics[column] = {
        "mean": float(X_train[column].mean()),
        "std": float(X_train[column].std()),
        "min": float(X_train[column].min()),
        "max": float(X_train[column].max())
    }

# Explanation:
# We store training-time feature distributions.
# Later, incoming production data will be compared against these.


# =========================
# 8. SAVE MODEL TO DISK
# =========================

model_path = os.path.join(MODEL_DIR, "model.joblib")

joblib.dump(model, model_path)

# joblib is preferred for sklearn models because it handles numpy arrays efficiently


# =========================
# 9. SAVE REPORTS TO DISK
# =========================

# save performance metrics
with open(os.path.join(REPORT_DIR, "performance_metrics.json"), "w") as f:
    json.dump(performance_metrics, f, indent=4)

# save feature statistics
with open(os.path.join(REPORT_DIR, "feature_stats.json"), "w") as f:
    json.dump(feature_statistics, f, indent=4)


# =========================
# 10. FINAL OUTPUT
# =========================

print("✅ Baseline model training complete.")
print("📦 Model saved to:", model_path)
print("📊 Performance metrics:", performance_metrics)
