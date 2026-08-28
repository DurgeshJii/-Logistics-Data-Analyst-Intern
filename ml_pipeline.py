"""
Week 4 Task: Build and Validate a Basic Machine Learning Model
Dataset: Wisconsin Breast Cancer Diagnostic dataset (built into scikit-learn,
publicly available, no download required, well suited to a binary
classification demonstration).

Pipeline:
1. Load & inspect data
2. Clean / preprocess (missing values, feature scaling, train/test split)
3. Train two candidate models (Logistic Regression, Decision Tree) and
   justify the final choice
4. Evaluate with accuracy, precision, recall, F1, ROC-AUC
5. Produce visualizations: confusion matrix heatmap, ROC curve,
   feature-importance / coefficient plot, train-vs-test accuracy bar
   (overfitting check)
6. Save all metrics + images to disk for the report
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score, classification_report
)

RANDOM_STATE = 42
OUT = "/home/claude"

# ----------------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------------
data = load_breast_cancer(as_frame=True)
df = data.frame.copy()
df.rename(columns={"target": "diagnosis"}, inplace=True)
# 0 = malignant, 1 = benign in sklearn's encoding
print("Shape:", df.shape)
print("Missing values total:", df.isna().sum().sum())
print("Class balance:\n", df["diagnosis"].value_counts())

# ----------------------------------------------------------------------
# 2. DATA PREPARATION / PREPROCESSING
# ----------------------------------------------------------------------
# a) Check & handle missing values (none in this dataset, but we show the
#    defensive step any real pipeline should include)
df = df.dropna(axis=0, how="any")

# b) Remove exact duplicate rows if any
n_before = len(df)
df = df.drop_duplicates()
n_after = len(df)
print(f"Duplicates removed: {n_before - n_after}")

# c) Separate features / target
X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

# d) Train/test split (stratified to preserve class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# e) Feature scaling (fit ONLY on training data to avoid data leakage)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Train size:", X_train.shape, " Test size:", X_test.shape)

# ----------------------------------------------------------------------
# 3. MODEL SELECTION & TRAINING
# ----------------------------------------------------------------------
# Candidate A: Logistic Regression — chosen as the primary model because:
#   - the problem is binary classification on continuous, roughly linearly
#     separable features (after scaling)
#   - it is interpretable (coefficients show feature influence)
#   - it is fast to train and a strong, well-understood baseline
log_reg = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
log_reg.fit(X_train_scaled, y_train)

# Candidate B: Decision Tree — trained for comparison, to illustrate the
# trade-off between a linear model and a non-linear, more flexible one
# that is prone to overfitting if left unconstrained.
tree = DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE)
tree.fit(X_train, y_train)  # trees don't need scaling

# 5-fold cross-validation on the training set (robustness check)
cv_scores_lr = cross_val_score(log_reg, X_train_scaled, y_train, cv=5)
cv_scores_tree = cross_val_score(tree, X_train, y_train, cv=5)

# ----------------------------------------------------------------------
# 4. EVALUATION
# ----------------------------------------------------------------------
def evaluate(model, X_tr, y_tr, X_te, y_te, proba_X_te=None):
    train_pred = model.predict(X_tr)
    test_pred = model.predict(X_te)
    proba = model.predict_proba(proba_X_te if proba_X_te is not None else X_te)[:, 1]
    return {
        "train_accuracy": accuracy_score(y_tr, train_pred),
        "test_accuracy": accuracy_score(y_te, test_pred),
        "precision": precision_score(y_te, test_pred),
        "recall": recall_score(y_te, test_pred),
        "f1": f1_score(y_te, test_pred),
        "roc_auc": roc_auc_score(y_te, proba),
        "test_pred": test_pred,
        "proba": proba,
    }

res_lr = evaluate(log_reg, X_train_scaled, y_train, X_test_scaled, y_test)
res_tree = evaluate(tree, X_train, y_train, X_test, y_test)

metrics_summary = {
    "logistic_regression": {
        "train_accuracy": round(res_lr["train_accuracy"], 4),
        "test_accuracy": round(res_lr["test_accuracy"], 4),
        "precision": round(res_lr["precision"], 4),
        "recall": round(res_lr["recall"], 4),
        "f1": round(res_lr["f1"], 4),
        "roc_auc": round(res_lr["roc_auc"], 4),
        "cv_mean_accuracy": round(cv_scores_lr.mean(), 4),
        "cv_std": round(cv_scores_lr.std(), 4),
    },
    "decision_tree": {
        "train_accuracy": round(res_tree["train_accuracy"], 4),
        "test_accuracy": round(res_tree["test_accuracy"], 4),
        "precision": round(res_tree["precision"], 4),
        "recall": round(res_tree["recall"], 4),
        "f1": round(res_tree["f1"], 4),
        "roc_auc": round(res_tree["roc_auc"], 4),
        "cv_mean_accuracy": round(cv_scores_tree.mean(), 4),
        "cv_std": round(cv_scores_tree.std(), 4),
    },
}
print(json.dumps(metrics_summary, indent=2))

with open(f"{OUT}/metrics.json", "w") as f:
    json.dump(metrics_summary, f, indent=2)

print("\nClassification report (Logistic Regression):")
print(classification_report(y_test, res_lr["test_pred"], target_names=data.target_names))

# ----------------------------------------------------------------------
# 5. VISUALIZATIONS
# ----------------------------------------------------------------------
plt.rcParams.update({"font.size": 11})

# --- Viz 1: Confusion matrix (Logistic Regression) ---
cm = confusion_matrix(y_test, res_lr["test_pred"])
fig, ax = plt.subplots(figsize=(5, 4.5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(data.target_names); ax.set_yticklabels(data.target_names)
ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
ax.set_title("Confusion Matrix — Logistic Regression")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig(f"{OUT}/viz_confusion_matrix.png", dpi=150)
plt.close(fig)

# --- Viz 2: ROC curves for both models ---
fpr_lr, tpr_lr, _ = roc_curve(y_test, res_lr["proba"])
fpr_tree, tpr_tree, _ = roc_curve(y_test, res_tree["proba"])
fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC={res_lr['roc_auc']:.3f})", linewidth=2)
ax.plot(fpr_tree, tpr_tree, label=f"Decision Tree (AUC={res_tree['roc_auc']:.3f})", linewidth=2)
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve Comparison")
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig(f"{OUT}/viz_roc_curve.png", dpi=150)
plt.close(fig)

# --- Viz 3: Train vs Test accuracy (overfitting check) ---
fig, ax = plt.subplots(figsize=(5.5, 4.5))
labels = ["Logistic\nRegression", "Decision\nTree"]
train_accs = [res_lr["train_accuracy"], res_tree["train_accuracy"]]
test_accs = [res_lr["test_accuracy"], res_tree["test_accuracy"]]
x = np.arange(len(labels)); width = 0.35
ax.bar(x - width/2, train_accs, width, label="Train accuracy")
ax.bar(x + width/2, test_accs, width, label="Test accuracy")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylim(0.8, 1.02)
ax.set_ylabel("Accuracy")
ax.set_title("Train vs Test Accuracy (Overfitting Check)")
ax.legend()
for i, (tr, te) in enumerate(zip(train_accs, test_accs)):
    ax.text(i - width/2, tr + 0.005, f"{tr:.3f}", ha="center", fontsize=9)
    ax.text(i + width/2, te + 0.005, f"{te:.3f}", ha="center", fontsize=9)
fig.tight_layout()
fig.savefig(f"{OUT}/viz_train_test_accuracy.png", dpi=150)
plt.close(fig)

# --- Viz 4: Top logistic regression coefficients (feature influence) ---
coefs = pd.Series(log_reg.coef_[0], index=X.columns).sort_values(key=np.abs, ascending=False).head(10)
fig, ax = plt.subplots(figsize=(6.5, 5))
colors = ["#d62728" if v < 0 else "#1f77b4" for v in coefs.values[::-1]]
ax.barh(coefs.index[::-1], coefs.values[::-1], color=colors)
ax.set_xlabel("Standardized Coefficient")
ax.set_title("Top 10 Feature Coefficients — Logistic Regression")
fig.tight_layout()
fig.savefig(f"{OUT}/viz_feature_coefficients.png", dpi=150)
plt.close(fig)

print("\nAll visualizations saved.")
print("Files:", "viz_confusion_matrix.png, viz_roc_curve.png, viz_train_test_accuracy.png, viz_feature_coefficients.png")
