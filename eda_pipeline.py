"""
Week 5 support script: generates the additional exploratory-data-analysis
visualizations that represent the earlier-week (data exploration / cleaning)
stages of the project, to be integrated into the final capstone report
alongside the Week 4 model-performance visualizations.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

OUT = "/home/claude"
data = load_breast_cancer(as_frame=True)
df = data.frame.copy()
df.rename(columns={"target": "diagnosis"}, inplace=True)
df["diagnosis_label"] = df["diagnosis"].map({0: "Malignant", 1: "Benign"})

plt.rcParams.update({"font.size": 11})

# --- EDA Viz 1: Class distribution ---
fig, ax = plt.subplots(figsize=(5.5, 4.5))
color_map = {"Malignant": "#c0392b", "Benign": "#2471a3"}
counts = df["diagnosis_label"].value_counts().reindex(["Malignant", "Benign"])
colors = ["#c0392b", "#2471a3"]
bars = ax.bar(counts.index, counts.values, color=[color_map[c] for c in counts.index])
for b, v in zip(bars, counts.values):
    ax.text(b.get_x() + b.get_width()/2, v + 5, str(v), ha="center", fontsize=11)
ax.set_ylabel("Number of samples")
ax.set_title("Class Distribution — Diagnosis Outcomes")
ax.set_ylim(0, max(counts.values) * 1.15)
fig.tight_layout()
fig.savefig(f"{OUT}/eda_class_distribution.png", dpi=150)
plt.close(fig)

# --- EDA Viz 2: Correlation heatmap of top 12 features ---
top_feats = ["mean radius", "mean texture", "mean perimeter", "mean area",
             "mean smoothness", "mean compactness", "mean concavity",
             "mean concave points", "mean symmetry", "worst radius",
             "worst perimeter", "worst area"]
corr = df[top_feats].corr()
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(top_feats))); ax.set_yticks(range(len(top_feats)))
ax.set_xticklabels(top_feats, rotation=90, fontsize=8)
ax.set_yticklabels(top_feats, fontsize=8)
ax.set_title("Feature Correlation Matrix (Selected Measurements)")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig(f"{OUT}/eda_correlation_heatmap.png", dpi=150)
plt.close(fig)

# --- EDA Viz 3: Feature distribution by class (mean radius, mean concavity) ---
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
for ax, feat in zip(axes, ["mean radius", "mean concavity"]):
    for label, color in zip(["Malignant", "Benign"], colors):
        subset = df[df["diagnosis_label"] == label][feat]
        ax.hist(subset, bins=25, alpha=0.6, label=label, color=color)
    ax.set_title(feat.title())
    ax.set_xlabel(feat)
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
fig.suptitle("Feature Distributions by Diagnosis Class", y=1.03)
fig.tight_layout()
fig.savefig(f"{OUT}/eda_feature_distributions.png", dpi=150)
plt.close(fig)

print("EDA visualizations saved: eda_class_distribution.png, eda_correlation_heatmap.png, eda_feature_distributions.png")
print("\nSummary stats:")
print(df[top_feats].describe().T[["mean", "std", "min", "max"]].round(2))
