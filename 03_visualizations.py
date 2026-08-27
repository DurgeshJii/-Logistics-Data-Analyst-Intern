"""
03_visualizations.py
Generates all charts used in the Word report, saved as PNGs at 150 dpi.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid", context="talk")
PALETTE = {"Control": "#7A8B99", "Treatment": "#2E86AB"}
SEG_PALETTE = {"New": "#A8DADC", "Regular": "#457B9D", "Premium": "#1D3557"}

df = pd.read_csv("/home/claude/project/data/marketing_ab_test.csv")
buyers = df[df["purchased"] == 1].copy()
FIG = "/home/claude/project/figures"

# -----------------------------------------------------------------
# Fig 1: Box + strip plot of purchase amount by group (buyers)
# -----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
order = ["Control", "Treatment"]
sns.boxplot(data=buyers, x="group", y="purchase_amount", order=order,
            palette=PALETTE, ax=ax, width=0.5, showfliers=False)
sns.stripplot(data=buyers, x="group", y="purchase_amount", order=order,
              color="black", alpha=0.35, size=4, jitter=0.15, ax=ax)
means = buyers.groupby("group")["purchase_amount"].mean().reindex(order)
for i, m in enumerate(means):
    ax.scatter(i, m, marker="D", color="red", s=90, zorder=5, label="Mean" if i == 0 else None)
ax.set_title("Purchase Amount by Experimental Group\n(buyers only)", fontsize=16, weight="bold")
ax.set_xlabel("")
ax.set_ylabel("Purchase Amount (USD)")
ax.legend(loc="upper left")
plt.tight_layout()
plt.savefig(f"{FIG}/fig1_boxplot_group.png", dpi=150)
plt.close()

# -----------------------------------------------------------------
# Fig 2: Overlaid histogram / density of purchase amount by group
# -----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
for g in order:
    sns.histplot(buyers.loc[buyers["group"] == g, "purchase_amount"], bins=18,
                 color=PALETTE[g], label=g, kde=True, stat="density", alpha=0.45, ax=ax)
ax.set_title("Distribution of Purchase Amount by Group", fontsize=16, weight="bold")
ax.set_xlabel("Purchase Amount (USD)")
ax.set_ylabel("Density")
ax.legend(title="Group")
plt.tight_layout()
plt.savefig(f"{FIG}/fig2_histogram_group.png", dpi=150)
plt.close()

# -----------------------------------------------------------------
# Fig 3: Conversion rate bar chart with 95% CI error bars
# -----------------------------------------------------------------
conv = df.groupby("group")["purchased"].agg(["mean", "count"]).reindex(order)
conv["se"] = np.sqrt(conv["mean"] * (1 - conv["mean"]) / conv["count"])
conv["ci95"] = 1.96 * conv["se"]

fig, ax = plt.subplots(figsize=(7, 6))
bars = ax.bar(order, conv["mean"], yerr=conv["ci95"], capsize=8,
              color=[PALETTE[g] for g in order], edgecolor="black", linewidth=0.8)
for i, (g, row) in enumerate(conv.iterrows()):
    ax.text(i, row["mean"] + row["ci95"] + 0.01, f"{row['mean']*100:.1f}%",
            ha="center", fontsize=13, weight="bold")
ax.set_title("Conversion Rate by Group\n(error bars = 95% CI)", fontsize=16, weight="bold")
ax.set_ylabel("Conversion Rate")
ax.set_ylim(0, max(conv["mean"] + conv["ci95"]) + 0.08)
plt.tight_layout()
plt.savefig(f"{FIG}/fig3_conversion_bar.png", dpi=150)
plt.close()

# -----------------------------------------------------------------
# Fig 4: Boxplot of purchase amount by segment (ANOVA)
# -----------------------------------------------------------------
seg_order = ["New", "Regular", "Premium"]
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(data=buyers, x="segment", y="purchase_amount", order=seg_order,
            palette=SEG_PALETTE, ax=ax, width=0.55, showfliers=False)
sns.stripplot(data=buyers, x="segment", y="purchase_amount", order=seg_order,
              color="black", alpha=0.3, size=4, jitter=0.15, ax=ax)
ax.set_title("Purchase Amount by Customer Segment\n(One-Way ANOVA)", fontsize=16, weight="bold")
ax.set_xlabel("Customer Segment")
ax.set_ylabel("Purchase Amount (USD)")
plt.tight_layout()
plt.savefig(f"{FIG}/fig4_boxplot_segment.png", dpi=150)
plt.close()

# -----------------------------------------------------------------
# Fig 5: Q-Q plots for normality check (Treatment vs Control)
# -----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
for ax, g in zip(axes, order):
    data = buyers.loc[buyers["group"] == g, "purchase_amount"]
    stats.probplot(data, dist="norm", plot=ax)
    ax.set_title(f"Q-Q Plot: {g}", fontsize=14, weight="bold")
    ax.get_lines()[0].set_markerfacecolor(PALETTE[g])
    ax.get_lines()[0].set_markeredgecolor(PALETTE[g])
    ax.get_lines()[1].set_color("red")
plt.tight_layout()
plt.savefig(f"{FIG}/fig5_qqplots.png", dpi=150)
plt.close()

# -----------------------------------------------------------------
# Fig 6: Mean purchase amount by group with 95% CI (point estimate plot)
# -----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))
stats_by_group = buyers.groupby("group")["purchase_amount"].agg(["mean", "std", "count"]).reindex(order)
stats_by_group["se"] = stats_by_group["std"] / np.sqrt(stats_by_group["count"])
stats_by_group["ci95"] = 1.96 * stats_by_group["se"]

ax.errorbar(order, stats_by_group["mean"], yerr=stats_by_group["ci95"],
            fmt="o", markersize=12, capsize=10, capthick=2, elinewidth=2,
            color="#2E86AB", ecolor="#2E86AB")
for i, (g, row) in enumerate(stats_by_group.iterrows()):
    ax.text(i + 0.05, row["mean"], f"${row['mean']:.2f}", fontsize=12, va="center")
ax.set_title("Mean Purchase Amount by Group\n(error bars = 95% CI)", fontsize=16, weight="bold")
ax.set_ylabel("Mean Purchase Amount (USD)")
ax.set_xlim(-0.5, 1.5)
plt.tight_layout()
plt.savefig(f"{FIG}/fig6_mean_ci_group.png", dpi=150)
plt.close()

print("All figures saved to", FIG)
import os
for f in sorted(os.listdir(FIG)):
    print(" -", f)
