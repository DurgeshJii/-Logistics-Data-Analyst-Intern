"""
02_analysis.py

Performs the full hypothesis-testing workflow for the marketing A/B test:

  H1 (Welch's two-sample t-test): Treatment customers who purchased spend
      more per transaction, on average, than Control customers who purchased.
  H2 (Chi-square test of independence): Conversion (purchased yes/no) is
      associated with experimental group.
  H3 (One-way ANOVA + Tukey HSD): Average purchase amount (among buyers)
      differs across customer segments (New / Regular / Premium).

Assumption checks (Shapiro-Wilk, Levene's test) and effect sizes
(Cohen's d, Cramer's V, eta-squared) are reported alongside each test.
All numeric results are written to results.txt for use in the report.
"""

import numpy as np
import pandas as pd
from scipy import stats
import itertools

df = pd.read_csv("/home/claude/project/data/marketing_ab_test.csv")
buyers = df[df["purchased"] == 1].copy()

out = []
def log(*args):
    line = " ".join(str(a) for a in args)
    print(line)
    out.append(line)

log("=" * 70)
log("DATASET OVERVIEW")
log("=" * 70)
log(f"Total customers: {len(df)}")
log(f"Buyers (purchased == 1): {len(buyers)}")
log(df.groupby("group").size().to_string())
log("")

# =================================================================
# H1: Welch's two-sample t-test on purchase amount (buyers only)
# =================================================================
log("=" * 70)
log("H1: TWO-SAMPLE T-TEST -- Purchase amount, Treatment vs Control (buyers)")
log("=" * 70)
log("H0: mu_treatment = mu_control")
log("H1: mu_treatment > mu_control  (one-sided)")
log("")

ctrl_amt = buyers.loc[buyers["group"] == "Control", "purchase_amount"]
treat_amt = buyers.loc[buyers["group"] == "Treatment", "purchase_amount"]

log(f"Control:   n={len(ctrl_amt)}, mean=${ctrl_amt.mean():.2f}, sd=${ctrl_amt.std(ddof=1):.2f}")
log(f"Treatment: n={len(treat_amt)}, mean=${treat_amt.mean():.2f}, sd=${treat_amt.std(ddof=1):.2f}")

# --- Assumption checks ---
sw_ctrl = stats.shapiro(ctrl_amt)
sw_treat = stats.shapiro(treat_amt)
levene = stats.levene(ctrl_amt, treat_amt)
log("")
log("Assumption checks:")
log(f"  Shapiro-Wilk (Control):   W={sw_ctrl.statistic:.4f}, p={sw_ctrl.pvalue:.4g}")
log(f"  Shapiro-Wilk (Treatment): W={sw_treat.statistic:.4f}, p={sw_treat.pvalue:.4g}")
log(f"  Levene's test (equal variances): W={levene.statistic:.4f}, p={levene.pvalue:.4g}")
log("  -> Because normality is questionable and n is reasonably large (CLT applies)")
log("     and/or variances may be unequal, we use Welch's t-test (unequal variances,")
log("     robust to mild non-normality) and corroborate with the nonparametric")
log("     Mann-Whitney U test below.")

# --- Welch's t-test (one-sided: treatment > control) ---
t_res = stats.ttest_ind(treat_amt, ctrl_amt, equal_var=False, alternative="greater")
log("")
log(f"Welch's t-test: t={t_res.statistic:.4f}, df~{t_res.df:.1f}, one-sided p={t_res.pvalue:.4g}")

# 95% CI for the difference in means (Welch-Satterthwaite)
n1, n2 = len(treat_amt), len(ctrl_amt)
m1, m2 = treat_amt.mean(), ctrl_amt.mean()
v1, v2 = treat_amt.var(ddof=1), ctrl_amt.var(ddof=1)
se_diff = np.sqrt(v1 / n1 + v2 / n2)
dof = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
diff = m1 - m2
tcrit = stats.t.ppf(0.975, dof)
ci_low, ci_high = diff - tcrit * se_diff, diff + tcrit * se_diff
log(f"Mean difference (Treatment - Control): ${diff:.2f}")
log(f"95% CI for the difference: [${ci_low:.2f}, ${ci_high:.2f}]")

# Cohen's d (pooled SD)
sp = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
cohend = diff / sp
log(f"Cohen's d (effect size): {cohend:.3f}")

# Nonparametric robustness check
mw = stats.mannwhitneyu(treat_amt, ctrl_amt, alternative="greater")
log(f"Mann-Whitney U (robustness check): U={mw.statistic:.1f}, one-sided p={mw.pvalue:.4g}")

alpha = 0.05
log("")
if t_res.pvalue < alpha:
    log(f"DECISION: p={t_res.pvalue:.4g} < alpha=0.05 -> REJECT H0. Treatment customers")
    log("spend significantly more per transaction than Control customers.")
else:
    log(f"DECISION: p={t_res.pvalue:.4g} >= alpha=0.05 -> FAIL TO REJECT H0.")
log("")

# =================================================================
# H2: Chi-square test of independence -- Group x Purchased
# =================================================================
log("=" * 70)
log("H2: CHI-SQUARE TEST OF INDEPENDENCE -- Group vs Conversion")
log("=" * 70)
log("H0: Conversion (purchased yes/no) is independent of experimental group")
log("H1: Conversion is associated with experimental group")
log("")

ct = pd.crosstab(df["group"], df["purchased"])
ct.columns = ["Not Purchased", "Purchased"]
log("Contingency table (observed counts):")
log(ct.to_string())
log("")

chi2, p_chi2, dof_chi2, expected = stats.chi2_contingency(ct, correction=True)
log(f"Chi-square statistic: {chi2:.4f}, df={dof_chi2}, p={p_chi2:.4g}")
log("Expected counts:")
log(pd.DataFrame(expected, index=ct.index, columns=ct.columns).round(2).to_string())

# Cramer's V (effect size for chi-square)
n_total = ct.values.sum()
cramers_v = np.sqrt(chi2 / (n_total * (min(ct.shape) - 1)))
log(f"Cramer's V (effect size): {cramers_v:.3f}")

conv_rate = df.groupby("group")["purchased"].mean()
log("")
log("Conversion rate by group:")
log(conv_rate.round(4).to_string())
# 95% CI for difference in proportions
p1 = conv_rate["Treatment"]; p2 = conv_rate["Control"]
n1p = (df["group"] == "Treatment").sum(); n2p = (df["group"] == "Control").sum()
se_p = np.sqrt(p1*(1-p1)/n1p + p2*(1-p2)/n2p)
diff_p = p1 - p2
ci_p_low, ci_p_high = diff_p - 1.96*se_p, diff_p + 1.96*se_p
log(f"Difference in conversion rate (Treatment - Control): {diff_p:.4f}")
log(f"95% CI for the difference in proportions: [{ci_p_low:.4f}, {ci_p_high:.4f}]")

log("")
if p_chi2 < alpha:
    log(f"DECISION: p={p_chi2:.4g} < alpha=0.05 -> REJECT H0. Conversion rate is")
    log("significantly associated with (higher in) the Treatment group.")
else:
    log(f"DECISION: p={p_chi2:.4g} >= alpha=0.05 -> FAIL TO REJECT H0.")
log("")

# =================================================================
# H3: One-way ANOVA -- purchase amount across customer segments
# =================================================================
log("=" * 70)
log("H3: ONE-WAY ANOVA -- Purchase amount across customer Segments (buyers)")
log("=" * 70)
log("H0: mu_New = mu_Regular = mu_Premium")
log("H1: at least one segment mean differs")
log("")

seg_groups = [buyers.loc[buyers["segment"] == s, "purchase_amount"] for s in ["New", "Regular", "Premium"]]
seg_labels = ["New", "Regular", "Premium"]

for lbl, g in zip(seg_labels, seg_groups):
    log(f"{lbl:8s}: n={len(g)}, mean=${g.mean():.2f}, sd=${g.std(ddof=1):.2f}")

log("")
levene_seg = stats.levene(*seg_groups)
log(f"Levene's test (equal variances across segments): W={levene_seg.statistic:.4f}, p={levene_seg.pvalue:.4g}")
for lbl, g in zip(seg_labels, seg_groups):
    sw = stats.shapiro(g)
    log(f"Shapiro-Wilk ({lbl}): W={sw.statistic:.4f}, p={sw.pvalue:.4g}")

f_stat, p_anova = stats.f_oneway(*seg_groups)
log("")
log(f"One-way ANOVA: F={f_stat:.4f}, p={p_anova:.4g}")

# Effect size: eta-squared
grand_mean = buyers["purchase_amount"].mean()
ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in seg_groups)
ss_total = ((buyers["purchase_amount"] - grand_mean) ** 2).sum()
eta_sq = ss_between / ss_total
log(f"Eta-squared (effect size): {eta_sq:.3f}")

log("")
if p_anova < alpha:
    log(f"DECISION: p={p_anova:.4g} < alpha=0.05 -> REJECT H0. At least one segment's")
    log("mean purchase amount differs significantly.")
    log("")
    log("Post-hoc pairwise comparisons (Tukey-style, Bonferroni-corrected t-tests):")
    pairs = list(itertools.combinations(range(len(seg_labels)), 2))
    m = len(pairs)
    for i, j in pairs:
        a, b = seg_groups[i], seg_groups[j]
        t_p, p_p = stats.ttest_ind(a, b, equal_var=False)
        p_adj = min(p_p * m, 1.0)  # Bonferroni correction
        sig = "SIGNIFICANT" if p_adj < alpha else "not significant"
        log(f"  {seg_labels[i]:8s} vs {seg_labels[j]:8s}: "
            f"mean diff=${a.mean()-b.mean():.2f}, t={t_p:.3f}, raw p={p_p:.4g}, "
            f"Bonferroni-adj p={p_adj:.4g} -> {sig}")
else:
    log(f"DECISION: p={p_anova:.4g} >= alpha=0.05 -> FAIL TO REJECT H0.")

log("")
log("=" * 70)
log("SUMMARY TABLE")
log("=" * 70)
summary = pd.DataFrame([
    ["H1", "Welch's t-test (one-sided)", "Purchase amount ~ Group", f"{t_res.pvalue:.4g}", "Reject H0" if t_res.pvalue < alpha else "Fail to reject"],
    ["H2", "Chi-square test of independence", "Purchased ~ Group", f"{p_chi2:.4g}", "Reject H0" if p_chi2 < alpha else "Fail to reject"],
    ["H3", "One-way ANOVA", "Purchase amount ~ Segment", f"{p_anova:.4g}", "Reject H0" if p_anova < alpha else "Fail to reject"],
], columns=["Hypothesis", "Test", "Variables", "p-value", "Decision (alpha=0.05)"])
log(summary.to_string(index=False))

with open("/home/claude/project/results.txt", "w") as f:
    f.write("\n".join(out))

print("\n\nSaved full results to results.txt")
