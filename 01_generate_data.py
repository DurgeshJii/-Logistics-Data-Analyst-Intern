"""
01_generate_data.py

Generates a self-generated (synthetic) e-commerce marketing dataset that
mimics the output of an A/B test run by a retail company's marketing team.

Business scenario
------------------
An online retailer ran a 30-day A/B test on its email marketing program.
Customers were randomly assigned to:
  - Control group   : received the standard, generic promotional email
  - Treatment group : received a personalized product-recommendation email

For every customer in the test we recorded:
  - group            : "Control" or "Treatment"
  - segment          : customer tier ("New", "Regular", "Premium")
  - region           : sales region ("North", "South", "East", "West")
  - purchase_amount  : amount spent (in USD) during the 30-day window
                        (0 for customers who did not purchase)
  - purchased        : 1 if the customer made at least one purchase, else 0

Random seed is fixed for reproducibility.
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

N = 640  # total customers in the experiment

# ---------------------------------------------------------------
# 1. Random assignment to experimental group (balanced randomization)
# ---------------------------------------------------------------
group = rng.choice(["Control", "Treatment"], size=N, p=[0.5, 0.5])

# ---------------------------------------------------------------
# 2. Customer attributes (assigned independently of group,
#    exactly as they would be in a properly randomized experiment)
# ---------------------------------------------------------------
segment = rng.choice(["New", "Regular", "Premium"], size=N, p=[0.35, 0.45, 0.20])
region = rng.choice(["North", "South", "East", "West"], size=N)

# ---------------------------------------------------------------
# 3. Conversion (purchased yes/no)
#    Treatment lifts conversion probability by ~9 points
# ---------------------------------------------------------------
base_conv_prob = 0.28
treatment_lift = 0.09
segment_lift = {"New": -0.05, "Regular": 0.0, "Premium": 0.10}

conv_prob = np.array([
    base_conv_prob
    + (treatment_lift if g == "Treatment" else 0.0)
    + segment_lift[s]
    for g, s in zip(group, segment)
])
conv_prob = np.clip(conv_prob, 0.03, 0.97)
purchased = rng.binomial(1, conv_prob)

# ---------------------------------------------------------------
# 4. Purchase amount (only defined for those who purchased)
#    Segment strongly affects spend level (drives the ANOVA);
#    Treatment adds a modest additional spend lift (drives the t-test).
# ---------------------------------------------------------------
segment_mean = {"New": 42, "Regular": 58, "Premium": 88}
treatment_bump = 9.0

purchase_amount = np.zeros(N)
for i in range(N):
    if purchased[i] == 1:
        mu = segment_mean[segment[i]] + (treatment_bump if group[i] == "Treatment" else 0.0)
        sigma = 0.35 * mu  # realistic right-skewed-ish spend variability
        amt = rng.normal(mu, sigma)
        purchase_amount[i] = max(5.0, amt)  # floor at $5
    else:
        purchase_amount[i] = 0.0

df = pd.DataFrame({
    "customer_id": [f"C{100000+i}" for i in range(N)],
    "group": group,
    "segment": segment,
    "region": region,
    "purchased": purchased,
    "purchase_amount": np.round(purchase_amount, 2),
})

df.to_csv("/home/claude/project/data/marketing_ab_test.csv", index=False)

print(df.head(10).to_string(index=False))
print("\nShape:", df.shape)
print("\nGroup counts:\n", df["group"].value_counts())
print("\nOverall conversion rate:", df["purchased"].mean().round(4))
print("\nConversion rate by group:\n", df.groupby("group")["purchased"].mean().round(4))
