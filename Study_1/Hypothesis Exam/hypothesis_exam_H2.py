# ============================================================
# H2: HVLE versus HVHE fourth-purchase churn
#
# Hypothesis:
# Among high-value customers, low-engagement customers (HVLE)
# have a higher fourth-purchase churn rate than high-engagement
# customers (HVHE).
#
# DV:
# churn_yn = 1 if the customer did not complete purchase 4
# churn_yn = 0 otherwise
#
# Reference group:
# HVHE (high value, high engagement)
# ============================================================

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from scipy.stats import chi2_contingency

from quadrant_utils import load_hypothesis_data


def logistic_result(model, predictor):
    """Return the key logistic-regression statistics."""
    coefficient = model.params[predictor]
    ci_low, ci_high = model.conf_int().loc[predictor]

    return {
        "coefficient": coefficient,
        "odds_ratio": np.exp(coefficient),
        "ci_low": np.exp(ci_low),
        "ci_high": np.exp(ci_high),
        "p_value": model.pvalues[predictor]
    }


data = load_hypothesis_data()
analysis_df = data["analysis_df"].copy()

required_columns = [
    "high_value",
    "high_engagement",
    "churn_yn",
    "order_unit_price"
]
missing_columns = [
    column
    for column in required_columns
    if column not in analysis_df.columns
]

if missing_columns:
    raise ValueError(
        f"H2 required columns are missing: {missing_columns}"
    )

for column in ["high_value", "high_engagement", "churn_yn"]:
    missing_count = analysis_df[column].isna().sum()
    invalid_values = sorted(
        analysis_df.loc[
            analysis_df[column].notna()
            & ~analysis_df[column].isin([0, 1]),
            column
        ].unique().tolist()
    )

    if missing_count or invalid_values:
        raise ValueError(
            f"{column} must contain only 0/1 with no missing values; "
            f"missing={missing_count}, invalid={invalid_values}"
        )

invalid_value_mask = (
    analysis_df["order_unit_price"].isna()
    | ~np.isfinite(analysis_df["order_unit_price"])
    | (analysis_df["order_unit_price"] <= 0)
)

if invalid_value_mask.any():
    raise ValueError(
        "order_unit_price must be finite, positive, and non-missing; "
        f"invalid rows={int(invalid_value_mask.sum())}"
    )

# H2 focuses on customers classified as high value.
h2_df = analysis_df[
    analysis_df["high_value"] == 1
].copy()

if h2_df.empty:
    raise ValueError(
        "H2 high-value sample is empty. Check the value threshold."
    )

# HVLE = 1 and HVHE = 0. A positive coefficient therefore indicates
# higher churn among HVLE customers relative to HVHE customers.
h2_df["HVLE_yn"] = (
    h2_df["high_engagement"] == 0
).astype(int)
h2_df["group"] = h2_df["HVLE_yn"].map({
    0: "HVHE",
    1: "HVLE"
})
h2_df["log_order_unit_price"] = np.log1p(
    h2_df["order_unit_price"]
)

if set(h2_df["HVLE_yn"].unique()) != {0, 1}:
    raise ValueError(
        "The H2 sample must contain both HVHE (0) and HVLE (1)."
    )

if set(h2_df["churn_yn"].unique()) != {0, 1}:
    raise ValueError(
        "The H2 sample must contain both retained (0) and churned (1) "
        "customers."
    )


# ------------------------------------------------------------
# 1. Descriptive comparison
# ------------------------------------------------------------

group_summary = (
    h2_df
    .groupby("group", observed=True)
    .agg(
        n=("churn_yn", "size"),
        churn_n=("churn_yn", "sum"),
        churn_rate=("churn_yn", "mean")
    )
    .reindex(["HVHE", "HVLE"])
    .reset_index()
)

group_summary["churn_rate_percent"] = (
    group_summary["churn_rate"] * 100
)
group_summary["retention_rate_percent"] = (
    (1 - group_summary["churn_rate"]) * 100
)

group_churn_rates = group_summary.set_index("group")["churn_rate"]
churn_rate_difference_pp = (
    group_churn_rates["HVLE"] - group_churn_rates["HVHE"]
) * 100

print("=" * 80)
print("H2: HVLE VERSUS HVHE FOURTH-PURCHASE CHURN")
print("=" * 80)
print("High-value analysis sample:", len(h2_df))
print("Value threshold: median order_unit_price")
print("Reference group: HVHE")

print("\nGROUP SUMMARY")
print(
    group_summary[
        [
            "group",
            "n",
            "churn_n",
            "churn_rate_percent",
            "retention_rate_percent"
        ]
    ]
    .round(2)
    .to_string(index=False)
)
print(
    "HVLE minus HVHE churn-rate difference: "
    f"{churn_rate_difference_pp:.2f} percentage points"
)


# ------------------------------------------------------------
# 2. Chi-square test
# ------------------------------------------------------------

contingency_table = (
    pd.crosstab(h2_df["group"], h2_df["churn_yn"])
    .reindex(index=["HVHE", "HVLE"], columns=[0, 1])
)

chi_square, chi_square_p, chi_square_df, expected = (
    # Use the uncorrected Pearson chi-square consistently across H2-H5.
    # Logistic regression is the primary inferential test.
    chi2_contingency(contingency_table, correction=False)
)

print("\nCHI-SQUARE TEST")
print(contingency_table.to_string())
print(f"Chi-square: {chi_square:.4f}")
print(f"Degrees of freedom: {chi_square_df}")
print(f"P-value: {chi_square_p:.6g}")
print(f"Minimum expected frequency: {expected.min():.4f}")


# ------------------------------------------------------------
# 3. Logistic regression models
# ------------------------------------------------------------

unadjusted_model = smf.logit(
    "churn_yn ~ HVLE_yn",
    data=h2_df
).fit(disp=False)

value_adjusted_model = smf.logit(
    "churn_yn ~ HVLE_yn + log_order_unit_price",
    data=h2_df
).fit(disp=False)

unadjusted_result = logistic_result(
    unadjusted_model,
    "HVLE_yn"
)
adjusted_result = logistic_result(
    value_adjusted_model,
    "HVLE_yn"
)

model_results = pd.DataFrame([
    {
        "model": "Model 1: Unadjusted",
        **unadjusted_result
    },
    {
        "model": "Model 2: Value adjusted",
        **adjusted_result
    }
])

print("\nLOGISTIC REGRESSION RESULTS: HVLE VERSUS HVHE")
print(
    model_results[
        [
            "model",
            "coefficient",
            "odds_ratio",
            "ci_low",
            "ci_high",
            "p_value"
        ]
    ]
    .to_string(
        index=False,
        formatters={
            "coefficient": lambda x: f"{x:.4f}",
            "odds_ratio": lambda x: f"{x:.4f}",
            "ci_low": lambda x: f"{x:.4f}",
            "ci_high": lambda x: f"{x:.4f}",
            "p_value": lambda x: f"{x:.6g}"
        }
    )
)


# ------------------------------------------------------------
# 4. H2 decision
# ------------------------------------------------------------

h2_supported = (
    unadjusted_result["coefficient"] > 0
    and unadjusted_result["p_value"] < 0.05
    and adjusted_result["coefficient"] > 0
    and adjusted_result["p_value"] < 0.05
)

print("\nH2 FINAL DECISION")
if h2_supported:
    print(
        "H2 is supported. Among high-value customers, HVLE "
        "customers have significantly higher fourth-purchase "
        "churn than HVHE customers, including after adjustment "
        "for early transaction value."
    )
else:
    print(
        "H2 is not fully supported. Review the unadjusted and "
        "value-adjusted estimates."
    )
