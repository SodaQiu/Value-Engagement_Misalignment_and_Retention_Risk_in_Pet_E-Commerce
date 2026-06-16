# ============================================================
# H1 TEST
# Early observable engagement signals and fourth-purchase churn
#
# Analysis sample:
# Customers who completed their first three purchases and have
# valid early transaction value, engagement signals, and churn
# outcome variables.
#
# DV:
# churn_yn = 1 if the user did not complete a fourth purchase
# churn_yn = 0 otherwise
#
# Main predictor:
# engagement_count = review_written_yn + push_notification_consent_yn
# ============================================================

import numpy as np
import statsmodels.formula.api as smf

from quadrant_utils import load_hypothesis_data


data = load_hypothesis_data()

third_purchase_df = data["third_purchase_df"]
analysis_df = data["analysis_df"].copy()
analysis_value_threshold = data["analysis_value_threshold"]
binary_cols = data["binary_cols"]
engagement_summary = data["engagement_summary"]

print("=" * 80)
print("BINARY VARIABLE CHECK")
print("=" * 80)
print("All binary variables contain only 0/1 values:")
print(binary_cols)

print("\n" + "=" * 80)
print("H1 ANALYSIS SAMPLE")
print("=" * 80)
print(
    "Completed-third-purchase sample:",
    len(third_purchase_df)
)
print(
    "Rows used for H1 analysis:",
    len(analysis_df)
)
print(
    "Rows excluded due to missing or invalid H1 variables:",
    len(third_purchase_df) - len(analysis_df)
)
print("High-value threshold: Median order_unit_price")
print("Analysis-sample median order_unit_price:", analysis_value_threshold)

print("\n" + "=" * 80)
print("ENGAGEMENT COUNT DISTRIBUTION")
print("=" * 80)
print(
    engagement_summary[
        [
            "engagement_count",
            "n",
            "churn_n",
            "churn_rate_percent",
            "retention_rate_percent"
        ]
    ]
    .round(2)
    .to_string(index=False)
)


# ============================================================
# H1 Trend Test
# Does churn decrease as engagement_count increases?
# ============================================================

h1_trend_model = smf.logit(
    formula="churn_yn ~ engagement_count",
    data=analysis_df
).fit(disp=False)

coef = h1_trend_model.params["engagement_count"]
p_value = h1_trend_model.pvalues["engagement_count"]

ci_low, ci_high = h1_trend_model.conf_int().loc[
    "engagement_count"
]

odds_ratio = np.exp(coef)
or_ci_low = np.exp(ci_low)
or_ci_high = np.exp(ci_high)

print("\n" + "=" * 80)
print("H1 TREND TEST: ENGAGEMENT COUNT -> CHURN")
print("=" * 80)
print(f"Coefficient: {coef:.4f}")
print(f"Odds Ratio: {odds_ratio:.4f}")
print(f"95% CI for OR: [{or_ci_low:.4f}, {or_ci_high:.4f}]")
print(f"P-value: {p_value:.6g}")

if coef < 0 and p_value < 0.05:
    print(
        "Conclusion: Higher engagement_count is significantly "
        "associated with lower fourth-purchase churn."
    )
else:
    print(
        "Conclusion: H1 is not supported by the trend model."
    )


# ============================================================
# H1 Adjusted Logistic Regression
# Control for early transaction value
# ============================================================

analysis_df["log_order_unit_price"] = np.log1p(
    analysis_df["order_unit_price"]
)

h1_adjusted_model = smf.logit(
    formula=(
        "churn_yn ~ engagement_count "
        "+ log_order_unit_price"
    ),
    data=analysis_df
).fit(disp=False)

coef = h1_adjusted_model.params["engagement_count"]
p_value = h1_adjusted_model.pvalues["engagement_count"]

ci_low, ci_high = h1_adjusted_model.conf_int().loc[
    "engagement_count"
]

odds_ratio = np.exp(coef)
or_ci_low = np.exp(ci_low)
or_ci_high = np.exp(ci_high)

print("\n" + "=" * 80)
print("H1 ADJUSTED MODEL")
print("=" * 80)
print(f"Coefficient: {coef:.4f}")
print(f"Odds Ratio: {odds_ratio:.4f}")
print(f"95% CI for OR: [{or_ci_low:.4f}, {or_ci_high:.4f}]")
print(f"P-value: {p_value:.6g}")

if coef < 0 and p_value < 0.05:
    print(
        "Conclusion: After controlling for early transaction value, "
        "higher engagement_count is significantly associated with "
        "lower fourth-purchase churn. H1 is supported."
    )
else:
    print(
        "Conclusion: After controlling for early transaction value, "
        "H1 is not supported by the adjusted model."
    )
