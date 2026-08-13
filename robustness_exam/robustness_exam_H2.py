"""H2 robustness: vary the value cutoff while holding engagement fixed.

H2:
Among high-value customers, low-engagement customers (HVLE) have a higher
probability of failing to complete the fourth purchase than high-engagement
customers (HVHE).

Robustness logic
----------------
Only the high-value cutoff changes.  Engagement is always defined as:

    engagement_count = review_written_yn + push_notification_consent_yn
    low engagement  = engagement_count == 0
    high engagement = engagement_count >= 1

The median cutoff is the prespecified primary definition.  The 40th and 60th
percentile cutoffs are sensitivity checks.  Continuous order value is not
added as a regression control because this script tests the robustness of
the value-based sample definition itself.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency
from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationError


PROJECT_DIR = Path(__file__).resolve().parents[1]
STUDY_1_DIR = PROJECT_DIR / "Study_1" / "Hypothesis Exam"

if str(STUDY_1_DIR) not in sys.path:
    sys.path.append(str(STUDY_1_DIR))

from quadrant_utils import load_hypothesis_data


# The median is the primary H2 definition. The surrounding cutoffs alter only
# the monetary boundary and leave the engagement definition unchanged.
VALUE_CUTOFFS = [
    ("Sensitivity: 40th percentile", 0.40),
    ("Primary: median", 0.50),
    ("Sensitivity: 60th percentile", 0.60),
]
PRIMARY_QUANTILE = 0.50


def prepare_analysis_data():
    """Load the common H1/H2 cohort and verify the engagement definition."""
    loaded = load_hypothesis_data()
    df = loaded["analysis_df"].copy()

    required_cols = [
        "order_unit_price",
        "review_written_yn",
        "push_notification_consent_yn",
        "engagement_count",
        "high_engagement",
        "churn_yn",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"H2 required columns are missing: {missing}")

    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    invalid_binary = {}
    for col in [
        "review_written_yn",
        "push_notification_consent_yn",
        "high_engagement",
        "churn_yn",
    ]:
        invalid = df.loc[df[col].notna() & ~df[col].isin([0, 1]), col]
        if not invalid.empty:
            invalid_binary[col] = sorted(invalid.unique().tolist())
    if invalid_binary:
        raise ValueError(f"Invalid binary values: {invalid_binary}")

    model_cols = required_cols
    df = df.dropna(subset=model_cols).copy()
    df = df[df["order_unit_price"] > 0].copy()

    expected_count = (
        df["review_written_yn"]
        + df["push_notification_consent_yn"]
    )
    expected_high_engagement = expected_count.ge(1).astype(int)

    if not expected_count.equals(df["engagement_count"]):
        raise ValueError(
            "Stored engagement_count does not match the fixed H2 definition."
        )
    if not expected_high_engagement.equals(df["high_engagement"].astype(int)):
        raise ValueError(
            "Stored high_engagement does not match engagement_count >= 1."
        )

    if df.empty:
        raise ValueError("The H2 analysis cohort is empty.")

    return df, loaded["analysis_value_threshold"]


def fit_h2_model(high_value_df, model_name):
    """Estimate the HVLE-versus-HVHE churn difference at one value cutoff."""
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model = smf.logit(
                "churn_yn ~ HVLE_yn",
                data=high_value_df,
            ).fit(
                disp=False,
                maxiter=200,
                cov_type="HC1",
            )
    except (PerfectSeparationError, np.linalg.LinAlgError) as exc:
        raise RuntimeError(
            f"{model_name} could not be estimated because of separation "
            "or a singular matrix."
        ) from exc

    convergence_messages = [
        str(item.message)
        for item in caught
        if issubclass(item.category, ConvergenceWarning)
    ]
    if not model.mle_retvals.get("converged", False) or convergence_messages:
        raise RuntimeError(
            f"{model_name} did not converge: {convergence_messages}"
        )

    return model


def analyze_cutoff(base_df, cutoff_name, quantile):
    """Construct one high-value sample and return descriptive/model results."""
    cutoff_value = base_df["order_unit_price"].quantile(quantile)
    high_value_df = base_df[
        base_df["order_unit_price"] >= cutoff_value
    ].copy()

    # Engagement remains fixed across all cutoff analyses. Within each
    # high-value sample, HVLE versus HVHE is therefore engagement-only.
    high_value_df["HVLE_yn"] = (
        high_value_df["high_engagement"].eq(0).astype(int)
    )
    high_value_df["group"] = high_value_df["HVLE_yn"].map(
        {0: "HVHE", 1: "HVLE"}
    )

    if set(high_value_df["HVLE_yn"].unique()) != {0, 1}:
        raise ValueError(f"{cutoff_name} lacks either HVHE or HVLE customers.")
    if set(high_value_df["churn_yn"].unique()) != {0, 1}:
        raise ValueError(f"{cutoff_name} lacks one churn outcome class.")

    contingency = (
        pd.crosstab(high_value_df["group"], high_value_df["churn_yn"])
        .reindex(index=["HVHE", "HVLE"], columns=[0, 1], fill_value=0)
    )
    if (contingency == 0).any().any():
        raise ValueError(f"{cutoff_name} has an empty group-by-outcome cell.")

    # Use the same uncorrected Pearson chi-square specification as the main
    # H2-H5 hypothesis tests; avoid SciPy's automatic Yates correction for
    # 2x2 tables so the method does not vary implicitly with table dimensions.
    chi_square, chi_square_p, _, expected = chi2_contingency(
        contingency,
        correction=False,
    )
    model = fit_h2_model(high_value_df, cutoff_name)

    coefficient = model.params["HVLE_yn"]
    ci_low, ci_high = model.conf_int().loc["HVLE_yn"]

    group_summary = (
        high_value_df.groupby("group", observed=True)
        .agg(
            n=("churn_yn", "size"),
            churn_n=("churn_yn", "sum"),
            churn_rate=("churn_yn", "mean"),
        )
        .reindex(["HVHE", "HVLE"])
    )
    hvhe_rate = group_summary.loc["HVHE", "churn_rate"]
    hvle_rate = group_summary.loc["HVLE", "churn_rate"]

    result = {
        "cutoff_definition": cutoff_name,
        "value_quantile": quantile,
        "cutoff_value": cutoff_value,
        "high_value_n": len(high_value_df),
        "HVHE_n": int(group_summary.loc["HVHE", "n"]),
        "HVLE_n": int(group_summary.loc["HVLE", "n"]),
        "HVHE_churn_percent": hvhe_rate * 100,
        "HVLE_churn_percent": hvle_rate * 100,
        "churn_difference_pp": (hvle_rate - hvhe_rate) * 100,
        "HVLE_beta": coefficient,
        "HVLE_OR": np.exp(coefficient),
        "CI_lower": np.exp(ci_low),
        "CI_upper": np.exp(ci_high),
        "p_value": model.pvalues["HVLE_yn"],
        "chi_square": chi_square,
        "chi_square_p": chi_square_p,
        "minimum_expected_count": expected.min(),
    }

    return {
        "analysis_df": high_value_df,
        "group_summary": group_summary,
        "contingency_table": contingency,
        "model": model,
        "result": result,
    }


def run_h2_robustness():
    """Run H2 under alternative value cutoffs with engagement held fixed."""
    base_df, primary_loader_threshold = prepare_analysis_data()

    analyses = {}
    result_rows = []
    previous_high_value_n = None

    for cutoff_name, quantile in VALUE_CUTOFFS:
        analysis = analyze_cutoff(base_df, cutoff_name, quantile)
        analyses[quantile] = analysis
        result_rows.append(analysis["result"])

        current_n = analysis["result"]["high_value_n"]
        if previous_high_value_n is not None and current_n > previous_high_value_n:
            raise RuntimeError(
                "High-value sample size increased under a stricter cutoff."
            )
        previous_high_value_n = current_n

    results_df = pd.DataFrame(result_rows)
    primary_cutoff = base_df["order_unit_price"].quantile(PRIMARY_QUANTILE)
    if not np.isclose(primary_cutoff, primary_loader_threshold):
        raise RuntimeError(
            "The primary median cutoff differs from the main H1/H2 loader."
        )

    print("=" * 120)
    print("H2 VALUE-CUTOFF ROBUSTNESS: ENGAGEMENT DEFINITION HELD FIXED")
    print("=" * 120)
    print(f"Base third-purchase cohort: {len(base_df)}")
    print(
        "Fixed engagement rule: low = 0 signals; high = at least 1 of "
        "review writing or push-notification consent."
    )
    print(
        "Only the order_unit_price cutoff changes; no continuous value "
        "control is included in the regression."
    )

    display_cols = [
        "cutoff_definition",
        "cutoff_value",
        "high_value_n",
        "HVHE_n",
        "HVLE_n",
        "HVHE_churn_percent",
        "HVLE_churn_percent",
        "churn_difference_pp",
        "HVLE_OR",
        "CI_lower",
        "CI_upper",
        "p_value",
    ]
    print("\nH2 results across value definitions:")
    print(
        results_df[display_cols].to_string(
            index=False,
            formatters={
                "cutoff_value": lambda x: f"{x:.2f}",
                "HVHE_churn_percent": lambda x: f"{x:.2f}",
                "HVLE_churn_percent": lambda x: f"{x:.2f}",
                "churn_difference_pp": lambda x: f"{x:.2f}",
                "HVLE_OR": lambda x: f"{x:.4f}",
                "CI_lower": lambda x: f"{x:.4f}",
                "CI_upper": lambda x: f"{x:.4f}",
                "p_value": lambda x: f"{x:.6g}",
            },
        )
    )

    primary_result = results_df.loc[
        np.isclose(results_df["value_quantile"], PRIMARY_QUANTILE)
    ].iloc[0]
    primary_supported = (
        primary_result["HVLE_OR"] > 1
        and primary_result["p_value"] < 0.05
    )
    direction_stable = bool((results_df["HVLE_OR"] > 1).all())
    significance_stable = bool((results_df["p_value"] < 0.05).all())

    print("\nPrimary H2 decision (median cutoff):")
    if primary_supported:
        print(
            "H2 is supported: HVLE customers have significantly higher "
            "fourth-purchase churn odds than HVHE customers."
        )
    else:
        print("H2 is not supported at the prespecified median cutoff.")

    print("\nValue-cutoff sensitivity conclusion:")
    print(f"Direction stable across all cutoffs: {direction_stable}")
    print(f"Statistically significant across all cutoffs: {significance_stable}")

    return {
        "base_df": base_df,
        "cutoff_analyses": analyses,
        "results_df": results_df,
        "primary_quantile": PRIMARY_QUANTILE,
        "primary_supported": primary_supported,
        "direction_stable": direction_stable,
        "significance_stable": significance_stable,
    }


if __name__ == "__main__":
    run_h2_robustness()
