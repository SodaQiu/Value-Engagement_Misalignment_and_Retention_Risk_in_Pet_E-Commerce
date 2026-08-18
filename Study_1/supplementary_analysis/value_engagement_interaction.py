# ============================================================
# Study 1 supplemental test:
# Value-engagement misalignment and fourth-purchase outcome
#
# Purpose:
# Directly assess whether early transaction value and observable
# engagement jointly predict fourth-purchase completion beyond the
# main effect of engagement alone.
#
# DV:
# churn_yn = 1 if the customer did not complete purchase 4
# churn_yn = 0 if the customer completed purchase 4
#
# Main model:
# churn_yn ~ centered_log_order_unit_price * high_engagement
#
# Reported probabilities:
# Because the fitted model predicts churn, fourth-purchase probability
# is reported as 1 - predicted churn probability.
# ============================================================

import warnings
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2
from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationError

PROJECT_DIR = Path(__file__).resolve().parents[2]
STUDY_1_DIR = PROJECT_DIR / "Study_1" / "Hypothesis Exam"

if str(STUDY_1_DIR) not in sys.path:
    sys.path.append(str(STUDY_1_DIR))

from quadrant_utils import load_hypothesis_data


QUADRANT_ORDER = ["HVHE", "HVLE", "LVHE", "LVLE"]


def format_p_value(value):
    if value < 0.001:
        return "< .001"
    return f"{value:.4f}"


def make_or_table(model):
    conf = model.conf_int()
    return pd.DataFrame({
        "predictor": model.params.index,
        "beta": model.params.values,
        "odds_ratio": np.exp(model.params.values),
        "ci_lower": np.exp(conf[0].values),
        "ci_upper": np.exp(conf[1].values),
        "p_value": model.pvalues.values,
    })


def fit_logit(formula, data, model_name):
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model = smf.logit(
                formula=formula,
                data=data,
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


def likelihood_ratio_test(full_model, reduced_model, df_diff):
    lr_chi2 = 2 * (full_model.llf - reduced_model.llf)
    return {
        "lr_chi2": lr_chi2,
        "df": df_diff,
        "p_value": chi2.sf(lr_chi2, df_diff),
    }


def prepare_analysis_data():
    loaded = load_hypothesis_data()
    df = loaded["analysis_df"].copy()

    required_cols = [
        "churn_yn",
        "order_unit_price",
        "engagement_count",
        "high_engagement",
        "high_value",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for supplemental test: {missing}")

    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required_cols).copy()
    df = df[df["order_unit_price"] > 0].copy()

    for col in ["churn_yn", "high_engagement", "high_value"]:
        invalid = df.loc[~df[col].isin([0, 1]), col]
        if not invalid.empty:
            raise ValueError(
                f"{col} must contain only 0/1 values; "
                f"invalid={sorted(invalid.unique().tolist())}"
            )

    df["log_order_unit_price"] = np.log1p(df["order_unit_price"])
    df["centered_log_order_unit_price"] = (
        df["log_order_unit_price"]
        - df["log_order_unit_price"].mean()
    )

    df["quadrant"] = np.select(
        [
            (df["high_value"] == 1) & (df["high_engagement"] == 1),
            (df["high_value"] == 1) & (df["high_engagement"] == 0),
            (df["high_value"] == 0) & (df["high_engagement"] == 1),
            (df["high_value"] == 0) & (df["high_engagement"] == 0),
        ],
        QUADRANT_ORDER,
        default="Undefined",
    )

    if (df["quadrant"] == "Undefined").any():
        raise ValueError("Some rows could not be assigned to a quadrant.")

    return df


def summarize_observed_quadrants(df):
    summary = (
        df.groupby("quadrant", observed=True)
        .agg(
            n=("churn_yn", "size"),
            churn_n=("churn_yn", "sum"),
            observed_churn_probability=("churn_yn", "mean"),
            median_order_unit_price=("order_unit_price", "median"),
            median_log_order_unit_price=("log_order_unit_price", "median"),
        )
        .reindex(QUADRANT_ORDER)
        .reset_index()
    )

    summary["observed_fourth_purchase_probability"] = (
        1 - summary["observed_churn_probability"]
    )

    return summary


def predict_quadrant_probabilities(model, df):
    prediction_df = df.copy()
    prediction_df["predicted_churn_probability"] = model.predict(prediction_df)
    prediction_df["predicted_fourth_purchase_probability"] = (
        1 - prediction_df["predicted_churn_probability"]
    )

    adjusted = (
        prediction_df.groupby("quadrant", observed=True)
        .agg(
            n=("churn_yn", "size"),
            adjusted_churn_probability=(
                "predicted_churn_probability",
                "mean",
            ),
            adjusted_fourth_purchase_probability=(
                "predicted_fourth_purchase_probability",
                "mean",
            ),
        )
        .reindex(QUADRANT_ORDER)
        .reset_index()
    )

    return adjusted


def run_misalignment_supplemental_test():
    df = prepare_analysis_data()

    full_formula = (
        "churn_yn ~ centered_log_order_unit_price * high_engagement"
    )
    reduced_formula = (
        "churn_yn ~ centered_log_order_unit_price + high_engagement"
    )

    full_model = fit_logit(full_formula, df, "interaction model")
    reduced_model = fit_logit(reduced_formula, df, "main-effects model")
    lr_result = likelihood_ratio_test(full_model, reduced_model, df_diff=1)

    or_table = make_or_table(full_model)
    interaction_term = "centered_log_order_unit_price:high_engagement"
    interaction_row = or_table.loc[
        or_table["predictor"] == interaction_term
    ].iloc[0]

    observed_summary = summarize_observed_quadrants(df)
    adjusted_summary = predict_quadrant_probabilities(full_model, df)

    quadrant_summary = observed_summary.merge(
        adjusted_summary,
        on=["quadrant", "n"],
        how="left",
    )

    print("=" * 90)
    print("STUDY 1 SUPPLEMENTAL TEST: VALUE-ENGAGEMENT MISALIGNMENT")
    print("=" * 90)
    print(f"Analysis sample size: {len(df)}")
    print(
        "Model: churn_yn ~ centered_log_order_unit_price * high_engagement"
    )
    print(
        "Value transformation: log_order_unit_price was mean-centered "
        "before creating the interaction term."
    )
    print(
        "Engagement definition: high_engagement = 1 if engagement_count >= 1; "
        "0 if engagement_count = 0."
    )
    print(
        "Value definition for quadrants: high_value = 1 if order_unit_price "
        "is at or above the Study 1 analysis-sample median."
    )

    print("\nLogistic regression odds-ratio table")
    print(
        or_table.to_string(
            index=False,
            formatters={
                "beta": lambda x: f"{x:.4f}",
                "odds_ratio": lambda x: f"{x:.4f}",
                "ci_lower": lambda x: f"{x:.4f}",
                "ci_upper": lambda x: f"{x:.4f}",
                "p_value": lambda x: format_p_value(x),
            },
        )
    )

    print("\nInteraction test")
    print(
        "Term: centered_log_order_unit_price x high_engagement\n"
        f"Beta = {interaction_row['beta']:.4f}, "
        f"OR = {interaction_row['odds_ratio']:.4f}, "
        f"p = {format_p_value(interaction_row['p_value'])}"
    )
    print(
        "Likelihood-ratio test comparing interaction model to main-effects "
        f"model: chi2({lr_result['df']}) = {lr_result['lr_chi2']:.4f}, "
        f"p = {format_p_value(lr_result['p_value'])}"
    )

    print("\nObserved and model-adjusted quadrant probabilities")
    display_cols = [
        "quadrant",
        "n",
        "median_order_unit_price",
        "observed_fourth_purchase_probability",
        "adjusted_fourth_purchase_probability",
        "observed_churn_probability",
        "adjusted_churn_probability",
    ]
    print(
        quadrant_summary[display_cols].to_string(
            index=False,
            formatters={
                "median_order_unit_price": lambda x: f"{x:.2f}",
                "observed_fourth_purchase_probability": lambda x: f"{x:.4f}",
                "adjusted_fourth_purchase_probability": lambda x: f"{x:.4f}",
                "observed_churn_probability": lambda x: f"{x:.4f}",
                "adjusted_churn_probability": lambda x: f"{x:.4f}",
            },
        )
    )

    if interaction_row["p_value"] < 0.05:
        print(
            "\nInterpretation: The interaction is statistically significant, "
            "supporting the view that the engagement-outcome association "
            "varies by early transaction value."
        )
    else:
        print(
            "\nInterpretation: The interaction is not statistically "
            "significant. The four-quadrant results should therefore be "
            "interpreted as a managerially useful diagnostic segmentation "
            "rather than as evidence of a statistical interaction effect."
        )

    return {
        "analysis_df": df,
        "model": full_model,
        "main_effects_model": reduced_model,
        "or_table": or_table,
        "interaction_lr_test": lr_result,
        "quadrant_summary": quadrant_summary,
    }


if __name__ == "__main__":
    run_misalignment_supplemental_test()
