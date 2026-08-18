"""
Supplemental H1 test: zero versus two observable signals.

This script keeps the same H1 analysis sample as hypothesis_exam_H1.py, but
uses a stricter contrast by comparing only customers with engagement_count = 0
against customers with engagement_count = 2.

DV:
churn_yn = 1 if fourth-purchase non-completion is observed
churn_yn = 0 otherwise

Main predictor:
both_engagement_signals_yn = 1 if engagement_count == 2
both_engagement_signals_yn = 0 if engagement_count == 0
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


PROJECT_DIR = Path(__file__).resolve().parents[2]
STUDY_1_DIR = PROJECT_DIR / "Study_1" / "Hypothesis Exam"

if str(STUDY_1_DIR) not in sys.path:
    sys.path.append(str(STUDY_1_DIR))

from quadrant_utils import load_hypothesis_data


def logistic_result(model, predictor):
    """Return the main logistic-regression statistics for one predictor."""
    coefficient = model.params[predictor]
    ci_low, ci_high = model.conf_int().loc[predictor]

    return {
        "coefficient": coefficient,
        "odds_ratio": np.exp(coefficient),
        "ci_low": np.exp(ci_low),
        "ci_high": np.exp(ci_high),
        "p_value": model.pvalues[predictor],
    }


def fit_logit(formula, data, predictor, model_name):
    model = smf.logit(
        formula=formula,
        data=data,
    ).fit(disp=False)

    result = logistic_result(model, predictor)

    print("\n" + "=" * 80)
    print(model_name)
    print("=" * 80)
    print("Formula:", formula)
    print("N:", len(data))
    print(f"Coefficient: {result['coefficient']:.4f}")
    print(f"Odds Ratio: {result['odds_ratio']:.4f}")
    print(
        "95% CI for OR: "
        f"[{result['ci_low']:.4f}, {result['ci_high']:.4f}]"
    )
    print(f"P-value: {result['p_value']:.6g}")

    return {
        "model": model,
        "result": result,
    }


def run_h1_zero_vs_two_signals():
    data = load_hypothesis_data()
    analysis_df = data["analysis_df"].copy()

    required_cols = [
        "churn_yn",
        "engagement_count",
        "order_unit_price",
    ]
    missing_cols = [
        column
        for column in required_cols
        if column not in analysis_df.columns
    ]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    for column in required_cols:
        analysis_df[column] = pd.to_numeric(
            analysis_df[column],
            errors="coerce",
        )

    middle_signal_n = int(
        analysis_df["engagement_count"].eq(1).sum()
    )

    contrast_df = analysis_df[
        analysis_df["engagement_count"].isin([0, 2])
    ].copy()
    contrast_df = contrast_df.dropna(subset=required_cols).copy()
    contrast_df = contrast_df[contrast_df["order_unit_price"] > 0].copy()

    if set(contrast_df["engagement_count"].unique()) != {0, 2}:
        raise ValueError(
            "The stricter H1 contrast requires both engagement_count = 0 "
            "and engagement_count = 2."
        )

    if set(contrast_df["churn_yn"].unique()) != {0, 1}:
        raise ValueError(
            "The stricter H1 contrast requires both churn outcome classes."
        )

    contrast_df["both_engagement_signals_yn"] = (
        contrast_df["engagement_count"] == 2
    ).astype(int)
    contrast_df["signal_group"] = contrast_df[
        "both_engagement_signals_yn"
    ].map({
        0: "no_observable_signals",
        1: "both_observable_signals",
    })
    contrast_df["log_order_unit_price"] = np.log1p(
        contrast_df["order_unit_price"]
    )

    group_summary = (
        contrast_df
        .groupby("signal_group", observed=True)
        .agg(
            n=("churn_yn", "size"),
            noncompletion_n=("churn_yn", "sum"),
            noncompletion_rate=("churn_yn", "mean"),
        )
        .reindex(["no_observable_signals", "both_observable_signals"])
        .reset_index()
    )
    group_summary["noncompletion_rate_percent"] = (
        group_summary["noncompletion_rate"] * 100
    )
    group_summary["completion_rate_percent"] = (
        (1 - group_summary["noncompletion_rate"]) * 100
    )

    print("=" * 80)
    print(
        "H1 SUPPLEMENTAL TEST: 0 VS 2 OBSERVABLE "
        "ENGAGEMENT-RELATED SIGNALS"
    )
    print("=" * 80)
    print("Original H1 analysis sample:", len(analysis_df))
    print("Stricter contrast sample:", len(contrast_df))
    print(
        "Excluded middle-signal rows "
        "(engagement_count = 1):",
        middle_signal_n,
    )
    print(
        "Reference group: no_observable_signals "
        "(engagement_count = 0)"
    )
    print(
        "Comparison group: both_observable_signals "
        "(engagement_count = 2)"
    )

    print("\nGROUP SUMMARY")
    print(
        group_summary[
            [
                "signal_group",
                "n",
                "noncompletion_n",
                "noncompletion_rate_percent",
                "completion_rate_percent",
            ]
        ]
        .round(2)
        .to_string(index=False)
    )

    unadjusted = fit_logit(
        formula="churn_yn ~ both_engagement_signals_yn",
        data=contrast_df,
        predictor="both_engagement_signals_yn",
        model_name="MODEL 1: UNADJUSTED 0 VS 2 SIGNALS",
    )

    value_adjusted = fit_logit(
        formula=(
            "churn_yn ~ both_engagement_signals_yn "
            "+ log_order_unit_price"
        ),
        data=contrast_df,
        predictor="both_engagement_signals_yn",
        model_name="MODEL 2: VALUE-ADJUSTED 0 VS 2 SIGNALS",
    )

    summary = pd.DataFrame([
        {
            "model": "Model 1: Unadjusted",
            **unadjusted["result"],
        },
        {
            "model": "Model 2: Value adjusted",
            **value_adjusted["result"],
        },
    ])

    print("\n" + "=" * 80)
    print("STRICTER H1 CONTRAST SUMMARY: 0 VS 2 SIGNALS")
    print("=" * 80)
    print(
        summary[
            [
                "model",
                "coefficient",
                "odds_ratio",
                "ci_low",
                "ci_high",
                "p_value",
            ]
        ]
        .to_string(
            index=False,
            formatters={
                "coefficient": lambda value: f"{value:.4f}",
                "odds_ratio": lambda value: f"{value:.4f}",
                "ci_low": lambda value: f"{value:.4f}",
                "ci_high": lambda value: f"{value:.4f}",
                "p_value": lambda value: f"{value:.6g}",
            },
        )
    )

    supported = (
        unadjusted["result"]["coefficient"] < 0
        and unadjusted["result"]["p_value"] < 0.05
        and value_adjusted["result"]["coefficient"] < 0
        and value_adjusted["result"]["p_value"] < 0.05
    )

    print("\nSTRICTER H1 CONTRAST DECISION")
    if supported:
        print(
            "The stricter H1 contrast supports H1: customers with both "
            "observable engagement-related signals have significantly "
            "lower odds of fourth-purchase non-completion than customers "
            "with no observable engagement-related signals, including "
            "after controlling for early transaction value."
        )
    else:
        print(
            "The stricter H1 contrast does not fully support H1. Review "
            "the unadjusted and value-adjusted estimates."
        )

    return {
        "contrast_df": contrast_df,
        "group_summary": group_summary,
        "unadjusted": unadjusted,
        "value_adjusted": value_adjusted,
        "summary": summary,
        "supported": supported,
    }


if __name__ == "__main__":
    run_h1_zero_vs_two_signals()
