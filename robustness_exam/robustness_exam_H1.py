import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


PROJECT_DIR = Path(__file__).resolve().parents[1]
STUDY_1_DIR = PROJECT_DIR / "Study_1" / "Hypothesis Exam"

if str(STUDY_1_DIR) not in sys.path:
    sys.path.append(str(STUDY_1_DIR))

from quadrant_utils import load_hypothesis_data


def make_or_table(model):
    params = model.params
    conf = model.conf_int()

    return pd.DataFrame({
        "predictor": params.index,
        "beta": params.values,
        "odds_ratio": np.exp(params.values),
        "ci_lower": np.exp(conf[0].values),
        "ci_upper": np.exp(conf[1].values),
        "p_value": model.pvalues.values
    })


def fit_signal_model(df, signal_col, adjusted):
    model_name = (
        f"{signal_col}: value-adjusted"
        if adjusted
        else f"{signal_col}: unadjusted"
    )

    model_cols = [
        "churn_yn",
        signal_col
    ]

    formula = f"churn_yn ~ {signal_col}"

    if adjusted:
        model_cols.append("log_order_unit_price")
        formula += " + log_order_unit_price"

    model_df = df.dropna(subset=model_cols).copy()
    model = smf.logit(formula=formula, data=model_df).fit(disp=False)
    or_table = make_or_table(model)

    signal_row = or_table[
        or_table["predictor"] == signal_col
    ].copy()

    signal_row.insert(0, "model", model_name)
    signal_row.insert(1, "n", len(model_df))

    print("\n" + "-" * 80)
    print(model_name)
    print("N:", len(model_df))
    print("Formula:", formula)
    print("\nOdds ratios:")
    print(or_table.round(4).to_string(index=False))

    return {
        "model": model,
        "or_table": or_table,
        "key_result": signal_row
    }


def run_h1_robustness():
    """
    H1 robustness:
    Replace engagement_count with each observable engagement signal.

    This tests whether the composite engagement_count result is robust
    to using review writing and push notification consent separately.
    """

    data = load_hypothesis_data()
    df = data["analysis_df"].copy()

    required_cols = [
        "churn_yn",
        "order_unit_price",
        "review_written_yn",
        "push_notification_consent_yn"
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["order_unit_price"] > 0].copy()
    df["log_order_unit_price"] = np.log1p(df["order_unit_price"])

    print("=" * 80)
    print("=== H1 Robustness: Alternative Engagement Definitions ===")
    print("=" * 80)
    print("Prepared analysis sample size:", len(df))

    signal_cols = [
        "review_written_yn",
        "push_notification_consent_yn"
    ]

    results = {}
    key_rows = []

    for signal_col in signal_cols:
        for adjusted in [False, True]:
            result = fit_signal_model(df, signal_col, adjusted)
            results[(signal_col, adjusted)] = result
            key_rows.append(result["key_result"])

    key_results = pd.concat(key_rows, ignore_index=True)

    print("\n" + "=" * 80)
    print("H1 robustness summary")
    print("=" * 80)
    print(key_results.round(4).to_string(index=False))

    return {
        "analysis_df": df,
        "model_results": results,
        "key_results": key_results
    }


if __name__ == "__main__":
    run_h1_robustness()
