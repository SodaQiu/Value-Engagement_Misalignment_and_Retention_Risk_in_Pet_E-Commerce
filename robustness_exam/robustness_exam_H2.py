import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score
)
from sklearn.model_selection import StratifiedKFold


PROJECT_DIR = Path(__file__).resolve().parents[1]
STUDY_1_DIR = PROJECT_DIR / "Study_1" / "Hypothesis Exam"

if str(STUDY_1_DIR) not in sys.path:
    sys.path.append(str(STUDY_1_DIR))

from quadrant_utils import load_hypothesis_data


N_SPLITS = 5
RANDOM_STATE = 42


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


def fit_nested_models(df, value_proxy):
    model_cols = [
        "churn_yn",
        value_proxy,
        "engagement_count"
    ]

    model_df = df.dropna(subset=model_cols).copy()

    value_formula = f"churn_yn ~ {value_proxy}"
    engagement_formula = (
        f"churn_yn ~ {value_proxy} + engagement_count"
    )

    value_only_model = smf.logit(
        formula=value_formula,
        data=model_df
    ).fit(disp=False)

    value_engagement_model = smf.logit(
        formula=engagement_formula,
        data=model_df
    ).fit(disp=False)

    lr_stat = 2 * (
        value_engagement_model.llf
        - value_only_model.llf
    )

    lr_df = int(
        value_engagement_model.df_model
        - value_only_model.df_model
    )

    lr_p_value = chi2.sf(lr_stat, lr_df)

    engagement_or_table = make_or_table(value_engagement_model)
    engagement_row = engagement_or_table[
        engagement_or_table["predictor"] == "engagement_count"
    ].iloc[0]

    return {
        "model_df": model_df,
        "value_only_model": value_only_model,
        "value_engagement_model": value_engagement_model,
        "lr_stat": lr_stat,
        "lr_df": lr_df,
        "lr_p_value": lr_p_value,
        "engagement_or": engagement_row["odds_ratio"],
        "engagement_ci_lower": engagement_row["ci_lower"],
        "engagement_ci_upper": engagement_row["ci_upper"],
        "engagement_p_value": engagement_row["p_value"],
        "engagement_or_table": engagement_or_table
    }


def evaluate_oof_performance(model_df, value_proxy):
    y = model_df["churn_yn"].astype(int).to_numpy()

    x_value_only = model_df[[value_proxy]].astype(float)
    x_value_engagement = model_df[
        [
            value_proxy,
            "engagement_count"
        ]
    ].astype(float)

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    pred_value_only = np.full(len(model_df), np.nan)
    pred_value_engagement = np.full(len(model_df), np.nan)

    for train_idx, test_idx in cv.split(x_value_only, y):
        y_train = y[train_idx]

        x_train_value_only = sm.add_constant(
            x_value_only.iloc[train_idx],
            has_constant="add"
        )
        x_test_value_only = sm.add_constant(
            x_value_only.iloc[test_idx],
            has_constant="add"
        )

        cv_value_only_model = sm.Logit(
            y_train,
            x_train_value_only
        ).fit(disp=False)

        pred_value_only[test_idx] = cv_value_only_model.predict(
            x_test_value_only
        )

        x_train_value_engagement = sm.add_constant(
            x_value_engagement.iloc[train_idx],
            has_constant="add"
        )
        x_test_value_engagement = sm.add_constant(
            x_value_engagement.iloc[test_idx],
            has_constant="add"
        )

        cv_value_engagement_model = sm.Logit(
            y_train,
            x_train_value_engagement
        ).fit(disp=False)

        pred_value_engagement[test_idx] = (
            cv_value_engagement_model.predict(
                x_test_value_engagement
            )
        )

    if (
        np.isnan(pred_value_only).any()
        or np.isnan(pred_value_engagement).any()
    ):
        raise ValueError("Out-of-fold predictions contain missing values.")

    value_only_auc = roc_auc_score(y, pred_value_only)
    value_engagement_auc = roc_auc_score(y, pred_value_engagement)

    value_only_log_loss = log_loss(
        y,
        np.clip(pred_value_only, 1e-15, 1 - 1e-15)
    )
    value_engagement_log_loss = log_loss(
        y,
        np.clip(pred_value_engagement, 1e-15, 1 - 1e-15)
    )

    value_only_brier = brier_score_loss(y, pred_value_only)
    value_engagement_brier = brier_score_loss(
        y,
        pred_value_engagement
    )

    return {
        "value_only_auc": value_only_auc,
        "value_engagement_auc": value_engagement_auc,
        "auc_delta": value_engagement_auc - value_only_auc,
        "value_only_log_loss": value_only_log_loss,
        "value_engagement_log_loss": value_engagement_log_loss,
        "log_loss_delta": (
            value_only_log_loss
            - value_engagement_log_loss
        ),
        "value_only_brier": value_only_brier,
        "value_engagement_brier": value_engagement_brier,
        "brier_delta": value_only_brier - value_engagement_brier
    }


def run_h2_robustness():
    """
    H2 robustness:
    Test whether engagement_count improves model fit beyond the available
    transaction-value proxy, log_order_unit_price.
    """

    data = load_hypothesis_data()
    df = data["analysis_df"].copy()

    required_cols = [
        "churn_yn",
        "engagement_count",
        "order_unit_price"
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["order_unit_price"] > 0].copy()
    df["log_order_unit_price"] = np.log1p(df["order_unit_price"])

    print("=" * 80)
    print("=== H2 Robustness: Engagement Increment Beyond Value ===")
    print("=" * 80)
    print("Base analysis sample size:", len(df))

    summary_rows = []
    model_results = {}

    value_proxy = "log_order_unit_price"
    source_col = "order_unit_price"

    nested = fit_nested_models(df, value_proxy)
    performance = evaluate_oof_performance(
        nested["model_df"],
        value_proxy
    )

    row = {
        "value_proxy": value_proxy,
        "source_col": source_col,
        "n": len(nested["model_df"]),
        "lr_chi2": nested["lr_stat"],
        "lr_df": nested["lr_df"],
        "lr_p_value": nested["lr_p_value"],
        "engagement_or": nested["engagement_or"],
        "engagement_ci_lower": nested["engagement_ci_lower"],
        "engagement_ci_upper": nested["engagement_ci_upper"],
        "engagement_p_value": nested["engagement_p_value"],
        **performance
    }

    summary_rows.append(row)
    model_results[value_proxy] = {
        **nested,
        "performance": performance
    }

    print("\n" + "-" * 80)
    print(f"Value proxy: {value_proxy} from {source_col}")
    print("N:", row["n"])
    print(
        "LR test: "
        f"chi2 = {row['lr_chi2']:.4f}, "
        f"df = {row['lr_df']}, "
        f"p = {row['lr_p_value']:.6g}"
    )
    print(
        "engagement_count OR: "
        f"{row['engagement_or']:.4f}, "
        f"95% CI = "
        f"[{row['engagement_ci_lower']:.4f}, "
        f"{row['engagement_ci_upper']:.4f}], "
        f"p = {row['engagement_p_value']:.6g}"
    )

    summary_df = pd.DataFrame(summary_rows)

    print("\n" + "=" * 80)
    print("H2 robustness summary table")
    print("=" * 80)
    print(
        summary_df[
            [
                "value_proxy",
                "source_col",
                "n",
                "lr_chi2",
                "lr_p_value",
                "engagement_or",
                "engagement_ci_lower",
                "engagement_ci_upper",
                "engagement_p_value",
                "value_only_auc",
                "value_engagement_auc",
                "auc_delta",
                "log_loss_delta",
                "brier_delta"
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    return {
        "analysis_df": df,
        "model_results": model_results,
        "summary_df": summary_df
    }


if __name__ == "__main__":
    run_h2_robustness()
