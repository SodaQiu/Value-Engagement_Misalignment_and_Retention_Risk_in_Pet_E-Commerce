"""Additional H1 analyses for observable platform relationship signals.

This script keeps the existing H1 definitions unchanged and adds:
1. categorical 0/1/2 signal-count models,
2. four signal-configuration adjusted probabilities,
3. incremental diagnostic value beyond early transaction value.
"""

from pathlib import Path
import contextlib
import io
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import build_design_matrices
from scipy.stats import chi2
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold


PROJECT_DIR = Path(__file__).resolve().parents[2]
STUDY_1_DIR = PROJECT_DIR / "Study_1" / "Hypothesis Exam"
RANDOM_STATE = 42
N_SPLITS = 5

if str(STUDY_1_DIR) not in sys.path:
    sys.path.append(str(STUDY_1_DIR))

from quadrant_utils import load_hypothesis_data 


SIGNAL_ORDER = ["None", "Review only", "Push only", "Both"]


def format_p_value(value):
    if value < 0.001:
        return "< .001"
    return f"{value:.4f}"


def format_ci(row):
    return f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]"


def make_or_table(model, model_name, contrast_prefix=None):
    conf = model.conf_int()
    table = pd.DataFrame(
        {
            "model": model_name,
            "predictor": model.params.index,
            "beta": model.params.values,
            "odds_ratio": np.exp(model.params.values),
            "ci_lower": np.exp(conf[0].values),
            "ci_upper": np.exp(conf[1].values),
            "p_value": model.pvalues.values,
        }
    )
    if contrast_prefix is not None:
        table = table[table["predictor"].str.contains(contrast_prefix, regex=False)].copy()
    return table.reset_index(drop=True)


def likelihood_ratio_test(reduced_model, full_model):
    df_diff = int(full_model.df_model - reduced_model.df_model)
    lr_chi2 = 2 * (full_model.llf - reduced_model.llf)
    return {
        "lr_chi2": lr_chi2,
        "df": df_diff,
        "p_value": chi2.sf(lr_chi2, df_diff),
        "ll_reduced": reduced_model.llf,
        "ll_full": full_model.llf,
    }


def prepare_data():
    with contextlib.redirect_stdout(io.StringIO()):
        loaded = load_hypothesis_data()

    df = loaded["analysis_df"].copy()
    df["engagement_count"] = df["engagement_count"].astype(int)
    df["log_order_unit_price"] = np.log1p(df["order_unit_price"])

    conditions = [
        (df["review_written_yn"] == 0) & (df["push_notification_consent_yn"] == 0),
        (df["review_written_yn"] == 1) & (df["push_notification_consent_yn"] == 0),
        (df["review_written_yn"] == 0) & (df["push_notification_consent_yn"] == 1),
        (df["review_written_yn"] == 1) & (df["push_notification_consent_yn"] == 1),
    ]
    df["signal_config"] = np.select(conditions, SIGNAL_ORDER, default=np.nan)
    df["signal_config"] = pd.Categorical(
        df["signal_config"],
        categories=SIGNAL_ORDER,
        ordered=True,
    )

    return df


def observed_rate_table(df, group_col, order=None):
    summary = (
        df.groupby(group_col, observed=True)
        .agg(
            n=("churn_yn", "size"),
            noncompletion_n=("churn_yn", "sum"),
            observed_noncompletion_rate=("churn_yn", "mean"),
        )
        .reset_index()
    )
    if order is not None:
        summary[group_col] = pd.Categorical(summary[group_col], categories=order, ordered=True)
        summary = summary.sort_values(group_col).reset_index(drop=True)
    return summary


def categorical_signal_count_models(df):
    unadjusted = smf.logit(
        "churn_yn ~ C(engagement_count, Treatment(reference=0))",
        data=df,
    ).fit(disp=False)
    adjusted = smf.logit(
        "churn_yn ~ C(engagement_count, Treatment(reference=0)) + log_order_unit_price",
        data=df,
    ).fit(disp=False)

    observed = observed_rate_table(df, "engagement_count", order=[0, 1, 2])
    or_table = pd.concat(
        [
            make_or_table(
                unadjusted,
                "Unadjusted categorical count",
                "C(engagement_count, Treatment(reference=0))",
            ),
            make_or_table(
                adjusted,
                "Value-adjusted categorical count",
                "C(engagement_count, Treatment(reference=0))",
            ),
        ],
        ignore_index=True,
    )
    or_table["contrast"] = or_table["predictor"].map(
        {
            "C(engagement_count, Treatment(reference=0))[T.1]": "1 signal vs 0 signals",
            "C(engagement_count, Treatment(reference=0))[T.2]": "2 signals vs 0 signals",
        }
    )
    direct_contrast = direct_two_vs_one_contrast(adjusted)
    return observed, or_table, direct_contrast, unadjusted, adjusted


def direct_two_vs_one_contrast(model):
    term_1 = "C(engagement_count, Treatment(reference=0))[T.1]"
    term_2 = "C(engagement_count, Treatment(reference=0))[T.2]"
    beta = model.params[term_2] - model.params[term_1]
    cov = model.cov_params()
    variance = (
        cov.loc[term_2, term_2]
        + cov.loc[term_1, term_1]
        - 2 * cov.loc[term_2, term_1]
    )
    se = float(np.sqrt(variance))
    z_value = beta / se
    p_value = chi2.sf(z_value ** 2, 1)
    ci_low = beta - 1.96 * se
    ci_high = beta + 1.96 * se
    return pd.DataFrame(
        [
            {
                "model": "Value-adjusted categorical count",
                "contrast": "2 signals vs 1 signal",
                "beta": beta,
                "odds_ratio": np.exp(beta),
                "ci_lower": np.exp(ci_low),
                "ci_upper": np.exp(ci_high),
                "p_value": p_value,
            }
        ]
    )
 

def adjusted_probability_for_group(model, df, group_value):
    design_info = model.model.data.design_info
    scenario = df.copy()
    scenario["signal_config"] = group_value
    scenario["signal_config"] = pd.Categorical(
        scenario["signal_config"],
        categories=SIGNAL_ORDER,
        ordered=True,
    )
    design = build_design_matrices([design_info], scenario, return_type="dataframe")[0]
    x_matrix = design.to_numpy()
    params = model.params.to_numpy()
    cov = model.cov_params().to_numpy()

    eta = x_matrix @ params
    pred = 1 / (1 + np.exp(-eta))
    probability = pred.mean()
    gradient = (pred * (1 - pred))[:, None] * x_matrix
    gradient = gradient.mean(axis=0)
    se = float(np.sqrt(gradient @ cov @ gradient.T))
    ci_low = max(0.0, probability - 1.96 * se)
    ci_high = min(1.0, probability + 1.96 * se)
    return probability, ci_low, ci_high


def signal_configuration_model(df):
    model = smf.logit(
        "churn_yn ~ C(signal_config, Treatment(reference='None')) + log_order_unit_price",
        data=df,
    ).fit(disp=False)

    observed = observed_rate_table(df, "signal_config", order=SIGNAL_ORDER)
    adjusted_rows = []
    for group in SIGNAL_ORDER:
        probability, ci_low, ci_high = adjusted_probability_for_group(model, df, group)
        adjusted_rows.append(
            {
                "signal_config": group,
                "adjusted_noncompletion_probability": probability,
                "ci_lower": ci_low,
                "ci_upper": ci_high,
            }
        )
    probability_table = observed.merge(pd.DataFrame(adjusted_rows), on="signal_config")

    or_table = make_or_table(
        model,
        "Value-adjusted signal configuration",
        "C(signal_config, Treatment(reference='None'))",
    )
    or_table["contrast"] = or_table["predictor"].str.extract(r"\[T\.(.*)\]")[0] + " vs None"
    return probability_table, or_table, model


def cross_validated_predictions(df, formulas):
    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    predictions = {
        label: pd.Series(index=df.index, dtype=float)
        for label in formulas
    }

    for train_idx, test_idx in cv.split(df, df["churn_yn"]):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        for label, formula in formulas.items():
            fold_model = smf.logit(formula, data=train_df).fit(disp=False)
            predictions[label].iloc[test_idx] = fold_model.predict(test_df)

    for label, values in predictions.items():
        if values.isna().any():
            raise RuntimeError(f"Missing cross-validated predictions for {label}.")
    return predictions


def model_performance(model, label, df, cv_pred):
    return {
        "model": label,
        "n": len(df),
        "auc_roc_cv": roc_auc_score(df["churn_yn"], cv_pred),
        "brier_score_cv": brier_score_loss(df["churn_yn"], cv_pred),
        "log_likelihood": model.llf,
        "df_model": model.df_model,
    }


def incremental_value_models(df):
    formulas = {
        "Model 1: value only": "churn_yn ~ log_order_unit_price",
        "Model 2: value + signal configuration": (
            "churn_yn ~ log_order_unit_price + "
            "C(signal_config, Treatment(reference='None'))"
        ),
        "Secondary: value + categorical count": (
            "churn_yn ~ log_order_unit_price + "
            "C(engagement_count, Treatment(reference=0))"
        ),
    }
    value_only = smf.logit(
        formulas["Model 1: value only"],
        data=df,
    ).fit(disp=False)
    value_plus_config = smf.logit(
        formulas["Model 2: value + signal configuration"],
        data=df,
    ).fit(disp=False)
    value_plus_count = smf.logit(
        formulas["Secondary: value + categorical count"],
        data=df,
    ).fit(disp=False)
    cv_predictions = cross_validated_predictions(df, formulas)

    performance = pd.DataFrame(
        [
            model_performance(
                value_only,
                "Model 1: value only",
                df,
                cv_predictions["Model 1: value only"],
            ),
            model_performance(
                value_plus_config,
                "Model 2: value + signal configuration",
                df,
                cv_predictions["Model 2: value + signal configuration"],
            ),
            model_performance(
                value_plus_count,
                "Secondary: value + categorical count",
                df,
                cv_predictions["Secondary: value + categorical count"],
            ),
        ]
    )
    baseline = performance.iloc[0]
    comparison = performance.iloc[1:].copy()
    comparison["delta_auc_cv"] = comparison["auc_roc_cv"] - baseline["auc_roc_cv"]
    comparison["delta_brier_cv"] = (
        comparison["brier_score_cv"] - baseline["brier_score_cv"]
    )

    lr_tests = pd.DataFrame(
        [
            {
                "comparison": "Value only vs value + signal configuration",
                "n_reduced": int(value_only.nobs),
                "n_full": int(value_plus_config.nobs),
                **likelihood_ratio_test(value_only, value_plus_config),
            },
            {
                "comparison": "Value only vs value + categorical count",
                "n_reduced": int(value_only.nobs),
                "n_full": int(value_plus_count.nobs),
                **likelihood_ratio_test(value_only, value_plus_count),
            },
        ]
    )
    return performance, comparison, lr_tests, value_only, value_plus_config, value_plus_count


def validation_checks(df, lr_tests):
    engagement_matches = (
        df["engagement_count"]
        == df["review_written_yn"] + df["push_notification_consent_yn"]
    ).all()
    binary_valid = (
        df["review_written_yn"].isin([0, 1]).all()
        and df["push_notification_consent_yn"].isin([0, 1]).all()
    )
    lr_same_n = (lr_tests["n_reduced"] == lr_tests["n_full"]).all()

    return pd.DataFrame(
        [
            {
                "check": "engagement_count equals review_written_yn + push_notification_consent_yn",
                "passed": bool(engagement_matches),
            },
            {
                "check": "review_written_yn and push_notification_consent_yn contain only 0/1",
                "passed": bool(binary_valid),
            },
            {
                "check": "nested likelihood-ratio models use identical N",
                "passed": bool(lr_same_n),
            },
        ]
    )


def print_or_table(title, table):
    display = table.copy()
    display["95% CI"] = display.apply(format_ci, axis=1)
    display["p"] = display["p_value"].map(format_p_value)
    display = display[["model", "contrast", "beta", "odds_ratio", "95% CI", "p"]]
    print("\n" + title)
    print(
        display.to_string(
            index=False,
            formatters={
                "beta": lambda value: f"{value:.4f}",
                "odds_ratio": lambda value: f"{value:.4f}",
            },
        )
    )


def print_probability_table(title, table):
    display = table.copy()
    display["observed_noncompletion_rate"] *= 100
    display["adjusted_noncompletion_probability"] *= 100
    display["95% CI"] = (
        display["ci_lower"].map(lambda value: f"{value * 100:.2f}")
        + "-"
        + display["ci_upper"].map(lambda value: f"{value * 100:.2f}")
    )
    print("\n" + title)
    print(
        display[
            [
                "signal_config",
                "n",
                "observed_noncompletion_rate",
                "adjusted_noncompletion_probability",
                "95% CI",
            ]
        ].to_string(
            index=False,
            formatters={
                "observed_noncompletion_rate": lambda value: f"{value:.2f}",
                "adjusted_noncompletion_probability": lambda value: f"{value:.2f}",
            },
        )
    )


def print_incremental_summary(performance, comparison, lr_tests):
    display_perf = performance.copy()
    print("\nIncremental diagnostic value")
    print(
        display_perf[["model", "n", "auc_roc_cv", "brier_score_cv", "log_likelihood"]]
        .to_string(
            index=False,
            formatters={
                "auc_roc_cv": lambda value: f"{value:.4f}",
                "brier_score_cv": lambda value: f"{value:.4f}",
                "log_likelihood": lambda value: f"{value:.4f}",
            },
        )
    )

    display_comp = comparison.copy()
    print("\nIncremental change relative to value-only model")
    print(
        display_comp[["model", "delta_auc_cv", "delta_brier_cv"]].to_string(
            index=False,
            formatters={
                "delta_auc_cv": lambda value: f"{value:.4f}",
                "delta_brier_cv": lambda value: f"{value:.4f}",
            },
        )
    )

    display_lr = lr_tests.copy()
    display_lr["p"] = display_lr["p_value"].map(format_p_value)
    print("\nLikelihood-ratio tests")
    print(
        display_lr[["comparison", "lr_chi2", "df", "p"]].to_string(
            index=False,
            formatters={"lr_chi2": lambda value: f"{value:.4f}"},
        )
    )


def main():
    df = prepare_data()

    count_observed, count_or, count_direct, count_unadj, count_adj = (
        categorical_signal_count_models(df)
    )
    config_prob, config_or, config_model = signal_configuration_model(df)
    performance, comparison, lr_tests, value_only, value_config, value_count = (
        incremental_value_models(df)
    )
    checks = validation_checks(df, lr_tests)

    print("=" * 80)
    print("Additional H1 analyses: observable platform relationship signals")
    print("=" * 80)
    print("Analysis sample:", len(df))
    print("Outcome: fourth-purchase non-completion = 1")
    print("Value covariate: log_order_unit_price = ln(1 + order_unit_price)")

    count_display = count_observed.copy()
    count_display["observed_noncompletion_rate"] *= 100
    print("\nObserved non-completion by signal count")
    print(
        count_display.to_string(
            index=False,
            formatters={"observed_noncompletion_rate": lambda value: f"{value:.2f}"},
        )
    )
    print_or_table("Categorical signal-count logistic models", count_or)
    print_or_table("Direct adjusted contrast", count_direct)

    print_probability_table("Signal-configuration adjusted probabilities", config_prob)
    print_or_table("Signal-configuration logistic model", config_or)

    print_incremental_summary(performance, comparison, lr_tests)

    print("\nValidation checks")
    print(checks.to_string(index=False))


if __name__ == "__main__":
    main()
