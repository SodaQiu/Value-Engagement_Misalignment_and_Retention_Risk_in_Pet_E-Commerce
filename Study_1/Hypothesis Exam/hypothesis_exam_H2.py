# ============================================================
# H2 TEST
# Incremental value of observable early engagement signals
#
# Hypothesis:
# Compared with a value-only model, a model incorporating
# both early transaction value and observable early engagement
# signals provides better identification of fourth-purchase
# churn.
#
# DV:
# churn_yn = 1 if the user did not complete a fourth purchase
# churn_yn = 0 otherwise
#
# Model 1:
# churn_yn ~ log_order_unit_price
#
# Model 2:
# churn_yn ~ log_order_unit_price + engagement_count
# ============================================================

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

from quadrant_utils import load_hypothesis_data


# ------------------------------------------------------------
# 0. Settings
# ------------------------------------------------------------

N_SPLITS = 5
RANDOM_STATE = 42
N_BOOTSTRAP = 2000


# ------------------------------------------------------------
# 1. Load shared analysis dataset
# ------------------------------------------------------------

data = load_hypothesis_data()

h2_df = data["analysis_df"].copy()
file_path = Path(data["file_path"])
output_dir = file_path.parent

h2_df["log_order_unit_price"] = np.log1p(
    h2_df["order_unit_price"]
)

required_h2_cols = [
    "churn_yn",
    "order_unit_price",
    "log_order_unit_price",
    "engagement_count"
]

h2_df = (
    h2_df[required_h2_cols]
    .dropna()
    .copy()
)

if h2_df.empty:
    raise ValueError(
        "H2 analysis sample is empty. Check the input data."
    )

if h2_df["churn_yn"].nunique() != 2:
    raise ValueError(
        "churn_yn must contain both 0 and 1 for logistic regression."
    )

print("=" * 80)
print("H2 DATASET LOADED")
print("=" * 80)
print("Data file:", file_path)
print("Analysis sample size:", len(h2_df))
print(
    "Fourth-purchase churn rate:",
    f"{h2_df['churn_yn'].mean() * 100:.2f}%"
)
print(
    "Median order_unit_price:",
    h2_df["order_unit_price"].median()
)


# ------------------------------------------------------------
# 2. Fit nested logistic regression models
#
# Model 1:
# Value-only model
#
# Model 2:
# Value + engagement model
# ------------------------------------------------------------

value_only_model = smf.logit(
    formula="churn_yn ~ log_order_unit_price",
    data=h2_df
).fit(disp=False)

value_engagement_model = smf.logit(
    formula=(
        "churn_yn ~ log_order_unit_price "
        "+ engagement_count"
    ),
    data=h2_df
).fit(disp=False)


# ------------------------------------------------------------
# 3. Likelihood-ratio test
#
# H0:
# Adding engagement_count does not improve model fit.
#
# H1:
# Adding engagement_count improves model fit.
# ------------------------------------------------------------

lr_stat = 2 * (
    value_engagement_model.llf
    - value_only_model.llf
)

lr_df = int(
    value_engagement_model.df_model
    - value_only_model.df_model
)

lr_p_value = chi2.sf(
    lr_stat,
    lr_df
)


# ------------------------------------------------------------
# 4. Calculate McFadden pseudo-R-squared
# ------------------------------------------------------------

value_only_pseudo_r2 = (
    1
    - value_only_model.llf
    / value_only_model.llnull
)

value_engagement_pseudo_r2 = (
    1
    - value_engagement_model.llf
    / value_engagement_model.llnull
)


# ------------------------------------------------------------
# 5. In-sample model comparison table
# ------------------------------------------------------------

fit_comparison = pd.DataFrame([
    {
        "model": "Model 1: Value only",
        "predictors": "log_order_unit_price",
        "log_likelihood": value_only_model.llf,
        "aic": value_only_model.aic,
        "bic": value_only_model.bic,
        "mcfadden_pseudo_r2": value_only_pseudo_r2
    },
    {
        "model": "Model 2: Value + engagement",
        "predictors": (
            "log_order_unit_price + engagement_count"
        ),
        "log_likelihood": value_engagement_model.llf,
        "aic": value_engagement_model.aic,
        "bic": value_engagement_model.bic,
        "mcfadden_pseudo_r2": value_engagement_pseudo_r2
    }
])

print("\n" + "=" * 80)
print("H2 IN-SAMPLE MODEL COMPARISON")
print("=" * 80)

print(
    fit_comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print("\n" + "=" * 80)
print("H2 LIKELIHOOD-RATIO TEST")
print("=" * 80)
print(f"LR chi-square statistic: {lr_stat:.4f}")
print(f"Degrees of freedom: {lr_df}")
print(f"P-value: {lr_p_value:.6g}")

if lr_p_value < 0.05:
    print(
        "Conclusion: Adding engagement_count significantly "
        "improves model fit."
    )
else:
    print(
        "Conclusion: Adding engagement_count does not "
        "significantly improve model fit."
    )


# ------------------------------------------------------------
# 6. Report the incremental engagement coefficient
# ------------------------------------------------------------

engagement_coef = (
    value_engagement_model
    .params["engagement_count"]
)

engagement_p_value = (
    value_engagement_model
    .pvalues["engagement_count"]
)

engagement_ci_low, engagement_ci_high = (
    value_engagement_model
    .conf_int()
    .loc["engagement_count"]
)

engagement_or = np.exp(
    engagement_coef
)

engagement_or_ci_low = np.exp(
    engagement_ci_low
)

engagement_or_ci_high = np.exp(
    engagement_ci_high
)

print("\n" + "=" * 80)
print("ENGAGEMENT COUNT IN MODEL 2")
print("=" * 80)
print(f"Coefficient: {engagement_coef:.4f}")
print(f"Odds Ratio: {engagement_or:.4f}")
print(
    "95% CI for OR: "
    f"[{engagement_or_ci_low:.4f}, "
    f"{engagement_or_ci_high:.4f}]"
)
print(f"P-value: {engagement_p_value:.6g}")


# ------------------------------------------------------------
# 7. Stratified five-fold cross-validation
#
# Use out-of-fold predictions:
# Each user receives a predicted probability from a model
# that was not trained using that user's row.
# ------------------------------------------------------------

y = h2_df["churn_yn"].astype(int).to_numpy()

x_value_only = (
    h2_df[
        ["log_order_unit_price"]
    ]
    .astype(float)
)

x_value_engagement = (
    h2_df[
        [
            "log_order_unit_price",
            "engagement_count"
        ]
    ]
    .astype(float)
)

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

oof_pred_value_only = np.full(
    len(h2_df),
    np.nan
)

oof_pred_value_engagement = np.full(
    len(h2_df),
    np.nan
)

fold_results = []

for fold, (train_idx, test_idx) in enumerate(
    cv.split(x_value_only, y),
    start=1
):
    y_train = y[train_idx]
    y_test = y[test_idx]

    # Model 1: Value only
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

    pred_value_only = cv_value_only_model.predict(
        x_test_value_only
    )

    # Model 2: Value + engagement
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

    pred_value_engagement = (
        cv_value_engagement_model.predict(
            x_test_value_engagement
        )
    )

    oof_pred_value_only[test_idx] = (
        pred_value_only
    )

    oof_pred_value_engagement[test_idx] = (
        pred_value_engagement
    )

    # Numerical safety for log loss
    pred_value_only_clipped = np.clip(
        pred_value_only,
        1e-15,
        1 - 1e-15
    )

    pred_value_engagement_clipped = np.clip(
        pred_value_engagement,
        1e-15,
        1 - 1e-15
    )

    fold_results.append(
        {
            "fold": fold,
            "value_only_auc": roc_auc_score(
                y_test,
                pred_value_only
            ),
            "value_engagement_auc": roc_auc_score(
                y_test,
                pred_value_engagement
            ),
            "auc_improvement": (
                roc_auc_score(
                    y_test,
                    pred_value_engagement
                )
                - roc_auc_score(
                    y_test,
                    pred_value_only
                )
            ),
            "value_only_brier": brier_score_loss(
                y_test,
                pred_value_only
            ),
            "value_engagement_brier": brier_score_loss(
                y_test,
                pred_value_engagement
            ),
            "brier_improvement": (
                brier_score_loss(
                    y_test,
                    pred_value_only
                )
                - brier_score_loss(
                    y_test,
                    pred_value_engagement
                )
            ),
            "value_only_log_loss": log_loss(
                y_test,
                pred_value_only_clipped
            ),
            "value_engagement_log_loss": log_loss(
                y_test,
                pred_value_engagement_clipped
            ),
            "log_loss_improvement": (
                log_loss(
                    y_test,
                    pred_value_only_clipped
                )
                - log_loss(
                    y_test,
                    pred_value_engagement_clipped
                )
            )
        }
    )

fold_results_df = pd.DataFrame(
    fold_results
)

if (
    np.isnan(oof_pred_value_only).any()
    or np.isnan(oof_pred_value_engagement).any()
):
    raise ValueError(
        "交叉验证预测结果存在缺失值，请检查代码。"
    )


# ------------------------------------------------------------
# 8. Overall out-of-fold performance
# ------------------------------------------------------------

oof_value_only_auc = roc_auc_score(
    y,
    oof_pred_value_only
)

oof_value_engagement_auc = roc_auc_score(
    y,
    oof_pred_value_engagement
)

oof_auc_improvement = (
    oof_value_engagement_auc
    - oof_value_only_auc
)

oof_value_only_brier = brier_score_loss(
    y,
    oof_pred_value_only
)

oof_value_engagement_brier = brier_score_loss(
    y,
    oof_pred_value_engagement
)

oof_brier_improvement = (
    oof_value_only_brier
    - oof_value_engagement_brier
)

oof_value_only_log_loss = log_loss(
    y,
    np.clip(
        oof_pred_value_only,
        1e-15,
        1 - 1e-15
    )
)

oof_value_engagement_log_loss = log_loss(
    y,
    np.clip(
        oof_pred_value_engagement,
        1e-15,
        1 - 1e-15
    )
)

oof_log_loss_improvement = (
    oof_value_only_log_loss
    - oof_value_engagement_log_loss
)

oof_performance = pd.DataFrame([
    {
        "model": "Model 1: Value only",
        "roc_auc": oof_value_only_auc,
        "brier_score": oof_value_only_brier,
        "log_loss": oof_value_only_log_loss
    },
    {
        "model": "Model 2: Value + engagement",
        "roc_auc": oof_value_engagement_auc,
        "brier_score": oof_value_engagement_brier,
        "log_loss": oof_value_engagement_log_loss
    }
])

print("\n" + "=" * 80)
print("H2 FIVE-FOLD CROSS-VALIDATION RESULTS")
print("=" * 80)

print(
    fold_results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print("\n" + "=" * 80)
print("H2 OUT-OF-FOLD PERFORMANCE")
print("=" * 80)

print(
    oof_performance.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print("\n" + "=" * 80)
print("H2 OUT-OF-FOLD IMPROVEMENT")
print("=" * 80)
print(
    "ROC-AUC improvement: "
    f"{oof_auc_improvement:.6f}"
)
print(
    "Brier Score improvement: "
    f"{oof_brier_improvement:.6f}"
)
print(
    "Log Loss improvement: "
    f"{oof_log_loss_improvement:.6f}"
)


# ------------------------------------------------------------
# 9. Paired bootstrap confidence intervals
#
# Positive improvement values mean:
# - higher ROC-AUC
# - lower Brier Score
# - lower Log Loss
#
# for Model 2 compared with Model 1.
# ------------------------------------------------------------

rng = np.random.default_rng(
    RANDOM_STATE
)

auc_improvements = []
brier_improvements = []
log_loss_improvements = []

n = len(y)

for _ in range(N_BOOTSTRAP):
    sample_idx = rng.integers(
        0,
        n,
        size=n
    )

    y_boot = y[sample_idx]

    # ROC-AUC requires both classes
    if np.unique(y_boot).size < 2:
        continue

    pred_value_only_boot = (
        oof_pred_value_only[sample_idx]
    )

    pred_value_engagement_boot = (
        oof_pred_value_engagement[sample_idx]
    )

    auc_improvements.append(
        roc_auc_score(
            y_boot,
            pred_value_engagement_boot
        )
        - roc_auc_score(
            y_boot,
            pred_value_only_boot
        )
    )

    brier_improvements.append(
        brier_score_loss(
            y_boot,
            pred_value_only_boot
        )
        - brier_score_loss(
            y_boot,
            pred_value_engagement_boot
        )
    )

    log_loss_improvements.append(
        log_loss(
            y_boot,
            np.clip(
                pred_value_only_boot,
                1e-15,
                1 - 1e-15
            )
        )
        - log_loss(
            y_boot,
            np.clip(
                pred_value_engagement_boot,
                1e-15,
                1 - 1e-15
            )
        )
    )

auc_improvements = np.array(
    auc_improvements
)

brier_improvements = np.array(
    brier_improvements
)

log_loss_improvements = np.array(
    log_loss_improvements
)


def summarize_bootstrap(
    metric_name,
    values
):
    observed = {
        "roc_auc_improvement": oof_auc_improvement,
        "brier_score_improvement": oof_brier_improvement,
        "log_loss_improvement": oof_log_loss_improvement
    }[metric_name]

    ci_low, ci_high = np.quantile(
        values,
        [0.025, 0.975]
    )

    one_sided_p_value = (
        1
        + np.sum(values <= 0)
    ) / (
        len(values)
        + 1
    )

    return {
        "metric": metric_name,
        "observed_improvement": observed,
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "one_sided_p_value": one_sided_p_value
    }


bootstrap_summary = pd.DataFrame([
    summarize_bootstrap(
        "roc_auc_improvement",
        auc_improvements
    ),
    summarize_bootstrap(
        "brier_score_improvement",
        brier_improvements
    ),
    summarize_bootstrap(
        "log_loss_improvement",
        log_loss_improvements
    )
])

print("\n" + "=" * 80)
print("H2 PAIRED BOOTSTRAP RESULTS")
print("=" * 80)

print(
    bootstrap_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ------------------------------------------------------------
# 10. Final hypothesis decision
#
# Primary inferential criterion:
# LR test p < .05
#
# Primary predictive criterion:
# ROC-AUC improvement > 0
# and bootstrap lower CI > 0
#
# Brier Score and Log Loss are supplementary metrics.
# ------------------------------------------------------------

explanatory_improvement_supported = (
    lr_p_value < 0.05
)

auc_bootstrap_low = bootstrap_summary.loc[
    bootstrap_summary["metric"]
    == "roc_auc_improvement",
    "bootstrap_ci_low"
].iloc[0]

predictive_improvement_supported = (
    oof_auc_improvement > 0
    and auc_bootstrap_low > 0
)

print("\n" + "=" * 80)
print("H2 FINAL DECISION")
print("=" * 80)

if explanatory_improvement_supported:
    print(
        "Explanatory evidence: Supported. "
        "Adding engagement_count significantly improves "
        "model fit."
    )
else:
    print(
        "Explanatory evidence: Not supported."
    )

if predictive_improvement_supported:
    print(
        "Predictive evidence: Supported. "
        "Adding engagement_count significantly improves "
        "out-of-fold ROC-AUC."
    )
else:
    print(
        "Predictive evidence: The ROC-AUC improvement is "
        "not sufficiently stable under paired bootstrap."
    )

if (
    explanatory_improvement_supported
    and predictive_improvement_supported
):
    print(
        "Overall conclusion: H2 is supported."
    )
else:
    print(
        "Overall conclusion: H2 is only partially supported "
        "or not supported. Review the detailed metrics."
    )
