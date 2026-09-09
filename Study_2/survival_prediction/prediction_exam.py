"""Study 2B non-completion risk model comparison.

Runs repeated stratified cross-validation for the five configured models and
prints the results to the console only.
"""

from pathlib import Path
import contextlib
import io
import sys
import warnings

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from stage3_modeling_utils import (  # noqa: E402
    RANDOM_STATE,
    TOP_RATE,
    build_models,
    build_preprocessor,
    build_study_2b_retention_features,
    split_feature_types,
)


warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


N_SPLITS = 5
N_REPEATS = 20
MODEL_NAMES = ["LGBM", "RF", "DT", "LR", "MLP"]


def available_models():
    models = build_models()
    selected = {name: models[name] for name in MODEL_NAMES if name in models}
    missing = [name for name in MODEL_NAMES if name not in selected]
    if missing:
        print("Skipped unavailable models:", ", ".join(missing))
    return selected


def build_pipeline(model, numeric_features, categorical_features):
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
            ("model", clone(model)),
        ]
    )


def risk_ranking_metrics(y_true, risk_score, top_rate=TOP_RATE):
    ranking = pd.DataFrame(
        {
            "noncompletion_yn": np.asarray(y_true).astype(int),
            "predicted_noncompletion_risk": np.asarray(risk_score),
        }
    ).sort_values("predicted_noncompletion_risk", ascending=False)

    top_n = max(1, int(np.ceil(len(ranking) * top_rate)))
    baseline_risk = ranking["noncompletion_yn"].mean()
    precision_at_top = ranking.head(top_n)["noncompletion_yn"].mean()
    lift_at_top = precision_at_top / baseline_risk if baseline_risk > 0 else np.nan

    return {
        "baseline_noncompletion_rate": baseline_risk,
        "risk_precision_at_20pct": precision_at_top,
        "lift_at_20pct": lift_at_top,
    }


def evaluate_repeated_cv(model_name, model, X, y, numeric_features, categorical_features):
    rows = []

    for repeat in range(1, N_REPEATS + 1):
        cv = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE + repeat - 1,
        )
        oof_score = pd.Series(index=y.index, dtype=float)

        for train_idx, test_idx in cv.split(X, y):
            pipeline = build_pipeline(model, numeric_features, categorical_features)
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train = y.iloc[train_idx]

            pipeline.fit(X_train, y_train)
            oof_score.iloc[test_idx] = pipeline.predict_proba(X_test)[:, 1]

        if oof_score.isna().any():
            raise RuntimeError(f"Missing OOF predictions for {model_name}, repeat {repeat}.")

        rows.append(
            {
                "model": model_name,
                "repeat": repeat,
                "evaluated_n": len(y),
                "auc": roc_auc_score(y, oof_score),
                "pr_auc": average_precision_score(y, oof_score),
                "brier_score": brier_score_loss(y, oof_score),
                **risk_ranking_metrics(y, oof_score),
            }
        )

    return pd.DataFrame(rows)


def summarize_metric(values):
    values = pd.Series(values).dropna()
    return pd.Series(
        {
            "mean": values.mean(),
            "sd": values.std(ddof=1),
            "p2_5": values.quantile(0.025),
            "p97_5": values.quantile(0.975),
        }
    )


def summarize_repeated_cv(repeat_metrics):
    metric_cols = [
        "auc",
        "pr_auc",
        "risk_precision_at_20pct",
        "lift_at_20pct",
        "brier_score",
    ]
    rows = []

    for model_name, model_df in repeat_metrics.groupby("model"):
        row = {"model": model_name, "repeats": len(model_df)}
        for metric in metric_cols:
            for key, value in summarize_metric(model_df[metric]).items():
                row[f"{metric}_{key}"] = value
        rows.append(row)

    return pd.DataFrame(rows).sort_values("auc_mean", ascending=False)


def format_mean_sd(row, metric):
    return f"{row[f'{metric}_mean']:.4f} +/- {row[f'{metric}_sd']:.4f}"


def build_summary_display(summary_df):
    return pd.DataFrame(
        {
            "Model": summary_df["model"],
            "AUC-ROC": summary_df.apply(lambda row: format_mean_sd(row, "auc"), axis=1),
            "PR-AUC": summary_df.apply(lambda row: format_mean_sd(row, "pr_auc"), axis=1),
            "Risk Precision@20%": summary_df.apply(
                lambda row: format_mean_sd(row, "risk_precision_at_20pct"),
                axis=1,
            ),
            "Lift@20%": summary_df.apply(
                lambda row: format_mean_sd(row, "lift_at_20pct"),
                axis=1,
            ),
            "Brier score": summary_df.apply(
                lambda row: format_mean_sd(row, "brier_score"),
                axis=1,
            ),
            "AUC empirical range": (
                summary_df["auc_p2_5"].map(lambda value: f"{value:.4f}")
                + "-"
                + summary_df["auc_p97_5"].map(lambda value: f"{value:.4f}")
            ),
        }
    )


def main():
    with contextlib.redirect_stdout(io.StringIO()):
        feature_data = build_study_2b_retention_features()

    modeling_df = feature_data["modeling_df"].copy()
    feature_cols = feature_data["feature_cols"]
    source_target_col = feature_data["target_col"]
    target_col = "noncompletion_yn"

    modeling_df[source_target_col] = pd.to_numeric(
        modeling_df[source_target_col],
        errors="coerce",
    )
    modeling_df = modeling_df.dropna(subset=[source_target_col]).copy()
    modeling_df[source_target_col] = modeling_df[source_target_col].astype(int)
    modeling_df[target_col] = 1 - modeling_df[source_target_col]

    numeric_features, categorical_features = split_feature_types(modeling_df, feature_cols)
    X = modeling_df[feature_cols].copy()
    y = modeling_df[target_col].copy()

    models = available_models()
    if not models:
        raise RuntimeError("None of the requested models are available.")

    print("\n" + "=" * 80)
    print("=== Repeated Stratified 5-fold CV for Non-completion Risk ===")
    print("=" * 80)
    print("Target: noncompletion_yn = 1 - survive_yn")
    print("Positive class: fourth-purchase non-completion = 1")
    print("HVLE sample size:", len(modeling_df))
    print("Baseline non-completion rate:", f"{y.mean() * 100:.2f}%")
    print("Models:", ", ".join(models.keys()))
    print("Design:", f"{N_REPEATS} repeats x {N_SPLITS} folds")

    repeated_cv_df = pd.concat(
        [
            evaluate_repeated_cv(
                model_name,
                model,
                X,
                y,
                numeric_features,
                categorical_features,
            )
            for model_name, model in models.items()
        ],
        ignore_index=True,
    )
    summary_df = summarize_repeated_cv(repeated_cv_df)

    print("\nRepeated-CV stability summary")
    print(build_summary_display(summary_df).to_string(index=False))


if __name__ == "__main__":
    main()
