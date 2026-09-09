from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, make_scorer, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None


STUDY_2_DIR = Path(__file__).resolve().parents[1]
if str(STUDY_2_DIR) not in sys.path:
    sys.path.append(str(STUDY_2_DIR))

from quadrant_utils import load_hvle_data


warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RANDOM_STATE = 42
N_SPLITS = 5
TOP_RATE = 0.20
METRIC_COLS = ["recall", "f1", "auc"]

TEXT_COLS = {
    "first_order_coupon_name",
    "second_order_coupon_name",
    "third_order_coupon_name",
    "pet_breed_detailed",
}

SOURCE_COLS = [
    "survive_yn",
    "days_to_third_purchase_from_signup",
    "days_from_second_to_third_purchase",
    "purchased_supplies_yn",
    "purchased_snacks_yn",
    "purchased_feed_yn",
    "purchased_essentials_yn",
    "purchased_supplies_count",
    "purchased_snacks_count",
    "purchased_feed_count",
    "purchased_essentials_count",
    "pb_purchase_yn",
    "pb_purchase_count",
    "nb_purchase_count",
    "days_to_first_purchase_from_signup",
    "days_from_first_to_second_purchase",
    "first_order_coupon_name",
    "second_order_coupon_name",
    "third_order_coupon_name",
    "first_order_delivery_time_hours",
    "second_order_delivery_time_hours",
    "third_order_delivery_time_hours",
    "sameday_delivery_orders",
    "dawn_delivery_orders",
    "platform_delivery_orders",
    "weekday_order_count",
    "weekend_order_count",
    "pet_species",
    "pet_breed_detailed",
    "pet_age_months",
    "pet_registered_before_first_purchase_yn",
    "first_purchase_month",
]

FEATURE_COLS = [
    "purchased_supplies_yn",
    "purchased_snacks_yn",
    "purchased_feed_yn",
    "purchased_essentials_yn",
    "category_flag_count",
    "pb_purchase_yn",
    "pb_purchase_count_ratio",
    "days_to_first_purchase_from_signup",
    "days_to_third_purchase_from_signup",
    "purchase_interval_mean",
    "purchase_interval_change",
    "coupon_used_count_first3",
    "coupon_dropoff_1_to_3",
    "delivery_time_mean_first3",
    "delivery_time_max_first3",
    "delivery_time_change_1_to_3",
    "sameday_delivery_ratio",
    "dawn_delivery_ratio",
    "weekend_order_ratio",
    "pet_age_group",
    "pet_species",
    "pet_breed_detailed",
    "pet_registered_before_first_purchase_yn",
    "first_purchase_quarter",
    "is_year_end_first_purchase",
]


def require_columns(df, cols, context):
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for {context}: {missing}")


def safe_ratio(numerator, denominator):
    return numerator / denominator.replace(0, np.nan)


def coupon_used(series):
    values = series.astype("string").str.strip().str.lower()
    missing = {"", "0", "0.0", "nan", "none", "null", "no", "missing", "-1"}
    return (~values.isna() & ~values.isin(missing)).astype(int)


def format_mean_sd(mean_value, sd_value):
    return f"{mean_value:.4f} ± {sd_value:.4f}"


def build_study_2b_retention_features(file_path=None):
    if file_path is None:
        file_path = Path(__file__).resolve().parents[2] / "output" / "pet_data_clean_all_variables.csv"
    file_path = Path(file_path)

    loaded = load_hvle_data(file_path)
    hvle_index = loaded["hv_df"].index[loaded["hv_df"]["HVLE_yn"] == 1]
    source = pd.read_csv(file_path, low_memory=False)
    require_columns(source, SOURCE_COLS, "Study 2B source data")

    df = source.loc[hvle_index, SOURCE_COLS].copy()
    for col in [col for col in df.columns if col not in TEXT_COLS]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["survive_yn"]).copy()
    df["survive_yn"] = df["survive_yn"].astype(int)

    add_timing_features(df)
    add_category_features(df)
    add_brand_features(df)
    add_coupon_features(df)
    add_delivery_features(df)
    add_pet_calendar_features(df)

    require_columns(df, FEATURE_COLS, "Study 2B model features")
    feature_cols = FEATURE_COLS.copy()
    modeling_df = df[feature_cols + ["survive_yn"]].copy()

    print("=" * 70)
    print("=== Study 2B Leakage-Aware HVLE Retention Feature Dataset ===")
    print("=" * 70)
    print("Data file:", file_path)
    print("HVLE sample size:", len(modeling_df))
    print("HVLE survival rate:", round(modeling_df["survive_yn"].mean() * 100, 2), "%")
    print("Candidate feature count:", len(feature_cols))

    return {
        "hvle_df": df,
        "modeling_df": modeling_df,
        "feature_cols": feature_cols,
        "target_col": "survive_yn",
        "value_threshold": loaded["value_threshold"],
    }


def add_timing_features(df):
    interval_cols = ["days_from_first_to_second_purchase", "days_from_second_to_third_purchase"]
    require_columns(df, interval_cols, "timing features")
    df["purchase_interval_mean"] = df[interval_cols].mean(axis=1)
    df["purchase_interval_change"] = (
        df["days_from_second_to_third_purchase"] - df["days_from_first_to_second_purchase"]
    )


def add_category_features(df):
    flags = ["purchased_supplies_yn", "purchased_snacks_yn", "purchased_feed_yn", "purchased_essentials_yn"]
    require_columns(df, flags, "category features")
    df[flags] = df[flags].fillna(0)
    df["category_flag_count"] = df[flags].sum(axis=1)


def add_brand_features(df):
    count_cols = ["pb_purchase_count", "nb_purchase_count"]
    require_columns(df, count_cols, "brand features")
    df[count_cols] = df[count_cols].fillna(0)
    total = df[count_cols].sum(axis=1)
    df["pb_purchase_count_ratio"] = safe_ratio(df["pb_purchase_count"], total)


def add_coupon_features(df):
    coupon_cols = ["first_order_coupon_name", "second_order_coupon_name", "third_order_coupon_name"]
    require_columns(df, coupon_cols, "coupon features")
    used_cols = []
    for col in coupon_cols:
        used_col = col.replace("_coupon_name", "_coupon_used_yn")
        df[used_col] = coupon_used(df[col])
        used_cols.append(used_col)
    df["coupon_used_count_first3"] = df[used_cols].sum(axis=1)
    df["coupon_dropoff_1_to_3"] = df["first_order_coupon_used_yn"] - df["third_order_coupon_used_yn"]


def add_delivery_features(df):
    time_cols = ["first_order_delivery_time_hours", "second_order_delivery_time_hours", "third_order_delivery_time_hours"]
    require_columns(df, time_cols, "delivery-time features")
    df["delivery_time_mean_first3"] = df[time_cols].mean(axis=1)
    df["delivery_time_max_first3"] = df[time_cols].max(axis=1)
    df["delivery_time_change_1_to_3"] = df["third_order_delivery_time_hours"] - df["first_order_delivery_time_hours"]

    method_cols = ["sameday_delivery_orders", "dawn_delivery_orders", "platform_delivery_orders"]
    require_columns(df, method_cols, "delivery-method features")
    df[method_cols] = df[method_cols].fillna(0)
    total = df[method_cols].sum(axis=1)
    for col in method_cols:
        df[col.replace("_orders", "_ratio")] = safe_ratio(df[col], total)

    day_cols = ["weekday_order_count", "weekend_order_count"]
    require_columns(df, day_cols, "weekday/weekend delivery features")
    df[day_cols] = df[day_cols].fillna(0)
    df["weekend_order_ratio"] = safe_ratio(df["weekend_order_count"], df[day_cols].sum(axis=1))


def add_pet_calendar_features(df):
    require_columns(df, ["pet_age_months", "first_purchase_month"], "pet and calendar features")
    df["pet_age_group"] = pd.cut(
        df["pet_age_months"].where(df["pet_age_months"] >= 0),
        bins=[0, 6, 24, 84, np.inf],
        labels=["baby_0_5m", "young_6_23m", "adult_2_6y", "senior_7y_plus"],
        right=False,
    )

    first_purchase_month = pd.to_numeric(df["first_purchase_month"], errors="coerce")
    if first_purchase_month.notna().sum() == 0:
        raise ValueError("first_purchase_month must be stored as Unix seconds.")
    purchase_dt = pd.to_datetime(first_purchase_month, unit="s", errors="coerce")
    if purchase_dt.notna().sum() == 0:
        raise ValueError("Unable to parse first_purchase_month as Unix seconds.")
    df["first_purchase_quarter"] = purchase_dt.dt.quarter.astype("Int64").astype("string")
    df["is_year_end_first_purchase"] = purchase_dt.dt.month.isin([11, 12]).astype(int)


def split_feature_types(df, feature_cols):
    categorical = [
        col
        for col in feature_cols
        if pd.api.types.is_object_dtype(df[col])
        or pd.api.types.is_string_dtype(df[col])
        or isinstance(df[col].dtype, pd.CategoricalDtype)
        or col in ["pet_age_group", "pet_species", "pet_breed_detailed", "first_purchase_quarter"]
    ]
    numeric = [col for col in feature_cols if col not in categorical]
    return numeric, categorical


def build_preprocessor(numeric_features, categorical_features):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])


def build_models():
    models = {
        "RF": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "DT": DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "LR": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            alpha=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=RANDOM_STATE,
        ),
    }
    if LGBMClassifier is not None:
        models = {
            "LGBM": LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=1,
                verbosity=-1,
            ),
            **models,
        }
    return models


def retention_potential_ranking_metrics(y_true, survival_score):
    ranking = pd.DataFrame({
        "survive_yn": np.asarray(y_true).astype(int),
        "predicted_survival_probability": np.asarray(survival_score),
    })
    ranking = ranking.sort_values("predicted_survival_probability", ascending=False)

    top_n = max(1, int(np.ceil(len(ranking) * TOP_RATE)))
    top_group = ranking.head(top_n)
    baseline = ranking["survive_yn"].mean()
    precision_at_20 = top_group["survive_yn"].mean()
    return {
        "evaluated_n": len(ranking),
        "top_n": top_n,
        "baseline_retention_rate": baseline,
        "precision_at_20pct": precision_at_20,
        "actual_retained_n_in_top_20pct": int(top_group["survive_yn"].sum()),
    }


def validation_fold_oof_metrics(X, y, survival_score, cv):
    y_array = np.asarray(y).astype(int)
    score_array = np.asarray(survival_score)
    fold_rows = []

    for fold_idx, (_, valid_idx) in enumerate(cv.split(X, y), start=1):
        y_valid = y_array[valid_idx]
        score_valid = score_array[valid_idx]
        ranking = retention_potential_ranking_metrics(y_valid, score_valid)
        fold_rows.append({
            "fold": fold_idx,
            "brier_score": brier_score_loss(y_valid, score_valid),
            **ranking,
        })

    return pd.DataFrame(fold_rows)


def evaluate_model(model_name, classifier, preprocessor, X, y, cv, scoring):
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", classifier)])
    cv_result = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=1)
    oof_survival_score = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    oof_fold_metrics = validation_fold_oof_metrics(X, y, oof_survival_score, cv)

    result = {"model": model_name}
    for metric in METRIC_COLS:
        values = cv_result[f"test_{metric}"]
        result[f"{metric}_mean"] = values.mean()
        result[f"{metric}_sd"] = values.std(ddof=1)
    for metric in ["brier_score", "precision_at_20pct"]:
        values = oof_fold_metrics[metric]
        result[f"{metric}_mean"] = values.mean()
        result[f"{metric}_sd"] = values.std(ddof=1)

    ranking = {
        "model": model_name,
        "top_n_mean": oof_fold_metrics["top_n"].mean(),
        "top_n_sd": oof_fold_metrics["top_n"].std(ddof=1),
        "baseline_retention_rate_mean": oof_fold_metrics["baseline_retention_rate"].mean(),
        "baseline_retention_rate_sd": oof_fold_metrics["baseline_retention_rate"].std(ddof=1),
        "precision_at_20pct_mean": result["precision_at_20pct_mean"],
        "precision_at_20pct_sd": result["precision_at_20pct_sd"],
        "actual_retained_n_in_top_20pct_mean": oof_fold_metrics["actual_retained_n_in_top_20pct"].mean(),
        "actual_retained_n_in_top_20pct_sd": oof_fold_metrics["actual_retained_n_in_top_20pct"].std(ddof=1),
    }
    folds = pd.DataFrame({
        "model": model_name,
        "fold": np.arange(1, N_SPLITS + 1),
        "recall": cv_result["test_recall"],
        "f1": cv_result["test_f1"],
        "auc": cv_result["test_auc"],
    })
    folds = folds.merge(
        oof_fold_metrics[
            [
                "fold",
                "brier_score",
                "precision_at_20pct",
                "actual_retained_n_in_top_20pct",
                "top_n",
            ]
        ],
        on="fold",
    )
    return result, ranking, folds


def select_overall_best_model(results_df, ranking_df):
    selection = results_df.merge(
        ranking_df[
            [
                "model",
                "actual_retained_n_in_top_20pct_mean",
                "actual_retained_n_in_top_20pct_sd",
            ]
        ],
        on="model",
    )
    selection = selection.sort_values(
        ["auc_mean", "precision_at_20pct_mean", "f1_mean", "recall_mean"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    selection["overall_rank"] = np.arange(1, len(selection) + 1)
    return selection


def run_model_comparison(modeling_df, feature_cols, target_col):
    model_df = modeling_df[feature_cols + [target_col]].dropna(subset=[target_col]).copy()
    model_df[target_col] = pd.to_numeric(model_df[target_col], errors="coerce")
    model_df = model_df.dropna(subset=[target_col]).copy()
    model_df[target_col] = model_df[target_col].astype(int)
    if model_df[target_col].nunique() != 2:
        raise ValueError(f"{target_col} must contain both 0 and 1.")

    numeric, categorical = split_feature_types(model_df, feature_cols)
    X, y = model_df[feature_cols], model_df[target_col]
    preprocessor = build_preprocessor(numeric, categorical)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "auc": "roc_auc",
    }

    models = build_models()
    outputs = [
        evaluate_model(name, model, preprocessor, X, y, cv, scoring)
        for name, model in models.items()
    ]
    results_df = pd.DataFrame([item[0] for item in outputs]).sort_values("auc_mean", ascending=False).reset_index(drop=True)
    ranking_df = (
        pd.DataFrame([item[1] for item in outputs])
        .sort_values("precision_at_20pct_mean", ascending=False)
        .reset_index(drop=True)
    )
    fold_results_df = pd.concat([item[2] for item in outputs], ignore_index=True)
    selection_df = select_overall_best_model(results_df, ranking_df)
    final_model = selection_df.iloc[0]["model"]

    print_results(target_col, numeric, categorical, results_df, ranking_df, fold_results_df, selection_df)
    return {
        "model_df": model_df,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "results_df": results_df,
        "fold_results_df": fold_results_df,
        "ranking_results_df": ranking_df,
        "overall_selection_df": selection_df,
        "shap_model_name": final_model,
        "final_model_name": final_model,
        "best_top20_model": ranking_df.iloc[0]["model"],
        "models": models,
        "random_state": RANDOM_STATE,
        "n_splits": N_SPLITS,
    }


def print_results(target_col, numeric, categorical, results_df, ranking_df, fold_df, selection_df):
    print("\n" + "=" * 70)
    print("=== HVLE Survival Model Comparison ===")
    print("=" * 70)
    print("Target:", target_col)
    print("Positive class: survive_yn = 1")
    print("Random state:", RANDOM_STATE)
    print("CV folds:", N_SPLITS)
    print("\nNumeric features:")
    for col in numeric:
        print("-", col)
    print("\nCategorical features:")
    for col in categorical:
        print("-", col)

    print("\nFold-level metrics:")
    for fold_idx in range(1, N_SPLITS + 1):
        current = fold_df.loc[
            fold_df["fold"] == fold_idx,
            [
                "model",
                "recall",
                "f1",
                "auc",
                "brier_score",
                "precision_at_20pct",
            ],
        ]
        print("\n" + "-" * 70)
        print(f"Fold {fold_idx}")
        print("-" * 70)
        print(current.round(4).to_string(index=False))

    print("\n" + "=" * 70)
    print("=== Model Performance Summary ===")
    print("=" * 70)
    summary = results_df.copy()
    summary_display = pd.DataFrame({
        "Model": summary["model"],
        "Recall": [
            format_mean_sd(mean, sd)
            for mean, sd in zip(
                summary["recall_mean"],
                summary["recall_sd"],
            )
        ],
        "F1-score": [
            format_mean_sd(mean, sd)
            for mean, sd in zip(
                summary["f1_mean"],
                summary["f1_sd"],
            )
        ],
        "AUC": [
            format_mean_sd(mean, sd)
            for mean, sd in zip(
                summary["auc_mean"],
                summary["auc_sd"],
            )
        ],
        "Brier score": [
            format_mean_sd(mean, sd)
            for mean, sd in zip(
                summary["brier_score_mean"],
                summary["brier_score_sd"],
            )
        ],
        "Precision@20%": [
            format_mean_sd(mean, sd)
            for mean, sd in zip(
                summary["precision_at_20pct_mean"],
                summary["precision_at_20pct_sd"],
            )
        ],
    })
    print(summary_display.to_string(index=False))
    print("Interpretation note: SHAP values are model-specific predictive explanations.")


if __name__ == "__main__":
    feature_data = build_study_2b_retention_features()
    run_model_comparison(
        modeling_df=feature_data["modeling_df"],
        feature_cols=feature_data["feature_cols"],
        target_col=feature_data["target_col"],
    )
