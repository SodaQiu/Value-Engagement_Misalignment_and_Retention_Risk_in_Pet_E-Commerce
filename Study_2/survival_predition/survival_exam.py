from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    make_scorer,
    precision_score,
    recall_score
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_SPLITS = 5


def build_preprocessor(numeric_features, categorical_features):
    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                drop="first"
            )
        )
    ])

    return ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ])


def split_feature_types(modeling_df, feature_cols):
    categorical_features = [
        col
        for col in feature_cols
        if (
            modeling_df[col].dtype.name in ["object", "category", "string"]
            or col in [
                "pet_age_group",
                "pet_species",
                "purchase_structure",
                "purchase_structure_binary"
            ]
        )
    ]

    numeric_features = [
        col
        for col in feature_cols
        if col not in categorical_features
    ]

    return numeric_features, categorical_features


def build_models():
    models = {
        "RF": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1
        ),
        "DT": DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        "LR": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )
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
                verbosity=-1
            ),
            **models
        }

    return models


def run_model_comparison(modeling_df, feature_cols, target_col):
    model_df = modeling_df[
        feature_cols
        + [target_col]
    ].copy()

    model_df = model_df.dropna(
        subset=[target_col]
    ).copy()

    model_df[target_col] = pd.to_numeric(
        model_df[target_col],
        errors="coerce"
    )

    model_df = model_df.dropna(
        subset=[target_col]
    ).copy()

    model_df[target_col] = (
        model_df[target_col]
        .astype(int)
    )

    if model_df[target_col].nunique() != 2:
        raise ValueError(
            f"{target_col} must contain both 0 and 1."
        )

    numeric_features, categorical_features = split_feature_types(
        model_df,
        feature_cols
    )

    X = model_df[feature_cols].copy()
    y = model_df[target_col].copy()

    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features
    )

    models = build_models()

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    scoring = {
        "precision": make_scorer(
            precision_score,
            zero_division=0
        ),
        "recall": make_scorer(
            recall_score,
            zero_division=0
        ),
        "f1": make_scorer(
            f1_score,
            zero_division=0
        ),
        "auc": "roc_auc"
    }

    results = []
    fold_results = []

    for model_name, classifier in models.items():
        pipeline = Pipeline([
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                classifier
            )
        ])

        cv_results = cross_validate(
            estimator=pipeline,
            X=X,
            y=y,
            cv=cv,
            scoring=scoring,
            n_jobs=1
        )

        for fold_idx in range(N_SPLITS):
            fold_results.append({
                "model": model_name,
                "fold": fold_idx + 1,
                "precision": cv_results["test_precision"][fold_idx],
                "recall": cv_results["test_recall"][fold_idx],
                "f1": cv_results["test_f1"][fold_idx],
                "auc": cv_results["test_auc"][fold_idx]
            })

        results.append({
            "model": model_name,
            "precision_mean": cv_results["test_precision"].mean(),
            "recall_mean": cv_results["test_recall"].mean(),
            "f1_mean": cv_results["test_f1"].mean(),
            "auc_mean": cv_results["test_auc"].mean()
        })

    results_df = (
        pd.DataFrame(results)
        .sort_values(
            by="auc_mean",
            ascending=False
        )
        .reset_index(drop=True)
    )

    fold_results_df = pd.DataFrame(
        fold_results
    )

    print("\n" + "=" * 70)
    print("=== HVLE Survival Model Comparison ===")
    print("=" * 70)
    print("Target:", target_col)
    print("Positive class: survive_yn = 1")
    print("Random state:", RANDOM_STATE)
    print("CV folds:", N_SPLITS)

    print("\nNumeric features:")
    for col in numeric_features:
        print("-", col)

    print("\nCategorical features:")
    for col in categorical_features:
        print("-", col)

    print("\nFold-level metrics:")
    for fold_idx in range(1, N_SPLITS + 1):
        fold_df = (
            fold_results_df[
                fold_results_df["fold"] == fold_idx
            ]
            .loc[
                :,
                [
                    "model",
                    "precision",
                    "recall",
                    "f1",
                    "auc"
                ]
            ]
        )

        print("\n" + "-" * 70)
        print(f"Fold {fold_idx}")
        print("-" * 70)
        print(
            fold_df
            .round(4)
            .to_string(index=False)
        )

    print("\n" + "=" * 70)
    print("=== Mean Cross-Validation Model Comparison ===")
    print("=" * 70)
    print(
        results_df
        .round(4)
        .to_string(index=False)
    )

    return {
        "model_df": model_df,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "results_df": results_df,
        "fold_results_df": fold_results_df,
        "models": models,
        "random_state": RANDOM_STATE,
        "n_splits": N_SPLITS
    }


def build_hvle_retention_features(file_path=None):
    """
    Exploratory Analysis:
    Explore feature engineering for fourth-purchase survival among HVLE customers.

    Target:
    survive_yn = 1 means completion of fourth purchase.
    survive_yn = 0 means failure to complete fourth purchase.

    Analysis sample:
    HVLE customers only:
    high_value = 1
    high_engagement = 0
    """

    # ========================================================
    # 1. Load data
    # ========================================================

    if file_path is None:
        file_path = (
            Path(__file__).resolve().parents[2]
            / "output"
            / "pet_data_clean_all_variables.csv"
        )
    else:
        file_path = Path(file_path)

    df = pd.read_csv(file_path, low_memory=False)

    # ========================================================
    # 2. Required columns
    # ========================================================

    required_cols = [
        "days_to_third_purchase_from_signup",
        "days_from_second_to_third_purchase",
        "order_unit_price",
        "review_written_yn",
        "push_notification_consent_yn",
        "survive_yn"
    ]

    optional_feature_cols = [
        "days_to_first_purchase_from_signup",
        "days_from_first_to_second_purchase",
        "pet_age_months",
        "pet_species",
        "pet_registered_before_first_purchase_yn",
        "purchased_feed_count",
        "purchased_snacks_count",
        "purchased_supplies_count",
        "purchased_essentials_count",
        "pet_registration_yn",
        "review_count"
    ]

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    available_optional_feature_cols = [
        c for c in optional_feature_cols
        if c in df.columns
    ]

    # ========================================================
    # 3. Convert basic variables
    # ========================================================

    for c in required_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ========================================================
    # 4. Restrict to customers who completed third purchase
    # ========================================================

    df = df[
        (df["days_to_third_purchase_from_signup"] >= 0) &
        (df["days_from_second_to_third_purchase"] >= 0)
    ].copy()

    # ========================================================
    # 5. Basic value x engagement construction
    # ========================================================

    df = df[
        required_cols
        + available_optional_feature_cols
    ].dropna(subset=required_cols).copy()
    df = df[df["order_unit_price"] > 0].copy()

    df["engagement_count"] = (
        df["review_written_yn"] +
        df["push_notification_consent_yn"]
    )

    df["high_engagement"] = (
        df["engagement_count"] >= 1
    ).astype(int)

    value_threshold = df["order_unit_price"].median()

    df["high_value"] = (
        df["order_unit_price"] >= value_threshold
    ).astype(int)

    # ========================================================
    # 6. Keep HVLE customers only
    # ========================================================

    hvle_df = df[
        (df["high_value"] == 1) &
        (df["high_engagement"] == 0)
    ].copy()

    # ========================================================
    # 7. Feature engineering
    # ========================================================

    # Log-transformed monetary value
    hvle_df["log_order_unit_price"] = np.log1p(
        hvle_df["order_unit_price"]
    )

    # Purchase timing features
    if "days_to_first_purchase_from_signup" in hvle_df.columns:
        hvle_df["days_to_first_purchase_from_signup"] = pd.to_numeric(
            hvle_df["days_to_first_purchase_from_signup"],
            errors="coerce"
        )

    if "days_from_first_to_second_purchase" in hvle_df.columns:
        hvle_df["days_from_first_to_second_purchase"] = pd.to_numeric(
            hvle_df["days_from_first_to_second_purchase"],
            errors="coerce"
        )

    if "days_from_second_to_third_purchase" in hvle_df.columns:
        hvle_df["days_from_second_to_third_purchase"] = pd.to_numeric(
            hvle_df["days_from_second_to_third_purchase"],
            errors="coerce"
        )

    # Average purchase interval
    interval_cols = [
        c for c in [
            "days_from_first_to_second_purchase",
            "days_from_second_to_third_purchase"
        ]
        if c in hvle_df.columns
    ]

    if len(interval_cols) > 0:
        hvle_df["purchase_interval_mean"] = (
            hvle_df[interval_cols].mean(axis=1)
        )

    if len(interval_cols) == 2:
        hvle_df["purchase_interval_change"] = (
            hvle_df["days_from_second_to_third_purchase"] -
            hvle_df["days_from_first_to_second_purchase"]
        )

    # Pet age group
    if "pet_age_months" in hvle_df.columns:
        hvle_df["pet_age_months"] = pd.to_numeric(
            hvle_df["pet_age_months"],
            errors="coerce"
        )

        hvle_df["pet_age_group"] = pd.cut(
            hvle_df["pet_age_months"].where(
                hvle_df["pet_age_months"] >= 0,
                np.nan
            ),
            bins=[0, 6, 24, 84, np.inf],
            labels=[
                "baby_0_5m",
                "young_6_23m",
                "adult_2_6y",
                "senior_7y_plus"
            ]
        )

    # Purchase structure from category count variables
    category_cols = [
        "purchased_feed_count",
        "purchased_snacks_count",
        "purchased_supplies_count",
        "purchased_essentials_count"
    ]

    existing_category_cols = [
        c for c in category_cols if c in hvle_df.columns
    ]

    if len(existing_category_cols) > 0:
        for c in existing_category_cols:
            hvle_df[c] = pd.to_numeric(
                hvle_df[c],
                errors="coerce"
            ).fillna(0)

        hvle_df["category_used_count"] = (
            hvle_df[existing_category_cols] > 0
        ).sum(axis=1)

        def assign_purchase_structure(row):
            if row["category_used_count"] >= 2:
                return "multi_category"

            if "purchased_feed_count" in existing_category_cols and row["purchased_feed_count"] > 0:
                return "feed_only"

            if "purchased_snacks_count" in existing_category_cols and row["purchased_snacks_count"] > 0:
                return "snacks_only"

            if "purchased_supplies_count" in existing_category_cols and row["purchased_supplies_count"] > 0:
                return "supplies_only"

            if "purchased_essentials_count" in existing_category_cols and row["purchased_essentials_count"] > 0:
                return "essentials_only"

            return np.nan

        hvle_df["purchase_structure"] = hvle_df.apply(
            assign_purchase_structure,
            axis=1
        )

        hvle_df["purchase_structure_binary"] = np.where(
            hvle_df["category_used_count"] >= 2,
            "multi_category",
            "single_category"
        )

    # ========================================================
    # 8. Select candidate features
    # ========================================================

    candidate_features = [
        # monetary value
        "log_order_unit_price",

        # timing
        "days_to_third_purchase_from_signup",
        "days_to_first_purchase_from_signup",
        "purchase_interval_mean",
        "purchase_interval_change",

        # pet information
        "pet_age_months",
        "pet_species",
        "pet_registered_before_first_purchase_yn",

        # purchase structure
        "category_used_count",
        "purchase_structure",
        "purchase_structure_binary",

        # service / app related
        "pet_registration_yn",
        "review_count"
    ]

    feature_cols = [
        c for c in candidate_features
        if c in hvle_df.columns
    ]

    modeling_df = hvle_df[
        feature_cols + ["survive_yn"]
    ].copy()

    # Remove rows without target
    modeling_df = modeling_df.dropna(subset=["survive_yn"]).copy()

    # ========================================================
    # 9. Print summary
    # ========================================================

    print("=" * 70)
    print("=== HVLE Retention Prediction Feature Dataset ===")
    print("=" * 70)

    print("Data file:", file_path)
    print("HVLE sample size:", len(modeling_df))
    print(
        "HVLE survival rate:",
        round(modeling_df["survive_yn"].mean() * 100, 2),
        "%"
    )

    print("\nSelected feature columns:")
    for c in feature_cols:
        print("-", c)

    print("\nTarget distribution:")
    print(
        modeling_df["survive_yn"]
        .value_counts()
        .sort_index()
        .rename(index={0: "not_survived", 1: "survived"})
    )

    return {
        "hvle_df": hvle_df,
        "modeling_df": modeling_df,
        "feature_cols": feature_cols,
        "target_col": "survive_yn",
        "value_threshold": value_threshold
    }


if __name__ == "__main__":
    feature_data = build_hvle_retention_features()
    run_model_comparison(
        modeling_df=feature_data["modeling_df"],
        feature_cols=feature_data["feature_cols"],
        target_col=feature_data["target_col"]
    )
