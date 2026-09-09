"""LR SHAP analysis for Study 2B fourth-purchase non-completion risk."""

from pathlib import Path
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from stage3_modeling_utils import (
    build_models,
    build_preprocessor,
    build_study_2b_retention_features,
    split_feature_types,
)

try:
    import shap
except ImportError as exc:
    raise ImportError("Install the 'shap' package before running this script.") from exc


warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

TEST_SIZE = 0.20
SHAP_SAMPLE_SIZE = 2000
BACKGROUND_SAMPLE_SIZE = 500
TOP_N = 20
RANDOM_STATE = 42
SHAP_MODEL_NAME = "LR"

OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "output"
    / "study_2b_hvle_noncompletion_shap"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


FEATURE_FAMILIES = {
    "purchase_cadence": [
        "days_to_first_purchase_from_signup",
        "days_to_third_purchase_from_signup",
        "purchase_interval_mean",
        "purchase_interval_change",
    ],
    "purchase_category": [
        "purchased_supplies_yn",
        "purchased_snacks_yn",
        "purchased_feed_yn",
        "purchased_essentials_yn",
        "category_flag_count",
    ],
    "brand": [
        "pb_purchase_yn",
        "pb_purchase_count_ratio",
    ],
    "coupon": [
        "coupon_used_count_first3",
        "coupon_dropoff_1_to_3",
    ],
    "delivery": [
        "delivery_time_mean_first3",
        "delivery_time_max_first3",
        "delivery_time_change_1_to_3",
        "sameday_delivery_ratio",
        "dawn_delivery_ratio",
        "weekend_order_ratio",
    ],
    "pet_characteristics": [
        "pet_age_group",
        "pet_species",
        "pet_breed_detailed",
        "pet_registered_before_first_purchase_yn",
    ],
    "calendar": [
        "first_purchase_quarter",
        "is_year_end_first_purchase",
    ],
}


def to_dense(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else matrix


def normalize_binary_shap(raw_values):
    """Return SHAP values for the positive class, noncompletion_yn = 1."""
    if isinstance(raw_values, list):
        return np.asarray(raw_values[1] if len(raw_values) > 1 else raw_values[0])

    values = np.asarray(raw_values)
    if values.ndim == 3:
        return values[:, :, 1] if values.shape[2] > 1 else values[:, :, 0]
    return values


def make_ascii_feature_names(raw_feature_names):
    """Replace non-English category labels with stable English display labels."""
    display_names = []
    breed_lookup = {}

    for raw_name in raw_feature_names:
        name = raw_name.replace("numeric__", "").replace("categorical__", "")

        if name.startswith("pet_breed_detailed_"):
            original_breed = name.replace("pet_breed_detailed_", "", 1)
            if not original_breed.isascii():
                if original_breed not in breed_lookup:
                    breed_lookup[original_breed] = f"breed_{len(breed_lookup) + 1}"
                name = f"pet_breed_detailed_{breed_lookup[original_breed]}"

        if not name.isascii():
            name = name.encode("ascii", "ignore").decode("ascii")
            name = name.strip("_") or "non_english_category"

        display_names.append(name)

    return display_names


def aggregate_onehot_importance(shap_importance_df, source_features):
    grouped = shap_importance_df.copy()
    ordered_sources = sorted(source_features, key=len, reverse=True)

    def source_name(transformed_name):
        for source in ordered_sources:
            if transformed_name == source or transformed_name.startswith(source + "_"):
                return source
        return transformed_name

    grouped["source_feature"] = grouped["feature"].map(source_name)
    return (
        grouped.groupby("source_feature", as_index=False)
        .agg(mean_abs_shap=("mean_abs_shap", "sum"))
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def feature_family(source_feature):
    for family, members in FEATURE_FAMILIES.items():
        if source_feature in members:
            return family
    return "other"


def build_model_pipeline(model_name, numeric_features, categorical_features):
    models = build_models()
    if model_name not in models:
        raise ValueError(f"{model_name} is not available in build_models().")

    return Pipeline(
        [
            (
                "preprocessor",
                build_preprocessor(numeric_features, categorical_features),
            ),
            ("model", models[model_name]),
        ]
    )


def explain_model(fitted_model, X_train_transformed, X_shap):
    if not isinstance(fitted_model, LogisticRegression):
        raise TypeError("This script is configured for LR SHAP only.")

    background_n = min(BACKGROUND_SAMPLE_SIZE, len(X_train_transformed))
    rng = np.random.default_rng(RANDOM_STATE)
    background_positions = rng.choice(
        len(X_train_transformed),
        size=background_n,
        replace=False,
    )
    background = X_train_transformed[background_positions]
    explainer = shap.LinearExplainer(fitted_model, background)
    return normalize_binary_shap(explainer.shap_values(X_shap)), background_n


def main():
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

    numeric_features, categorical_features = split_feature_types(
        modeling_df,
        feature_cols,
    )
    X = modeling_df[feature_cols].copy()
    y = modeling_df[target_col].copy()

    X_train, X_test, y_train, _ = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    if not X_train.index.intersection(X_test.index).empty:
        raise RuntimeError("Training and SHAP test samples overlap.")

    pipeline = build_model_pipeline(
        SHAP_MODEL_NAME,
        numeric_features,
        categorical_features,
    )
    pipeline.fit(X_train, y_train)

    preprocessor = pipeline.named_steps["preprocessor"]
    fitted_model = pipeline.named_steps["model"]
    X_train_transformed = to_dense(preprocessor.transform(X_train))
    X_test_transformed = to_dense(preprocessor.transform(X_test))

    feature_names = make_ascii_feature_names(preprocessor.get_feature_names_out())
    X_test_transformed_df = pd.DataFrame(
        X_test_transformed,
        columns=feature_names,
        index=X_test.index,
    )
    X_shap = (
        X_test_transformed_df.sample(
            n=SHAP_SAMPLE_SIZE,
            random_state=RANDOM_STATE,
        )
        if len(X_test_transformed_df) > SHAP_SAMPLE_SIZE
        else X_test_transformed_df.copy()
    )

    shap_values, background_n = explain_model(
        fitted_model,
        X_train_transformed,
        X_shap,
    )
    if shap_values.shape != X_shap.shape:
        raise ValueError("SHAP values do not match the transformed test matrix.")

    shap_importance_df = (
        pd.Series(np.abs(shap_values).mean(axis=0), index=feature_names)
        .rename_axis("feature")
        .reset_index(name="mean_abs_shap")
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    total = shap_importance_df["mean_abs_shap"].sum()
    shap_importance_df["shap_share"] = (
        shap_importance_df["mean_abs_shap"] / total if total > 0 else np.nan
    )

    grouped_importance_df = aggregate_onehot_importance(
        shap_importance_df,
        feature_cols,
    )
    grouped_total = grouped_importance_df["mean_abs_shap"].sum()
    grouped_importance_df["shap_share"] = (
        grouped_importance_df["mean_abs_shap"] / grouped_total
        if grouped_total > 0
        else np.nan
    )
    grouped_importance_df["feature_family"] = grouped_importance_df[
        "source_feature"
    ].map(feature_family)

    family_importance_df = (
        grouped_importance_df.groupby("feature_family", as_index=False)
        .agg(mean_abs_shap=("mean_abs_shap", "sum"))
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    family_total = family_importance_df["mean_abs_shap"].sum()
    family_importance_df["shap_share"] = (
        family_importance_df["mean_abs_shap"] / family_total
        if family_total > 0
        else np.nan
    )

    model_slug = SHAP_MODEL_NAME.lower()
    beeswarm_plot_path = OUTPUT_DIR / f"{model_slug}_shap_summary_beeswarm.png"

    plt.figure()
    shap.summary_plot(
        shap_values,
        X_shap,
        max_display=TOP_N,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(beeswarm_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("\n" + "=" * 80)
    print("=== STUDY 2B LR SHAP FOR NON-COMPLETION RISK ===")
    print("=" * 80)
    print("Selected SHAP model: LR")
    print("Target: noncompletion_yn")
    print("Positive class: fourth-purchase non-completion = 1")
    print("Positive SHAP values indicate higher fourth-purchase non-completion risk.")
    print("Holdout design: stratified 80/20 split")
    print("Random state:", RANDOM_STATE)
    print("Full sample size:", len(X))
    print("Training sample size:", len(X_train))
    print("Held-out test sample size:", len(X_test))
    print("SHAP sample size:", len(X_shap))
    print("Training/SHAP overlap: 0")
    print("\nLR feature-level mean absolute SHAP:")
    print(shap_importance_df.head(TOP_N).round(6).to_string(index=False))
    print("\nLR grouped source features:")
    print(grouped_importance_df.head(TOP_N).round(6).to_string(index=False))
    print("\nLR grouped feature-family SHAP share:")
    print(family_importance_df.round(6).to_string(index=False))

    print("\nSaved:")
    print(beeswarm_plot_path)


if __name__ == "__main__":
    main()
