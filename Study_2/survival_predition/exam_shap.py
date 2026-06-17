# ============================================================
# SHAP Analysis for HVLE Fourth-Purchase Survival Prediction
#
# Target:
# survive_yn = 1 means completion of fourth purchase.
# survive_yn = 0 means failure to complete fourth purchase.
# ============================================================

from pathlib import Path
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from survival_exam import (
    RANDOM_STATE,
    build_hvle_retention_features,
    build_preprocessor,
    split_feature_types
)

try:
    import shap
except ImportError as exc:
    raise ImportError(
        "The 'shap' package is not installed in this environment. "
        "Install it first, then rerun this script. "
        "Example: pip install shap"
    ) from exc

warnings.filterwarnings("ignore")


# ============================================================
# 0. Settings
# ============================================================

SHAP_SAMPLE_SIZE = 2000
TOP_N = 20

OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "output"
    / "study_2_hvle_survival_shap"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 1. Build modeling dataset
# ============================================================

feature_data = build_hvle_retention_features()

modeling_df = feature_data["modeling_df"].copy()
feature_cols = feature_data["feature_cols"]
target_col = feature_data["target_col"]

modeling_df[target_col] = pd.to_numeric(
    modeling_df[target_col],
    errors="coerce"
)

modeling_df = (
    modeling_df
    .dropna(subset=[target_col])
    .copy()
)

modeling_df[target_col] = (
    modeling_df[target_col]
    .astype(int)
)

if modeling_df[target_col].nunique() != 2:
    raise ValueError(
        f"{target_col} must contain both 0 and 1."
    )

numeric_features, categorical_features = split_feature_types(
    modeling_df,
    feature_cols
)

X = modeling_df[feature_cols].copy()
y = modeling_df[target_col].copy()


# ============================================================
# 2. Train SHAP model
# ============================================================

model_name = "LR"
classifier = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=RANDOM_STATE
)

pipeline = Pipeline([
    (
        "preprocessor",
        build_preprocessor(
            numeric_features,
            categorical_features
        )
    ),
    (
        "model",
        classifier
    )
])

pipeline.fit(
    X,
    y
)

fitted_preprocessor = pipeline.named_steps[
    "preprocessor"
]

fitted_model = pipeline.named_steps[
    "model"
]


# ============================================================
# 3. Prepare transformed feature matrix
# ============================================================

X_transformed = fitted_preprocessor.transform(
    X
)

if hasattr(X_transformed, "toarray"):
    X_transformed = X_transformed.toarray()

feature_names = (
    fitted_preprocessor
    .get_feature_names_out()
)

feature_names = [
    name
    .replace("numeric__", "")
    .replace("categorical__", "")
    for name in feature_names
]

X_transformed_df = pd.DataFrame(
    X_transformed,
    columns=feature_names,
    index=X.index
)

if len(X_transformed_df) > SHAP_SAMPLE_SIZE:
    X_shap = X_transformed_df.sample(
        n=SHAP_SAMPLE_SIZE,
        random_state=RANDOM_STATE
    )
else:
    X_shap = X_transformed_df.copy()


# ============================================================
# 4. Compute SHAP values for survive_yn = 1
# ============================================================

explainer = shap.LinearExplainer(
    fitted_model,
    X_shap
)

shap_values = explainer.shap_values(
    X_shap
)

if isinstance(shap_values, list):
    shap_values_for_survival = (
        shap_values[1]
        if len(shap_values) > 1
        else shap_values[0]
    )
else:
    shap_values_for_survival = shap_values


# ============================================================
# 5. SHAP importance table
# ============================================================

mean_abs_shap = np.abs(
    shap_values_for_survival
).mean(axis=0)

shap_importance_df = (
    pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    })
    .sort_values(
        by="mean_abs_shap",
        ascending=False
    )
    .reset_index(drop=True)
)

total_shap = shap_importance_df[
    "mean_abs_shap"
].sum()

if total_shap > 0:
    shap_importance_df["shap_share"] = (
        shap_importance_df["mean_abs_shap"]
        / total_shap
    )
else:
    shap_importance_df["shap_share"] = np.nan


print("\n" + "=" * 80)
print("=== SHAP Feature Importance: HVLE Survival ===")
print("=" * 80)
print("Model:", model_name)
print("Target:", target_col)
print("Positive class: survive_yn = 1")
print("SHAP sample size:", len(X_shap))

print(
    shap_importance_df
    .loc[
        :,
        [
            "feature",
            "mean_abs_shap",
            "shap_share"
        ]
    ]
    .head(TOP_N)
    .round(6)
    .to_string(index=False)
)


# ============================================================
# 6. Save SHAP outputs
# ============================================================

bar_plot_path = OUTPUT_DIR / "shap_summary_bar.png"
beeswarm_plot_path = OUTPUT_DIR / "shap_summary_beeswarm.png"

shap_importance_df.to_csv(
    index=False,
    encoding="utf-8-sig"
)

plt.figure()
shap.summary_plot(
    shap_values_for_survival,
    X_shap,
    plot_type="bar",
    max_display=TOP_N,
    show=False
)
plt.tight_layout()
plt.savefig(
    bar_plot_path,
    dpi=300,
    bbox_inches="tight"
)
plt.close()

plt.figure()
shap.summary_plot(
    shap_values_for_survival,
    X_shap,
    max_display=TOP_N,
    show=False
)
plt.tight_layout()
plt.savefig(
    beeswarm_plot_path,
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("\n" + "=" * 80)
print("=== SHAP Outputs Saved ===")
print("=" * 80)
print(bar_plot_path)
print(beeswarm_plot_path)
