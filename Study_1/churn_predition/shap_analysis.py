# ============================================================
# SHAP Analysis for Fourth-Purchase Churn Prediction
#
# Uses the shared feature-engineering pipeline from
# churn_feature_utils.py.
# ============================================================

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from churn_feature_utils import (
    build_churn_model_data,
    build_preprocessor
)

try:
    import shap
except ImportError as exc:
    raise ImportError(
        "The 'shap' package is not installed in this environment. "
        "Install it first, then rerun this script. "
        "Example: pip install shap"
    ) from exc

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

warnings.filterwarnings("ignore")


# ============================================================
# 0. Settings
# ============================================================

RANDOM_STATE = 42
SHAP_SAMPLE_SIZE = 2000
TOP_N = 20

OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "output"
    / "study_1_shap"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 1. Build modeling dataset
# ============================================================

data = build_churn_model_data()

X = data["X"]
y = data["y"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=RANDOM_STATE
)


# ============================================================
# 2. Train model
# ============================================================

if LGBMClassifier is not None:
    model_name = "LGBM"
    classifier = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbosity=-1
    )
else:
    model_name = "RF"
    classifier = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=1
    )

pipeline = Pipeline([
    (
        "preprocessor",
        build_preprocessor()
    ),
    (
        "model",
        classifier
    )
])

pipeline.fit(
    X_train,
    y_train
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
    X_test
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
    index=X_test.index
)

if len(X_transformed_df) > SHAP_SAMPLE_SIZE:
    X_shap = X_transformed_df.sample(
        n=SHAP_SAMPLE_SIZE,
        random_state=RANDOM_STATE
    )
else:
    X_shap = X_transformed_df.copy()


# ============================================================
# 4. Compute SHAP values
# ============================================================

explainer = shap.TreeExplainer(
    fitted_model
)

shap_values = explainer.shap_values(
    X_shap
)

if isinstance(shap_values, list):
    shap_values_for_churn = (
        shap_values[1]
        if len(shap_values) > 1
        else shap_values[0]
    )
elif (
    isinstance(shap_values, np.ndarray)
    and shap_values.ndim == 3
):
    shap_values_for_churn = shap_values[:, :, 1]
else:
    shap_values_for_churn = shap_values


# ============================================================
# 5. SHAP importance table
# ============================================================

mean_abs_shap = np.abs(
    shap_values_for_churn
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
print("=== SHAP Feature Importance ===")
print("=" * 80)
print("Model:", model_name)
print("Training sample size:", len(X_train))
print("Test sample size:", len(X_test))
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
# 6. Save SHAP plots
# ============================================================

bar_plot_path = OUTPUT_DIR / "shap_summary_bar.png"
beeswarm_plot_path = OUTPUT_DIR / "shap_summary_beeswarm.png"

plt.figure()
shap.summary_plot(
    shap_values_for_churn,
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
    shap_values_for_churn,
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
print("=== SHAP Plots Saved ===")
print("=" * 80)
print(bar_plot_path)
print(beeswarm_plot_path)
