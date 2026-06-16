# ============================================================
# Fourth-Purchase Churn Prediction
# Sample: customers who completed the first three purchases
# Target: churn_yn = 1 - survive_yn
#
# Models:
# - LightGBM
# - Random Forest
# - Decision Tree
# - Logistic Regression
# ============================================================

import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    make_scorer,
    precision_score,
    recall_score
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from churn_feature_utils import (
    build_churn_model_data,
    build_preprocessor
)

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

warnings.filterwarnings("ignore")


# ============================================================
# 0. Settings
# ============================================================

RANDOM_STATE = 42
N_SPLITS = 5


# ============================================================
# 1. Build modeling dataset
# ============================================================

data = build_churn_model_data()

X = data["X"]
y = data["y"]

preprocessor = build_preprocessor()


# ============================================================
# 2. Models
# ============================================================

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


# ============================================================
# 3. Five-fold cross-validation
# ============================================================

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

    results.append({
        "model": model_name,
        "precision_mean": cv_results["test_precision"].mean(),
        "recall_mean": cv_results["test_recall"].mean(),
        "f1_mean": cv_results["test_f1"].mean(),
        "auc_mean": cv_results["test_auc"].mean()
    })


# ============================================================
# 4. Model-comparison table
# ============================================================

results_df = (
    pd.DataFrame(results)
    .sort_values(
        by="auc_mean",
        ascending=False
    )
    .reset_index(drop=True)
)

print("\n" + "=" * 80)
print("=== Five-Fold Cross-Validation Model Comparison ===")
print("=" * 80)

print(
    results_df
    .round(4)
    .to_string(index=False)
)


# ============================================================
# 5. Feature importance for the best model
# ============================================================

best_model_name = results_df.loc[
    0,
    "model"
]

best_pipeline = Pipeline([
    (
        "preprocessor",
        build_preprocessor()
    ),
    (
        "model",
        models[best_model_name]
    )
])

best_pipeline.fit(
    X,
    y
)

fitted_preprocessor = best_pipeline.named_steps[
    "preprocessor"
]

fitted_classifier = best_pipeline.named_steps[
    "model"
]

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

if hasattr(fitted_classifier, "feature_importances_"):
    importance_values = (
        fitted_classifier
        .feature_importances_
    )
elif hasattr(fitted_classifier, "coef_"):
    importance_values = np.abs(
        fitted_classifier
        .coef_[0]
    )
else:
    raise ValueError(
        f"{best_model_name} does not support direct feature-importance output."
    )

importance_df = (
    pd.DataFrame({
        "feature": feature_names,
        "importance": importance_values
    })
    .sort_values(
        by="importance",
        ascending=False
    )
    .reset_index(drop=True)
)

importance_total = importance_df["importance"].sum()

if importance_total > 0:
    importance_df["importance_share"] = (
        importance_df["importance"]
        / importance_total
    )
else:
    importance_df["importance_share"] = np.nan


print("\n" + "=" * 80)
print("=== Feature Importance: Best Model ===")
print("=" * 80)
print("Best model:", best_model_name)

print(
    importance_df
    .loc[
        :,
        [
            "feature",
            "importance_share"
        ]
    ]
    .head(20)
    .round(6)
    .to_string(index=False)
)
