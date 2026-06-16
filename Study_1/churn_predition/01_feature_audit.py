# ============================================================
# FEATURE AUDIT FOR FULL CHURN PREDICTION
#
# Goal:
# Inspect the cleaned dataset before building prediction models.
#
# DV:
# churn_yn = 1 if the user did not complete a fourth purchase
# churn_yn = 0 otherwise
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# 0. Load cleaned dataset
# ------------------------------------------------------------

base_dir = Path(__file__).resolve().parents[2]

file_path = (
    base_dir
    / "output"
    / "pet_data_clean_all_variables.csv"
)

df = pd.read_csv(
    file_path,
    low_memory=False
)

print("=" * 100)
print("DATASET LOADED")
print("=" * 100)
print("Data file:", file_path)
print("Rows:", len(df))
print("Columns:", len(df.columns))


# ------------------------------------------------------------
# 1. Construct churn_yn if necessary
# ------------------------------------------------------------

if (
    "churn_yn" not in df.columns
    and "survive_yn" in df.columns
):
    df["churn_yn"] = (
        1
        - pd.to_numeric(
            df["survive_yn"],
            errors="coerce"
        )
    )

if "churn_yn" not in df.columns:
    raise ValueError(
        "数据中不存在 churn_yn 或 survive_yn，"
        "无法构造预测目标。"
    )

df["churn_yn"] = pd.to_numeric(
    df["churn_yn"],
    errors="coerce"
)

invalid_target_values = sorted(
    df.loc[
        df["churn_yn"].notna()
        & ~df["churn_yn"].isin([0, 1]),
        "churn_yn"
    ]
    .unique()
    .tolist()
)

if invalid_target_values:
    raise ValueError(
        "churn_yn 存在非 0/1 值："
        f"{invalid_target_values}"
    )


# ------------------------------------------------------------
# 2. Target-variable summary
# ------------------------------------------------------------

target_summary = (
    df["churn_yn"]
    .value_counts(
        dropna=False
    )
    .rename_axis(
        "churn_yn"
    )
    .reset_index(
        name="n"
    )
)

target_summary["share_percent"] = (
    target_summary["n"]
    / len(df)
    * 100
)

print("\n" + "=" * 100)
print("TARGET SUMMARY")
print("=" * 100)

print(
    target_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)


# ------------------------------------------------------------
# 3. Build column-level audit table
# ------------------------------------------------------------

audit_rows = []

for col in df.columns:
    series = df[col]

    non_missing_n = (
        series.notna().sum()
    )

    missing_n = (
        series.isna().sum()
    )

    unique_n = (
        series.nunique(
            dropna=True
        )
    )

    sample_values = (
        series
        .dropna()
        .astype(str)
        .drop_duplicates()
        .head(5)
        .tolist()
    )

    audit_rows.append(
        {
            "column": col,
            "dtype": str(series.dtype),
            "non_missing_n": non_missing_n,
            "missing_n": missing_n,
            "missing_rate_percent": (
                missing_n
                / len(df)
                * 100
            ),
            "unique_n": unique_n,
            "sample_values": " | ".join(
                sample_values
            )
        }
    )

audit_df = pd.DataFrame(
    audit_rows
)


# ------------------------------------------------------------
# 4. Detect potential leakage variables
#
# This is an automatic warning list.
# Each variable must still be checked manually.
# ------------------------------------------------------------

leakage_keywords = [
    "churn",
    "survive",
    "survival",
    "withdraw",
    "withdrawal",
    "fourth",
    "4th",
    "four_purchase",
    "fourth_purchase",
    "purchase_4",
    "order_4",
    "after_4",
    "final",
    "lifetime",
    "total_purchase",
    "total_order"
]

potential_leakage_cols = []

for col in df.columns:
    col_lower = col.lower()

    if any(
        keyword in col_lower
        for keyword in leakage_keywords
    ):
        potential_leakage_cols.append(
            col
        )


# ------------------------------------------------------------
# 5. Detect ID-like columns
#
# ID columns should usually not be used as predictive features.
# ------------------------------------------------------------

id_keywords = [
    "id",
    "key",
    "user",
    "member",
    "customer",
    "index",
    "no"
]

potential_id_cols = []

for col in df.columns:
    col_lower = col.lower()

    name_match = any(
        keyword in col_lower
        for keyword in id_keywords
    )

    unique_ratio = (
        df[col].nunique(
            dropna=True
        )
        / max(
            df[col].notna().sum(),
            1
        )
    )

    if (
        name_match
        or unique_ratio > 0.95
    ):
        potential_id_cols.append(
            col
        )


# ------------------------------------------------------------
# 6. Detect likely numeric and categorical variables
# ------------------------------------------------------------

numeric_cols = (
    df
    .select_dtypes(
        include=[
            "number"
        ]
    )
    .columns
    .tolist()
)

categorical_cols = (
    df
    .select_dtypes(
        exclude=[
            "number"
        ]
    )
    .columns
    .tolist()
)


# ------------------------------------------------------------
# 7. Print results
# ------------------------------------------------------------

print("\n" + "=" * 100)
print("ALL COLUMNS")
print("=" * 100)

for idx, col in enumerate(
    df.columns,
    start=1
):
    print(
        f"{idx:>3}. {col}"
    )


print("\n" + "=" * 100)
print("POTENTIAL DATA-LEAKAGE VARIABLES")
print("=" * 100)

if potential_leakage_cols:
    for col in potential_leakage_cols:
        print(col)
else:
    print("No automatically detected leakage variables")


print("\n" + "=" * 100)
print("POTENTIAL ID-LIKE VARIABLES")
print("=" * 100)

if potential_id_cols:
    for col in potential_id_cols:
        print(col)
else:
    print("No automatically detected ID-like variables")


print("\n" + "=" * 100)
print("NUMERIC COLUMNS")
print("=" * 100)

for col in numeric_cols:
    print(col)


print("\n" + "=" * 100)
print("CATEGORICAL COLUMNS")
print("=" * 100)

for col in categorical_cols:
    print(col)


print("\n" + "=" * 100)
print("COLUMN AUDIT TABLE")
print("=" * 100)

print(
    audit_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)


# ------------------------------------------------------------
# 8. Save audit results
# ------------------------------------------------------------

audit_output_path = (
    base_dir
    / "output"
    / "full_prediction_feature_audit.csv"
)

audit_df.to_csv(
    audit_output_path,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 100)
print("AUDIT FILE SAVED")
print("=" * 100)
print(audit_output_path)
