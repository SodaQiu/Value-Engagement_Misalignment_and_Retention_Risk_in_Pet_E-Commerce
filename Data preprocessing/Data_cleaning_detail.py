from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "original_data_english.csv"

MISSING_LIKE_VALUES = [
    "",
    " ",
    "nan",
    "NaN",
    "NAN",
    "None",
    "NONE",
    "none",
    "null",
    "NULL",
    "Null",
    "-",
    "--",
]

BINARY_MAP = {"Y": 1, "N": 0}


def clean_missing_values(df):
    object_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in object_cols:
        df[col] = df[col].astype("string").str.strip()
    return df.replace(MISSING_LIKE_VALUES, np.nan)


def add_step(rows, step, before_n, after_n, reason):
    rows.append(
        {
            "step": step,
            "n_remaining": after_n,
            "n_excluded": before_n - after_n,
            "exclusion_reason": reason,
        }
    )


def main():
    rows = []
    df = clean_missing_values(pd.read_csv(DATA_PATH, low_memory=False))

    add_step(
        rows,
        "Original customer-level records",
        len(df),
        len(df),
        "Starting sample",
    )

    before = len(df)
    df = df[df["days_to_first_purchase_from_signup"].notna()].copy()
    add_step(
        rows,
        "Customers with first-purchase timing",
        before,
        len(df),
        "Missing days_to_first_purchase_from_signup",
    )

    before = len(df)
    df = df[df["pet_species"].notna()].copy()
    add_step(
        rows,
        "Customers with pet species",
        before,
        len(df),
        "Missing pet_species",
    )

    before = len(df)
    df = df[df["pet_registration_yn"].eq("Y")].copy()
    add_step(
        rows,
        "Cleaned customer-level records",
        before,
        len(df),
        "pet_registration_yn != Y",
    )

    for col in [
        "days_to_third_purchase_from_signup",
        "days_from_second_to_third_purchase",
        "order_unit_price",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["survive_yn", "review_written_yn", "push_notification_consent_yn"]:
        df[col] = df[col].map(BINARY_MAP)

    before = len(df)
    df = df[
        (df["days_to_third_purchase_from_signup"] >= 0)
        & (df["days_from_second_to_third_purchase"] >= 0)
    ].copy()
    add_step(
        rows,
        "Completed first three purchases",
        before,
        len(df),
        "Fewer than three completed purchases or invalid third-purchase timing",
    )

    before = len(df)
    df = df.dropna(
        subset=[
            "order_unit_price",
            "review_written_yn",
            "push_notification_consent_yn",
            "survive_yn",
        ]
    ).copy()
    df = df[df["order_unit_price"] > 0].copy()
    add_step(
        rows,
        "Final Study 1 analytical sample",
        before,
        len(df),
        "Missing/invalid value, engagement, or outcome variables",
    )

    df["engagement_count"] = (
        df["review_written_yn"] + df["push_notification_consent_yn"]
    )
    df["high_engagement"] = (df["engagement_count"] >= 1).astype(int)
    value_threshold = df["order_unit_price"].median()
    df["high_value"] = (df["order_unit_price"] >= value_threshold).astype(int)

    before = len(df)
    high_value_df = df[df["high_value"] == 1].copy()
    add_step(
        rows,
        "High-value subset",
        before,
        len(high_value_df),
        f"Below median early transaction value; median={value_threshold:.4f}",
    )

    before = len(high_value_df)
    hvle_df = high_value_df[high_value_df["high_engagement"] == 0].copy()
    add_step(
        rows,
        "HVLE prediction sample",
        before,
        len(hvle_df),
        "High-value customers classified as HVHE",
    )

    print("\nFigure 3 sample-construction counts")
    print("=" * 80)
    print(pd.DataFrame(rows).to_string(index=False))

    print("\nOutcome check among final Study 1 analytical sample")
    print("=" * 80)
    print(df["survive_yn"].value_counts().rename(index={0: "No", 1: "Yes"}))


if __name__ == "__main__":
    main()
