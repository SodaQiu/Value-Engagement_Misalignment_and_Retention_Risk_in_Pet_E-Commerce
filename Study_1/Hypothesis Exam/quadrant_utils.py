from pathlib import Path

import pandas as pd


# ============================================================
# Required variables
# ============================================================

COHORT_COLS = [
    "days_to_third_purchase_from_signup",
    "days_from_second_to_third_purchase"
]

ANALYSIS_COLS = [
    "order_unit_price",
    "review_written_yn",
    "push_notification_consent_yn",
    "churn_yn"
]

BINARY_COLS = [
    "review_written_yn",
    "push_notification_consent_yn",
    "churn_yn"
]

def load_hypothesis_data(file_path=None):
    """
    Load the cleaned dataset and construct the hypothesis-analysis sample.

    Main analysis cohort:
    Customers who completed the third purchase.

    Analysis sample:
    Customers in the main cohort with valid early transaction value,
    engagement signals, and fourth-purchase churn outcome.

    Outcome:
    churn_yn = 1 means failure to complete a fourth purchase.
    churn_yn = 0 means completion of a fourth purchase.
    """

    # ========================================================
    # 1. Load data
    # ========================================================

    if file_path is None:
        base_dir = Path(__file__).resolve().parents[1]

        file_path = (
            base_dir
            / "output"
            / "pet_data_clean_all_variables.csv"
        )
    else:
        file_path = Path(file_path)

    raw_df = pd.read_csv(
        file_path,
        low_memory=False
    )

    # ========================================================
    # 2. Create churn target when needed
    # ========================================================

    if (
        "churn_yn" not in raw_df.columns
        and "survive_yn" in raw_df.columns
    ):
        raw_df["churn_yn"] = (
            1
            - pd.to_numeric(
                raw_df["survive_yn"],
                errors="coerce"
            )
        )

    # ========================================================
    # 3. Check required columns
    # ========================================================

    required_cols = (
        COHORT_COLS
        + ANALYSIS_COLS
    )

    missing_cols = [
        col
        for col in required_cols
        if col not in raw_df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"以下变量不存在，请检查列名：{missing_cols}"
        )

    # ========================================================
    # 4. Convert required variables to numeric
    # ========================================================

    for col in required_cols:
        raw_df[col] = pd.to_numeric(
            raw_df[col],
            errors="coerce"
        )

    # ========================================================
    # 5. Restrict sample to customers who completed purchase 3
    # ========================================================

    third_purchase_df = raw_df[
        (
            raw_df[
                "days_to_third_purchase_from_signup"
            ] >= 0
        )
        & (
            raw_df[
                "days_from_second_to_third_purchase"
            ] >= 0
        )
    ].copy()

    # ========================================================
    # 6. Validate binary variables in third-purchase cohort
    # ========================================================

    invalid_binary_values = {}

    for col in BINARY_COLS:

        valid_mask = (
            third_purchase_df[col].isna()
            | third_purchase_df[col].isin([0, 1])
        )

        if not valid_mask.all():
            invalid_binary_values[col] = sorted(
                third_purchase_df
                .loc[
                    ~valid_mask,
                    col
                ]
                .dropna()
                .unique()
                .tolist()
            )

    if invalid_binary_values:
        raise ValueError(
            "以下二元变量存在非 0/1 的取值，请检查编码："
            f"{invalid_binary_values}"
        )

    # ========================================================
    # 7. Prepare hypothesis-analysis dataset
    # ========================================================

    analysis_df = (
        third_purchase_df[
            ANALYSIS_COLS
        ]
        .dropna()
        .copy()
    )

    analysis_df = analysis_df[
        analysis_df["order_unit_price"] > 0
    ].copy()

    # ========================================================
    # 8. Construct engagement variables
    # ========================================================

    analysis_df["engagement_count"] = (
        analysis_df["review_written_yn"]
        + analysis_df[
            "push_notification_consent_yn"
        ]
    )

    analysis_df["high_engagement"] = (
        analysis_df["engagement_count"] >= 1
    ).astype(int)

    # ========================================================
    # 9. Define high-value threshold within analysis sample
    # ========================================================

    value_threshold = (
        analysis_df[
            "order_unit_price"
        ]
        .median()
    )

    analysis_df["high_value"] = (
        analysis_df[
            "order_unit_price"
        ] >= value_threshold
    ).astype(int)

    # ========================================================
    # 10. Create engagement summary table
    # ========================================================

    engagement_summary = (
        analysis_df
        .groupby(
            "engagement_count",
            observed=True
        )
        .agg(
            n=(
                "churn_yn",
                "size"
            ),
            churn_n=(
                "churn_yn",
                "sum"
            ),
            churn_rate=(
                "churn_yn",
                "mean"
            )
        )
        .reset_index()
    )

    engagement_summary["churn_rate_percent"] = (
        engagement_summary["churn_rate"] * 100
    )

    engagement_summary["retention_rate_percent"] = (
        (1 - engagement_summary["churn_rate"]) * 100
    )

    # ========================================================
    # 11. Print basic information
    # ========================================================

    print("=" * 70)
    print("=== Value and Engagement Hypothesis Dataset ===")
    print("=" * 70)

    print("Data file:", file_path)
    print("Original cleaned sample:", len(raw_df))
    print(
        "Completed-third-purchase sample:",
        len(third_purchase_df)
    )
    print(
        "Hypothesis-analysis sample:",
        len(analysis_df)
    )
    print(
        "Churn rate:",
        round(
            analysis_df[
                "churn_yn"
            ].mean() * 100,
            2
        ),
        "%"
    )
    print(
        "Value median:",
        value_threshold
    )

    print("\nEngagement summary:")
    print(
        engagement_summary[
            [
                "engagement_count",
                "n",
                "churn_n",
                "churn_rate_percent",
                "retention_rate_percent"
            ]
        ]
        .round(2)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # 12. Return data
    # ========================================================

    return {
        # Original full cleaned dataset
        "raw_df": raw_df,

        # Main cohort for fourth-purchase analyses
        # Keep the key "df" for compatibility with downstream code
        "df": third_purchase_df,

        # Explicit alias
        "third_purchase_df": third_purchase_df,

        # Dataset used for H1/H2 analyses
        "analysis_df": analysis_df,

        # Summary statistics
        "engagement_summary": engagement_summary,
        "value_threshold": value_threshold,

        # Metadata
        "file_path": file_path,
        "binary_cols": BINARY_COLS,
        "cohort_cols": COHORT_COLS,
        "analysis_cols": ANALYSIS_COLS
    }
