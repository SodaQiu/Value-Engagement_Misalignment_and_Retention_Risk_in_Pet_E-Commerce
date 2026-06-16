import pandas as pd
import numpy as np
from pathlib import Path


PURCHASE_CATEGORY_COLS = [
    "purchased_feed_count",
    "purchased_snacks_count",
    "purchased_supplies_count",
    "purchased_essentials_count"
]

PURCHASE_CATEGORY_LABELS = {
    "purchased_feed_count": "feed",
    "purchased_snacks_count": "snacks",
    "purchased_supplies_count": "supplies",
    "purchased_essentials_count": "essentials"
}


def classify_purchase_structure(row):
    purchased_categories = [
        PURCHASE_CATEGORY_LABELS[col]
        for col in PURCHASE_CATEGORY_COLS
        if row[col] > 0
    ]

    if len(purchased_categories) == 0:
        return "no_category"

    if len(purchased_categories) == 1:
        return f"{purchased_categories[0]}_only"

    return "multi_category"


def load_hvle_data(file_path=None):
    if file_path is None:
        file_path = (
            Path(__file__).resolve().parents[1]
            / "output"
            / "pet_data_clean_all_variables.csv"
        )
    else:
        file_path = Path(file_path)

    df = pd.read_csv(file_path, low_memory=False)

    # 生成 churn_yn
    if "churn_yn" not in df.columns and "survive_yn" in df.columns:
        df["churn_yn"] = 1 - pd.to_numeric(df["survive_yn"], errors="coerce")

    # 需要的变量
    cols = [
        "days_to_third_purchase_from_signup",
        "days_from_second_to_third_purchase",
        "order_unit_price",
        "review_written_yn",
        "push_notification_consent_yn",
        "churn_yn",
        "pet_age_months",
        *PURCHASE_CATEGORY_COLS
    ]

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # 数值变量转换
    num_cols = [
        "days_to_third_purchase_from_signup",
        "days_from_second_to_third_purchase",
        "order_unit_price",
        "review_written_yn",
        "push_notification_consent_yn",
        "churn_yn",
        "pet_age_months",
        *PURCHASE_CATEGORY_COLS
    ]

    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 限定完成前三次购买的用户
    df = df[
        (df["days_to_third_purchase_from_signup"] >= 0) &
        (df["days_from_second_to_third_purchase"] >= 0)
    ].copy()

    # 保留 H3/H4 所需有效样本
    analysis_df = df[cols].dropna().copy()
    analysis_df = analysis_df[analysis_df["order_unit_price"] > 0].copy()

    analysis_df["purchase_structure"] = analysis_df.apply(
        classify_purchase_structure,
        axis=1
    )

    # 构造 engagement
    analysis_df["engagement_count"] = (
        analysis_df["review_written_yn"] +
        analysis_df["push_notification_consent_yn"]
    )

    analysis_df["high_engagement"] = (
        analysis_df["engagement_count"] >= 1
    ).astype(int)

    # 构造 value
    value_threshold = analysis_df["order_unit_price"].median()

    analysis_df["high_value"] = (
        analysis_df["order_unit_price"] >= value_threshold
    ).astype(int)

    # 构造四象限
    analysis_df["quadrant"] = np.select(
        [
            (analysis_df["high_value"] == 1) & (analysis_df["high_engagement"] == 1),
            (analysis_df["high_value"] == 1) & (analysis_df["high_engagement"] == 0),
            (analysis_df["high_value"] == 0) & (analysis_df["high_engagement"] == 1),
            (analysis_df["high_value"] == 0) & (analysis_df["high_engagement"] == 0),
        ],
        ["HVHE", "HVLE", "LVHE", "LVLE"],
        default="Undefined"
    )

    # 四象限分布
    quadrant_summary = (
        analysis_df
        .groupby("quadrant")
        .agg(
            n=("churn_yn", "size"),
            churn_n=("churn_yn", "sum"),
            churn_rate=("churn_yn", "mean")
        )
        .reindex(["HVHE", "HVLE", "LVHE", "LVLE"])
        .reset_index()
    )

    quadrant_summary["sample_percent"] = (
        quadrant_summary["n"] / len(analysis_df) * 100
    )

    quadrant_summary["churn_rate_percent"] = (
        quadrant_summary["churn_rate"] * 100
    )

    quadrant_summary["retention_rate_percent"] = (
        100 - quadrant_summary["churn_rate_percent"]
    )

    # H3/H4 样本：只保留高价值客户
    hv_df = analysis_df[analysis_df["high_value"] == 1].copy()

    # H3/H4 的因变量：1 = HVLE, 0 = HVHE
    hv_df["HVLE_yn"] = (
        hv_df["high_engagement"] == 0
    ).astype(int)

    # 高价值客户内部 HVLE / HVHE 分布
    hvle_summary = (
        hv_df
        .groupby("HVLE_yn")
        .agg(n=("HVLE_yn", "size"))
        .reset_index()
    )

    hvle_summary["group"] = hvle_summary["HVLE_yn"].map({
        0: "HVHE",
        1: "HVLE"
    })

    hvle_summary["percent"] = (
        hvle_summary["n"] / len(hv_df) * 100
    )

    print("\n=== Quadrant Summary ===")
    print(quadrant_summary.round(2).to_string(index=False))

    print("\n=== High-value Customer Summary ===")
    print(hvle_summary[["group", "n", "percent"]].round(2).to_string(index=False))

    return {
        "analysis_df": analysis_df,
        "hv_df": hv_df,
        "quadrant_summary": quadrant_summary,
        "hvle_summary": hvle_summary,
        "value_threshold": value_threshold
    }
