from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 以下变量均为数值变量 可直接进入模型
NUMERIC_FEATURES = [
    "review_written_yn",
    "push_notification_consent_yn",
    "log_order_unit_price",
    "pet_registration_yn",
    "pet_age_months",
    "pet_registered_before_first_purchase_yn",
    "days_to_first_purchase_from_signup",
    "days_between_purchase_mean",
    "days_between_purchase_change",
    "purchased_feed_ratio",
    "purchased_snacks_ratio",
    "purchased_supplies_ratio",
    "pb_purchase_ratio",
    "first_order_coupon_used_yn",
    "second_order_coupon_used_yn",
    "third_order_coupon_used_yn",
    "weekend_order_ratio",
    "sameday_delivery_ratio",
    "dawn_delivery_ratio",
    "log_average_delivery_time_hours",
    "delivery_time_std"
]

# one-hot encoding变量
CATEGORICAL_FEATURES = [
    "first_purchase_quarter",
    "first_purchase_season",
    "delivery_address",
    "pet_age_group"
]

TARGET = "churn_yn"

FEATURES = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)

# 变量不一定全部直接进入模型 部分用于构造新变量
REQUIRED_COLS = [
    "survive_yn",
    "days_to_third_purchase_from_signup",
    "days_from_second_to_third_purchase",
    "review_written_yn",
    "push_notification_consent_yn",
    "order_unit_price",
    "pet_registration_yn",
    "pet_age_months",
    "pet_registered_before_first_purchase_yn",
    "days_to_first_purchase_from_signup",
    "days_from_first_to_second_purchase",
    "purchased_feed_count",
    "purchased_snacks_count",
    "purchased_supplies_count",
    "purchased_essentials_count",
    "pb_purchase_count",
    "nb_purchase_count",
    "first_order_coupon_name",
    "second_order_coupon_name",
    "third_order_coupon_name",
    "sameday_delivery_orders",
    "dawn_delivery_orders",
    "platform_delivery_orders",
    "first_order_delivery_time_hours",
    "second_order_delivery_time_hours",
    "third_order_delivery_time_hours",
    "average_delivery_time_hours",
    "weekday_order_count",
    "weekend_order_count",
    "first_purchase_month",
    "delivery_address"
]

BINARY_COLS = [
    "review_written_yn",
    "push_notification_consent_yn",
    "pet_registration_yn",
    "pet_registered_before_first_purchase_yn"
]

# 不能为负数的变量
NONNEGATIVE_NUMERIC_COLS = [
    "order_unit_price",
    "pet_age_months",
    "days_to_first_purchase_from_signup",
    "days_from_first_to_second_purchase",
    "days_from_second_to_third_purchase",
    "purchased_feed_count",
    "purchased_snacks_count",
    "purchased_supplies_count",
    "purchased_essentials_count",
    "pb_purchase_count",
    "nb_purchase_count",
    "sameday_delivery_orders",
    "dawn_delivery_orders",
    "platform_delivery_orders",
    "first_order_delivery_time_hours",
    "second_order_delivery_time_hours",
    "third_order_delivery_time_hours",
    "average_delivery_time_hours",
    "weekday_order_count",
    "weekend_order_count"
]

SENTINEL_CHECK_COLS = [
    "days_to_third_purchase_from_signup",
    "days_from_second_to_third_purchase",
    "third_order_delivery_time_hours",
    "average_delivery_time_hours"
]


def default_data_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "output"
        / "pet_data_clean_all_variables.csv"
    )


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    ratio = numerator / denominator

    return (
        ratio
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )


def clean_nonnegative(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(
        series,
        errors="coerce"
    )

    return numeric.where(
        numeric >= 0,
        np.nan
    )


def clean_binary(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(
        series,
        errors="coerce"
    )

    return numeric.where(
        numeric.isin([0, 1]),
        np.nan
    )


def coupon_to_binary(series: pd.Series) -> pd.Series:
    no_coupon_values = {
        "",
        "0",
        "0.0",
        "-1",
        "-1.0",
        "nan",
        "none",
        "null",
        "\uc5c6\uc74c",
        "\ubbf8\uc0ac\uc6a9",
        "no coupon"
    }

    normalized = (
        series
        .astype("string")
        .str.strip()
        .str.lower()
    )

    return (
        normalized.notna()
        & ~normalized.isin(no_coupon_values)
    ).astype(int)


def pet_age_to_group(series: pd.Series) -> pd.Series:
    age_months = clean_nonnegative(series)

    age_group = pd.cut(
        age_months,
        bins=[0, 6, 24, 84, np.inf],
        labels=[
            "baby_0_5m",
            "young_6_23m",
            "adult_2_6y",
            "senior_7y_plus"
        ],
        right=False
    )

    return (
        age_group
        .astype("string")
        .fillna("unknown")
    )


def delivery_address_to_label(series: pd.Series) -> pd.Series:
    normalized = (
        series
        .astype("string")
        .str.strip()
    )

    return (
        normalized
        .map({
            "\uc11c\uc6b8": "seoul",
            "\uacbd\uae30": "gyeonggi",
            "\uadf8\uc678": "other"
        })
        .fillna("unknown")
    )


def parse_first_purchase_month(series: pd.Series) -> pd.Series:
    timestamp_like = pd.to_numeric(
        series,
        errors="coerce"
    )

    return pd.to_datetime(
        timestamp_like * 1000,
        unit="s",
        errors="coerce"
    )


def first_purchase_quarter_to_label(series: pd.Series) -> pd.Series:
    parsed_month = parse_first_purchase_month(series)
    valid_month = parsed_month.notna()

    quarter = pd.Series(
        "unknown",
        index=series.index,
        dtype="string"
    )

    quarter.loc[valid_month] = (
        "q"
        + parsed_month.loc[valid_month]
        .dt.quarter
        .astype("Int64")
        .astype("string")
    )

    return quarter


def first_purchase_season_to_label(series: pd.Series) -> pd.Series:
    parsed_month = parse_first_purchase_month(series)
    month = parsed_month.dt.month

    season = pd.Series(
        "unknown",
        index=series.index,
        dtype="string"
    )

    season.loc[month.isin([3, 4, 5])] = "spring"
    season.loc[month.isin([6, 7, 8])] = "summer"
    season.loc[month.isin([9, 10, 11])] = "fall"
    season.loc[month.isin([12, 1, 2])] = "winter"

    return (
        season
        .astype("string")
        .fillna("unknown")
    )


def build_churn_model_data(data_path=None):
    if data_path is None:
        data_path = default_data_path()
    else:
        data_path = Path(data_path)

    df = pd.read_csv(data_path)

    missing_cols = [
        col
        for col in REQUIRED_COLS
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            "The following required columns are missing. Check the input column names:\n"
            + "\n".join(
                f"- {col}"
                for col in missing_cols
            )
        )

    df["days_to_third_purchase_from_signup"] = pd.to_numeric(
        df["days_to_third_purchase_from_signup"],
        errors="coerce"
    )

    df["days_from_second_to_third_purchase"] = pd.to_numeric(
        df["days_from_second_to_third_purchase"],
        errors="coerce"
    )

    model_df = df[
        (df["days_to_third_purchase_from_signup"] >= 0)
        & (df["days_from_second_to_third_purchase"] >= 0)
    ].copy()

    model_df["survive_yn"] = clean_binary(
        model_df["survive_yn"]
    )

    model_df = model_df.dropna(
        subset=["survive_yn"]
    ).copy()

    model_df["survive_yn"] = (
        model_df["survive_yn"]
        .astype(int)
    )

    model_df["churn_yn"] = (
        1 - model_df["survive_yn"]
    )

    for col in BINARY_COLS:
        model_df[col] = clean_binary(
            model_df[col]
        )

    for col in NONNEGATIVE_NUMERIC_COLS:
        model_df[col] = clean_nonnegative(
            model_df[col]
        )

    model_df["log_order_unit_price"] = np.log1p(
        model_df["order_unit_price"]
    )

    model_df["pet_age_group"] = pet_age_to_group(
        model_df["pet_age_months"]
    )

    model_df["delivery_address"] = delivery_address_to_label(
        model_df["delivery_address"]
    )

    model_df["first_purchase_quarter"] = first_purchase_quarter_to_label(
        model_df["first_purchase_month"]
    )

    model_df["first_purchase_season"] = first_purchase_season_to_label(
        model_df["first_purchase_month"]
    )

    model_df["days_between_purchase_mean"] = (
        model_df["days_from_first_to_second_purchase"]
        + model_df["days_from_second_to_third_purchase"]
    ) / 2

    model_df["days_between_purchase_change"] = (
        model_df["days_from_second_to_third_purchase"]
        - model_df["days_from_first_to_second_purchase"]
    )

    model_df["category_count_total"] = (
        model_df["purchased_feed_count"]
        + model_df["purchased_snacks_count"]
        + model_df["purchased_supplies_count"]
        + model_df["purchased_essentials_count"]
    )

    model_df["purchased_feed_ratio"] = safe_ratio(
        model_df["purchased_feed_count"],
        model_df["category_count_total"]
    )

    model_df["purchased_snacks_ratio"] = safe_ratio(
        model_df["purchased_snacks_count"],
        model_df["category_count_total"]
    )

    model_df["purchased_supplies_ratio"] = safe_ratio(
        model_df["purchased_supplies_count"],
        model_df["category_count_total"]
    )

    model_df["purchased_essentials_ratio"] = safe_ratio(
        model_df["purchased_essentials_count"],
        model_df["category_count_total"]
    )

    model_df["brand_count_total"] = (
        model_df["pb_purchase_count"]
        + model_df["nb_purchase_count"]
    )

    model_df["pb_purchase_ratio"] = safe_ratio(
        model_df["pb_purchase_count"],
        model_df["brand_count_total"]
    )

    model_df["first_order_coupon_used_yn"] = coupon_to_binary(
        model_df["first_order_coupon_name"]
    )

    model_df["second_order_coupon_used_yn"] = coupon_to_binary(
        model_df["second_order_coupon_name"]
    )

    model_df["third_order_coupon_used_yn"] = coupon_to_binary(
        model_df["third_order_coupon_name"]
    )

    model_df["coupon_count"] = (
        model_df["first_order_coupon_used_yn"]
        + model_df["second_order_coupon_used_yn"]
        + model_df["third_order_coupon_used_yn"]
    )

    model_df["weekend_order_ratio"] = safe_ratio(
        model_df["weekend_order_count"],
        model_df["weekday_order_count"]
        + model_df["weekend_order_count"]
    )

    model_df["delivery_count_total"] = (
        model_df["sameday_delivery_orders"]
        + model_df["dawn_delivery_orders"]
        + model_df["platform_delivery_orders"]
    )

    model_df["sameday_delivery_ratio"] = safe_ratio(
        model_df["sameday_delivery_orders"],
        model_df["delivery_count_total"]
    )

    model_df["dawn_delivery_ratio"] = safe_ratio(
        model_df["dawn_delivery_orders"],
        model_df["delivery_count_total"]
    )

    model_df["platform_delivery_ratio"] = safe_ratio(
        model_df["platform_delivery_orders"],
        model_df["delivery_count_total"]
    )

    delivery_time_cols = [
        "first_order_delivery_time_hours",
        "second_order_delivery_time_hours",
        "third_order_delivery_time_hours"
    ]

    model_df["log_average_delivery_time_hours"] = np.log1p(
        model_df["average_delivery_time_hours"]
    )

    model_df["delivery_time_std"] = (
        model_df[delivery_time_cols]
        .std(
            axis=1,
            ddof=0
        )
    )

    for col in SENTINEL_CHECK_COLS:
        negative_n = (
            pd.to_numeric(
                model_df[col],
                errors="coerce"
            ) < 0
        ).sum()

        if negative_n > 0:
            raise ValueError(
                f"{col} still has {negative_n} negative values. Check sentinel-value handling."
            )

    X = model_df[FEATURES].copy()
    y = model_df[TARGET].copy()

    return {
        "data_path": data_path,
        "model_df": model_df,
        "X": X,
        "y": y,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "features": FEATURES,
        "target": TARGET
    }


def build_preprocessor():
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
            NUMERIC_FEATURES
        ),
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES
        )
    ])
