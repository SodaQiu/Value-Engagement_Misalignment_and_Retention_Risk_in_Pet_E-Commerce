import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# 0. 文件路径
# ============================================================

input_candidates = [
    Path("original_data_english.csv"),
    Path("data") / "original_data_english.csv",
]
input_path = next((path for path in input_candidates if path.exists()), input_candidates[0])
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

output_path = output_dir / "pet_data_clean_all_variables.csv"

# ============================================================
# 1. 读取数据
# ============================================================

df = pd.read_csv(input_path, low_memory=False)

print("=" * 70)
print("1. 原始数据")
print("=" * 70)
print("原始数据规模:", df.shape)
print("原始变量数:", len(df.columns))
print(df.columns.tolist())


def parse_datetime_series(series: pd.Series) -> pd.Series:
    """Parse Korean AM/PM date strings and fallback to pandas inference."""
    cleaned = (
        series.astype("string")
        .str.strip()
        .str.replace("오전", "AM", regex=False)
        .str.replace("오후", "PM", regex=False)
    )

    parsed = pd.to_datetime(
        cleaned,
        format="%Y. %m. %d. %p %I:%M:%S",
        errors="coerce",
    )

    fallback_mask = parsed.isna() & cleaned.notna()
    if fallback_mask.any():
        parsed.loc[fallback_mask] = pd.to_datetime(
            cleaned.loc[fallback_mask],
            errors="coerce",
        )

    return parsed


# ============================================================
# 2. 统一空值格式
# ============================================================
# strip() 用于处理 " Y "、" N "、" 없음 " 等前后存在空格的情况

missing_like_values = [
    "", " ", "nan", "NaN", "NAN",
    "None", "NONE", "none",
    "null", "NULL", "Null",
    "등록없음", "없음", "미등록",
    "-", "--"
]

# 仅对字符串列去除首尾空格
object_cols = df.select_dtypes(include=["object", "string"]).columns

for col in object_cols:
    df[col] = df[col].astype("string").str.strip()

df = df.replace(missing_like_values, np.nan)


# ============================================================
# 3. 删除关键列缺失样本，并排除未注册宠物
# ============================================================
# 论文标准：
# 1) 至少存在一次购买记录
# 2) 排除未注册宠物或宠物种类缺失的样本

required_cols = [
    "days_to_first_purchase_from_signup",
    "pet_species"
]

for col in required_cols:
    if col in df.columns:
        before = len(df)
        df = df[df[col].notna()].copy()
        removed = before - len(df)
        print(f"删除 {col} 缺失样本: {removed:,}")

if "pet_registration_yn" in df.columns:
    before = len(df)
    df = df[df["pet_registration_yn"].eq("Y")].copy()
    removed = before - len(df)
    print(f"删除未注册宠物样本 pet_registration_yn != Y: {removed:,}")


# ============================================================
# 4. Y/N 类变量编码为 1/0
# ============================================================

binary_map = {
    "Y": 1, "N": 0,
    "YES": 1, "NO": 0,
    "Yes": 1, "No": 0,
    "yes": 1, "no": 0,
    "TRUE": 1, "FALSE": 0,
    "True": 1, "False": 0,
    "true": 1, "false": 0,
    True: 1, False: 0
}

binary_encoded_cols = []

for col in df.columns:
    non_null_values = set(df[col].dropna().unique())

    # 只处理完全属于 Y/N、Yes/No、True/False 的变量
    if non_null_values and non_null_values.issubset(set(binary_map.keys())):
        df[col] = df[col].map(binary_map).astype("Int64")
        binary_encoded_cols.append(col)

print("\nY/N 二元编码变量:")
print(binary_encoded_cols)


# ============================================================
# 5. 宠物种类编码
# ============================================================
# 狗 = 0，猫 = 1
# 如果存在其他动物，不直接删除，而是保留为缺失值，
# 最后统一填充为 -1，表示 unknown / other

if "pet_species" in df.columns:
    species_map = {
        "dog": 0,
        "Dog": 0,
        "DOG": 0,
        "개": 0,
        "강아지": 0,
        "cat": 1,
        "Cat": 1,
        "CAT": 1,
        "고양이": 1
    }

    original_species = df["pet_species"].copy()
    df["pet_species"] = df["pet_species"].map(species_map)

    unmatched_species = original_species[df["pet_species"].isna()].value_counts()

    print("\n宠物种类编码完成: dog=0, cat=1")

    if len(unmatched_species) > 0:
        print("未匹配的宠物种类，将在最后填充为 -1:")
        print(unmatched_species)


# ============================================================
# 6. Coupon 变量转换为二元存在性指标
# ============================================================
# 处理逻辑：
# - 如果 coupon 列本身已经是 0/1，则保持不变
# - 如果 coupon 列保存的是优惠券名称，则存在值 = 1，缺失值 = 0
# - 不额外生成变量，直接覆盖原始 coupon 列

coupon_cols = [
    col for col in df.columns
    if "coupon" in col.lower()
]

print("\nCoupon 相关变量:")
print(coupon_cols)

for col in coupon_cols:
    non_null_values = set(df[col].dropna().unique())

    # 已经是明确的 0/1 变量
    if non_null_values.issubset({0, 1, "0", "1"}):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("Int64")

    # coupon name 或 coupon code：存在即为 1，缺失即为 0
    else:
        df[col] = df[col].notna().astype("Int64")


# ============================================================
# 7. Count 类变量缺失填 0
# ============================================================
# 业务逻辑：
# 缺失表示该行为没有发生，例如没有评论、没有购买某类产品。
#
# 注意：
# 不再直接使用关键词 "order"，避免误处理 order_unit_price。

explicit_count_cols = [
    # 可以根据实际数据集继续补充变量名
    "pb_purchase_count",
    "nb_purchase_count",
    "purchased_essentials_count",
    "orders_within_2months_of_first_purchase",
    "review_count",
    "delivery_orders"
]

pattern_count_cols = [
    col for col in df.columns
    if (
        col.lower().endswith("_count")
        or col.lower().endswith("_counts")
        or col.lower().endswith("_orders")
        or "within_2months" in col.lower()
    )
]

count_cols = sorted(
    set(explicit_count_cols + pattern_count_cols)
    .intersection(df.columns)
)

for col in count_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(0)

print("\nCount 类填充为 0 的变量:")
print(count_cols)


# ============================================================
# 8. 时间类变量缺失填 -1
# ============================================================
# 业务逻辑：
# - 真实的 0 天或 0 小时具有实际含义
# - 缺失表示该行为未发生或无法观测
# - 因此使用 -1 区分缺失值和真实 0

time_keywords = [
    "days_to_",
    "days_from_",
    "delivery_time",
    "time_hours",
    "hours",
    "elapsed",
    "interval"
]

time_cols = [
    col for col in df.columns
    if any(keyword in col.lower() for keyword in time_keywords)
]

for col in time_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(-1)

print("\n时间类填充为 -1 的变量:")
print(time_cols)


# ============================================================
# 9. 日期变量转换为 Unix timestamp
# ============================================================
# 不额外生成变量，直接覆盖原日期列。
# 无法识别的日期最终填充为 -1。

date_cols = [
    col for col in df.columns
    if (
        "date" in col.lower()
        or col.lower().endswith("_month")
        or col.lower().endswith("_signup")
        or col.lower() == "signup"
    )
    and "days_to_" not in col.lower()
]

converted_date_cols = []

for col in date_cols:
    parsed = parse_datetime_series(df[col])
    valid_mask = parsed.notna()

    # 先创建全部为 -1 的列，避免 NaT 被转换为极端负数
    timestamp_values = pd.Series(-1, index=df.index, dtype="int64")

    timestamp_values.loc[valid_mask] = (
        parsed.loc[valid_mask].astype("int64") // 10**9
    )

    df[col] = timestamp_values
    converted_date_cols.append(col)

print("\n转换为 Unix timestamp 的日期变量:")
print(converted_date_cols)


# ============================================================
# 10. 地址变量地理编码
# ============================================================
# 当前 delivery_address 是地区级别文本：
# - 서울：首尔代表性经纬度
# - 경기：京畿道代表性经纬度
# - 그외 / 缺失：无法精确定位，填 -1
# 保留原始 delivery_address，同时新增经纬度变量，避免用无距离含义的整数覆盖地址。

if "delivery_address" in df.columns:
    address_geocode_map = {
        "서울": (37.5665, 126.9780),
        "경기": (37.4138, 127.5183),
    }

    df["delivery_address_latitude"] = (
        df["delivery_address"].map(lambda value: address_geocode_map.get(value, (-1, -1))[0])
    )
    df["delivery_address_longitude"] = (
        df["delivery_address"].map(lambda value: address_geocode_map.get(value, (-1, -1))[1])
    )

    print("\n收货地址地理编码完成:")
    print(df[["delivery_address_latitude", "delivery_address_longitude"]].value_counts())

coordinate_keywords = [
    "latitude", "longitude", "lat", "lon", "lng"
]

coordinate_cols = [
    col for col in df.columns
    if col.lower() in coordinate_keywords
    or "latitude" in col.lower()
    or "longitude" in col.lower()
]

for col in coordinate_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(-1)

print("\n经纬度类填充为 -1 的变量:")
print(coordinate_cols)


# ============================================================
# 11. 剩余缺失值统一填充
# ============================================================
# 这是本次修改的关键部分。
#
# 已经根据业务逻辑处理过：
# - count 类变量：0
# - time 类变量：-1
# - coupon 类变量：0/1
#
# 剩余无法根据业务含义进一步推断的缺失值：
# - 数值变量：填 -1
# - 分类和文本变量：填 "-1"
#
# 统一使用 -1 表示 unknown / unavailable。

remaining_missing_before = int(df.isna().sum().sum())

numeric_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.columns.difference(numeric_cols)

df[numeric_cols] = df[numeric_cols].fillna(-1)
df[categorical_cols] = df[categorical_cols].fillna("-1")

remaining_missing_after = int(df.isna().sum().sum())

print("\n剩余缺失值统一填充:")
print(f"填充前剩余缺失值数量: {remaining_missing_before:,}")
print(f"填充后剩余缺失值数量: {remaining_missing_after:,}")


# ============================================================
# 12. 异常值保留
# ============================================================
# 论文标准：不直接删除 outliers。
# 后续 EDA 阶段单独检查异常值分布。

print("\n异常值处理: 不删除，保留原始购买行为。")


# ============================================================
# 13. 清洗结果检查
# ============================================================

print("\n" + "=" * 70)
print("清洗后数据")
print("=" * 70)
print("清洗后数据规模:", df.shape)
print("清洗后变量数:", len(df.columns))
print("最终缺失值总数:", int(df.isna().sum().sum()))

missing_report = (
    df.isna()
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)

missing_report.columns = ["variable", "missing_count"]
missing_report["missing_ratio_percent"] = (
    missing_report["missing_count"] / len(df) * 100
)

print("\n缺失值最多的前 30 个变量:")
print(missing_report.head(30))


# ============================================================
# 14. 保存数据
# ============================================================

df.to_csv(output_path, index=False, encoding="utf-8-sig")
# missing_report.to_csv(missing_report_path, index=False, encoding="utf-8-sig")

print("\n已保存清洗后数据:", output_path)
# print("已保存缺失值报告:", missing_report_path)
