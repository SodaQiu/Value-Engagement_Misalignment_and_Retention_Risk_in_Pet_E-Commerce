from pathlib import Path

import pandas as pd


data_path = Path(__file__).resolve().parents[1] / "output" / "pet_data_clean_all_variables.csv"
df = pd.read_csv(data_path, low_memory=False)

order_unit_price = pd.to_numeric(df["order_unit_price"], errors="coerce")

study_sample = df[
    (pd.to_numeric(df["days_to_third_purchase_from_signup"], errors="coerce") >= 0)
    & (pd.to_numeric(df["days_from_second_to_third_purchase"], errors="coerce") >= 0)
].copy()

order_unit_price = pd.to_numeric(
    study_sample["order_unit_price"],
    errors="coerce",
).dropna()

print("Order unit price EDA")
print("N:", len(order_unit_price))
print("Mean:", round(order_unit_price.mean(), 2))
print("Median:", round(order_unit_price.median(), 2))
print("Skewness:", round(order_unit_price.skew(), 3))
print("99th percentile:", round(order_unit_price.quantile(0.99), 2))
print("Maximum:", round(order_unit_price.max(), 0))
