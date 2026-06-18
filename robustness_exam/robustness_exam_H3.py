import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


PROJECT_DIR = Path(__file__).resolve().parents[1]
STUDY_2_DIR = PROJECT_DIR / "Study_2"

if str(STUDY_2_DIR) not in sys.path:
    sys.path.append(str(STUDY_2_DIR))

from quadrant_utils import load_hvle_data


def pet_age_to_group(series):
    age_months = pd.to_numeric(series, errors="coerce")
    age_months = age_months.where(age_months >= 0, np.nan)

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

    return age_group.astype("string").fillna("unknown")


def pet_species_to_label(series):
    species = pd.to_numeric(series, errors="coerce")

    labels = species.map({
        0: "dog",
        1: "cat"
    })

    return labels.fillna("unknown")


def add_purchase_structure_binary(df):
    valid_structures = [
        "multi_category",
        "snacks_only",
        "supplies_only",
        "feed_only",
        "essentials_only"
    ]

    binary_df = df[df["purchase_structure"].isin(valid_structures)].copy()

    binary_df["purchase_structure_binary"] = np.where(
        binary_df["purchase_structure"] == "multi_category",
        "multi_category",
        "single_category"
    )

    return binary_df


def make_or_table(model):
    params = model.params
    conf = model.conf_int()

    return pd.DataFrame({
        "predictor": params.index,
        "beta": params.values,
        "odds_ratio": np.exp(params.values),
        "ci_lower": np.exp(conf[0].values),
        "ci_upper": np.exp(conf[1].values),
        "p_value": model.pvalues.values
    })


def build_model_df(df, model_cols):
    model_df = df.dropna(subset=model_cols).copy()

    if "pet_age_group" in model_cols:
        model_df = model_df[
            model_df["pet_age_group"] != "unknown"
        ].copy()

    if "pet_species_label" in model_cols:
        model_df = model_df[
            model_df["pet_species_label"] != "unknown"
        ].copy()

    return model_df


def fit_logit_model(df, model_name, formula, model_cols):
    model_df = build_model_df(df, model_cols)

    model = smf.logit(formula, data=model_df).fit(disp=False)
    or_table = make_or_table(model)

    print("\n" + "-" * 70)
    print(model_name)
    print("N:", len(model_df))
    print("Formula:")
    print(formula)
    print("\nOdds ratios:")
    print(or_table.round(4).to_string(index=False))

    return {
        "analysis_df": model_df,
        "n": len(model_df),
        "model": model,
        "or_table": or_table
    }


def run_h3_robustness(hv_df):
    """
    H3 robustness:
    Pet age group and HVLE formation among high-value customers.

    Main adjusted model excludes order_unit_price because high-value
    status is already defined by order_unit_price. Model 3 adds
    log_order_unit_price as a robustness check.
    """

    df = hv_df.copy()

    required_cols = [
        "HVLE_yn",
        "pet_age_months",
        "pet_species",
        "purchase_structure",
        "days_to_first_purchase_from_signup",
        "days_from_first_to_second_purchase",
        "days_from_second_to_third_purchase",
        "order_unit_price"
    ]

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for col in [
        "HVLE_yn",
        "pet_age_months",
        "pet_species",
        "days_to_first_purchase_from_signup",
        "days_from_first_to_second_purchase",
        "days_from_second_to_third_purchase",
        "order_unit_price"
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["pet_age_group"] = pet_age_to_group(df["pet_age_months"])
    df["pet_species_label"] = pet_species_to_label(df["pet_species"])
    df = add_purchase_structure_binary(df)

    df["log_order_unit_price"] = np.log1p(df["order_unit_price"])

    print("=" * 70)
    print("=== H3 Robustness: Pet Age Group and HVLE Formation ===")
    print("=" * 70)
    print("Prepared high-value sample size:", len(df))

    print("\nPet age distribution:")
    print(df["pet_age_group"].value_counts().to_string())

    formulas = {
        "Model 1: Unadjusted": {
            "formula": (
                "HVLE_yn ~ "
                "C(pet_age_group, Treatment(reference='adult_2_6y'))"
            ),
            "model_cols": [
                "HVLE_yn",
                "pet_age_group"
            ]
        },
        "Model 2: Adjusted without spending intensity": {
            "formula": (
                "HVLE_yn ~ "
                "C(pet_age_group, Treatment(reference='adult_2_6y')) + "
                "C(pet_species_label, Treatment(reference='dog')) + "
                "C(purchase_structure_binary, Treatment(reference='multi_category')) + "
                "days_to_first_purchase_from_signup + "
                "days_from_first_to_second_purchase + "
                "days_from_second_to_third_purchase"
            ),
            "model_cols": [
                "HVLE_yn",
                "pet_age_group",
                "pet_species_label",
                "purchase_structure_binary",
                "days_to_first_purchase_from_signup",
                "days_from_first_to_second_purchase",
                "days_from_second_to_third_purchase"
            ]
        },
        "Model 3: Robustness with log order unit price": {
            "formula": (
                "HVLE_yn ~ "
                "C(pet_age_group, Treatment(reference='adult_2_6y')) + "
                "C(pet_species_label, Treatment(reference='dog')) + "
                "C(purchase_structure_binary, Treatment(reference='multi_category')) + "
                "days_to_first_purchase_from_signup + "
                "days_from_first_to_second_purchase + "
                "days_from_second_to_third_purchase + "
                "log_order_unit_price"
            ),
            "model_cols": [
                "HVLE_yn",
                "pet_age_group",
                "pet_species_label",
                "purchase_structure_binary",
                "days_to_first_purchase_from_signup",
                "days_from_first_to_second_purchase",
                "days_from_second_to_third_purchase",
                "log_order_unit_price"
            ]
        }
    }

    results = {}

    for model_name, spec in formulas.items():
        results[model_name] = fit_logit_model(
            df,
            model_name,
            spec["formula"],
            spec["model_cols"]
        )

    key_rows = []

    for model_name, result in results.items():
        or_table = result["or_table"]
        key = or_table[
            or_table["predictor"].str.contains(
                "C\\(pet_age_group",
                regex=True
            )
        ].copy()
        key.insert(0, "model", model_name)
        key.insert(1, "n", result["n"])
        key_rows.append(key)

    key_results = pd.concat(key_rows, ignore_index=True)

    print("\n" + "=" * 70)
    print("H3 key pet-age coefficients across models")
    print("=" * 70)
    print(key_results.round(4).to_string(index=False))

    return {
        "analysis_df": df,
        "model_results": results,
        "key_results": key_results
    }


if __name__ == "__main__":
    hvle_data = load_hvle_data()
    run_h3_robustness(hvle_data["hv_df"])
