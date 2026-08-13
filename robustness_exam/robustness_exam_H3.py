import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2


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


def format_p_value(value):
    if value < 0.001:
        return "< .001"
    return f"{value:.4f}"


def format_ci(row):
    return f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]"


def short_model_name(model_name):
    return {
        "Model 1: Unadjusted": "M1 Unadjusted",
        "Model 2: Adjusted for purchase timing": "M2 + timing",
        "Model 3: Additionally adjusted for species": "M3 + timing + species",
        "Model 4: Additionally adjusted for age": "M4 + timing + species + age",
    }.get(model_name, model_name)


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

    return {
        "analysis_df": model_df,
        "n": len(model_df),
        "model": model,
        "or_table": or_table
    }


def likelihood_ratio_test(full_model, reduced_formula, analysis_df):
    reduced_model = smf.logit(reduced_formula, data=analysis_df).fit(disp=False)
    lr_chi2 = 2 * (full_model.llf - reduced_model.llf)

    return {
        "lr_chi2": lr_chi2,
        "df": 1,
        "p_value": chi2.sf(lr_chi2, 1),
        "reduced_model": reduced_model
    }


def run_h3_robustness(hv_df):
    """
    H3 robustness:
    Purchase structure and HVLE formation among high-value customers.

    Models sequentially adjust for purchase timing, pet species,
    and pet age.
    """

    df = hv_df.copy()

    required_cols = [
        "HVLE_yn",
        "purchase_structure",
        "pet_age_months",
        "pet_species",
        "days_to_first_purchase_from_signup",
        "days_from_first_to_second_purchase",
        "days_from_second_to_third_purchase"
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
        "days_from_second_to_third_purchase"
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["pet_age_group"] = pet_age_to_group(df["pet_age_months"])
    df["pet_species_label"] = pet_species_to_label(df["pet_species"])
    df = add_purchase_structure_binary(df)

    print("=" * 70)
    print("=== H3 Robustness: Purchase Structure and HVLE Formation ===")
    print("=" * 70)
    print("Model legend:")
    print("M1 Unadjusted = HVLE_yn ~ purchase_structure_binary")
    print("M2 + timing = M1 + early purchase timing controls")
    print("M3 + timing + species = M2 + pet species")
    print("M4 + timing + species + age = M3 + pet age group")
    counts = df["purchase_structure_binary"].value_counts()
    print(f"N = {len(df)}")
    print(
        "Purchase structure: "
        f"multi-category = {counts.get('multi_category', 0)}, "
        f"single-category = {counts.get('single_category', 0)}"
    )

    formulas = {
        "Model 1: Unadjusted": {
            "formula": (
                "HVLE_yn ~ "
                "C(purchase_structure_binary, Treatment(reference='multi_category'))"
            ),
            "model_cols": [
                "HVLE_yn",
                "purchase_structure_binary"
            ],
            "reduced_formula": "HVLE_yn ~ 1"
        },
        "Model 2: Adjusted for purchase timing": {
            "formula": (
                "HVLE_yn ~ "
                "C(purchase_structure_binary, Treatment(reference='multi_category')) + "
                "days_to_first_purchase_from_signup + "
                "days_from_first_to_second_purchase + "
                "days_from_second_to_third_purchase"
            ),
            "model_cols": [
                "HVLE_yn",
                "purchase_structure_binary",
                "days_to_first_purchase_from_signup",
                "days_from_first_to_second_purchase",
                "days_from_second_to_third_purchase"
            ],
            "reduced_formula": (
                "HVLE_yn ~ "
                "days_to_first_purchase_from_signup + "
                "days_from_first_to_second_purchase + "
                "days_from_second_to_third_purchase"
            )
        },
        "Model 3: Additionally adjusted for species": {
            "formula": (
                "HVLE_yn ~ "
                "C(purchase_structure_binary, Treatment(reference='multi_category')) + "
                "C(pet_species_label, Treatment(reference='dog')) + "
                "days_to_first_purchase_from_signup + "
                "days_from_first_to_second_purchase + "
                "days_from_second_to_third_purchase"
            ),
            "model_cols": [
                "HVLE_yn",
                "purchase_structure_binary",
                "pet_species_label",
                "days_to_first_purchase_from_signup",
                "days_from_first_to_second_purchase",
                "days_from_second_to_third_purchase"
            ],
            "reduced_formula": (
                "HVLE_yn ~ "
                "C(pet_species_label, Treatment(reference='dog')) + "
                "days_to_first_purchase_from_signup + "
                "days_from_first_to_second_purchase + "
                "days_from_second_to_third_purchase"
            )
        },
        "Model 4: Additionally adjusted for age": {
            "formula": (
                "HVLE_yn ~ "
                "C(purchase_structure_binary, Treatment(reference='multi_category')) + "
                "C(pet_species_label, Treatment(reference='dog')) + "
                "C(pet_age_group, Treatment(reference='adult_2_6y')) + "
                "days_to_first_purchase_from_signup + "
                "days_from_first_to_second_purchase + "
                "days_from_second_to_third_purchase"
            ),
            "model_cols": [
                "HVLE_yn",
                "purchase_structure_binary",
                "pet_species_label",
                "pet_age_group",
                "days_to_first_purchase_from_signup",
                "days_from_first_to_second_purchase",
                "days_from_second_to_third_purchase"
            ],
            "reduced_formula": (
                "HVLE_yn ~ "
                "C(pet_species_label, Treatment(reference='dog')) + "
                "C(pet_age_group, Treatment(reference='adult_2_6y')) + "
                "days_to_first_purchase_from_signup + "
                "days_from_first_to_second_purchase + "
                "days_from_second_to_third_purchase"
            )
        }
    }

    results = {}

    for model_name, spec in formulas.items():
        result = fit_logit_model(
            df,
            model_name,
            spec["formula"],
            spec["model_cols"]
        )
        result["likelihood_ratio_test"] = likelihood_ratio_test(
            result["model"],
            spec["reduced_formula"],
            result["analysis_df"]
        )
        results[model_name] = result

    key_rows = []

    for model_name, result in results.items():
        or_table = result["or_table"]
        key = or_table[
            or_table["predictor"].str.contains(
                "C\\(purchase_structure_binary",
                regex=True
            )
        ].copy()
        key.insert(0, "model", model_name)
        key.insert(1, "n", result["n"])
        key_rows.append(key)

    key_results = pd.concat(key_rows, ignore_index=True)

    lr_results = pd.DataFrame([
        {
            "model": model_name,
            "n": result["n"],
            "lr_chi2": result["likelihood_ratio_test"]["lr_chi2"],
            "df": result["likelihood_ratio_test"]["df"],
            "p_value": result["likelihood_ratio_test"]["p_value"]
        }
        for model_name, result in results.items()
    ])

    display_results = key_results.copy()
    display_results["model"] = display_results["model"].map(short_model_name)
    display_results["95% CI"] = display_results.apply(format_ci, axis=1)
    display_results["p"] = display_results["p_value"].map(format_p_value)
    display_results = display_results[
        ["model", "n", "beta", "odds_ratio", "95% CI", "p"]
    ].rename(
        columns={
            "model": "Model",
            "n": "N",
            "beta": "Beta",
            "odds_ratio": "OR",
        }
    )

    print("\nKey effect: single-category vs multi-category")
    print(
        display_results.to_string(
            index=False,
            formatters={
                "Beta": lambda x: f"{x:.4f}",
                "OR": lambda x: f"{x:.4f}",
            },
        )
    )

    lr_display = lr_results.copy()
    lr_display["model"] = lr_display["model"].map(short_model_name)
    lr_display["p"] = lr_display["p_value"].map(format_p_value)
    lr_display = lr_display[["model", "lr_chi2", "df", "p"]].rename(
        columns={
            "model": "Model",
            "lr_chi2": "LR chi-square",
            "df": "df",
        }
    )

    print("\nLikelihood-ratio test for purchase structure")
    print(
        lr_display.to_string(
            index=False,
            formatters={"LR chi-square": lambda x: f"{x:.4f}"},
        )
    )

    return {
        "analysis_df": df,
        "model_results": results,
        "key_results": key_results,
        "lr_results": lr_results
    }


if __name__ == "__main__":
    hvle_data = load_hvle_data()
    run_h3_robustness(hvle_data["hv_df"])
