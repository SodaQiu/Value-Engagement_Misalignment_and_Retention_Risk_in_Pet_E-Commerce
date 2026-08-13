import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2, chi2_contingency, fisher_exact


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
    return species.map({0: "dog", 1: "cat"}).fillna("unknown")


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


def load_h4_data(file_path=None):
    """Load the common HVLE sample and append PB purchase variables."""
    if file_path is None:
        file_path = PROJECT_DIR / "output" / "pet_data_clean_all_variables.csv"
    else:
        file_path = Path(file_path)

    loaded = load_hvle_data(file_path)
    hv_df = loaded["hv_df"].copy()
    source_df = pd.read_csv(file_path, low_memory=False)

    pb_cols = ["pb_purchase_yn", "pb_purchase_count"]
    missing = [col for col in pb_cols if col not in source_df.columns]
    if missing:
        raise ValueError(f"Missing PB columns in source data: {missing}")

    # load_hvle_data preserves source-row indices after filtering.
    hv_df[pb_cols] = source_df.loc[hv_df.index, pb_cols]
    loaded["hv_df"] = hv_df
    return loaded


def make_or_table(model):
    conf = model.conf_int()
    return pd.DataFrame({
        "predictor": model.params.index,
        "beta": model.params.values,
        "odds_ratio": np.exp(model.params.values),
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
        "Model 3: Additionally adjusted for pet characteristics": (
            "M3 + timing + pet"
        ),
        "Model 4: Additionally adjusted for purchase structure": (
            "M4 + timing + pet + structure"
        ),
    }.get(model_name, model_name)


def build_model_df(df, model_cols):
    model_df = df.dropna(subset=model_cols).copy()
    model_df = model_df[model_df["pb_purchase_yn"].isin([0, 1])].copy()

    if "pet_species_label" in model_cols:
        model_df = model_df[
            model_df["pet_species_label"].isin(["cat", "dog"])
        ].copy()

    if "pet_age_group" in model_cols:
        model_df = model_df[model_df["pet_age_group"] != "unknown"].copy()

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


def run_h4_robustness(hv_df):
    """
    H4 robustness:
    Among high-value customers, PB purchasers are less likely to be HVLE
    than customers who do not purchase PB products.

    Models progressively adjust for early purchase timing, pet
    characteristics, and purchase structure. Order value is excluded
    because it defines the high-value analysis sample.
    """
    df = hv_df.copy()
    required_cols = [
        "HVLE_yn",
        "pb_purchase_yn",
        "pet_species",
        "pet_age_months",
        "purchase_structure",
        "days_to_first_purchase_from_signup",
        "days_from_first_to_second_purchase",
        "days_from_second_to_third_purchase"
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    numeric_cols = [
        "HVLE_yn",
        "pb_purchase_yn",
        "pet_species",
        "pet_age_months",
        "days_to_first_purchase_from_signup",
        "days_from_first_to_second_purchase",
        "days_from_second_to_third_purchase"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["pet_species_label"] = pet_species_to_label(df["pet_species"])
    df["pet_age_group"] = pet_age_to_group(df["pet_age_months"])
    df = add_purchase_structure_binary(df)

    print("=" * 70)
    print("=== H4 Robustness: PB Purchase and HVLE Formation ===")
    print("=" * 70)
    print("Model legend:")
    print("M1 Unadjusted = HVLE_yn ~ pb_purchase_yn")
    print("M2 + timing = M1 + early purchase timing controls")
    print("M3 + timing + pet = M2 + pet species and pet age group")
    print("M4 + timing + pet + structure = M3 + purchase structure")
    print(f"N = {len(df)}")

    unadjusted_df = df.dropna(subset=["HVLE_yn", "pb_purchase_yn"])
    unadjusted_df = unadjusted_df[
        unadjusted_df["pb_purchase_yn"].isin([0, 1])
    ]
    contingency = pd.crosstab(
        unadjusted_df["pb_purchase_yn"], unadjusted_df["HVLE_yn"]
    ).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    pearson_chi2, pearson_p, pearson_df, expected = chi2_contingency(
        contingency, correction=False
    )
    fisher_or, fisher_p = fisher_exact(
        contingency.to_numpy(), alternative="two-sided"
    )

    unadjusted_summary = (
        unadjusted_df.groupby("pb_purchase_yn")
        .agg(
            n=("HVLE_yn", "size"),
            hvle_n=("HVLE_yn", "sum"),
            hvle_rate=("HVLE_yn", "mean"),
        )
        .rename(index={0: "no_PB", 1: "PB"})
        .reset_index()
    )
    unadjusted_summary["hvle_rate_percent"] = (
        unadjusted_summary["hvle_rate"] * 100
    )

    print("\nPB purchase summary")
    print(
        unadjusted_summary[
            ["pb_purchase_yn", "n", "hvle_n", "hvle_rate_percent"]
        ].to_string(
            index=False,
            formatters={"hvle_rate_percent": lambda x: f"{x:.2f}"},
        )
    )
    print(
        "\nPearson chi-square: "
        f"chi2({pearson_df}) = {pearson_chi2:.4f}, "
        f"p = {format_p_value(pearson_p)}"
    )
    print(
        "Fisher exact test: "
        f"OR = {fisher_or:.4f}, p = {format_p_value(fisher_p)}"
    )
    print(f"Minimum expected frequency: {expected.min():.4f}")

    pb_term = "C(pb_purchase_yn, Treatment(reference=0))"
    timing_terms = (
        "days_to_first_purchase_from_signup + "
        "days_from_first_to_second_purchase + "
        "days_from_second_to_third_purchase"
    )
    pet_terms = (
        "C(pet_species_label, Treatment(reference='dog')) + "
        "C(pet_age_group, Treatment(reference='adult_2_6y'))"
    )
    structure_term = (
        "C(purchase_structure_binary, Treatment(reference='multi_category'))"
    )

    formulas = {
        "Model 1: Unadjusted": {
            "formula": f"HVLE_yn ~ {pb_term}",
            "model_cols": ["HVLE_yn", "pb_purchase_yn"],
            "reduced_formula": "HVLE_yn ~ 1"
        },
        "Model 2: Adjusted for purchase timing": {
            "formula": f"HVLE_yn ~ {pb_term} + {timing_terms}",
            "model_cols": [
                "HVLE_yn", "pb_purchase_yn",
                "days_to_first_purchase_from_signup",
                "days_from_first_to_second_purchase",
                "days_from_second_to_third_purchase"
            ],
            "reduced_formula": f"HVLE_yn ~ {timing_terms}"
        },
        "Model 3: Additionally adjusted for pet characteristics": {
            "formula": f"HVLE_yn ~ {pb_term} + {pet_terms} + {timing_terms}",
            "model_cols": [
                "HVLE_yn", "pb_purchase_yn", "pet_species_label",
                "pet_age_group", "days_to_first_purchase_from_signup",
                "days_from_first_to_second_purchase",
                "days_from_second_to_third_purchase"
            ],
            "reduced_formula": f"HVLE_yn ~ {pet_terms} + {timing_terms}"
        },
        "Model 4: Additionally adjusted for purchase structure": {
            "formula": (
                f"HVLE_yn ~ {pb_term} + {pet_terms} + "
                f"{structure_term} + {timing_terms}"
            ),
            "model_cols": [
                "HVLE_yn", "pb_purchase_yn", "pet_species_label",
                "pet_age_group", "purchase_structure_binary",
                "days_to_first_purchase_from_signup",
                "days_from_first_to_second_purchase",
                "days_from_second_to_third_purchase"
            ],
            "reduced_formula": (
                f"HVLE_yn ~ {pet_terms} + {structure_term} + {timing_terms}"
            )
        }
    }

    results = {}
    for model_name, spec in formulas.items():
        result = fit_logit_model(
            df, model_name, spec["formula"], spec["model_cols"]
        )
        result["likelihood_ratio_test"] = likelihood_ratio_test(
            result["model"], spec["reduced_formula"], result["analysis_df"]
        )
        results[model_name] = result

    key_rows = []
    for model_name, result in results.items():
        key = result["or_table"][
            result["or_table"]["predictor"].str.contains(
                "C\\(pb_purchase_yn", regex=True
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

    print("\nKey effect: PB buyers vs non-PB buyers")
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

    print("\nLikelihood-ratio test for PB purchase")
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
        "lr_results": lr_results,
        "contingency_table": contingency,
        "pearson_chi2": pearson_chi2,
        "pearson_p": pearson_p,
        "fisher_or": fisher_or,
        "fisher_p": fisher_p
    }


if __name__ == "__main__":
    h4_data = load_h4_data()
    run_h4_robustness(h4_data["hv_df"])
