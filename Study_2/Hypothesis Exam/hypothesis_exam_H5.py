# ============================================================
# H5: Pet species and low-engagement status among high-value customers
#
# Hypothesis:
# Among high-value customers, the likelihood of exhibiting
# low-engagement status differs significantly by pet species.
#
# DV:
#     HVLE_yn = 1 means HVLE
#     HVLE_yn = 0 means HVHE
#
# Sample:
#     high-value customers only, loaded from H3/H4 shared utils
# ============================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import statsmodels.formula.api as smf


STUDY_2_DIR = Path(__file__).resolve().parents[1]

if str(STUDY_2_DIR) not in sys.path:
    sys.path.append(str(STUDY_2_DIR))

from quadrant_utils import load_hvle_data


def pet_species_to_label(series):
    species = pd.to_numeric(series, errors="coerce")

    labels = species.map({
        0: "dog",
        1: "cat"
    })

    return labels.fillna("unknown")


def logistic_result_table(model):
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


def test_h5_pet_species(hv_df):
    """
    H5:
    Among high-value customers, pet species is associated with HVLE status.

    DV:
    HVLE_yn = 1 means HVLE
    HVLE_yn = 0 means HVHE
    """

    df = hv_df.copy()

    required_cols = [
        "HVLE_yn",
        "pet_species"
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.dropna(subset=required_cols).copy()
    df["pet_species_label"] = pet_species_to_label(df["pet_species"])
    df = df[df["pet_species_label"].isin(["cat", "dog"])].copy()
    df["pet_species_label"] = df["pet_species_label"].astype("category")

    print("=" * 70)
    print("=== H5: Pet Species and HVLE Formation ===")
    print("=" * 70)
    print(f"\nHigh-value customer sample size: {len(df):,}")

    print("\nPet species distribution:")
    print(df["pet_species_label"].value_counts(dropna=False).to_string())

    # --------------------------------------------------------
    # 1. Descriptive summary
    # --------------------------------------------------------

    h5_summary = (
        df
        .groupby("pet_species_label", observed=True)
        .agg(
            n=("HVLE_yn", "size"),
            hvle_n=("HVLE_yn", "sum"),
            hvle_rate=("HVLE_yn", "mean")
        )
        .reset_index()
    )

    h5_summary["hvle_rate_percent"] = h5_summary["hvle_rate"] * 100

    print("\nH5 Summary:")
    print(
        h5_summary[
            [
                "pet_species_label",
                "n",
                "hvle_n",
                "hvle_rate_percent"
            ]
        ]
        .round(2)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # 2. Chi-square test
    # --------------------------------------------------------

    contingency = pd.crosstab(
        df["pet_species_label"],
        df["HVLE_yn"]
    )

    chi2, chi2_p, dof, expected = chi2_contingency(contingency)
    expected_min = expected.min()
    expected_warning = expected_min < 5

    print("\nChi-square test:")
    print("Contingency table:")
    print(contingency)
    print("Chi-square:", round(chi2, 4))
    print("df:", dof)
    print("p-value:", round(chi2_p, 6))
    print("Minimum expected frequency:", round(expected_min, 4))

    if expected_warning:
        print("Warning: Some expected frequencies are below 5. Interpret chi-square results cautiously.")

    # --------------------------------------------------------
    # 3. Logistic regression: unadjusted model
    # --------------------------------------------------------

    unadjusted_model = smf.logit(
        "HVLE_yn ~ C(pet_species_label, Treatment(reference='dog'))",
        data=df
    ).fit(disp=False)

    unadjusted_or_table = logistic_result_table(unadjusted_model)

    print("\nUnadjusted logistic regression odds ratios:")
    print(unadjusted_or_table.round(4).to_string(index=False))

    species_effect = unadjusted_or_table[
        unadjusted_or_table["predictor"].str.contains("pet_species_label", regex=False)
    ]

    print("\nH5 key result: pet species effect")
    if species_effect.empty:
        print("No pet species coefficient found. Please check category coding.")
    else:
        print(species_effect.round(4).to_string(index=False))

    print("\nInterpretation guide:")
    if chi2_p < 0.05:
        print("Chi-square result: pet species and HVLE status are significantly associated.")
    else:
        print("Chi-square result: no significant association between pet species and HVLE status.")

    for _, row in species_effect.iterrows():
        print(f"\nUnadjusted logistic result for {row['predictor']}:")
        print(
            "OR = "
            f"{row['odds_ratio']:.4f}, 95% CI = "
            f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}], "
            f"p = {row['p_value']:.4f}"
        )

        if row["p_value"] < 0.05:
            if row["odds_ratio"] > 1:
                print("Interpretation: compared with dog owners, this group is more likely to be HVLE.")
            else:
                print("Interpretation: compared with dog owners, this group is less likely to be HVLE.")
        else:
            print("Interpretation: the difference is not statistically significant.")

    return {
        "h5_data": df,
        "h5_summary": h5_summary,
        "contingency_table": contingency,
        "expected": expected,
        "chi2": chi2,
        "chi2_p": chi2_p,
        "dof": dof,
        "expected_min": expected_min,
        "expected_warning": expected_warning,
        "unadjusted_model": unadjusted_model,
        "unadjusted_or_table": unadjusted_or_table,
        "species_effect": species_effect
    }


if __name__ == "__main__":
    hvle_data = load_hvle_data()
    test_h5_pet_species(hvle_data["hv_df"])
