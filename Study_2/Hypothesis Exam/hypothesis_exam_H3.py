import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import statsmodels.formula.api as smf
import sys
from pathlib import Path


STUDY_2_DIR = Path(__file__).resolve().parents[1]

if str(STUDY_2_DIR) not in sys.path:
    sys.path.append(str(STUDY_2_DIR))

from H3_H4_quadrant_utils import load_hvle_data


def pet_age_to_group(series):
    age_months = pd.to_numeric(
        series,
        errors="coerce"
    )

    age_months = age_months.where(
        age_months >= 0,
        np.nan
    )

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


def test_h3_pet_age_group(hv_df):
    """
    H3:
    Among high-value customers, the likelihood of being classified as HVLE
    rather than HVHE varies significantly across pet age groups.

    DV:
    HVLE_yn = 1 means HVLE
    HVLE_yn = 0 means HVHE
    """

    df = hv_df.copy()

    # --------------------------------------------------------
    # 1. Check required columns
    # --------------------------------------------------------

    required_cols = [
        "HVLE_yn",
        "pet_age_months"
    ]

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.dropna(subset=required_cols).copy()
    df["pet_age_group"] = pet_age_to_group(
        df["pet_age_months"]
    )

    # --------------------------------------------------------
    # 2. Descriptive summary
    # --------------------------------------------------------

    h3_summary = (
        df
        .groupby("pet_age_group")
        .agg(
            n=("HVLE_yn", "size"),
            hvle_n=("HVLE_yn", "sum"),
            hvle_rate=("HVLE_yn", "mean")
        )
        .reset_index()
    )

    h3_summary["hvle_rate_percent"] = (
        h3_summary["hvle_rate"] * 100
    )

    # --------------------------------------------------------
    # 3. Chi-square test
    # --------------------------------------------------------

    contingency = pd.crosstab(
        df["pet_age_group"],
        df["HVLE_yn"]
    )

    chi2, chi2_p, dof, expected = chi2_contingency(contingency)

    # --------------------------------------------------------
    # 4. Logistic regression
    # --------------------------------------------------------

    model = smf.logit(
        "HVLE_yn ~ C(pet_age_group)",
        data=df
    ).fit(disp=False)

    params = model.params
    conf = model.conf_int()

    or_table = pd.DataFrame({
        "beta": params,
        "odds_ratio": np.exp(params),
        "ci_lower": np.exp(conf[0]),
        "ci_upper": np.exp(conf[1]),
        "p_value": model.pvalues
    }).reset_index().rename(columns={"index": "predictor"})

    # --------------------------------------------------------
    # 5. Print results
    # --------------------------------------------------------

    print("=" * 70)
    print("=== H3: Pet Age Group and HVLE Formation ===")
    print("=" * 70)

    print("\nH3 Summary:")
    print(
        h3_summary[
            [
                "pet_age_group",
                "n",
                "hvle_n",
                "hvle_rate_percent"
            ]
        ]
        .round(2)
        .to_string(index=False)
    )

    print("\nChi-square test:")
    print("Chi-square:", round(chi2, 4))
    print("df:", dof)
    print("p-value:", round(chi2_p, 6))

    print("\nLogistic regression odds ratios:")
    print(
        or_table
        .round(4)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # 6. Return results
    # --------------------------------------------------------

    return {
        "h3_data": df,
        "h3_summary": h3_summary,
        "contingency_table": contingency,
        "chi2": chi2,
        "chi2_p": chi2_p,
        "dof": dof,
        "expected": expected,
        "model": model,
        "or_table": or_table
    }


if __name__ == "__main__":
    hvle_data = load_hvle_data()
    test_h3_pet_age_group(hvle_data["hv_df"])
