import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import statsmodels.formula.api as smf
import sys
from pathlib import Path


STUDY_2_DIR = Path(__file__).resolve().parents[1]

if str(STUDY_2_DIR) not in sys.path:
    sys.path.append(str(STUDY_2_DIR))

from H3_H4_quadrant_utils import load_hvle_data


def test_h4_purchase_structure(hv_df, structure_col="purchase_structure"):
    """
    H4:
    在高价值客户中，购买结构会显著影响用户是否表现为低参与状态。

    DV:
    HVLE_yn = 1 means HVLE
    HVLE_yn = 0 means HVHE

    IV:
    purchase structure based on the first three purchases
    """

    df = hv_df.copy()

    required_cols = ["HVLE_yn", structure_col]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.dropna(subset=required_cols).copy()

    # --------------------------------------------------------
    # 1. Descriptive summary
    # --------------------------------------------------------

    h4_summary = (
        df
        .groupby(structure_col)
        .agg(
            n=("HVLE_yn", "size"),
            hvle_n=("HVLE_yn", "sum"),
            hvle_rate=("HVLE_yn", "mean")
        )
        .reset_index()
    )

    h4_summary["hvle_rate_percent"] = (
        h4_summary["hvle_rate"] * 100
    )

    # --------------------------------------------------------
    # 2. Chi-square test
    # --------------------------------------------------------

    contingency = pd.crosstab(
        df[structure_col],
        df["HVLE_yn"]
    )

    chi2, chi2_p, dof, expected = chi2_contingency(contingency)

    expected_min = expected.min()
    expected_warning = expected_min < 5

    # --------------------------------------------------------
    # 3. Logistic regression
    # --------------------------------------------------------

    formula = f"HVLE_yn ~ C({structure_col})"

    model = smf.logit(
        formula,
        data=df
    ).fit(disp=False)

    params = model.params
    conf = model.conf_int()

    or_table = pd.DataFrame({
        "predictor": params.index,
        "beta": params.values,
        "odds_ratio": np.exp(params.values),
        "ci_lower": np.exp(conf[0].values),
        "ci_upper": np.exp(conf[1].values),
        "p_value": model.pvalues.values
    })

    # --------------------------------------------------------
    # 4. Print results
    # --------------------------------------------------------

    print("=" * 70)
    print("=== H4: Purchase Structure and HVLE Formation ===")
    print("=" * 70)

    print("\nH4 Summary:")
    print(
        h4_summary[
            [
                structure_col,
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
    print("Minimum expected frequency:", round(expected_min, 4))

    if expected_warning:
        print("Warning: Some expected frequencies are below 5. Interpret chi-square results cautiously.")

    print("\nLogistic regression odds ratios:")
    print(or_table.round(4).to_string(index=False))

    # --------------------------------------------------------
    # 5. Binary robustness check:
    # multi_category vs. single_category
    # --------------------------------------------------------

    binary_df = df[
        df[structure_col].isin(
            [
                "multi_category",
                "snacks_only",
                "supplies_only",
                "feed_only",
                "essentials_only"
            ]
        )
    ].copy()

    binary_df["purchase_structure_binary"] = np.where(
        binary_df[structure_col] == "multi_category",
        "multi_category",
        "single_category"
    )

    binary_summary = (
        binary_df
        .groupby("purchase_structure_binary")
        .agg(
            n=("HVLE_yn", "size"),
            hvle_n=("HVLE_yn", "sum"),
            hvle_rate=("HVLE_yn", "mean")
        )
        .reset_index()
    )

    binary_summary["hvle_rate_percent"] = (
        binary_summary["hvle_rate"] * 100
    )

    binary_contingency = pd.crosstab(
        binary_df["purchase_structure_binary"],
        binary_df["HVLE_yn"]
    )

    (
        binary_chi2,
        binary_chi2_p,
        binary_dof,
        binary_expected
    ) = chi2_contingency(binary_contingency)

    binary_expected_min = binary_expected.min()
    binary_expected_warning = binary_expected_min < 5

    binary_model = smf.logit(
        "HVLE_yn ~ C(purchase_structure_binary)",
        data=binary_df
    ).fit(disp=False)

    binary_params = binary_model.params
    binary_conf = binary_model.conf_int()

    binary_or_table = pd.DataFrame({
        "predictor": binary_params.index,
        "beta": binary_params.values,
        "odds_ratio": np.exp(binary_params.values),
        "ci_lower": np.exp(binary_conf[0].values),
        "ci_upper": np.exp(binary_conf[1].values),
        "p_value": binary_model.pvalues.values
    })

    print("\n" + "=" * 70)
    print("=== H4 Binary Check: Multi vs Single Category ===")
    print("=" * 70)

    print("\nBinary H4 Summary:")
    print(
        binary_summary[
            [
                "purchase_structure_binary",
                "n",
                "hvle_n",
                "hvle_rate_percent"
            ]
        ]
        .round(2)
        .to_string(index=False)
    )

    print("\nBinary chi-square test:")
    print("Chi-square:", round(binary_chi2, 4))
    print("df:", binary_dof)
    print("p-value:", round(binary_chi2_p, 6))
    print(
        "Minimum expected frequency:",
        round(binary_expected_min, 4)
    )

    if binary_expected_warning:
        print(
            "Warning: Some expected frequencies are below 5. "
            "Interpret chi-square results cautiously."
        )

    print("\nBinary logistic regression odds ratios:")
    print(binary_or_table.round(4).to_string(index=False))

    return {
        "h4_data": df,
        "h4_summary": h4_summary,
        "contingency_table": contingency,
        "expected": expected,
        "chi2": chi2,
        "chi2_p": chi2_p,
        "dof": dof,
        "expected_min": expected_min,
        "expected_warning": expected_warning,
        "model": model,
        "or_table": or_table,
        "binary_h4_data": binary_df,
        "binary_h4_summary": binary_summary,
        "binary_contingency_table": binary_contingency,
        "binary_expected": binary_expected,
        "binary_chi2": binary_chi2,
        "binary_chi2_p": binary_chi2_p,
        "binary_dof": binary_dof,
        "binary_expected_min": binary_expected_min,
        "binary_expected_warning": binary_expected_warning,
        "binary_model": binary_model,
        "binary_or_table": binary_or_table
    }


if __name__ == "__main__":
    hvle_data = load_hvle_data()
    test_h4_purchase_structure(hvle_data["hv_df"])
