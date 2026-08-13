import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact
import statsmodels.formula.api as smf
import sys
from pathlib import Path


STUDY_2_DIR = Path(__file__).resolve().parents[1]

if str(STUDY_2_DIR) not in sys.path:
    sys.path.append(str(STUDY_2_DIR))

from quadrant_utils import load_hvle_data


def test_h3_purchase_structure(hv_df, structure_col="purchase_structure"):
    """
    H3:
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
    ).reindex(
        index=["multi_category", "single_category"],
        columns=[0, 1],
        fill_value=0
    )

    (
        binary_chi2,
        binary_chi2_p,
        binary_dof,
        binary_expected
    ) = chi2_contingency(
        binary_contingency,
        correction=False  # Uncorrected Pearson chi-square for H2-H5.
    )

    binary_expected_min = binary_expected.min()
    binary_expected_warning = binary_expected_min < 5

    # Exact two-sided robustness test for the imbalanced 2x2 table.
    # With rows ordered as multi then single and columns as non-HVLE then
    # HVLE, an odds ratio above 1 means higher HVLE odds for single-category
    # customers relative to multi-category customers.
    fisher_odds_ratio, fisher_p_value = fisher_exact(
        binary_contingency.to_numpy(),
        alternative="two-sided"
    )

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

    print("=" * 70)
    print("=== H3: Multi vs Single Category Purchase Structure ===")
    print("=" * 70)

    print("\nBinary H3 Summary:")
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

    print("\nFisher's exact robustness test (two-sided):")
    print("Odds ratio (single vs multi):", round(fisher_odds_ratio, 4))
    print("p-value:", round(fisher_p_value, 6))

    print("\nBinary logistic regression odds ratios:")
    print(binary_or_table.round(4).to_string(index=False))

    return {
        "h3_data": df,
        # Original three-category H3 outputs are intentionally disabled.
        # "h4_summary": h4_summary,
        # "contingency_table": contingency,
        # "expected": expected,
        # "chi2": chi2,
        # "chi2_p": chi2_p,
        # "dof": dof,
        # "expected_min": expected_min,
        # "expected_warning": expected_warning,
        # "model": model,
        # "or_table": or_table,
        "binary_h3_data": binary_df,
        "binary_h3_summary": binary_summary,
        "binary_contingency_table": binary_contingency,
        "binary_expected": binary_expected,
        "binary_chi2": binary_chi2,
        "binary_chi2_p": binary_chi2_p,
        "binary_dof": binary_dof,
        "binary_expected_min": binary_expected_min,
        "binary_expected_warning": binary_expected_warning,
        "fisher_odds_ratio": fisher_odds_ratio,
        "fisher_p_value": fisher_p_value,
        "binary_model": binary_model,
        "binary_or_table": binary_or_table
    }


if __name__ == "__main__":
    hvle_data = load_hvle_data()
    test_h3_purchase_structure(hvle_data["hv_df"])
