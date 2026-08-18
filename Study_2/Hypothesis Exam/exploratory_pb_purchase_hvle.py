import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact
import statsmodels.formula.api as smf


PROJECT_DIR = Path(__file__).resolve().parents[2]
STUDY_2_DIR = PROJECT_DIR / "Study_2"

if str(STUDY_2_DIR) not in sys.path:
    sys.path.append(str(STUDY_2_DIR))

from quadrant_utils import load_hvle_data


def load_h4_pb_data(file_path=None):
    """Load the high-value HVLE/HVHE sample and append PB purchase variables."""
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


def test_h4_pb_purchase(hv_df):
    """
    H4:
    Among high-value customers, PB purchasers are less likely to be HVLE
    than customers who do not purchase PB products.

    DV:
    HVLE_yn = 1 means HVLE
    HVLE_yn = 0 means HVHE

    IV:
    pb_purchase_yn = 1 means the customer purchased at least one PB product.
    """
    df = hv_df.copy()
    required_cols = ["HVLE_yn", "pb_purchase_yn"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required_cols).copy()
    df = df[df["pb_purchase_yn"].isin([0, 1])].copy()

    summary = (
        df.groupby("pb_purchase_yn")
        .agg(
            n=("HVLE_yn", "size"),
            hvle_n=("HVLE_yn", "sum"),
            hvle_rate=("HVLE_yn", "mean"),
        )
        .rename(index={0: "no_PB", 1: "PB"})
        .reset_index()
    )
    summary["hvle_rate_percent"] = summary["hvle_rate"] * 100

    contingency = pd.crosstab(df["pb_purchase_yn"], df["HVLE_yn"]).reindex(
        index=[0, 1],
        columns=[0, 1],
        fill_value=0,
    )
    contingency.index = ["no_PB", "PB"]

    chi2_stat, chi2_p, dof, expected = chi2_contingency(
        contingency,
        correction=False,
    )
    expected_min = expected.min()
    expected_warning = expected_min < 5

    fisher_or, fisher_p = fisher_exact(
        contingency.to_numpy(),
        alternative="two-sided",
    )

    model = smf.logit(
        "HVLE_yn ~ C(pb_purchase_yn, Treatment(reference=0))",
        data=df,
    ).fit(disp=False)
    params = model.params
    conf = model.conf_int()
    or_table = pd.DataFrame({
        "predictor": params.index,
        "beta": params.values,
        "odds_ratio": np.exp(params.values),
        "ci_lower": np.exp(conf[0].values),
        "ci_upper": np.exp(conf[1].values),
        "p_value": model.pvalues.values,
    })

    print("=" * 70)
    print("=== H4: PB Purchase and HVLE Formation ===")
    print("=" * 70)

    print("\nH4 PB summary:")
    print(
        summary[
            ["pb_purchase_yn", "n", "hvle_n", "hvle_rate_percent"]
        ].round(2).to_string(index=False)
    )

    print("\nPB-purchase by HVLE contingency table:")
    print(contingency.to_string())

    print("\nChi-square test:")
    print("Chi-square:", round(chi2_stat, 4))
    print("df:", dof)
    print("p-value:", round(chi2_p, 6))
    print("Minimum expected frequency:", round(expected_min, 4))

    if expected_warning:
        print(
            "Warning: Some expected frequencies are below 5. "
            "Interpret chi-square results cautiously."
        )

    print("\nFisher's exact test (two-sided):")
    print("Odds ratio (PB vs no PB for HVLE):", round(fisher_or, 4))
    print("p-value:", round(fisher_p, 6))

    print("\nBinary logistic regression odds ratios:")
    print(or_table.round(4).to_string(index=False))

    return {
        "h4_data": df,
        "h4_summary": summary,
        "contingency_table": contingency,
        "expected": expected,
        "chi2": chi2_stat,
        "chi2_p": chi2_p,
        "dof": dof,
        "expected_min": expected_min,
        "expected_warning": expected_warning,
        "fisher_odds_ratio": fisher_or,
        "fisher_p_value": fisher_p,
        "model": model,
        "or_table": or_table,
    }


if __name__ == "__main__":
    h4_data = load_h4_pb_data()
    test_h4_pb_purchase(h4_data["hv_df"])
