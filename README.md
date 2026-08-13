# Value-Engagement Misalignment and Retention Risk in Pet E-Commerce

This repository contains the analysis code for a pet e-commerce retention study. The project examines whether early customer value and observable engagement become misaligned, and whether that misalignment is associated with higher retention risk.

The empirical setting follows customers through their first three purchases and studies two related outcomes:

- Fourth-purchase churn among customers who completed their first three purchases.
- Retention potential among high-value, low-engagement customers.

## Project Structure

```text
.
|-- Data preprocessing/
|   |-- data_cleaning.py
|   `-- order_unit_price_EDA.py
|-- Study_1/
|   `-- Hypothesis Exam/
|       |-- quadrant_utils.py
|       |-- hypothesis_exam_H1.py
|       `-- hypothesis_exam_H2.py
|-- Study_2/
|   |-- quadrant_utils.py
|   |-- Hypothesis Exam/
|   |   |-- hypothesis_exam_H3.py
|   |   `-- hypothesis_exam_H4.py
|   `-- survival_prediction/
|       |-- survival_exam.py
|       `-- study2b_shap.py
|-- robustness_exam/
|   |-- robustness_exam_H1.py
|   |-- robustness_exam_H2.py
|   |-- robustness_exam_H3.py
|   `-- robustness_exam_H4.py
|-- original_data_english.csv
`-- output/
```

## Data Pipeline

The preprocessing script reads the raw transaction-level file:

```text
original_data_english.csv
```

and writes the cleaned analysis dataset to:

```text
output/pet_data_clean_all_variables.csv
```

The cleaning logic in `Data preprocessing/data_cleaning.py`:

- standardizes missing-value encodings;
- filters to customers with required signup, purchase, and pet-registration information;
- converts binary Y/N variables to 0/1;
- encodes pet species;
- converts coupon fields into coupon-presence indicators;
- fills count-like variables with 0 and time-like missing values with -1;
- converts date variables to Unix timestamps;
- adds latitude and longitude fields for available delivery-address categories;
- preserves purchase-value outliers rather than deleting them.

`Data preprocessing/order_unit_price_EDA.py` provides a focused distribution check for `order_unit_price` within the third-purchase cohort.

## Core Definitions

Most hypothesis tests use customers who completed their first three purchases:

```text
days_to_third_purchase_from_signup >= 0
days_from_second_to_third_purchase >= 0
```

The main churn outcome is:

```text
churn_yn = 1 if the customer did not complete a fourth purchase
churn_yn = 0 if the customer completed a fourth purchase
```

Early observable engagement is defined from two signals:

```text
engagement_count = review_written_yn + push_notification_consent_yn
high_engagement = 1 if engagement_count >= 1
high_engagement = 0 if engagement_count == 0
```

High-value customers are identified using the median `order_unit_price` within the relevant analysis sample:

```text
high_value = 1 if order_unit_price >= sample median
high_value = 0 otherwise
```

The project uses four value-engagement quadrants:

```text
HVHE = high value, high engagement
HVLE = high value, low engagement
LVHE = low value, high engagement
LVLE = low value, low engagement
```

## Study 1: Fourth-Purchase Churn

Study 1 tests whether early engagement and value-engagement misalignment are associated with fourth-purchase churn.

`Study_1/Hypothesis Exam/hypothesis_exam_H1.py`

Tests whether customers with more early observable engagement signals have lower fourth-purchase churn.

Main models:

```text
churn_yn ~ engagement_count
churn_yn ~ engagement_count + log_order_unit_price
```

`Study_1/Hypothesis Exam/hypothesis_exam_H2.py`

Tests whether, among high-value customers, HVLE customers have higher fourth-purchase churn than HVHE customers.

Main models:

```text
churn_yn ~ HVLE_yn
churn_yn ~ HVLE_yn + log_order_unit_price
```

The script also reports group-level churn rates, chi-square tests, odds ratios, confidence intervals, and final hypothesis decisions.

## Study 2: HVLE Formation and Retention Potential

Study 2 focuses on the high-value sample and examines which behavioral patterns are associated with becoming HVLE rather than HVHE.

`Study_2/Hypothesis Exam/hypothesis_exam_H3.py`

Tests whether purchase structure is associated with HVLE formation among high-value customers.

The script classifies the first-three-purchase basket structure into:

```text
multi_category
single_category
```

and estimates whether single-category purchasers differ from multi-category purchasers in their probability of being HVLE.

`Study_2/Hypothesis Exam/hypothesis_exam_H4.py`

Tests whether PB purchase behavior is associated with HVLE formation among high-value customers.

Main predictor:

```text
pb_purchase_yn
```

The script reports descriptive summaries, chi-square tests, Fisher exact tests, and logistic-regression odds ratios.

## Study 2B: HVLE Retention Prediction

`Study_2/survival_prediction/survival_exam.py` builds a leakage-aware retention-prediction dataset for HVLE customers and compares multiple classifiers for predicting:

```text
survive_yn = 1
```

Candidate predictors include first-three-purchase category behavior, PB/NB purchase ratios, coupon use, delivery timing, purchase intervals, pet characteristics, and calendar timing.

The model comparison uses stratified 5-fold cross-validation and reports:

- precision;
- recall;
- F1-score;
- AUC;
- Brier score;
- precision and lift among the top 20% highest predicted retention-potential customers.

Available models include Random Forest, Decision Tree, Logistic Regression, MLP, and LightGBM when `lightgbm` is installed.

`Study_2/survival_prediction/study2b_shap.py`

Runs SHAP interpretation for the model selected by the Study 2B selection rule:

```text
AUC -> Precision@20% -> F1-score -> Recall
```

The SHAP script uses an 80/20 stratified holdout split and saves summary figures to:

```text
output/study_2b_hvle_retention_shap/
```

## Robustness Checks

The `robustness_exam/` directory contains sensitivity analyses aligned with the main hypotheses:

- `robustness_exam_H1.py`: replaces the composite engagement count with each engagement signal separately.
- `robustness_exam_H2.py`: varies the high-value cutoff across the 40th percentile, median, and 60th percentile while holding the engagement definition fixed.
- `robustness_exam_H3.py`: tests the purchase-structure effect with sequential controls for purchase timing, pet species, and pet age.
- `robustness_exam_H4.py`: tests the PB-purchase effect with sequential controls for purchase timing, pet characteristics, and purchase structure.

## How to Run

Run all commands from the project root:

```powershell
cd "D:\mum_baby\pet commerce"
```

First generate the cleaned dataset:

```powershell
.\.venv\Scripts\python.exe "Data preprocessing\data_cleaning.py"
```

Optional EDA:

```powershell
.\.venv\Scripts\python.exe "Data preprocessing\order_unit_price_EDA.py"
```

Run Study 1:

```powershell
.\.venv\Scripts\python.exe "Study_1\Hypothesis Exam\hypothesis_exam_H1.py"
.\.venv\Scripts\python.exe "Study_1\Hypothesis Exam\hypothesis_exam_H2.py"
```

Run Study 2:

```powershell
.\.venv\Scripts\python.exe "Study_2\Hypothesis Exam\hypothesis_exam_H3.py"
.\.venv\Scripts\python.exe "Study_2\Hypothesis Exam\hypothesis_exam_H4.py"
```

Run Study 2B prediction and SHAP analysis:

```powershell
.\.venv\Scripts\python.exe "Study_2\survival_prediction\survival_exam.py"
.\.venv\Scripts\python.exe "Study_2\survival_prediction\study2b_shap.py"
```

Run robustness checks:

```powershell
.\.venv\Scripts\python.exe "robustness_exam\robustness_exam_H1.py"
.\.venv\Scripts\python.exe "robustness_exam\robustness_exam_H2.py"
.\.venv\Scripts\python.exe "robustness_exam\robustness_exam_H3.py"
.\.venv\Scripts\python.exe "robustness_exam\robustness_exam_H4.py"
```

## Python Dependencies

The scripts use common scientific Python packages:

```text
pandas
numpy
scipy
statsmodels
scikit-learn
matplotlib
shap
lightgbm
```

`lightgbm` is optional for `survival_exam.py`; the script will still run without it, using the remaining models. `shap` is required for `study2b_shap.py`.

## Notes

The analysis code expects `output/pet_data_clean_all_variables.csv` to exist before running the hypothesis, robustness, prediction, or SHAP scripts. Generate it with `Data preprocessing/data_cleaning.py` whenever the raw data or cleaning rules change.

The raw data file may contain commercially sensitive platform data. Check data-sharing permissions before making the repository public.
