# Value-Engagement Misalignment and Retention Risk in Pet E-Commerce

---

This repository contains the analysis code used to support reproducibility for the study on **value-engagement misalignment and subsequent customer retention risk in pet e-commerce**.

The study examines whether customers with relatively high early transaction value but limited observable engagement signals exhibit different subsequent retention outcomes, and whether retention heterogeneity within this segment can be further identified using machine-learning methods.

---

## Research Overview

This study examines value-engagement misalignment by classifying customers according to early transaction value and observable engagement signals.

![Research overview](Figure%201.png)

The core segmentation framework is:

```text
HVHE = High Value, High Engagement
HVLE = High Value, Low Engagement
LVHE = Low Value, High Engagement
LVLE = Low Value, Low Engagement
```

The analysis first compares subsequent churn across engagement and value-engagement groups, then examines behavioral factors associated with HVLE membership and retention heterogeneity within the HVLE segment.

---

## Project Structure

```text
.
|-- Data preprocessing/
|   |-- data_cleaning.py
|   `-- order_unit_price_EDA.py
|
|-- Study_1/
|   |-- Hypothesis Exam/
|   |   |-- quadrant_utils.py
|   |   |-- hypothesis_exam_H1.py
|   |   `-- hypothesis_exam_H2.py
|   |
|   `-- supplementary_analysis/
|       |-- h1_zero_vs_two_signals.py
|       `-- value_engagement_interaction.py
|
|-- Study_2/
|   |-- quadrant_utils.py
|   |
|   |-- Hypothesis Exam/
|   |   |-- hypothesis_exam_H3.py
|   |   `-- exploratory_pb_purchase_hvle.py
|   |
|   `-- survival_prediction/
|       |-- survival_exam.py
|       `-- study2b_shap.py
|
|-- robustness_exam/
|   |-- robustness_exam_H1.py
|   |-- robustness_exam_H2.py
|   |-- robustness_exam_H3.py
|   `-- robustness_exam_H4.py
|
|-- output/
|
`-- README.md
```

---
## Data Source

This study uses a publicly available pet e-commerce dataset released by Song (2025).

* GitHub: [https://github.com/opusdeisong/Prediction-of-Private-Brand-Purchases-](https://github.com/opusdeisong/Prediction-of-Private-Brand-Purchases-)
* Zenodo: [https://doi.org/10.5281/zenodo.16296754](https://doi.org/10.5281/zenodo.16296754)

The dataset is not redistributed in this repository. Users can obtain it directly from the original GitHub repository or Zenodo archive.

---

## Data Preprocessing

The main preprocessing script is:

```text
Data preprocessing/data_cleaning.py
```

The script prepares the customer-level analysis dataset used in the subsequent analyses.

The processed dataset is saved to:

```text
output/pet_data_clean_all_variables.csv
```

The following script provides exploratory analysis of the early transaction-value variable:

```text
Data preprocessing/order_unit_price_EDA.py
```

It summarizes the distribution of:

```text
order_unit_price
```

within the third-purchase analysis cohort.

---

## Core Variable Definitions

### Analysis Cohort

The main hypothesis analyses focus on customers who completed their first three purchases.

Operationally, these observations satisfy the purchase-timing conditions required to identify a completed third purchase.

---

### Subsequent Retention Outcome

Within the third-purchase cohort, customer retention status is classified according to whether a fourth purchase is subsequently observed.

```text
survive_yn = 1  -> fourth purchase observed
survive_yn = 0  -> fourth purchase not observed

churn_yn = 1    -> fourth purchase not observed
churn_yn = 0    -> fourth purchase observed
```

---

### Early Observable Engagement

Two observable engagement-related signals are used:

```text
review_written_yn
push_notification_consent_yn
```

The combined engagement measure is:

```text
engagement_count =
review_written_yn + push_notification_consent_yn
```

Engagement status is defined as:

```text
high_engagement = 1 if engagement_count >= 1
high_engagement = 0 if engagement_count == 0
```

---

### Early Transaction Value

Early transaction value is measured using:

```text
order_unit_price
```

Among customers who completed their first three purchases, the median of `order_unit_price` is used as the primary cutoff for distinguishing relatively high-value and low-value customers.

```text
high_value = 1 if order_unit_price >= median
high_value = 0 otherwise
```

For regression analyses in which transaction value is entered as a continuous covariate, a logarithmic transformation is used to reduce the influence of the strongly right-skewed distribution.

---

# Study 1: Engagement and Subsequent Churn

Study 1 examines the relationship between observable early engagement-related signals and subsequent churn.

---

## H1 Analysis

Main script:

```text
Study_1/Hypothesis Exam/hypothesis_exam_H1.py
```

The analysis evaluates whether a greater number of observable early engagement signals is associated with a lower likelihood of subsequent churn.

Main logistic-regression specifications include:

```text
churn_yn ~ engagement_count
```

and:

```text
churn_yn ~ engagement_count + log_order_unit_price
```

Supplementary H1 scripts are provided in:

```text
Study_1/supplementary_analysis/
```

These scripts report a stricter zero-versus-two-signal contrast and a value-engagement interaction analysis.

---

## H2 Analysis

Main script:

```text
Study_1/Hypothesis Exam/hypothesis_exam_H2.py
```

This analysis focuses on high-value customers and compares HVLE customers with HVHE customers.

The principal model examines:

```text
churn_yn ~ HVLE_yn
```

with an additional specification controlling for continuous early transaction value:

```text
churn_yn ~ HVLE_yn + log_order_unit_price
```

---

# Study 2A: Factors Associated with HVLE Membership

Study 2A focuses on high-value customers and examines behavioral characteristics associated with belonging to the HVLE rather than HVHE segment.

The dependent variable is HVLE membership among customers already classified as high value:

```text
HVLE membership
```

---

## Purchase Structure Analysis

Main script:

```text
Study_2/Hypothesis Exam/hypothesis_exam_H3.py
```

Customers' purchase structures across their first three purchases are categorized as:

```text
single_category
multi_category
```

The analysis evaluates whether customers whose early purchases are concentrated in a single product category are more likely to belong to the HVLE segment.

---

## Exploratory Private-Brand Purchase Analysis

Main script:

```text
Study_2/Hypothesis Exam/exploratory_pb_purchase_hvle.py
```

The principal explanatory variable is:

```text
pb_purchase_yn
```

This variable identifies whether a customer purchased at least one private-brand product during the early purchase period.

This exploratory analysis evaluates the association between private-brand purchasing and HVLE membership.

---

# Study 2B: HVLE Retention Prediction

Study 2B examines retention heterogeneity within the HVLE customer segment.

Main prediction script:

```text
Study_2/survival_prediction/survival_exam.py
```

The prediction target is:

```text
survive_yn = 1
```

indicating that a subsequent fourth purchase is observed.

---

## Prediction Models

The following models are compared:

```text
Random Forest
LightGBM
Logistic Regression
Decision Tree
Multilayer Perceptron
```

LightGBM is included when the `lightgbm` package is available.

---

## SHAP Interpretation

Main script:

```text
Study_2/survival_prediction/study2b_shap.py
```

SHAP analysis is used to examine the contribution of individual predictors to model predictions within the HVLE segment.

The script generates feature-importance outputs and SHAP summary figures.

Outputs are saved to:

```text
output/study_2b_hvle_retention_shap/
```

---
