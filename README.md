# Beyond Transaction Value: Value–Engagement Misalignment and Retention Heterogeneity in Pet E-Commerce

## Description

This repository contains the analysis code used to support reproducibility for a study of **value–engagement misalignment and retention heterogeneity in pet e-commerce**.

The study examines whether observable early engagement signals provide additional information beyond early transaction value for distinguishing subsequent customer outcomes. It further identifies high-value, low-engagement (HVLE) customers as a value–engagement misalignment segment and examines behavioral characteristics and retention heterogeneity within this segment.

**Background:** Transaction-based customer segmentation can identify economically important customers, but relatively high early transaction value does not necessarily imply a continuing customer relationship. Observable relationship signals may therefore provide additional information for distinguishing subsequent retention outcomes.

**Framework:** The study proposes a sequential customer management framework of **transaction value identification, misalignment diagnosis, and internal retention stratification**. Transaction value is first used to identify high-value customers, observable early engagement signals are then used to diagnose value–engagement misalignment, and predictive models are subsequently used to distinguish retention potential within the HVLE segment.

**Key Findings:** A greater number of observable early engagement signals was associated with a lower likelihood of fourth-purchase non-completion. Among high-value customers, HVLE customers showed poorer subsequent retention outcomes than high-value, high-engagement (HVHE) customers. Single-category purchasing was associated with a higher likelihood of HVLE membership, whereas exploratory analysis showed that private-brand purchasing was associated with a lower likelihood of HVLE membership. Within the HVLE segment, purchase timing and progression features provided the strongest predictive information for subsequent retention.

## Research Framework

Customers are classified according to early transaction value and observable engagement signals:

```text
HVHE = High Value, High Engagement
HVLE = High Value, Low Engagement
LVHE = Low Value, High Engagement
LVLE = Low Value, Low Engagement
```

The analyses examine:

* the relationship between observable early engagement signals and subsequent churn;
* differences between HVLE and HVHE customers;
* behavioral characteristics associated with HVLE membership; and
* retention heterogeneity within the HVLE segment.

## Data Source

This study uses a publicly available pet e-commerce dataset released by Song (2025).

* GitHub: https://github.com/opusdeisong/Prediction-of-Private-Brand-Purchases-
* Zenodo: https://doi.org/10.5281/zenodo.16296754

The original dataset is not redistributed in this repository and can be obtained directly from the sources above.

## Core Variable Definitions

The main analyses focus on customers who completed their first three purchases.

### Subsequent Retention Outcome

```text
survive_yn = 1 -> fourth purchase observed
survive_yn = 0 -> fourth purchase not observed

churn_yn = 1   -> fourth purchase not observed
churn_yn = 0   -> fourth purchase observed
```

### Observable Early Engagement

Two engagement-related signals are used:

```text
review_written_yn
push_notification_consent_yn
```

They are combined as:

```text
engagement_count =
review_written_yn + push_notification_consent_yn

high_engagement = 1 if engagement_count >= 1
high_engagement = 0 if engagement_count == 0
```

### Early Transaction Value

Early transaction value is measured using:

```text
order_unit_price
```

The median among customers who completed their first three purchases is used as the primary cutoff:

```text
high_value = 1 if order_unit_price >= median
high_value = 0 otherwise
```

For regression models using transaction value as a continuous covariate, `order_unit_price` is log-transformed because of its strongly right-skewed distribution.

## Analysis Structure

### Study 1: Engagement and Subsequent Churn

Study 1 examines whether observable early engagement signals provide information about subsequent churn beyond early transaction value.

#### H1

```text
churn_yn ~ engagement_count
churn_yn ~ engagement_count + log_order_unit_price
```

Supplementary analyses include a zero-versus-two-signal comparison and a continuous value × engagement interaction.

#### H2

Among high-value customers, HVLE and HVHE customers are compared using:

```text
churn_yn ~ HVLE_yn
churn_yn ~ HVLE_yn + log_order_unit_price
```

### Study 2A: HVLE Membership

Study 2A examines behavioral characteristics associated with HVLE membership among high-value customers.

The analyses focus on:

* single-category versus multi-category purchasing; and
* exploratory associations between private-brand purchasing (`pb_purchase_yn`) and HVLE membership.

### Study 2B: HVLE Retention Prediction

Study 2B examines retention heterogeneity within the HVLE segment.

The prediction target is:

```text
survive_yn = 1
```

The following models are compared:

```text
Random Forest
LightGBM
Logistic Regression
Decision Tree
Multilayer Perceptron
```

SHAP is used to interpret the contribution of individual predictors to retention predictions within the HVLE segment.

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

## Conclusion

Based on the above results, this study proposes a customer management framework of **transaction value identification, misalignment diagnosis, and internal retention stratification**. This framework can provide a basis for pet e-commerce platforms to further differentiate high-value customers and optimize the allocation of limited customer retention resources.
