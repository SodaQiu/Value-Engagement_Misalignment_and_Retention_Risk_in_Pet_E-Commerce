# Study 1: Hypothesis Tests

This folder contains the Study 1 analyses examining fourth-purchase churn among customers who completed their first three purchases.

## Data

All scripts use the cleaned analysis dataset:

```text
output/pet_data_clean_all_variables.csv
```

The shared data-loading and sample-construction functions are implemented in:

```text
quadrant_utils.py
```

The Study 1 analysis sample includes customers with non-missing and usable values for:

* `order_unit_price`
* `review_written_yn`
* `push_notification_consent_yn`
* `churn_yn`

The dependent variable is defined as:

```text
churn_yn = 1 if the customer did not complete a fourth purchase
churn_yn = 0 if the customer completed a fourth purchase
```

Early observable engagement is measured as:

```text
engagement_count = review_written_yn + push_notification_consent_yn

high_engagement = 1 if engagement_count >= 1
high_engagement = 0 if engagement_count = 0
```

High-value customers are identified using the median `order_unit_price` within the Study 1 analysis sample.

The original transaction-level dataset is not included in this repository because it was provided by a commercial e-commerce platform and is subject to confidentiality restrictions.

## Files

### `hypothesis_exam_H1.py`

Tests whether a greater number of early observable engagement signals is associated with lower fourth-purchase churn.

Main models:

```text
churn_yn ~ engagement_count
churn_yn ~ engagement_count + log_order_unit_price
```

The script reports the estimated coefficients, odds ratios, confidence intervals, statistical significance, and relevant model-comparison statistics.

### `hypothesis_exam_H2.py`

Tests whether, among high-value customers, low-engagement customers (`HVLE`) have higher fourth-purchase churn than high-engagement customers (`HVHE`).

Main models:

```text
churn_yn ~ HVLE_yn
churn_yn ~ HVLE_yn + log_order_unit_price
```

The script also reports descriptive comparisons between the HVLE and HVHE groups.

### `misalignment_supplemental_test.py`

Provides a supplemental test of the interaction between transaction value and observable engagement.

Main model:

```text
churn_yn ~ log_order_unit_price * high_engagement
```

The script reports:

* the interaction coefficient and odds ratio
* a likelihood-ratio test comparing the interaction model with the corresponding main-effects model
* observed and model-adjusted fourth-purchase completion probabilities for the `HVHE`, `HVLE`, `LVHE`, and `LVLE` groups

Because the regression model predicts churn, the corresponding fourth-purchase completion probability is calculated as:

```text
1 - predicted churn probability
```

## Run

Run the scripts from the project root directory:

```powershell
.\.venv\Scripts\python.exe "Study_1\Hypothesis Exam\hypothesis_exam_H1.py"
.\.venv\Scripts\python.exe "Study_1\Hypothesis Exam\hypothesis_exam_H2.py"
.\.venv\Scripts\python.exe "Study_1\Hypothesis Exam\misalignment_supplemental_test.py"
```

Python package requirements and environment information are provided in the project-level `requirements.txt` and README files.
