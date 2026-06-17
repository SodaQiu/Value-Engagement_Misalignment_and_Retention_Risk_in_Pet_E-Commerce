# Pet E-commerce Analysis

This repository contains a customer analytics workflow for a pet e-commerce dataset. It covers data preprocessing, churn prediction, SHAP interpretation, and hypothesis tests around early engagement, customer value, pet lifecycle, purchase structure, and HVLE formation.

## Project Structure

```text
.
├── Data preprocessing/
│   └── data_cleaning.py
├── Study_1/
│   ├── Hypothesis Exam/
│   │   ├── hypothesis_exam_H1.py
│   │   ├── hypothesis_exam_H2.py
│   │   └── quadrant_utils.py
│   └── churn_predition/
│       ├── churn_exam.py
│       ├── churn_feature_utils.py
│       └── shap_analysis.py
├── Study_2/
│   ├── Hypothesis Exam/
│   │   ├── hypothesis_exam_H3.py
│   │   ├── hypothesis_exam_H4.py
│   │   └── hypothesis_exam_H5.py
│   ├── survival_predition/
│   │   ├── survival_exam.py
│   │   └── exam_shap.py
│   └── quadrant_utils.py
├── original_data_english.csv
├── requirements.txt
└── README.md
```

Generated analysis files are written to `output/`, which is intentionally excluded from Git.

## Environment Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

The project was most recently run with Python 3.11/3.12 and the package versions listed in `requirements.txt`.

## Data Preparation

The preprocessing script expects `original_data_english.csv` in the project root.

Run:

```powershell
python "Data preprocessing\data_cleaning.py"
```

This creates the cleaned analysis datasets under `output/`, including:

```text
output/pet_data_clean_all_variables.csv
output/pet_data_with_ve_variables.csv
```

## Study 1

Study 1 focuses on fourth-purchase churn among customers who completed the first three purchases.

Run hypothesis tests:

```powershell
python "Study_1\Hypothesis Exam\hypothesis_exam_H1.py"
python "Study_1\Hypothesis Exam\hypothesis_exam_H2.py"
```

Run churn prediction models:

```powershell
python "Study_1\churn_predition\churn_exam.py"
```

Run SHAP interpretation:

```powershell
python "Study_1\churn_predition\shap_analysis.py"
```

## Study 2

Study 2 focuses on high-value customers and HVLE formation. In this project:

```text
HVLE = high value, low engagement
HVHE = high value, high engagement
```

Run hypothesis tests:

```powershell
python "Study_2\Hypothesis Exam\hypothesis_exam_H3.py"
python "Study_2\Hypothesis Exam\hypothesis_exam_H4.py"
python "Study_2\Hypothesis Exam\hypothesis_exam_H5.py"
```

Run HVLE survival prediction:

```powershell
python "Study_2\survival_predition\survival_exam.py"
```

Run SHAP interpretation for the Study 2 survival model:

```powershell
python "Study_2\survival_predition\exam_shap.py"
```

## Notes

- Paths contain spaces, so keep quotation marks around script paths in terminal commands.
- `output/` and `.venv/` are ignored by Git.
- Some scripts use optional model libraries such as LightGBM and SHAP. They are included in `requirements.txt`.
