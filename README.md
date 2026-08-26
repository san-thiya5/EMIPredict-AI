# EMIPredict AI — Intelligent Financial Risk Assessment Platform

A FinTech risk-assessment platform that predicts EMI (installment loan) eligibility and affordability using machine learning, built as a 14-day capstone project.

**Live app:** [EMIPredict AI on Streamlit Cloud](https://san-emipredict-ai-26.streamlit.app/)

## What it does

Given a customer's financial profile, the platform answers two questions simultaneously:

1. **Classification** — Is this customer `Eligible`, `High_Risk`, or `Not_Eligible` for an EMI?
2. **Regression** — What is the maximum monthly EMI amount they can afford?

Both predictions are shown together with a confidence breakdown and an affordability comparison against the customer's requested loan.

## Dataset

- 404,800 records across 5 EMI scenarios: E-commerce Shopping, Home Appliances, Vehicle, Personal Loan, Education
- 22 raw input features spanning demographics, employment, housing, monthly expenses, and credit history

## Models

Three models were trained and compared for each task, tracked via MLflow:

| Task | Model | Key Metric | Result |
|---|---|---|---|
| Classification | **XGBoost** (best) | Accuracy / F1 | 97.49% / 0.973 |
| Classification | RandomForest | Accuracy | 87.53% |
| Classification | LogisticRegression | Accuracy | 78.27% |
| Regression | **XGBoost** (best) | RMSE | ₹664.59 |
| Regression | RandomForest | RMSE | ₹899.04 |
| Regression | LinearRegression | RMSE | ₹4,098.56 |

XGBoost was selected for both tasks, comfortably clearing the project's targets (>90% accuracy, <₹2,000 RMSE). See [PROJECT_REPORT.md](PROJECT_REPORT.md) for full methodology, EDA findings, and business impact analysis.

## App structure

The Streamlit app has 4 pages:

- **Home** — real-time prediction (both models, combined results)
- **EDA** — dataset exploration charts and key insights
- **Model Performance** — full comparison of all 6 trained models
- **Admin** — CRUD interface for managing customer records

## Tech stack

Python · pandas · scikit-learn · XGBoost · MLflow · Streamlit

## Running locally

```bash
# 1. Clone the repo
git clone https://github.com/san-thiya5/EMIPredict-AI.git
cd EMIPredict-AI

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\Activate.ps1   # Windows PowerShell

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app/Home.py
```

## Project pipeline (src/)

The `src/` folder contains the full ML pipeline in order:

1. `01_load_inspect.py` — load and inspect the raw dataset
2. `02_check_dtypes.py` — diagnose dtype/data-quality issues
3. `03_clean_data.py` — fix corrupted numeric columns
4. `04_handle_missing.py` — impute missing values
5. `05_eda.py` — generate exploratory data analysis charts
6. `06_feature_engineering.py` — build ratio features and encode categoricals
7. `07_train_classification.py` — train and compare 3 classification models
8. `08_train_regression.py` — train and compare 3 regression models
9. `09_register_models.py` — register winning models in MLflow
10. `10_export_models.py` — export models as standalone files for the app
11. `11_make_sample.py` — generate a lightweight sample dataset for the deployed app

## Author

Santhiya — B.Tech AI & Data Science, VSB Engineering College
