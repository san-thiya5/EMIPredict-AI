# EMIPredict AI — Project Report

## 1. Methodology

### 1.1 Problem Framing

The project required two coupled predictions from the same customer profile:

- **Classification**: `emi_eligibility` ∈ {Eligible, High_Risk, Not_Eligible}
- **Regression**: `max_monthly_emi` (INR) — the maximum affordable monthly installment

Both tasks share the same underlying feature set but are modeled independently, since eligibility is a categorical risk decision while affordability is a continuous financial estimate.

### 1.2 Data Pipeline

1. **Load & inspect** — 404,800 rows, 27 columns confirmed against the expected schema.
2. **Data cleaning** — a data-export bug had duplicated the decimal suffix on three numeric columns (e.g. `58.0` stored as `58.0.0`), and missing values in `bank_balance` were encoded as the literal string `"nan.0"`. Both were fixed with a targeted regex + numeric coercion pass.
3. **Missing value imputation** — five columns had ~0.6% missing values (`education`, `monthly_rent`, `credit_score`, `bank_balance`, `emergency_fund`). Categorical gaps were filled with the mode; numeric gaps with the median, chosen for robustness against the right-skewed financial distributions in this dataset.
4. **Feature engineering** — six new features were derived to capture affordability directly, rather than relying on raw income alone:
   - `debt_to_income` — existing EMI burden relative to salary
   - `expense_to_income` — total monthly expenses relative to salary
   - `disposable_income` — salary minus expenses minus existing EMI
   - `requested_to_disposable` — requested loan amount relative to disposable income
   - `savings_to_income` — emergency fund + bank balance relative to salary
   - `is_stable_employment` — binary flag for ≥2 years employment tenure

   Categorical variables were one-hot encoded, expanding the feature set from 27 to 50 columns.

### 1.3 Model Training

For each task, three models were trained on a 70/15/15 stratified train/validation/test split:

- **Classification**: Logistic Regression, Random Forest, XGBoost — with `class_weight="balanced"` to address the 4.3% minority `High_Risk` class.
- **Regression**: Linear Regression, Random Forest, XGBoost.

All experiments (parameters, metrics, and model artifacts) were tracked using MLflow, and the winning model for each task was registered in the MLflow Model Registry as `EMI_Eligibility_Classifier` and `EMI_Amount_Regressor`.

## 2. EDA Report — Key Findings

- **Class imbalance**: `Not_Eligible` accounts for 77.3% of records, `Eligible` 18.4%, and `High_Risk` only 4.3%. This imbalance directly informed the choice of `class_weight="balanced"` during training and stratified splitting.
- **Approval rates vary sharply by scenario**: Vehicle EMI (86.2% Not_Eligible) and Personal Loan EMI (85.2% Not_Eligible) are far harder to qualify for than E-commerce or Home Appliances EMI (~69% Not_Eligible each). `emi_scenario` is therefore a strong predictive signal.
- **Credit score alone is a weak separator**: mean credit score is 725.6 for Eligible customers vs. 694.1 for Not_Eligible — only a ~31-point gap. Credit score needed to be combined with income/expense signals rather than used in isolation.
- **Affordability correlates more with cash flow than gross income**: the strongest linear correlates of `max_monthly_emi` were `groceries_utilities` (0.48), `bank_balance` (0.46), and `travel_expenses` (0.44) — all higher than `monthly_salary` (0.38). This validated the decision to engineer disposable-income-based features rather than relying on salary alone.
- **No duplicate records** and minimal missingness (<1% per affected column) meant the dataset required targeted cleaning rather than large-scale row removal.

## 3. Model Comparison

### Classification (EMI Eligibility)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **XGBoost** | **97.49%** | **97.32%** | **97.49%** | **97.29%** | **99.86%** |
| Random Forest | 87.53% | 94.23% | 87.53% | 90.11% | 98.74% |
| Logistic Regression | 78.27% | 91.47% | 78.27% | 83.42% | 95.41% |

XGBoost's advantage was most pronounced on the minority `High_Risk` class: 84% precision and 58% recall, versus 24% precision for Random Forest and 14% for Logistic Regression at similar recall. This reflects XGBoost's ability to capture non-linear interactions between debt ratios, employment stability, and credit history that a linear model cannot.

**Known limitation**: `High_Risk` recall (58%) remains the weakest metric across all three models — a direct consequence of it being only 4.3% of the training data. This is disclosed transparently rather than masked, and is flagged below as a future improvement area.

### Regression (Max Monthly EMI)

| Model | RMSE (₹) | MAE (₹) | R² | MAPE |
|---|---|---|---|---|
| **XGBoost** | **664.59** | **222.69** | **0.9926** | 7.46% |
| Random Forest | 899.04 | 263.55 | 0.9865 | 5.38% |
| Linear Regression | 4,098.56 | 2,927.49 | 0.7197 | 191.97% |

XGBoost achieved the lowest RMSE and highest R², clearing the project's target of RMSE < ₹2,000 by a wide margin. Linear Regression's poor performance (R² 0.72) confirms that affordability is driven by non-linear feature interactions that the engineered ratio features alone don't fully linearize.

Note: XGBoost's MAPE (7.46%) is slightly higher than Random Forest's (5.38%), because MAPE is disproportionately sensitive to small true values (the dataset's minimum `max_monthly_emi` is ₹500). Since RMSE, MAE, and R² — the project's primary evaluation criteria — all favor XGBoost, it remains the correct model choice.

### Final Model Selection

**XGBoost was selected for both tasks**, exceeding both stated project targets:

| Target | Required | Achieved |
|---|---|---|
| Classification Accuracy | > 90% | 97.49% |
| Regression RMSE | < ₹2,000 | ₹664.59 |

## 4. Business Impact

- **Faster, more consistent underwriting**: automating eligibility and affordability scoring reduces reliance on manual credit assessment, enabling near-instant EMI pre-approval decisions across all 5 loan scenarios.
- **Reduced default risk exposure**: the affordability model (`max_monthly_emi`) provides a data-driven ceiling on approved EMI amounts, directly informed by disposable income rather than gross salary — reducing the risk of approving loans that exceed a customer's actual repayment capacity.
- **Segment-aware risk pricing**: the sharp difference in approval rates across EMI scenarios (e.g. Vehicle/Personal Loan vs. E-commerce/Home Appliances) suggests scenario-specific risk thresholds could further refine lending policy.
- **Transparency for High_Risk customers**: rather than a binary approve/reject, the three-class output (Eligible / High_Risk / Not_Eligible) allows for a middle-tier response — e.g. conditional approval with additional documentation — instead of an outright rejection.

## 5. Future Improvements

- **Address `High_Risk` recall**: techniques such as SMOTE oversampling or a tuned `scale_pos_weight` could improve recall on the minority class beyond the current 58%, at the cost of some precision.
- **Model monitoring**: as this is trained on a single historical snapshot, periodic retraining and drift monitoring would be needed for production use.
- **Explainability**: adding SHAP value analysis to the Streamlit app would let loan officers see which factors drove a specific eligibility decision, improving trust and regulatory auditability.
