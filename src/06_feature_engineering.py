import pandas as pd
import numpy as np

df = pd.read_csv("data/emi_prediction_final.csv", low_memory=False)

# --- Total monthly expenses (excluding current EMI, so we can compare separately) ---
expense_cols = [
    "monthly_rent", "school_fees", "college_fees",
    "travel_expenses", "groceries_utilities", "other_monthly_expenses"
]
df["total_monthly_expenses"] = df[expense_cols].sum(axis=1)

# --- Debt-to-Income ratio: existing EMI burden vs salary ---
df["debt_to_income"] = df["current_emi_amount"] / df["monthly_salary"].replace(0, np.nan)
df["debt_to_income"] = df["debt_to_income"].fillna(0)

# --- Expense-to-Income ratio: overall spending burden ---
df["expense_to_income"] = df["total_monthly_expenses"] / df["monthly_salary"].replace(0, np.nan)
df["expense_to_income"] = df["expense_to_income"].fillna(0)

# --- Disposable income: what's left after expenses + existing EMI ---
df["disposable_income"] = df["monthly_salary"] - df["total_monthly_expenses"] - df["current_emi_amount"]

# --- Affordability ratio: requested amount vs disposable income (higher = riskier ask) ---
df["requested_to_disposable"] = df["requested_amount"] / df["disposable_income"].replace(0, np.nan)
df["requested_to_disposable"] = df["requested_to_disposable"].fillna(df["requested_to_disposable"].median())
# cap extreme outliers (e.g. negative or huge disposable-income divisions)
df["requested_to_disposable"] = df["requested_to_disposable"].clip(
    lower=df["requested_to_disposable"].quantile(0.01),
    upper=df["requested_to_disposable"].quantile(0.99)
)

# --- Savings ratio: emergency fund + bank balance relative to salary (financial cushion) ---
df["savings_to_income"] = (df["emergency_fund"] + df["bank_balance"]) / df["monthly_salary"].replace(0, np.nan)
df["savings_to_income"] = df["savings_to_income"].fillna(0)

# --- Employment stability flag ---
df["is_stable_employment"] = (df["years_of_employment"] >= 2).astype(int)

# --- Encode binary categorical ---
df["existing_loans_flag"] = df["existing_loans"].map({"Yes": 1, "No": 0})
df = df.drop(columns=["existing_loans"])  # drop original text column now that it's encoded

# --- One-hot encode remaining categoricals ---
categorical_cols = ["gender", "marital_status", "education", "employment_type",
                     "company_type", "house_type", "emi_scenario"]
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print("New engineered features summary:")
print(df[["debt_to_income", "expense_to_income", "disposable_income",
          "requested_to_disposable", "savings_to_income", "is_stable_employment"]].describe())

print(f"\nShape before encoding: {df.shape}")
print(f"Shape after one-hot encoding: {df_encoded.shape}")

# Sanity check: confirm no object/text columns remain except target columns
non_numeric_remaining = df_encoded.select_dtypes(include=["object"]).columns.tolist()
print(f"\nRemaining non-numeric columns (should only be target col 'emi_eligibility'): {non_numeric_remaining}")

df_encoded.to_csv("data/emi_prediction_engineered.csv", index=False)
print("\nSaved to data/emi_prediction_engineered.csv")