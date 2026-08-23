import pandas as pd

df = pd.read_csv("data/emi_prediction_cleaned.csv", low_memory=False)

print("Missing values BEFORE imputation:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Categorical -> fill with mode (most frequent value)
df["education"] = df["education"].fillna(df["education"].mode()[0])

# Numeric -> fill with median (robust to outliers/skew, common for financial data)
numeric_cols_with_na = ["monthly_rent", "credit_score", "bank_balance", "emergency_fund"]
for col in numeric_cols_with_na:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)
    print(f"Filled {col} missing values with median: {median_val}")

print("\nMissing values AFTER imputation:")
print(df.isnull().sum().sum(), "total missing values remain")

df.to_csv("data/emi_prediction_final.csv", index=False)
print("\nSaved to data/emi_prediction_final.csv")
print(f"Final shape: {df.shape}")