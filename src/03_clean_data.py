import pandas as pd
import re

DATA_PATH = "data/emi_prediction_dataset.csv"
df = pd.read_csv(DATA_PATH, low_memory=False)

def fix_duplicated_decimal(series):
    """Collapse values like '58.0.0' -> '58.0', then convert to numeric.
    Handles the literal string 'nan.0' -> NaN automatically via to_numeric."""
    cleaned = series.astype(str).apply(lambda x: re.sub(r'(\.0){2,}$', '.0', x))
    return pd.to_numeric(cleaned, errors="coerce")

for col in ["age", "monthly_salary", "bank_balance"]:
    before_na = df[col].isna().sum()
    df[col] = fix_duplicated_decimal(df[col])
    after_na = df[col].isna().sum()
    print(f"{col}: NaNs before={before_na}, after cleaning={after_na}")

# Sanity check - these should now be fully numeric
print("\nDtypes after fix:")
print(df[["age", "monthly_salary", "bank_balance"]].dtypes)

print("\nAny remaining non-numeric rows check:")
for col in ["age", "monthly_salary", "bank_balance"]:
    print(f"{col}: {df[col].isna().sum()} missing values total")

# Save cleaned version for the next steps
df.to_csv("data/emi_prediction_cleaned.csv", index=False)
print("\nSaved cleaned file to data/emi_prediction_cleaned.csv")
print(f"Final shape: {df.shape}")