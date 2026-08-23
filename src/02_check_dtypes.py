import pandas as pd

DATA_PATH = "data/emi_prediction_dataset.csv"
df = pd.read_csv(DATA_PATH, low_memory=False)

cols_to_check = ["age", "monthly_salary", "bank_balance", "existing_loans"]

for col in cols_to_check:
    print("=" * 60)
    print(f"COLUMN: {col}")
    print("=" * 60)
    print("Sample unique values (first 20):")
    print(df[col].unique()[:20])
    print("\nValue counts of non-numeric-looking entries:")
    non_numeric = df[col][pd.to_numeric(df[col], errors="coerce").isna()]
    print(non_numeric.value_counts().head(20))
    print(f"\nTotal non-numeric rows in {col}: {non_numeric.shape[0]}")
    print()
    