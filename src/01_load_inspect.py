import pandas as pd

# Load the dataset
DATA_PATH = "data/emi_prediction_dataset.csv"
df = pd.read_csv(DATA_PATH)

# Basic shape and structure
print("=" * 60)
print("SHAPE")
print("=" * 60)
print(df.shape)

print("\n" + "=" * 60)
print("COLUMN NAMES & DTYPES")
print("=" * 60)
print(df.dtypes)

print("\n" + "=" * 60)
print("FIRST 5 ROWS")
print("=" * 60)
print(df.head())

print("\n" + "=" * 60)
print("MISSING VALUES PER COLUMN")
print("=" * 60)
print(df.isnull().sum())

print("\n" + "=" * 60)
print("DUPLICATE ROWS")
print("=" * 60)
print(df.duplicated().sum())

print("\n" + "=" * 60)
print("BASIC STATS (numeric columns)")
print("=" * 60)
print(df.describe())

print("\n" + "=" * 60)
print("TARGET COLUMN VALUE COUNTS")
print("=" * 60)
# Adjust these column names once you see the real ones printed above
if "emi_eligibility" in df.columns:
    print(df["emi_eligibility"].value_counts())
if "max_monthly_emi" in df.columns:
    print(df["max_monthly_emi"].describe())