import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/emi_prediction_final.csv", low_memory=False)

sns.set_style("whitegrid")

# 1. Target distribution - classification
plt.figure(figsize=(6, 4))
df["emi_eligibility"].value_counts().plot(kind="bar", color=["#4C72B0", "#DD8452", "#C44E52"])
plt.title("EMI Eligibility Distribution")
plt.xlabel("Eligibility Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("notebooks/eda_target_distribution.png")
plt.close()

# 2. Target distribution - regression
plt.figure(figsize=(6, 4))
sns.histplot(df["max_monthly_emi"], bins=50, kde=True)
plt.title("Max Monthly EMI Distribution")
plt.xlabel("Max Monthly EMI (INR)")
plt.tight_layout()
plt.savefig("notebooks/eda_emi_distribution.png")
plt.close()

# 3. Eligibility by EMI scenario
plt.figure(figsize=(8, 5))
pd.crosstab(df["emi_scenario"], df["emi_eligibility"], normalize="index").plot(kind="bar", stacked=True)
plt.title("Eligibility Rate by EMI Scenario")
plt.ylabel("Proportion")
plt.tight_layout()
plt.savefig("notebooks/eda_eligibility_by_scenario.png")
plt.close()

# 4. Correlation heatmap (numeric columns only)
numeric_df = df.select_dtypes(include=["float64", "int64"])
plt.figure(figsize=(14, 10))
sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0, annot=False)
plt.title("Correlation Heatmap - Numeric Features")
plt.tight_layout()
plt.savefig("notebooks/eda_correlation_heatmap.png")
plt.close()

# 5. Credit score vs eligibility
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="emi_eligibility", y="credit_score")
plt.title("Credit Score by Eligibility Class")
plt.tight_layout()
plt.savefig("notebooks/eda_credit_score_by_eligibility.png")
plt.close()

# 6. Salary vs max_monthly_emi
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df.sample(5000, random_state=42), x="monthly_salary", y="max_monthly_emi", alpha=0.4)
plt.title("Monthly Salary vs Max Monthly EMI (sampled 5000 rows)")
plt.tight_layout()
plt.savefig("notebooks/eda_salary_vs_emi.png")
plt.close()

print("All EDA plots saved to notebooks/ folder.")
print("\n--- Key summary stats ---")
print("\nEligibility rate by scenario:")
print(pd.crosstab(df["emi_scenario"], df["emi_eligibility"], normalize="index").round(3))
print("\nCredit score by eligibility (mean):")
print(df.groupby("emi_eligibility")["credit_score"].mean().round(1))
print("\nCorrelation of numeric features with max_monthly_emi (top 10):")
print(numeric_df.corr()["max_monthly_emi"].sort_values(ascending=False).head(11))