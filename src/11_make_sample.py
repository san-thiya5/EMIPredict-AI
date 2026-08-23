import pandas as pd

df = pd.read_csv("data/emi_prediction_final.csv", low_memory=False)
sample = df.sample(n=3000, random_state=42)
sample.to_csv("data/eda_sample.csv", index=False)
print(f"Saved sample: {sample.shape}")