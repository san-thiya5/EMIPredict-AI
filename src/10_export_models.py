import pandas as pd
import os
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

os.makedirs("app/models", exist_ok=True)

df = pd.read_csv("data/emi_prediction_engineered.csv", low_memory=False)

# ============================================================
# CLASSIFIER
# ============================================================
target_map = {"Not_Eligible": 0, "High_Risk": 1, "Eligible": 2}
y_clf = df["emi_eligibility"].map(target_map)
X_clf = df.drop(columns=["emi_eligibility", "max_monthly_emi"])

X_train, _, y_train, _ = train_test_split(
    X_clf, y_clf, test_size=0.30, random_state=42, stratify=y_clf
)

clf_model = XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    random_state=42, eval_metric="mlogloss", n_jobs=-1
)
clf_model.fit(X_train, y_train)
clf_model.save_model("app/models/emi_eligibility_classifier.json")
print("Classifier saved -> app/models/emi_eligibility_classifier.json")

# Save the exact feature column order/names - the Streamlit app needs these
with open("app/models/classifier_features.txt", "w") as f:
    f.write("\n".join(X_clf.columns.tolist()))
print("Classifier feature list saved -> app/models/classifier_features.txt")

# ============================================================
# REGRESSOR
# ============================================================
y_reg = df["max_monthly_emi"]
X_reg = df.drop(columns=["max_monthly_emi", "emi_eligibility"])

X_train, _, y_train, _ = train_test_split(
    X_reg, y_reg, test_size=0.30, random_state=42
)

reg_model = XGBRegressor(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    random_state=42, n_jobs=-1
)
reg_model.fit(X_train, y_train)
reg_model.save_model("app/models/emi_amount_regressor.json")
print("Regressor saved -> app/models/emi_amount_regressor.json")

with open("app/models/regressor_features.txt", "w") as f:
    f.write("\n".join(X_reg.columns.tolist()))
print("Regressor feature list saved -> app/models/regressor_features.txt")

print("\nBoth models exported successfully and ready for the Streamlit app.")