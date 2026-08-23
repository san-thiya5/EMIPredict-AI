import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score
)

df = pd.read_csv("data/emi_prediction_engineered.csv", low_memory=False)

# ============================================================
# CLASSIFICATION MODEL - register
# ============================================================
target_map = {"Not_Eligible": 0, "High_Risk": 1, "Eligible": 2}
y_clf = df["emi_eligibility"].map(target_map)
X_clf = df.drop(columns=["emi_eligibility", "max_monthly_emi"])

X_train, X_temp, y_train, y_temp = train_test_split(
    X_clf, y_clf, test_size=0.30, random_state=42, stratify=y_clf
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

mlflow.set_experiment("EMI_Classification")

with mlflow.start_run(run_name="XGBoost_Registered") as run:
    clf_model = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=42, eval_metric="mlogloss", n_jobs=-1
    )
    clf_model.fit(X_train, y_train)
    preds = clf_model.predict(X_val)
    probs = clf_model.predict_proba(X_val)

    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average="weighted")
    auc = roc_auc_score(y_val, probs, multi_class="ovr", average="weighted")

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", auc)

    # log_model + register in one call using registered_model_name
    mlflow.xgboost.log_model(
        clf_model,
        "model",
        registered_model_name="EMI_Eligibility_Classifier"
    )

    print(f"Classification model registered. Run ID: {run.info.run_id}")
    print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

# ============================================================
# REGRESSION MODEL - register
# ============================================================
y_reg = df["max_monthly_emi"]
X_reg = df.drop(columns=["max_monthly_emi", "emi_eligibility"])

X_train, X_temp, y_train, y_temp = train_test_split(
    X_reg, y_reg, test_size=0.30, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

mlflow.set_experiment("EMI_Regression")

with mlflow.start_run(run_name="XGBoost_Registered") as run:
    reg_model = XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=42, n_jobs=-1
    )
    reg_model.fit(X_train, y_train)
    preds = reg_model.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val, preds))
    r2 = r2_score(y_val, preds)

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

    mlflow.xgboost.log_model(
        reg_model,
        "model",
        registered_model_name="EMI_Amount_Regressor"
    )

    print(f"\nRegression model registered. Run ID: {run.info.run_id}")
    print(f"RMSE: {rmse:.2f}, R2: {r2:.4f}")

print("\n" + "=" * 50)
print("Both models registered in MLflow Model Registry:")
print("  - EMI_Eligibility_Classifier")
print("  - EMI_Amount_Regressor")
print("Run 'mlflow ui' to view them under the 'Models' tab.")
print("=" * 50)