import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix
)

df = pd.read_csv("data/emi_prediction_engineered.csv", low_memory=False)

# Target encoding
target_map = {"Not_Eligible": 0, "High_Risk": 1, "Eligible": 2}
inverse_map = {v: k for k, v in target_map.items()}
y = df["emi_eligibility"].map(target_map)

# Drop leakage-prone / non-feature columns
drop_cols = ["emi_eligibility", "max_monthly_emi"]
X = df.drop(columns=drop_cols)

# Train/val/test split: 70/15/15, stratified on target
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Scale numeric features (helps Logistic Regression; harmless for trees)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

mlflow.set_experiment("EMI_Classification")

models = {
    "LogisticRegression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=200, max_depth=15, class_weight="balanced",
        random_state=42, n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=42, eval_metric="mlogloss", n_jobs=-1
    ),
}

results = {}

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        if name == "LogisticRegression":
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_val_scaled)
            probs = model.predict_proba(X_val_scaled)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            probs = model.predict_proba(X_val)

        acc = accuracy_score(y_val, preds)
        prec = precision_score(y_val, preds, average="weighted")
        rec = recall_score(y_val, preds, average="weighted")
        f1 = f1_score(y_val, preds, average="weighted")
        auc = roc_auc_score(y_val, probs, multi_class="ovr", average="weighted")

        mlflow.log_param("model_type", name)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)
        mlflow.sklearn.log_model(model, "model")

        results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}

        print(f"\n{'=' * 50}")
        print(f"{name}")
        print(f"{'=' * 50}")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"ROC-AUC:   {auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_val, preds, target_names=["Not_Eligible", "High_Risk", "Eligible"]))

print("\n" + "=" * 50)
print("MODEL COMPARISON SUMMARY")
print("=" * 50)
results_df = pd.DataFrame(results).T.sort_values("f1", ascending=False)
print(results_df)

best_model_name = results_df.index[0]
print(f"\nBest model by F1 score: {best_model_name}")
print("\nRun 'mlflow ui' in a separate terminal (with venv active) to view the full dashboard.")