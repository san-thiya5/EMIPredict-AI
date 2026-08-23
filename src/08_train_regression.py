import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)

df = pd.read_csv("data/emi_prediction_engineered.csv", low_memory=False)

y = df["max_monthly_emi"]

# Drop target columns; emi_eligibility is also dropped to avoid leakage
# (it's derived from similar underlying signal as the EMI amount)
drop_cols = ["max_monthly_emi", "emi_eligibility"]
X = df.drop(columns=drop_cols)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

mlflow.set_experiment("EMI_Regression")

models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(
        n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
    ),
    "XGBoost": XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=42, n_jobs=-1
    ),
}

results = {}

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        if name == "LinearRegression":
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_val_scaled)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_val)

        rmse = np.sqrt(mean_squared_error(y_val, preds))
        mae = mean_absolute_error(y_val, preds)
        r2 = r2_score(y_val, preds)
        mape = mean_absolute_percentage_error(y_val, preds)

        mlflow.log_param("model_type", name)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mape", mape)

        if name == "XGBoost":
            mlflow.xgboost.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")

        results[name] = {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}

        print(f"\n{'=' * 50}")
        print(f"{name}")
        print(f"{'=' * 50}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE:  {mae:.2f}")
        print(f"R2:   {r2:.4f}")
        print(f"MAPE: {mape:.4f}")

print("\n" + "=" * 50)
print("MODEL COMPARISON SUMMARY")
print("=" * 50)
results_df = pd.DataFrame(results).T.sort_values("rmse")
print(results_df)

best_model_name = results_df.index[0]
print(f"\nBest model by RMSE: {best_model_name}")
print("\nProject target: RMSE < 2000 INR")