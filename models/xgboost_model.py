import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBRegressor

# Load dataset (handling path differences)
file_path = "circadian_dataset_final.csv"
if not os.path.exists(file_path):
    file_path = "models/circadian_dataset_final.csv"

df = pd.read_csv(file_path)
df = df.dropna()
if "SEQN" in df.columns:
    df = df.drop(columns=["SEQN"])

# Target
target = "Acrophase"

X = df.drop(columns=[target])
y = df[target]

# Convert to cyclic
y_sin = np.sin(2 * np.pi * y / 24)
y_cos = np.cos(2 * np.pi * y / 24)

# Split EVERYTHING together
X_train, X_test, y_train, y_test, y_sin_train, y_sin_test, y_cos_train, y_cos_test = train_test_split(
    X, y, y_sin, y_cos, test_size=0.2, random_state=42
)

# Proper Cyclic RMSE evaluation metric for final assessment
def cyclic_rmse(y_true, y_pred):
    diff = np.abs(y_true - y_pred)
    cyclic_diff = np.minimum(diff, 24 - diff)
    return np.sqrt(np.mean(cyclic_diff**2))

# Hyperparameter search space
param_distributions = {
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7, 9],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5]
}

# Models with Randomized Search to optimize accuracy
print("Tuning models...")
xgb_base = XGBRegressor(random_state=42)

# Train Sin Model
rs_sin = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_distributions,
    n_iter=15, # Number of parameter settings that are sampled
    scoring='neg_root_mean_squared_error',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)
rs_sin.fit(X_train, y_sin_train)
xgb_sin = rs_sin.best_estimator_

# Train Cos Model
rs_cos = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_distributions,
    n_iter=15,
    scoring='neg_root_mean_squared_error',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)
rs_cos.fit(X_train, y_cos_train)
xgb_cos = rs_cos.best_estimator_

print(f"\nBest params for Sin model: {rs_sin.best_params_}")
print(f"Best params for Cos model: {rs_cos.best_params_}")

# Predict
pred_sin = xgb_sin.predict(X_test)
pred_cos = xgb_cos.predict(X_test)

# Convert back
pred_phase = np.arctan2(pred_sin, pred_cos) * 24 / (2 * np.pi)
pred_phase[pred_phase < 0] += 24

# Evaluate (FIXED: Standard RMSE doesn't work for cyclic target 0-24, where 23 and 1 are only 2 hours apart)
old_incorrect_rmse = mean_squared_error(y_test, pred_phase, squared=False)
correct_cyclic_rmse = cyclic_rmse(y_test, pred_phase)

print("\nEvaluation:")
print("True XGBoost RMSE (Cyclic):", round(correct_cyclic_rmse, 4))