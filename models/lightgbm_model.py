import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import precision_recall_fscore_support
import warnings


warnings.filterwarnings('ignore', category=UserWarning)
file_path = "circadian_dataset_final.csv"
if not os.path.exists(file_path):
    file_path = "models/circadian_dataset_final.csv"

df = pd.read_csv(file_path)
df = df.dropna()
if "SEQN" in df.columns:
    df = df.drop(columns=["SEQN"])

target = "Acrophase"
X = df.drop(columns=[target])
y = df[target]

y_sin = np.sin(2 * np.pi * y / 24)
y_cos = np.cos(2 * np.pi * y / 24)

X_train, X_test, y_train, y_test, y_sin_train, y_sin_test, y_cos_train, y_cos_test = train_test_split(
    X, y, y_sin, y_cos, test_size=0.2, random_state=42
)

def cyclic_mae(y_true, y_pred):
    diff = np.abs(y_true - y_pred)
    cyclic_diff = np.minimum(diff, 24 - diff)
    return np.mean(cyclic_diff)

def cyclic_rmse(y_true, y_pred):
    diff = np.abs(y_true - y_pred)
    cyclic_diff = np.minimum(diff, 24 - diff)
    return np.sqrt(np.mean(cyclic_diff**2))

def tolerance_accuracy(y_true, y_pred, tolerance_hours=1.0):
    diff = np.abs(y_true - y_pred)
    cyclic_diff = np.minimum(diff, 24 - diff)
    successes = np.sum(cyclic_diff <= tolerance_hours)
    return (successes / len(y_true)) * 100

lgb_base = lgb.LGBMRegressor(random_state=42, verbose=-1)

param_distributions = {
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7, -1],
    'num_leaves': [15, 31, 50],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

rs_sin = RandomizedSearchCV(
    estimator=lgb_base,
    param_distributions=param_distributions,
    n_iter=15, 
    scoring='neg_root_mean_squared_error',
    cv=3,
    verbose=0,
    random_state=42,
    n_jobs=-1
)
rs_sin.fit(X_train, y_sin_train)
best_sin_model = rs_sin.best_estimator_

rs_cos = RandomizedSearchCV(
    estimator=lgb_base,
    param_distributions=param_distributions,
    n_iter=15,
    scoring='neg_root_mean_squared_error',
    cv=3,
    verbose=0,
    random_state=42,
    n_jobs=-1
)
rs_cos.fit(X_train, y_cos_train)
best_cos_model = rs_cos.best_estimator_

pred_sin = best_sin_model.predict(X_test)
pred_cos = best_cos_model.predict(X_test)

pred_phase = np.arctan2(pred_sin, pred_cos) * 24 / (2 * np.pi)
pred_phase[pred_phase < 0] += 24

mae_result = cyclic_mae(y_test, pred_phase)
rmse_result = cyclic_rmse(y_test, pred_phase)
accuracy_1h = tolerance_accuracy(y_test, pred_phase, tolerance_hours=1.0)
accuracy_2h = tolerance_accuracy(y_test, pred_phase, tolerance_hours=2.0)


def categorize_phase(phase_array):
    bins = [0, 6, 12, 18, 24]
    labels = [0, 1, 2, 3] 
    categorized = np.digitize(phase_array, bins, right=False) - 1
    categorized = np.clip(categorized, 0, 3)
    return categorized

y_test_binned = categorize_phase(y_test)
pred_phase_binned = categorize_phase(pred_phase)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test_binned, pred_phase_binned, average='weighted', zero_division=0
)

print(f"Cyclic MAE (Avg Error):      {mae_result:.3f} hours")
print(f"Cyclic RMSE (Std Error):     {rmse_result:.3f} hours")
print("Classification Metrics (Binned into 6-Hour Shifts):")
print(f"Weighted Precision:          {precision:.3f}")
print(f"Weighted Recall:             {recall:.3f}")
print(f"Weighted F1 Score:           {f1:.3f}")
