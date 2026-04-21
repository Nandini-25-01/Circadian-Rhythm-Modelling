import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, confusion_matrix, ConfusionMatrixDisplay

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Setup 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42); np.random.seed(42)

# --- CIRCULAR HELPERS ---
def to_circle(hours):
    a = 2 * np.pi * hours / 24
    return np.sin(a).astype(np.float32), np.cos(a).astype(np.float32)

def from_circle(s, c):
    return (np.arctan2(s, c) * 24 / (2 * np.pi)) % 24

def circ_diff(actual, pred):
    d = np.abs(actual - pred) % 24
    return np.minimum(d, 24 - d)

def circ_mae(actual, pred):
    return circ_diff(actual, pred).mean()

def circ_rmse(actual, pred):
    return np.sqrt((circ_diff(actual, pred) ** 2).mean())

def circ_r2(actual, pred):
    """
    Calculates R2 by accounting for the 24-hour wrap-around.
    Standard R2 penalizes midnight-crossing errors (e.g., 23.9 vs 0.1) as 23.8h instead of 0.2h.
    """
    mse_model = np.mean(circ_diff(actual, pred)**2)
    # Variance of a null model that always predicts the circular mean
    # We find the circular mean acrophase first
    s, c = np.sin(actual * 2 * np.pi / 24), np.cos(actual * 2 * np.pi / 24)
    circ_mean = from_circle(np.mean(s), np.mean(c))
    mse_null = np.mean(circ_diff(actual, circ_mean)**2)
    return 1 - (mse_model / (mse_null + 1e-8))

# Load data
import os
csv_path = "circadian_dataset_final.csv"
if not os.path.exists(csv_path):
    csv_path = "../circadian_dataset_final.csv"
df = pd.read_csv(csv_path).dropna()

# Quality filter 
df = df[df["cosinor_r2"] >= 0.2].copy()

# Feature engineering
df["morning_to_evening_ratio"] = df["morning_light"] / (df["evening_light"] + 1e-8)
df["rhythm_strength"]          = df["cosinor_amplitude"] * df["cosinor_r2"]
df["sleep_efficiency"]         = df["sleep_duration"] / 24.0
df["light_activity_spread"]    = df["std_light"] / (df["std_act"] + 1e-8)
df["light_volatility"]         = df["std_light"] / (df["mean_light"] + 1e-8)
df["act_volatility"]           = df["std_act"] / (df["mean_act"] + 1e-8)

# Prepare X and y 
drop_cols    = ["SEQN", "Acrophase", "cosinor_r2"]
feature_cols = [c for c in df.columns if c not in drop_cols]

X   = df[feature_cols].values.astype(np.float32)
y_h = df["Acrophase"].values.astype(np.float32)
s, c = to_circle(y_h)
y   = np.stack([s, c], axis=1).astype(np.float32)

print(f"Dataset: {X.shape[0]} participants, {X.shape[1]} features")

# Model Class
class CircadianMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            
            nn.Linear(64, 2),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.net(x)

def loader(X, y, shuffle=False):
    return DataLoader(TensorDataset(torch.tensor(X), torch.tensor(y)),
                      batch_size=64, shuffle=shuffle)

# K-Fold Cross-Validation
KFOLDS = 5
kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=42)

all_preds = np.zeros_like(y)
fold_metrics = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\nTraining Fold {fold+1}/{KFOLDS}...")
    
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    yh_val = y_h[val_idx]
    
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    
    tr_loader = loader(X_tr, y_tr, shuffle=True)
    val_loader = loader(X_val, y_val)
    
    model = CircadianMLP(X_tr.shape[1]).to(DEVICE)
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    best_fold_loss = float("inf")
    best_fold_weights = None
    no_improve = 0
    
    for epoch in range(1, 201):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_total += criterion(model(xb), yb).item() * xb.size(0)
        val_loss = val_total / len(val_loader.dataset)
        scheduler.step(val_loss)
        
        if val_loss < best_fold_loss:
            best_fold_loss = val_loss
            best_fold_weights = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            
        if no_improve >= 30:
            break
            
    model.load_state_dict(best_fold_weights)
    model.eval()
    
    preds = []
    with torch.no_grad():
        for xb, _ in val_loader:
            preds.append(model(xb.to(DEVICE)).cpu().numpy())
    preds = np.vstack(preds)
    all_preds[val_idx] = preds
    
    pred_hours = from_circle(preds[:, 0], preds[:, 1])
    mae = circ_mae(yh_val, pred_hours)
    print(f"  Fold {fold+1} Val MAE: {mae:.4f} hours")
    fold_metrics.append(mae)

# Overall Evaluation
pred_hours = from_circle(all_preds[:, 0], all_preds[:, 1])
total_mae  = circ_mae(y_h, pred_hours)
total_rmse = circ_rmse(y_h, pred_hours)
# Standard R2 (penalizes boundary errors)
std_r2     = r2_score(y_h, pred_hours)
# Circular R2 (correctly handles 24h wrap-around)
corrected_r2 = circ_r2(y_h, pred_hours)

print(f"\n--- Final K-Fold Results ---")
print(f"Mean Fold MAE : {np.mean(fold_metrics):.4f} ± {np.std(fold_metrics):.4f}")
print(f"Overall MAE   : {total_mae:.4f} hours")
print(f"Overall RMSE  : {total_rmse:.4f} hours")
print(f"Standard R²   : {std_r2:.4f} (Underestimates fit)")
print(f"Circular R²   : {corrected_r2:.4f} (Correct Fit Metric)")

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
ax1, ax2 = axs

ax1.scatter(y_h, pred_hours, alpha=0.3, s=10, color="forestgreen")
ax1.plot([0, 24], [0, 24], "r--", label="Perfect")
ax1.set(title=f"Predicted vs Actual (Circ-R²={corrected_r2:.2f})", xlabel="Actual (h)", ylabel="Predicted (h)")
ax1.legend(); ax1.grid(alpha=0.3)

actual_bins = np.round(y_h).astype(int) % 24
pred_bins = np.round(pred_hours).astype(int) % 24
unique_classes = np.unique(np.concatenate([actual_bins, pred_bins]))
cm = confusion_matrix(actual_bins, pred_bins, labels=unique_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_classes)
disp.plot(ax=ax2, cmap="Greens", colorbar=False)
ax2.set_title("Confusion Matrix (Rounded Hours)")

plt.tight_layout()
plt.show()