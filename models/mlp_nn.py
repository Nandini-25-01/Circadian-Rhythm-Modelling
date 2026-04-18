import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, confusion_matrix, ConfusionMatrixDisplay

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Setup 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42); np.random.seed(42)

# Circular time helpers 
def to_circle(hours):
    a = 2 * np.pi * hours / 24
    return np.sin(a).astype(np.float32), np.cos(a).astype(np.float32)

def from_circle(s, c):
    return (np.arctan2(s, c) * 24 / (2 * np.pi)) % 24

def circ_mae(actual, pred):
    d = np.abs(actual - pred) % 24
    return np.minimum(d, 24 - d).mean()

def circ_rmse(actual, pred):
    d = np.abs(actual - pred) % 24
    return np.sqrt((np.minimum(d, 24 - d) ** 2).mean())

# Load data
import os
csv_path = "circadian_dataset_final.csv"
if not os.path.exists(csv_path):
    csv_path = "../circadian_dataset_final.csv"
df = pd.read_csv(csv_path).dropna()

# Quality filter 
# Drop participants whose circadian rhythm was too noisy for a reliable label.
df = df[df["cosinor_r2"] >= 0.2].copy()

# Feature engineering
df["morning_to_evening_ratio"] = df["morning_light"] / (df["evening_light"] + 1e-8)
df["rhythm_strength"]          = df["cosinor_amplitude"] * df["cosinor_r2"]
df["sleep_efficiency"]         = df["sleep_duration"] / 24.0
df["light_activity_spread"]    = df["std_light"] / (df["std_act"] + 1e-8)

# Prepare X and y 
drop_cols    = ["SEQN", "Acrophase", "cosinor_r2"]
feature_cols = [c for c in df.columns if c not in drop_cols]

X   = df[feature_cols].values.astype(np.float32)
y_h = df["Acrophase"].values.astype(np.float32)       # raw hours (for evaluation)
s, c = to_circle(y_h)
y   = np.stack([s, c], axis=1).astype(np.float32)     # (sin, cos) targets

print(f"Dataset: {X.shape[0]} participants, {X.shape[1]} features")

# Split + scale
X_tr, X_te, y_tr, y_te, yh_tr, yh_te = train_test_split(
    X, y, y_h, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr).astype(np.float32)  # fit on train only
X_te = scaler.transform(X_te).astype(np.float32)

def loader(X, y, shuffle=False):
    return DataLoader(TensorDataset(torch.tensor(X), torch.tensor(y)),
                      batch_size=64, shuffle=shuffle)

# 3-layer MLP. Input → 256 → 128 → 2 (sin, cos).
model = nn.Sequential(
    nn.Linear(X_tr.shape[1], 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.25),
    nn.Linear(256, 128),           nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.15),
    nn.Linear(128, 2),             nn.Tanh()
).to(DEVICE)

print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters | Device: {DEVICE}")

# Training 
criterion = nn.HuberLoss(delta=1.0)
optimizer = torch.optim.Adam(model.parameters(), lr=8e-4, weight_decay=5e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=10, factor=0.5)   # halve LR if stuck for 10 epochs

tr_loader = loader(X_tr, y_tr, shuffle=True)
te_loader = loader(X_te, y_te)

best_loss, best_weights, no_improve = float("inf"), None, 0
train_hist, val_hist = [], []

for epoch in range(1, 201):

    # Training pass
    model.train()
    tr_total = 0.0
    for xb, yb in tr_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        tr_total += loss.item() * xb.size(0)
    tr_loss = tr_total / len(tr_loader.dataset)

    # Validation pass (no gradient needed)
    model.eval()
    val_total = 0.0
    with torch.no_grad():
        for xb, yb in te_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            val_total += criterion(model(xb), yb).item() * xb.size(0)
    val_loss = val_total / len(te_loader.dataset)

    train_hist.append(tr_loss); val_hist.append(val_loss)
    scheduler.step(val_loss)

    # Save the best weights seen so far
    if val_loss < best_loss:
        best_loss    = val_loss
        best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        no_improve   = 0
    else:
        no_improve  += 1

    if epoch % 20 == 0 or epoch == 1:
        print(f"  Epoch {epoch:4d} | train {tr_loss:.4f} | val {val_loss:.4f}")

    if no_improve >= 30:
        print(f"  Early stop at epoch {epoch}")
        break

model.load_state_dict(best_weights)

# Evaluate 
model.eval()
preds = []
with torch.no_grad():
    for xb, _ in te_loader:
        preds.append(model(xb.to(DEVICE)).cpu().numpy())
preds = np.vstack(preds)

pred_hours = from_circle(preds[:, 0], preds[:, 1])
mae  = circ_mae(yh_te, pred_hours)
rmse = circ_rmse(yh_te, pred_hours)
r2   = r2_score(yh_te, pred_hours)

print(f"\nCircular MAE  : {mae:.4f} hours")
print(f"Circular RMSE : {rmse:.4f} hours")
print(f"R²            : {r2:.4f}")

fig, axs = plt.subplots(1, 3, figsize=(18, 5))
ax1, ax2, ax3 = axs

ax1.plot(train_hist, label="Train"); ax1.plot(val_hist, label="Val", linestyle="--")
ax1.set(title="Loss over time", xlabel="Epoch", ylabel="Huber Loss")
ax1.legend(); ax1.grid(alpha=0.3)

ax2.scatter(yh_te, pred_hours, alpha=0.3, s=10, color="steelblue")
ax2.plot([0, 24], [0, 24], "r--", label="Perfect")
ax2.set(title=f"Predicted vs Actual (MAE={mae:.2f}h)", xlabel="Actual (h)", ylabel="Predicted (h)")
ax2.legend(); ax2.grid(alpha=0.3)

# Confusion matrix based on integer hours (0-23)
actual_bins = np.round(yh_te).astype(int) % 24
pred_bins = np.round(pred_hours).astype(int) % 24

# Create bins that actually exist in the data to avoid empty classes
unique_classes = np.unique(np.concatenate([actual_bins, pred_bins]))
cm = confusion_matrix(actual_bins, pred_bins, labels=unique_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_classes)
disp.plot(ax=ax3, cmap="Blues", colorbar=False)
ax3.set_title("Confusion Matrix (Rounded Hours)")

plt.tight_layout()
plt.show()