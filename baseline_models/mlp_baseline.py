"""
MLP Neural Network for Circadian Acrophase Prediction
Dataset: NHANES circadian multimodal dataset (6040 rows, 19 features)
Target: Acrophase (continuous regression)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

try:
    from numbers_parser import Document
    NUMBERS_AVAILABLE = True
except ImportError:
    NUMBERS_AVAILABLE = False

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 55)
print("STEP 1: Loading data")
print("=" * 55)

csv_path = "circadian_dataset_final.csv"
if not os.path.exists(csv_path):
    csv_path = "../circadian_dataset_final.csv"

if os.path.exists("data.numbers") and NUMBERS_AVAILABLE:
    print("Loading from data.numbers...")
    doc = Document("data.numbers")
    table = doc.sheets[0].tables[0]
    rows = list(table.iter_rows())
    headers = [cell.value for cell in rows[0]]
    data = [[cell.value for cell in row] for row in rows[1:]]
    df = pd.DataFrame(data, columns=headers)
else:
    print(f"Loading from {csv_path}...")
    df = pd.read_csv(csv_path)

print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Missing values:\n{df.isnull().sum()}")
print(f"\nTarget (Acrophase) stats:\n{df['Acrophase'].describe()}")

# ─────────────────────────────────────────────
# 2. PREPARE FEATURES & TARGET
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 2: Preparing features")
print("=" * 55)

# Drop participant ID — not a feature
df = df.drop(columns=["SEQN"])
df = df.dropna()

X = df.drop(columns=["Acrophase"]).values.astype(np.float32)
y = df["Acrophase"].values.astype(np.float32)

feature_names = df.drop(columns=["Acrophase"]).columns.tolist()
print(f"Features ({len(feature_names)}): {feature_names}")
print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"y range: [{y.min():.3f}, {y.max():.3f}]")

# ─────────────────────────────────────────────
# 3. TRAIN / VAL / TEST SPLIT
# ─────────────────────────────────────────────
# 70% train | 15% val | 15% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)
print(f"\nTrain: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

# ─────────────────────────────────────────────
# 4. SCALE FEATURES (z-score normalization)
#    Fit ONLY on train set to prevent data leakage
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_val   = scaler.transform(X_val).astype(np.float32)
X_test  = scaler.transform(X_test).astype(np.float32)

# ─────────────────────────────────────────────
# 5. PYTORCH DATASETS & DATALOADERS
# ─────────────────────────────────────────────
def to_tensors(X, y):
    return TensorDataset(torch.tensor(X), torch.tensor(y).unsqueeze(1))

train_ds = to_tensors(X_train, y_train)
val_ds   = to_tensors(X_val, y_val)
test_ds  = to_tensors(X_test, y_test)

BATCH_SIZE = 64
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

# ─────────────────────────────────────────────
# 6. MLP ARCHITECTURE
# ─────────────────────────────────────────────
class CircadianMLP(nn.Module):
    """
    3-hidden-layer MLP for acrophase regression.
    Architecture: 18 → 128 → 64 → 32 → 1
    Uses ReLU activations + BatchNorm + Dropout for regularization.
    """
    def __init__(self, input_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Layer 2
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Layer 3
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout / 2),  # less dropout near output
            # Output (no activation — regression)
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


input_dim = X_train.shape[1]
model = CircadianMLP(input_dim=input_dim, dropout=0.3)
print("\n" + "=" * 55)
print("STEP 5: Model Architecture")
print("=" * 55)
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")

# ─────────────────────────────────────────────
# 7. TRAINING SETUP
# ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")
model = model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
# ReduceLROnPlateau: halves LR if val loss doesn't improve for 10 epochs
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=10, factor=0.5
)

# ─────────────────────────────────────────────
# 8. TRAINING LOOP WITH EARLY STOPPING
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 6: Training")
print("=" * 55)

EPOCHS      = 200
PATIENCE    = 25          # early stopping patience
best_val    = float("inf")
best_state  = None
no_improve  = 0

train_losses, val_losses = [], []

for epoch in range(1, EPOCHS + 1):
    # ── Train ──
    model.train()
    running = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        running += loss.item() * xb.size(0)
    train_loss = running / len(train_loader.dataset)

    # ── Validate ──
    model.eval()
    val_running = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            val_running += criterion(pred, yb).item() * xb.size(0)
    val_loss = val_running / len(val_loader.dataset)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    scheduler.step(val_loss)

    # ── Early stopping ──
    if val_loss < best_val:
        best_val   = val_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        no_improve = 0
    else:
        no_improve += 1

    if epoch % 20 == 0 or epoch == 1:
        print(f"Epoch {epoch:4d} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    if no_improve >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
        break

# Restore best weights
model.load_state_dict(best_state)
print(f"\nBest val MSE: {best_val:.4f} (RMSE: {best_val**0.5:.4f})")

# ─────────────────────────────────────────────
# 9. EVALUATION ON TEST SET
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 7: Test Set Evaluation")
print("=" * 55)

model.eval()
preds, actuals = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        pred = model(xb).cpu().numpy().flatten()
        preds.extend(pred)
        actuals.extend(yb.numpy().flatten())

preds   = np.array(preds)
actuals = np.array(actuals)

mae  = mean_absolute_error(actuals, preds)
rmse = mean_squared_error(actuals, preds) ** 0.5
r2   = r2_score(actuals, preds)

print(f"MAE  : {mae:.4f}  (mean absolute error in acrophase units)")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# ─────────────────────────────────────────────
# 10. PLOTS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("MLP — Circadian Acrophase Prediction", fontsize=14, fontweight="bold")

# Loss curves
ax = axes[0]
ax.plot(train_losses, label="Train MSE", linewidth=2)
ax.plot(val_losses,   label="Val MSE",   linewidth=2, linestyle="--")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss")
ax.set_title("Training & Validation Loss")
ax.legend()
ax.grid(alpha=0.3)

# Predicted vs Actual
ax = axes[1]
ax.scatter(actuals, preds, alpha=0.3, s=10, color="steelblue")
mn, mx = min(actuals.min(), preds.min()), max(actuals.max(), preds.max())
ax.plot([mn, mx], [mn, mx], "r--", linewidth=2, label="Perfect prediction")
ax.set_xlabel("Actual Acrophase")
ax.set_ylabel("Predicted Acrophase")
ax.set_title(f"Predicted vs Actual  (R²={r2:.3f})")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("mlp_results.png", dpi=150, bbox_inches="tight")
print("\nPlot saved to mlp_results.png")

# Save model
torch.save({
    "model_state": model.state_dict(),
    "scaler":      scaler,
    "feature_names": feature_names,
    "metrics": {"mae": mae, "rmse": rmse, "r2": r2}
}, "mlp_circadian_model.pt")
print("Model saved to mlp_circadian_model.pt")