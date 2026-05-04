import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, f1_score, confusion_matrix, precision_score, recall_score
import os

def bin_acrophase(hours):
    """
    Bin continuous acrophase (0-24) into 4 categories for classification metrics:
    0-6: Night, 6-12: Morning, 12-18: Afternoon, 18-24: Evening
    """
    hours = np.array(hours) % 24
    bins = [0, 6, 12, 18, 24]
    labels = ['Night', 'Morning', 'Afternoon', 'Evening']
    categories = pd.cut(hours, bins=bins, labels=labels, right=False, include_lowest=True)
    return categories

def circular_rmse(y_true, y_pred):
    """
    Calculate the RMSE correctly for cyclic variables bounded 0-24. 
    Difference between 23 and 1 is 2 hours, not 22.
    """
    diff = np.abs(y_true - y_pred)
    circ_diff = np.minimum(diff, 24 - diff)
    return np.sqrt(np.mean(circ_diff**2))

def circular_r2(y_true, y_pred):
    """
    Standard R2 is problematic due to circular variance across midnight boundaries.
    Calculates pseudo-R2 based on the variance of the circular difference.
    """
    # SS_res
    diff = np.abs(y_true - y_pred)
    circ_diff = np.minimum(diff, 24 - diff)
    ss_res = np.sum(circ_diff**2)
    
    # SS_tot
    y_true_rad = y_true * (2 * np.pi / 24)
    # Find circular mean
    mean_rad = np.arctan2(np.mean(np.sin(y_true_rad)), np.mean(np.cos(y_true_rad)))
    mean_hr = (mean_rad * 24 / (2 * np.pi)) % 24
    
    diff_tot = np.abs(y_true - mean_hr)
    circ_diff_tot = np.minimum(diff_tot, 24 - diff_tot)
    ss_tot = np.sum(circ_diff_tot**2)
    
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

def main():
    file_path = 'circadian_dataset_final.csv'
    if not os.path.exists(file_path):
        file_path = '../circadian_dataset_final.csv'
        
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Features and Target
    X = df.drop(columns=['SEQN', 'Acrophase'])
    y = df['Acrophase']
    
    # Train-Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # --- CYCLIC TRANSFORMATION FOR TARGET ---
    # Predicting Sin and Cos logic instead of pure hours (0 vs 24)
    y_train_rad = y_train * (2 * np.pi / 24)
    y_train_sin = np.sin(y_train_rad)
    y_train_cos = np.cos(y_train_rad)
    
    # We train our multi-target KNN to output BOTH sine and cosine mapping
    y_train_sincos = np.column_stack((y_train_sin, y_train_cos)) 
    
    best_k = -1
    best_rmse = float('inf')
    best_r2 = -float('inf')
    best_model = None
    
    print("Iterating through different values of k (1 to 40) using cyclic predictions...")
    
    # Iterating through values of K
    for k in range(1, 41):
        # Adding distance-weighted voting inherently handles nearest neighbour density better
        knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
        
        # We predict BOTH sin and cos 
        knn.fit(X_train_scaled, y_train_sincos)
        y_pred_sincos = knn.predict(X_test_scaled)
        
        # 1. Reverse the transformation: arctan2 maps our [sin, cos] tuples directly to rad
        y_pred_rad = np.arctan2(y_pred_sincos[:, 0], y_pred_sincos[:, 1])
        
        # 2. Output is mapping of [-pi, pi], remap directly to cyclic bounded [0, 24] hour metrics
        y_pred_hr = (y_pred_rad * 24 / (2 * np.pi)) % 24
        
        # Calculate CYCLIC metrics correctly interpreting crossover boundaries 
        rmse = circular_rmse(y_test, y_pred_hr)
        r2 = circular_r2(y_test, y_pred_hr)
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_r2 = r2
            best_k = k
            best_model = knn
            
    print(f"\nBest K found: {best_k} with Cyclic RMSE: {best_rmse:.4f} and Cyclic R2: {best_r2:.4f}")
    
    # --- FULL EVALUATION ---
    y_pred_sincos_best = best_model.predict(X_test_scaled)
    y_pred_rad_best = np.arctan2(y_pred_sincos_best[:, 0], y_pred_sincos_best[:, 1])
    y_pred_best_hr = (y_pred_rad_best * 24 / (2 * np.pi)) % 24
    
    # Binning constraints assigning string representations categorising prediction buckets
    y_test_cat = bin_acrophase(y_test)
    y_pred_cat = bin_acrophase(y_pred_best_hr)
    
    labels = ['Night', 'Morning', 'Afternoon', 'Evening']
    
    # Generate multi-target evaluations
    precision = precision_score(y_test_cat, y_pred_cat, average='weighted', zero_division=0)
    recall = recall_score(y_test_cat, y_pred_cat, average='weighted', zero_division=0)
    f1 = f1_score(y_test_cat, y_pred_cat, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test_cat, y_pred_cat, labels=labels)
    
    print(f"\n--- Best Model Evaluation (K={best_k}) ---")
    print(f"Cyclic RMSE:        {best_rmse:.4f} hours")
    print(f"Cyclic R-squared:   {best_r2:.4f}")
    print(f"\nClassification Metrics (Weighted, 4-phase bins):")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    print("\nConfusion Matrix:")
    cm_df = pd.DataFrame(cm, index=[f'True {l}' for l in labels],
                         columns=[f'Pred {l}' for l in labels])
    print(cm_df)

if __name__ == "__main__":
    main()
