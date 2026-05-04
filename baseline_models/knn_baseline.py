import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os

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
    
    print("Training basic KNN baseline model (k=5)...")
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    
    y_pred = knn.predict(X_test_scaled)
    
    # Calculate basic linear metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n--- Baseline KNN Evaluation ---")
    print(f"Linear RMSE:      {rmse:.4f}")
    print(f"Linear R-squared: {r2:.4f}")

if __name__ == "__main__":
    main()