# Circadian Rhythm Monitoring

Machine learning project for predicting circadian acrophase from engineered wearable-style features.

## Project Status

- Dataset prepared: `circadian_dataset_final.csv`
- Feature engineering completed
- Target variable: `Acrophase`
- Current next step: train and compare regression models

## Recommended Model Plan

1. XGBoost Regressor
2. Random Forest Regressor
3. MLP Regressor / Feedforward Neural Network

These models fit the current tabular engineered dataset. LSTM should only be used if raw ordered time-series data is available.

## Repository Structure

- `circadian_dataset_final.csv`: final engineered dataset
- `notebooks/`: exploratory analysis and model training notebooks
- `src/`: reusable Python scripts
- `models/`: saved trained models
- `reports/`: result tables, plots, and presentation outputs

## Evaluation Metrics

Use both standard regression metrics and circular time-aware metrics:

- MAE
- RMSE
- R2
- Circular MAE
- Circular RMSE
