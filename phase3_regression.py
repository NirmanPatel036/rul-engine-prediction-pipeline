import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import os
import joblib
import math
import time

# Create output directory for plots
os.makedirs('phase3_plots', exist_ok=True)

print("Loading enhanced data...")
df = pd.read_csv('phase1_csv/train_enhanced_with_stages.csv')
df_clean = df.dropna(subset=['time_to_next_stage'])

print(f"Working with {len(df_clean)} samples with valid time_to_next_stage values")

exclude_cols = ['engine_id', 'cycle', 'RUL', 'total_lifetime', 'life_percentage', 
                'cluster', 'stage', 'stage_numeric', 'time_to_next_stage']
feature_cols = [col for col in df_clean.columns if col not in exclude_cols]

print(f"Using {len(feature_cols)} features for regression model")

X = df_clean[feature_cols]
y = df_clean['time_to_next_stage']

stage_map = df_clean['stage_numeric'].copy()
X = X.fillna(X.mean())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Scale the target variable
y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()
joblib.dump(y_scaler, 'saved_models/time_to_next_stage_scaler.pkl')

X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
    X_scaled, y_scaled, pd.Series(df_clean.index), test_size=0.25, random_state=42
)

plt.figure(figsize=(10, 6))
sns.histplot(y, bins=30, kde=True)
plt.title('Distribution of Time to Next Stage')
plt.xlabel('Cycles until Next Stage')
plt.ylabel('Frequency')
plt.savefig('phase3_plots/time_to_next_stage_distribution.png')
plt.close()

plt.figure(figsize=(12, 8))
sns.boxplot(x='stage_numeric', y='time_to_next_stage', data=df_clean)
plt.title('Time to Next Stage by Current Stage')
plt.xlabel('Current Degradation Stage')
plt.ylabel('Cycles until Next Stage')
plt.xticks(range(5), ['Normal', 'Slightly Deg', 'Moderately Deg', 'Critical', 'Failure'])
plt.savefig('phase3_plots/time_to_next_stage_by_stage.png')
plt.close()

print("\nTraining Random Forest Regressor...")
start_time = time.time()
rf_reg = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True,
    n_jobs=-1,
    random_state=42
)
rf_reg.fit(X_train, y_train)
rf_train_time = time.time() - start_time
print(f"RF training completed in {rf_train_time:.2f} seconds")

# Inverse-transform predictions and ground truth
y_pred_rf = rf_reg.predict(X_val)
y_pred_rf_inv = y_scaler.inverse_transform(y_pred_rf.reshape(-1, 1)).flatten()
y_val_inv = y_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()

print("Random Forest Regression Results:")
mae_rf = mean_absolute_error(y_val_inv, y_pred_rf_inv)
rmse_rf = math.sqrt(mean_squared_error(y_val_inv, y_pred_rf_inv))
r2_rf = r2_score(y_val_inv, y_pred_rf_inv)
print(f"MAE: {mae_rf:.6f} cycles")
print(f"RMSE: {rmse_rf:.6f} cycles")
print(f"R² Score: {r2_rf:.4f}")

print("\nTraining XGBoost Regressor...")
start_time = time.time()
xgb_reg = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=42,
    n_jobs=-1
)
xgb_reg.fit(X_train, y_train)
xgb_train_time = time.time() - start_time
print(f"XGB training completed in {xgb_train_time:.2f} seconds")

y_pred_xgb = xgb_reg.predict(X_val)
y_pred_xgb_inv = y_scaler.inverse_transform(y_pred_xgb.reshape(-1, 1)).flatten()

print("XGBoost Regression Results:")
mae_xgb = mean_absolute_error(y_val_inv, y_pred_xgb_inv)
rmse_xgb = math.sqrt(mean_squared_error(y_val_inv, y_pred_xgb_inv))
r2_xgb = r2_score(y_val_inv, y_pred_xgb_inv)
print(f"MAE: {mae_xgb:.6f} cycles")
print(f"RMSE: {rmse_xgb:.6f} cycles")
print(f"R² Score: {r2_xgb:.4f}")

print("\nTraining Ridge Regression...")
start_time = time.time()
ridge_reg = Ridge(alpha=1.0)
ridge_reg.fit(X_train, y_train)
ridge_train_time = time.time() - start_time
print(f"Ridge training completed in {ridge_train_time:.2f} seconds")

y_pred_ridge = ridge_reg.predict(X_val)
y_pred_ridge_inv = y_scaler.inverse_transform(y_pred_ridge.reshape(-1, 1)).flatten()

print("Ridge Regression Results:")
mae_ridge = mean_absolute_error(y_val_inv, y_pred_ridge_inv)
rmse_ridge = math.sqrt(mean_squared_error(y_val_inv, y_pred_ridge_inv))
r2_ridge = r2_score(y_val_inv, y_pred_ridge_inv)
print(f"MAE: {mae_ridge:.6f} cycles")
print(f"RMSE: {rmse_ridge:.6f} cycles")
print(f"R² Score: {r2_ridge:.4f}")

plt.figure(figsize=(12, 8))
plt.scatter(y_val_inv, y_pred_xgb_inv, alpha=0.5)
plt.plot([y_val_inv.min(), y_val_inv.max()], [y_val_inv.min(), y_val_inv.max()], 'r--')
plt.xlabel('Actual Time to Next Stage')
plt.ylabel('Predicted Time to Next Stage')
plt.title('XGBoost: Actual vs Predicted Time to Next Stage')
plt.savefig('phase3_plots/xgb_actual_vs_predicted.png')
plt.close()

val_stages = df_clean.loc[idx_val.values, 'stage_numeric'].values

df_val = pd.DataFrame({
    'actual': y_val_inv,
    'predicted': y_pred_xgb_inv,
    'error': np.abs(y_val_inv - y_pred_xgb_inv),
    'stage': val_stages
})

plt.figure(figsize=(12, 8))
sns.boxplot(x='stage', y='error', data=df_val)
plt.title('Prediction Error by Stage')
plt.xlabel('Current Stage')
plt.ylabel('Absolute Error (cycles)')
plt.xticks(range(5), ['Normal', 'Slightly Deg', 'Moderately Deg', 'Critical', 'Failure'])
plt.savefig('phase3_plots/prediction_error_by_stage.png')
plt.close()

print("\nPerforming hyperparameter tuning with RandomizedSearchCV...")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8],
    'colsample_bytree': [0.8]
}

start_time = time.time()
random_search = RandomizedSearchCV(
    estimator=xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    ),
    param_distributions=param_grid,
    n_iter=10,
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=1,
    verbose=2,
    random_state=42
)

random_search.fit(X_train, y_train)
search_time = time.time() - start_time
print(f"RandomizedSearchCV completed in {search_time:.2f} seconds")

print(f"Best parameters: {random_search.best_params_}")
print(f"Best cross-validation score: {-random_search.best_score_:.4f} (MSE)")

best_reg = random_search.best_estimator_
best_reg.fit(X_train, y_train)
y_pred_best = best_reg.predict(X_val)
y_pred_best_inv = y_scaler.inverse_transform(y_pred_best.reshape(-1, 1)).flatten()

print("\nOptimized XGBoost Regression Results:")
mae_best = mean_absolute_error(y_val_inv, y_pred_best_inv)
rmse_best = math.sqrt(mean_squared_error(y_val_inv, y_pred_best_inv))
r2_best = r2_score(y_val_inv, y_pred_best_inv)
print(f"MAE: {mae_best:.6f} cycles")
print(f"RMSE: {rmse_best:.6f} cycles")
print(f"R² Score: {r2_best:.4f}")

plt.figure(figsize=(12, 10))
importances = best_reg.feature_importances_
indices = np.argsort(importances)[-20:]
plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), [feature_cols[i] for i in indices])
plt.xlabel('Feature Importance')
plt.title('Top 20 Features for Time-to-Next-Stage Prediction')
plt.tight_layout()
plt.savefig('phase3_plots/regression_feature_importance.png')
plt.close()

joblib.dump(best_reg, 'saved_models/time_regression_model.pkl')
joblib.dump(scaler, 'saved_models/time_regression_scaler.pkl')
joblib.dump(feature_cols, 'saved_models/time_regression_features.pkl')

print("\nModel training complete. Best model and scaler saved for future use.")

available_engines = df_clean['engine_id'].unique()
print(f"Available engine IDs: {available_engines[:5]}... (total: {len(available_engines)})")

if len(available_engines) > 0:
    example_engine_id = available_engines[1]
    print(f"Using engine ID {example_engine_id} for example predictions")

    example_engine_data = df_clean[df_clean['engine_id'] == example_engine_id].sort_values('cycle')

    if len(example_engine_data) > 0:
        example_X = example_engine_data[feature_cols]
        example_X_scaled = scaler.transform(example_X)
        example_y_pred = best_reg.predict(example_X_scaled)
        example_y_pred_inv = y_scaler.inverse_transform(example_y_pred.reshape(-1, 1)).flatten()

        example_results = pd.DataFrame({
            'cycle': example_engine_data['cycle'].values,
            'actual_time': example_engine_data['time_to_next_stage'].values,
            'predicted_time': example_y_pred_inv,
            'stage': example_engine_data['stage_numeric'].values
        })

        plt.figure(figsize=(12, 8))
        plt.plot(example_results['cycle'], example_results['actual_time'], 'b-', label='Actual Time')
        plt.plot(example_results['cycle'], example_results['predicted_time'], 'r--', label='Predicted Time')
        plt.xlabel('Cycle')
        plt.ylabel('Time to Next Stage (cycles)')
        plt.title(f'Actual vs Predicted Time to Next Stage for Engine {example_engine_id}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('phase3_plots/example_engine_time_predictions.png')
        plt.close()
        print(f"Example predictions visualization saved for engine {example_engine_id}")
    else:
        print(f"No data found for engine ID {example_engine_id}. Skipping example predictions.")
else:
    print("No engines available with valid time_to_next_stage values. Skipping example predictions.")

print("\nPhase 3 (Regression Model) complete.")
