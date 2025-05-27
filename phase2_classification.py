import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import os
from collections import Counter
import time
import joblib

# Create output directory for plots and models
os.makedirs('phase2_plots', exist_ok=True)
os.makedirs('saved_models', exist_ok=True)

print("Loading enhanced data...")
# Load the enhanced data with degradation stages from Phase 1
df = pd.read_csv('phase1_csv/train_enhanced_with_stages.csv')

# Feature selection for classification model
# Exclude the target variables and non-feature columns
exclude_cols = ['engine_id', 'cycle', 'RUL', 'total_lifetime', 'life_percentage', 
                'cluster', 'stage', 'stage_numeric', 'time_to_next_stage']
feature_cols = [col for col in df.columns if col not in exclude_cols]

print(f"Using {len(feature_cols)} features for classification model")

# Prepare data for classification - we want to predict the degradation stage
X = df[feature_cols]
y = df['stage_numeric']

# Handle missing values if any
X = X.fillna(X.mean())

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data for training and validation
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.25, random_state=42, stratify=y)

# Check class distribution
print("Class distribution before balancing:")
print(Counter(y_train))

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print("Class distribution after balancing:")
print(Counter(y_train_balanced))

# Train a Random Forest Classifier
print("\nTraining Random Forest Classifier...")
rf_clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True,
    n_jobs=-1,
    random_state=42,
    class_weight='balanced'
)

start_time = time.time()
rf_clf.fit(X_train_balanced, y_train_balanced)
rf_train_time = time.time() - start_time
print(f"RF training completed in {rf_train_time:.2f} seconds")

y_pred_rf = rf_clf.predict(X_val)

# Evaluate the Random Forest model
print("Random Forest Classification Results:")
print(classification_report(y_val, y_pred_rf))

# Confusion matrix for Random Forest
cm_rf = confusion_matrix(y_val, y_pred_rf)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues',
           xticklabels=['Normal', 'Slightly Deg', 'Moderately Deg', 'Critical', 'Failure'],
           yticklabels=['Normal', 'Slightly Deg', 'Moderately Deg', 'Critical', 'Failure'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Random Forest Confusion Matrix')
plt.tight_layout()
plt.savefig('phase2_plots/rf_confusion_matrix.png')
plt.close()

# Train an XGBoost Classifier with more conservative settings
print("\nTraining XGBoost Classifier...")
xgb_clf = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softmax',
    num_class=5,
    random_state=42,
    n_jobs=-1  # Use all cores
)

start_time = time.time()
xgb_clf.fit(X_train_balanced, y_train_balanced)
xgb_train_time = time.time() - start_time
print(f"XGB training completed in {xgb_train_time:.2f} seconds")

y_pred_xgb = xgb_clf.predict(X_val)

# Evaluate the XGBoost model
print("XGBoost Classification Results:")
print(classification_report(y_val, y_pred_xgb))

# Confusion matrix for XGBoost
cm_xgb = confusion_matrix(y_val, y_pred_xgb)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues',
           xticklabels=['Normal', 'Slightly Deg', 'Moderately Deg', 'Critical', 'Failure'],
           yticklabels=['Normal', 'Slightly Deg', 'Moderately Deg', 'Critical', 'Failure'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('XGBoost Confusion Matrix')
plt.tight_layout()
plt.savefig('phase2_plots/xgb_confusion_matrix.png')
plt.close()

# Feature importance analysis
plt.figure(figsize=(12, 10))
importances = rf_clf.feature_importances_
indices = np.argsort(importances)[-20:]  # Get indices of top 20 features
plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), [feature_cols[i] for i in indices])
plt.xlabel('Feature Importance')
plt.title('Top 20 Features for Degradation Stage Classification')
plt.tight_layout()
plt.savefig('phase2_plots/feature_importance.png')
plt.close()

# Use RandomizedSearchCV instead of GridSearchCV for better efficiency
# and reduce the parameter space
print("\nPerforming hyperparameter tuning with RandomizedSearchCV...")
# Reduced parameter space for faster execution
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8],
    'colsample_bytree': [0.8]
}

# For even faster execution, you can sample data for hyperparameter tuning
# Uncomment these lines if you still have performance issues
# sample_size = min(10000, len(X_train_balanced))
# random_indices = np.random.choice(range(len(X_train_balanced)), size=sample_size, replace=False)
# X_train_sample = X_train_balanced[random_indices]
# y_train_sample = y_train_balanced[random_indices]

random_search = RandomizedSearchCV(
    estimator=xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=5,
        random_state=42,
        n_jobs=-1  # Use all cores
    ),
    param_distributions=param_grid,
    n_iter=10,  # Try only 10 combinations
    cv=3,
    scoring='f1_weighted',
    n_jobs=1,   # Only use 1 job since XGBoost will use all cores
    verbose=2,  # More verbose output to see progress
    random_state=42
)

print("Starting RandomizedSearchCV fit. This might take a while...")
start_time = time.time()
# Use X_train_sample and y_train_sample here if you're sampling
random_search.fit(X_train_balanced, y_train_balanced)
search_time = time.time() - start_time
print(f"RandomizedSearchCV completed in {search_time:.2f} seconds")

print(f"Best parameters: {random_search.best_params_}")
print(f"Best cross-validation score: {random_search.best_score_:.4f}")

# Train the best model from grid search
best_clf = random_search.best_estimator_
best_clf.fit(X_train_balanced, y_train_balanced)
y_pred_best = best_clf.predict(X_val)

print("\nOptimized XGBoost Classification Results:")
print(classification_report(y_val, y_pred_best))

# Confusion matrix for optimized model
cm_best = confusion_matrix(y_val, y_pred_best)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues',
           xticklabels=['Normal', 'Slightly Deg', 'Moderately Deg', 'Critical', 'Failure'],
           yticklabels=['Normal', 'Slightly Deg', 'Moderately Deg', 'Critical', 'Failure'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Optimized XGBoost Confusion Matrix')
plt.tight_layout()
plt.savefig('phase2_plots/best_xgb_confusion_matrix.png')
plt.close()

# Compare model performance on critical stages
print("\nComparing model performance on critical stages (Stage 3 & 4):")
stages = [3, 4]  # Critical and Failure stages
stage_mask = np.isin(y_val, stages)

rf_f1 = f1_score(y_val[stage_mask], y_pred_rf[stage_mask], average='weighted')
xgb_f1 = f1_score(y_val[stage_mask], y_pred_xgb[stage_mask], average='weighted')
best_f1 = f1_score(y_val[stage_mask], y_pred_best[stage_mask], average='weighted')

print(f"Random Forest F1 Score on critical stages: {rf_f1:.4f}")
print(f"XGBoost F1 Score on critical stages: {xgb_f1:.4f}")
print(f"Optimized XGBoost F1 Score on critical stages: {best_f1:.4f}")

# Get probability predictions for later risk score calculation
y_prob_best = best_clf.predict_proba(X_val)

# Save the best model and scaler for later use
joblib.dump(best_clf, 'saved_models/stage_classifier_model.pkl')
joblib.dump(scaler, 'saved_models/stage_classifier_scaler.pkl')
joblib.dump(feature_cols, 'saved_models/stage_classifier_features.pkl')

print("\nModel training complete. Best model and scaler saved for future use.")

""" Optional:
# Generate example predictions for a single engine
# First check which engine IDs are actually available in the dataset
available_engines = df['engine_id'].unique()
print(f"Available engine IDs: {available_engines[:5]}... (total: {len(available_engines)})")

# Choose the first available engine ID instead of hardcoding
example_engine_id = available_engines[0]
print(f"Using engine ID {example_engine_id} for example predictions")

example_engine_data = df[df['engine_id'] == example_engine_id].sort_values('cycle')
# Check if we actually have data for this engine
if len(example_engine_data) == 0:
    print(f"No data found for engine ID {example_engine_id}. Skipping example predictions.")
    # Skip the example prediction section if no data is found
else:
    example_X = example_engine_data[feature_cols]
    example_X_scaled = scaler.transform(example_X)
    example_y_pred = best_clf.predict(example_X_scaled)
    example_y_proba = best_clf.predict_proba(example_X_scaled)

    # Create a dataframe with the example predictions
    example_results = pd.DataFrame({
        'cycle': example_engine_data['cycle'].values,
        'actual_stage': example_engine_data['stage_numeric'].values,
        'predicted_stage': example_y_pred
    })

# Only process example prediction visualization if we have data
if 'example_results' in locals():
    # Add probability for each stage
    for i in range(5):
        example_results[f'prob_stage_{i}'] = example_y_proba[:, i]

    # Visualize the predictions for the example engine
    plt.figure(figsize=(12, 8))
    plt.plot(example_results['cycle'], example_results['actual_stage'], 'b-', label='Actual Stage')
    plt.plot(example_results['cycle'], example_results['predicted_stage'], 'r--', label='Predicted Stage')
    plt.xlabel('Cycle')
    plt.ylabel('Degradation Stage')
    plt.title(f'Actual vs Predicted Degradation Stages for Engine {example_engine_id}')
    plt.yticks(range(5), ['Normal', 'Slightly Deg', 'Moderately Deg', 'Critical', 'Failure'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('phase2_plots/example_engine_predictions.png')
    plt.close()
    print(f"Example predictions visualization saved for engine {example_engine_id}")
else:
    print("Skipped example predictions visualization due to lack of data")
    """

print("\nPhase 2 (Classification Model) complete.")