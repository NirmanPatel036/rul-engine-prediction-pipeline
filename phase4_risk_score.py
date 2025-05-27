import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from sklearn.metrics import precision_recall_curve, average_precision_score

# Create output directory for plots
os.makedirs('phase4_plots', exist_ok=True)
os.makedirs('phase4_csv', exist_ok=True)

print("Loading data and models...")
# Load the enhanced data
df = pd.read_csv('phase1_csv/train_enhanced_with_stages.csv')

# Load the classification model and scaler
clf_model = joblib.load('saved_models/stage_classifier_model.pkl')
clf_scaler = joblib.load('saved_models/stage_classifier_scaler.pkl')
clf_features = joblib.load('saved_models/stage_classifier_features.pkl')

# Load the regression model and scaler
reg_model = joblib.load('saved_models/time_regression_model.pkl')
reg_scaler = joblib.load('saved_models/time_regression_scaler.pkl')
reg_features = joblib.load('saved_models/time_regression_features.pkl')

def calculate_risk_score(engine_data, clf_model, clf_scaler, clf_features, 
                         reg_model, reg_scaler, reg_features):
    """
    Calculate risk score for an engine based on:
    1. Probability of being in failure stage (Stage 4)
    2. Predicted time until next stage (if not already in Stage 4)
    """
    # Prepare data for classification (stage prediction)
    X_clf = engine_data[clf_features].copy()
    X_clf = X_clf.fillna(X_clf.mean())
    X_clf_scaled = clf_scaler.transform(X_clf)
    
    # Get stage predictions and probabilities
    stage_probs = clf_model.predict_proba(X_clf_scaled)
    predicted_stages = clf_model.predict(X_clf_scaled)
    
    # Extract probability of failure (Stage 4)
    failure_prob = stage_probs[:, 4]  # Assuming Stage 4 is at index 4
    
    # Calculate time to failure
    time_to_failure = np.zeros(len(engine_data))
    
    # For points not already in Stage 4, predict time to next stage
    non_failure_mask = predicted_stages != 4
    
    if np.any(non_failure_mask):
        # Prepare data for regression (time prediction)
        X_reg = engine_data.loc[non_failure_mask, reg_features].copy()
        X_reg = X_reg.fillna(X_reg.mean())
        X_reg_scaled = reg_scaler.transform(X_reg)
        
        # Predict time to next stage
        time_to_next_stage = reg_model.predict(X_reg_scaled)
        
        # For simplicity, assume we need additional cycles to reach Stage 4 based on current stage
        # This is a simplification - a more sophisticated approach would be to use a separate model
        # to predict time to Stage 4 specifically
        current_stages = predicted_stages[non_failure_mask]
        stage_to_failure_multiplier = 4 - current_stages  # Stages remaining until failure
        
        # Estimate total time to failure - simple approach: multiply by stages remaining
        estimated_time_to_failure = time_to_next_stage * stage_to_failure_multiplier
        time_to_failure[non_failure_mask] = estimated_time_to_failure
    
    # For stages already at 4, time to failure is 0
    
    # Calculate raw risk score: failure_probability / (time_to_failure + small_constant)
    # Adding small constant to avoid division by zero
    small_constant = 1e-6
    risk_score_raw = failure_prob / (time_to_failure + small_constant)
    
    # Alternative risk score calculation: failure_probability * (1 - normalized_time_to_failure)
    # This gives higher weight to items that are not only likely to fail but also close to failure
    max_time = max(time_to_failure) if max(time_to_failure) > 0 else 1
    normalized_time = time_to_failure / max_time
    risk_score_alt = failure_prob * (1 - normalized_time)
    
    return pd.DataFrame({
        'cycle': engine_data['cycle'].values,
        'predicted_stage': predicted_stages,
        'actual_stage': engine_data['stage_numeric'].values,
        'failure_probability': failure_prob,
        'time_to_failure': time_to_failure,
        'risk_score_raw': risk_score_raw,
        'risk_score_alt': risk_score_alt
    })

# Calculate risk scores for all engines
print("Calculating risk scores for all engines...")
all_risk_scores = []

for engine_id in df['engine_id'].unique():
    engine_data = df[df['engine_id'] == engine_id].sort_values('cycle')
    
    # Skip engines with too few data points
    if len(engine_data) < 5:
        continue
    
    # Calculate risk scores
    risk_scores = calculate_risk_score(
        engine_data, clf_model, clf_scaler, clf_features,
        reg_model, reg_scaler, reg_features
    )
    
    # Add engine ID
    risk_scores['engine_id'] = engine_id
    
    all_risk_scores.append(risk_scores)

# Combine all risk scores
risk_scores_df = pd.concat(all_risk_scores)

# Normalize risk scores across all engines
scaler = MinMaxScaler()
risk_scores_df['risk_score_normalized'] = scaler.fit_transform(risk_scores_df[['risk_score_raw']])
risk_scores_df['risk_score_alt_normalized'] = scaler.fit_transform(risk_scores_df[['risk_score_alt']])

# Save risk scores to CSV
risk_scores_df.to_csv('phase4_csv/engine_risk_scores.csv', index=False)

print(f"Risk scores calculated for {len(df['engine_id'].unique())} engines")

# Define risk thresholds
risk_thresholds = {
    'low': 0.3,
    'medium': 0.6,
    'high': 0.8
}

# Add risk category based on normalized alternative risk score
def categorize_risk(risk_score):
    if risk_score < risk_thresholds['low']:
        return 'Low'
    elif risk_score < risk_thresholds['medium']:
        return 'Medium'
    elif risk_score < risk_thresholds['high']:
        return 'High'
    else:
        return 'Critical'

risk_scores_df['risk_category'] = risk_scores_df['risk_score_alt_normalized'].apply(categorize_risk)

# Count engines in each risk category (using the latest cycle for each engine)
latest_cycle_risks = risk_scores_df.loc[risk_scores_df.groupby('engine_id')['cycle'].idxmax()]
risk_category_counts = latest_cycle_risks['risk_category'].value_counts()

# Visualize risk category distribution
plt.figure(figsize=(10, 6))
# Fix: Assign x variable to hue and set legend=False
colors = ['green', 'yellow', 'orange', 'red']
risk_category_df = pd.DataFrame({'category': risk_category_counts.index, 'count': risk_category_counts.values})
sns.barplot(x='category', y='count', hue='category', data=risk_category_df, palette=colors[:len(risk_category_counts)], legend=False)
plt.title('Distribution of Risk Categories')
plt.xlabel('Risk Category')
plt.ylabel('Number of Engines')
plt.savefig('phase4_plots/risk_category_distribution.png')
plt.close()

# Visualize risk scores for a few sample engines
def plot_engine_risk_progression(engine_ids, risk_df, num_engines=5):
    plt.figure(figsize=(15, 10))
    
    # Limit to specified number of engines if needed
    if len(engine_ids) > num_engines:
        engine_ids = engine_ids[:num_engines]
    
    for i, engine_id in enumerate(engine_ids):
        engine_data = risk_df[risk_df['engine_id'] == engine_id].sort_values('cycle')
        
        plt.subplot(len(engine_ids), 1, i+1)
        
        # Plot risk score
        line = plt.plot(engine_data['cycle'], engine_data['risk_score_alt_normalized'], 
                 marker='o', markersize=4, label='Risk Score')
        color = line[0].get_color()
        
        # Add threshold lines
        plt.axhline(y=risk_thresholds['low'], color='green', linestyle='--', alpha=0.5, label='Low Risk Threshold')
        plt.axhline(y=risk_thresholds['medium'], color='orange', linestyle='--', alpha=0.5, label='Medium Risk Threshold')
        plt.axhline(y=risk_thresholds['high'], color='red', linestyle='--', alpha=0.5, label='High Risk Threshold')
        
        # Plot actual stage as background color
        for stage in range(5):
            stage_data = engine_data[engine_data['actual_stage'] == stage]
            if len(stage_data) > 0:
                min_cycle = stage_data['cycle'].min()
                max_cycle = stage_data['cycle'].max()
                plt.axvspan(min_cycle, max_cycle, alpha=0.2, color=plt.cm.viridis(stage/4))
        
        plt.title(f'Risk Score Progression for Engine {engine_id}')
        plt.xlabel('Cycle')
        plt.ylabel('Normalized Risk Score')
        plt.ylim(0, 1.1)
        
        # Add "maintenance alert" text for high risk points
        alert_points = engine_data[engine_data['risk_score_alt_normalized'] > risk_thresholds['high']]
        if len(alert_points) > 0:
            for idx, point in alert_points.iterrows():
                plt.text(point['cycle'], point['risk_score_alt_normalized'] + 0.05, 
                        "⚠️ Maintenance Alert!", color='red', fontweight='bold')
        
        # Add legend for first plot only
        if i == 0:
            plt.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig('phase4_plots/risk_score_progression.png')
    plt.close()

# Select engines in different risk categories for visualization
sample_engines = []
for category in ['Low', 'Medium', 'High', 'Critical']:
    category_engines = latest_cycle_risks[latest_cycle_risks['risk_category'] == category]['engine_id'].values
    if len(category_engines) > 0:
        sample_engines.append(category_engines[0])  # Add one engine from each category

# Plot risk progression for sample engines
plot_engine_risk_progression(sample_engines, risk_scores_df)

# Create a dashboard-style summary for all engines
def generate_fleet_summary(risk_df):
    plt.figure(figsize=(15, 12))
    
    # First plot: Risk Category Distribution
    plt.subplot(2, 2, 1)
    # Fix: Use DataFrame and set hue parameter
    risk_category_df = pd.DataFrame({'category': risk_category_counts.index, 'count': risk_category_counts.values})
    sns.barplot(x='category', y='count', hue='category', data=risk_category_df, palette=colors[:len(risk_category_counts)], legend=False)
    plt.title('Fleet Risk Distribution')
    plt.xlabel('Risk Category')
    plt.ylabel('Number of Engines')
    
    # Second plot: Stage Distribution
    plt.subplot(2, 2, 2)
    # Fix: Convert to DataFrame and use hue parameter
    stage_counts = latest_cycle_risks['predicted_stage'].value_counts().sort_index()
    stage_df = pd.DataFrame({'stage': stage_counts.index, 'count': stage_counts.values})
    sns.barplot(x='stage', y='count', hue='stage', data=stage_df, palette='viridis', legend=False)
    plt.title('Current Degradation Stage Distribution')
    plt.xlabel('Degradation Stage')
    plt.ylabel('Number of Engines')
    plt.xticks(range(5), ['Normal', 'Slightly', 'Moderate', 'Critical', 'Failure'])
    
    # Third plot: Risk Score vs. Time to Failure
    plt.subplot(2, 2, 3)
    plt.scatter(latest_cycle_risks['time_to_failure'], 
                latest_cycle_risks['risk_score_alt_normalized'],
                c=latest_cycle_risks['predicted_stage'], cmap='viridis',
                alpha=0.7)
    plt.colorbar(label='Predicted Stage')
    plt.title('Risk Score vs. Time to Failure')
    plt.xlabel('Estimated Time to Failure (cycles)')
    plt.ylabel('Normalized Risk Score')
    plt.grid(True, alpha=0.3)
    
    # Fourth plot: Top 10 Highest Risk Engines
    plt.subplot(2, 2, 4)
    top_risk_engines = latest_cycle_risks.sort_values('risk_score_alt_normalized', ascending=False).head(10)
    # Fix: Use hue parameter for color mapping
    sns.barplot(x='engine_id', y='risk_score_alt_normalized', hue='engine_id', data=top_risk_engines, palette='Reds_r', legend=False)
    plt.title('Top 10 Highest Risk Engines')
    plt.xlabel('Engine ID')
    plt.ylabel('Normalized Risk Score')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('phase4_plots/fleet_risk_summary.png')
    plt.close()

# Generate fleet summary
generate_fleet_summary(risk_scores_df)

# Evaluate the risk score as a failure predictor
# We'll define "actual failure" as being in Stage 3 or 4 within the next 30 cycles
def evaluate_risk_score_performance(risk_df):
    # Create a new dataframe with one row per engine-cycle
    eval_df = risk_df.copy()
    
    # For each engine, determine if it will reach Stage 3 or 4 within the next 30 cycles
    will_fail_soon = []
    
    for engine_id in eval_df['engine_id'].unique():
        engine_data = eval_df[eval_df['engine_id'] == engine_id].sort_values('cycle')
        
        for idx, row in engine_data.iterrows():
            current_cycle = row['cycle']
            future_data = engine_data[(engine_data['cycle'] > current_cycle) & 
                                      (engine_data['cycle'] <= current_cycle + 30)]
            
            # Check if engine will be in Stage 3 or 4 within next 30 cycles
            will_fail = any(future_data['actual_stage'].isin([3, 4]))
            will_fail_soon.append(will_fail)
    
    eval_df['will_fail_soon'] = will_fail_soon
    
    # Calculate precision-recall curve for risk score as predictor of imminent failure
    precision, recall, thresholds = precision_recall_curve(
        eval_df['will_fail_soon'], 
        eval_df['risk_score_alt_normalized']
    )
    
    # Calculate average precision score
    avg_precision = average_precision_score(
        eval_df['will_fail_soon'], 
        eval_df['risk_score_alt_normalized']
    )
    
    # Plot precision-recall curve
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, marker='.', label=f'Average Precision: {avg_precision:.3f}')
    plt.title('Precision-Recall Curve for Risk Score as Failure Predictor')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('phase4_plots/risk_score_precision_recall.png')
    plt.close()
    
    # Find optimal threshold based on F1 score
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    # Calculate confusion matrix metrics at optimal threshold
    predicted_failures = eval_df['risk_score_alt_normalized'] >= optimal_threshold
    true_failures = eval_df['will_fail_soon']
    
    true_positives = sum(predicted_failures & true_failures)
    false_positives = sum(predicted_failures & ~true_failures)
    true_negatives = sum(~predicted_failures & ~true_failures)
    false_negatives = sum(~predicted_failures & true_failures)
    
    precision_at_threshold = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall_at_threshold = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    print(f"Optimal risk threshold: {optimal_threshold:.3f}")
    print(f"Precision at optimal threshold: {precision_at_threshold:.3f}")
    print(f"Recall at optimal threshold: {recall_at_threshold:.3f}")
    
    return {
        'optimal_threshold': optimal_threshold,
        'precision': precision_at_threshold,
        'recall': recall_at_threshold,
        'avg_precision': avg_precision
    }

# Evaluate risk score performance
print("Evaluating risk score performance as failure predictor...")
performance_metrics = evaluate_risk_score_performance(risk_scores_df)

# Create a final combined visualization showing the risk score distribution by actual outcome
def plot_risk_score_distribution_by_outcome(risk_df):
    # Get latest state for each engine
    latest_data = risk_df.loc[risk_df.groupby('engine_id')['cycle'].idxmax()]
    
    # Create a plot of risk score distribution colored by actual stage
    plt.figure(figsize=(12, 6))
    
    # First subplot: Distribution by predicted stage
    plt.subplot(1, 2, 1)
    sns.boxplot(x='predicted_stage', y='risk_score_alt_normalized', data=latest_data)
    plt.title('Risk Score Distribution by Predicted Stage')
    plt.xlabel('Predicted Degradation Stage')
    plt.ylabel('Normalized Risk Score')
    plt.xticks(range(5), ['Normal', 'Slight', 'Moderate', 'Critical', 'Failure'])
    
    # Second subplot: Distribution by actual stage
    plt.subplot(1, 2, 2)
    sns.boxplot(x='actual_stage', y='risk_score_alt_normalized', data=latest_data)
    plt.title('Risk Score Distribution by Actual Stage')
    plt.xlabel('Actual Degradation Stage')
    plt.ylabel('Normalized Risk Score')
    plt.xticks(range(5), ['Normal', 'Slight', 'Moderate', 'Critical', 'Failure'])
    
    plt.tight_layout()
    plt.savefig('phase4_plots/risk_score_by_stage_distribution.png')
    plt.close()

# Plot risk score distribution
plot_risk_score_distribution_by_outcome(risk_scores_df)

# Function to apply risk-based maintenance recommendations
def generate_maintenance_recommendations(risk_df, performance_metrics):
    # Get latest data for each engine
    latest_data = risk_df.loc[risk_df.groupby('engine_id')['cycle'].idxmax()]
    
    # Use the optimal threshold from performance evaluation
    risk_threshold = performance_metrics['optimal_threshold']
    
    # Create recommendation categories
    def get_recommendation(row):
        if row['risk_score_alt_normalized'] >= risk_thresholds['high']:
            return "Immediate Maintenance Required"
        elif row['risk_score_alt_normalized'] >= risk_thresholds['medium']:
            return "Schedule Maintenance Soon"
        elif row['risk_score_alt_normalized'] >= risk_thresholds['low']:
            return "Monitor Closely"
        else:
            return "Normal Operation"
    
    latest_data['recommendation'] = latest_data.apply(get_recommendation, axis=1)
    
    # Create a more detailed recommendation with estimated cycles until maintenance needed
    def get_detailed_recommendation(row):
        if row['predicted_stage'] == 4:
            return "Engine has reached failure stage. Immediate maintenance required."
        elif row['recommendation'] == "Immediate Maintenance Required":
            return f"Critical risk level. Schedule maintenance within {max(1, int(row['time_to_failure']/2))} cycles."
        elif row['recommendation'] == "Schedule Maintenance Soon":
            return f"Elevated risk level. Plan maintenance within {max(5, int(row['time_to_failure']/1.5))} cycles."
        elif row['recommendation'] == "Monitor Closely":
            return f"Moderate risk level. Increase monitoring frequency. Expect maintenance needs in ~{int(row['time_to_failure'])} cycles."
        else:
            return f"Low risk level. Regular maintenance schedule. Estimated {int(row['time_to_failure'])} cycles until next stage."
    
    latest_data['detailed_recommendation'] = latest_data.apply(get_detailed_recommendation, axis=1)
    
    # Calculate fleet statistics
    total_engines = len(latest_data)
    immediate_maintenance = sum(latest_data['recommendation'] == "Immediate Maintenance Required")
    schedule_soon = sum(latest_data['recommendation'] == "Schedule Maintenance Soon")
    monitor = sum(latest_data['recommendation'] == "Monitor Closely")
    normal = sum(latest_data['recommendation'] == "Normal Operation")
    
    print("\n===== Fleet Maintenance Summary =====")
    print(f"Total engines in fleet: {total_engines}")
    print(f"Engines requiring immediate maintenance: {immediate_maintenance} ({immediate_maintenance/total_engines*100:.1f}%)")
    print(f"Engines to schedule maintenance soon: {schedule_soon} ({schedule_soon/total_engines*100:.1f}%)")
    print(f"Engines to monitor closely: {monitor} ({monitor/total_engines*100:.1f}%)")
    print(f"Engines in normal operation: {normal} ({normal/total_engines*100:.1f}%)")
    
    # Save recommendations to CSV
    latest_data[['engine_id', 'cycle', 'predicted_stage', 'risk_score_alt_normalized', 
                'risk_category', 'recommendation', 'detailed_recommendation', 
                'time_to_failure']].to_csv('phase4_csv/maintenance_recommendations.csv', index=False)
    
    # Visualize maintenance recommendations
    plt.figure(figsize=(10, 6))
    # Fix: Convert to DataFrame and use hue parameter
    recommendation_df = pd.DataFrame(latest_data['recommendation'].value_counts()).reset_index()
    recommendation_df.columns = ['recommendation', 'count']
    colors_map = {
        "Normal Operation": "green",
        "Monitor Closely": "yellow",
        "Schedule Maintenance Soon": "orange",
        "Immediate Maintenance Required": "red"
    }
    sns.barplot(x='recommendation', y='count', hue='recommendation', data=recommendation_df, palette=[colors_map.get(r, "blue") for r in recommendation_df['recommendation']], legend=False)
    plt.title('Maintenance Recommendation Distribution')
    plt.xlabel('Recommendation')
    plt.ylabel('Number of Engines')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig('phase4_plots/maintenance_recommendations.png')
    plt.close()
    
    return latest_data[['engine_id', 'recommendation', 'detailed_recommendation']]

# Generate maintenance recommendations
print("\nGenerating maintenance recommendations based on risk scores...")
recommendations = generate_maintenance_recommendations(risk_scores_df, performance_metrics)

# Print top critical engines
critical_engines = recommendations[recommendations['recommendation'] == "Immediate Maintenance Required"]
if len(critical_engines) > 0:
    print("\n===== Critical Engines Requiring Immediate Attention =====")
    for idx, row in critical_engines.iterrows():
        print(f"Engine {row['engine_id']}: {row['detailed_recommendation']}")

print("\nRisk analysis complete. Results saved to 'engine_risk_scores.csv' and 'maintenance_recommendations.csv'")
print("Visualizations saved to 'phase4_plots/' directory")

# Create a function to apply the risk scoring to new engine data
def predict_risk_for_new_data(engine_data, clf_model, clf_scaler, clf_features,
                             reg_model, reg_scaler, reg_features, risk_thresholds):
    """
    Apply risk scoring to new engine data
    
    Parameters:
    -----------
    engine_data : DataFrame
        New engine sensor data to analyze
    clf_model, clf_scaler, clf_features : 
        Classification model components for stage prediction
    reg_model, reg_scaler, reg_features : 
        Regression model components for time prediction
    risk_thresholds : dict
        Thresholds for risk categorization
        
    Returns:
    --------
    DataFrame with risk scores and recommendations
    """
    # Calculate risk scores
    risk_scores = calculate_risk_score(
        engine_data, clf_model, clf_scaler, clf_features,
        reg_model, reg_scaler, reg_features
    )
    
    # Add engine ID
    if 'engine_id' in engine_data.columns:
        risk_scores['engine_id'] = engine_data['engine_id'].iloc[0]
    
    # Normalize risk scores (using same approach as for training data)
    scaler = MinMaxScaler()
    risk_scores['risk_score_normalized'] = scaler.fit_transform(risk_scores[['risk_score_raw']])
    risk_scores['risk_score_alt_normalized'] = scaler.fit_transform(risk_scores[['risk_score_alt']])
    
    # Categorize risk
    risk_scores['risk_category'] = risk_scores['risk_score_alt_normalized'].apply(
        lambda x: categorize_risk(x)
    )
    
    # Add recommendations
    def get_recommendation(row):
        if row['risk_score_alt_normalized'] >= risk_thresholds['high']:
            return "Immediate Maintenance Required"
        elif row['risk_score_alt_normalized'] >= risk_thresholds['medium']:
            return "Schedule Maintenance Soon"
        elif row['risk_score_alt_normalized'] >= risk_thresholds['low']:
            return "Monitor Closely"
        else:
            return "Normal Operation"
    
    risk_scores['recommendation'] = risk_scores.apply(get_recommendation, axis=1)
    
    return risk_scores

print("\nAll functions and models have been saved for future use on new engine data.")
print("You can use 'predict_risk_for_new_data()' to assess risk for new engines.")