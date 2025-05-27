import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import SelectKBest, f_regression
import os
from scipy.signal import savgol_filter

# Create output directory for plots and csv's
os.makedirs('phase1_plots', exist_ok=True)
os.makedirs('phase1_csv', exist_ok=True)

# File paths - use relative paths for portability
train_file = '/Users/nirmanpatel36/Documents/Mahindra University/Semester_04/HPM Using CMAPSS/CMaps/train_data/train_FD001.txt'
test_file = '/Users/nirmanpatel36/Documents/Mahindra University/Semester_04/HPM Using CMAPSS/CMaps/test_data/test_FD001.txt'
rul_file = '/Users/nirmanpatel36/Documents/Mahindra University/Semester_04/HPM Using CMAPSS/CMaps/raw_sensor_data/RUL_FD001.txt'

# Column names for the dataset
column_names = [
    'engine_id', 'cycle', 'altitude', 'mach_number', 'TRA',
    'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5', 'sensor_6', 'sensor_7', 'sensor_8',
    'sensor_9', 'sensor_10', 'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15', 'sensor_16',
    'sensor_17', 'sensor_18', 'sensor_19', 'sensor_20', 'sensor_21'
]

print("Loading and preparing data...")
# Load data
train_df = pd.read_csv(train_file, sep=' ', header=None, names=column_names)
test_df = pd.read_csv(test_file, sep=' ', header=None, names=column_names)
rul_df = pd.read_csv(rul_file, sep=' ', header=None, names=['RUL'])

# Clean up the data (remove NaN columns that come from extra spaces in the text file)
train_df = train_df.dropna(axis=1)
test_df = test_df.dropna(axis=1)

print(f"Training set shape: {train_df.shape}")
print(f"Test set shape: {test_df.shape}")
print(f"RUL values shape: {rul_df.shape}")

# Calculate RUL for training data
def add_rul_column(df):
    # Group by engine_id and calculate max cycle for each engine
    max_cycles = df.groupby('engine_id')['cycle'].max().reset_index()
    max_cycles.columns = ['engine_id', 'max_cycle']
    
    # Merge with original dataframe
    df = df.merge(max_cycles, on='engine_id', how='left')
    
    # Calculate RUL (max_cycle - current_cycle)
    df['RUL'] = df['max_cycle'] - df['cycle']
    
    # Drop the max_cycle column
    df = df.drop('max_cycle', axis=1)
    
    return df

train_df = add_rul_column(train_df)

# Add total lifetime column to help with stage identification
train_df['total_lifetime'] = train_df.groupby('engine_id')['cycle'].transform('max')
train_df['life_percentage'] = (train_df['cycle'] / train_df['total_lifetime'] * 100).round()

# Visualize RUL distribution
plt.figure(figsize=(10, 6))
sns.histplot(train_df['RUL'], bins=30, kde=True)
plt.title('Distribution of Remaining Useful Life (RUL) in Training Data')
plt.xlabel('Remaining Useful Life (cycles)')
plt.ylabel('Frequency')
plt.savefig('phase1_plots/rul_distribution.png')
plt.close()

# Feature selection and normalization
# First, let's separate operational settings and sensor measurements
op_settings = ['altitude', 'mach_number', 'TRA']
sensor_cols = [col for col in train_df.columns if 'sensor' in col]

# Identify constant or nearly constant sensors
sensor_variance = train_df[sensor_cols].var()
low_variance_sensors = sensor_variance[sensor_variance < 0.001].index.tolist()
print(f"Low variance sensors that will be removed: {low_variance_sensors}")

# Remove low variance sensors
useful_sensors = [col for col in sensor_cols if col not in low_variance_sensors]
print(f"Useful sensors for analysis: {useful_sensors}")

# Visualize sensor trends for a single engine
def plot_sensor_trends(engine_id=1, sensors=None):
    if sensors is None:
        sensors = useful_sensors[:6]  # Default to first 6 useful sensors
    
    engine_data = train_df[train_df['engine_id'] == engine_id]
    
    # Check if engine_id exists in the dataset
    if len(engine_data) == 0:
        print(f"Warning: Engine ID {engine_id} not found in dataset")
        return
        
    # Check if we have any valid sensors to plot
    valid_sensors = [s for s in sensors if s in engine_data.columns]
    if not valid_sensors:
        print(f"Warning: No valid sensors found among {sensors}")
        return
    
    # Debug info    
    print(f"Plotting trends for engine {engine_id}, {len(engine_data)} data points")
    print(f"Sensors to plot: {valid_sensors}")
    
    plt.figure(figsize=(15, 10))
    for i, sensor in enumerate(valid_sensors):
        plt.subplot(3, 2, i+1)
        plt.plot(engine_data['cycle'], engine_data[sensor])
        plt.title(f'{sensor} vs Cycle for Engine {engine_id}')
        plt.xlabel('Cycle')
        plt.ylabel(sensor)
        
        # Debug the min/max values
        min_val = engine_data[sensor].min()
        max_val = engine_data[sensor].max()
        print(f"  {sensor}: min={min_val:.4f}, max={max_val:.4f}")
        
    plt.tight_layout()
    plt.savefig(f'phase1_plots/engine_{engine_id}_sensor_trends.png')
    plt.close()

# Plot for different engines to observe patterns
print("\nGenerating sensor trend plots...")
# Get a list of available engine IDs
available_engines = train_df['engine_id'].unique()
print(f"Available engine IDs: {available_engines[:10]}...")

# Make sure we're using valid engine IDs
engine_ids_to_plot = [available_engines[0], available_engines[2], available_engines[4]] 
print(f"Plotting engines: {engine_ids_to_plot}")

for engine_id in engine_ids_to_plot:
    plot_sensor_trends(engine_id=engine_id)

# Normalize sensor data
def normalize_data(df, cols_to_normalize):
    df_normalized = df.copy()
    scaler = MinMaxScaler()
    df_normalized[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])
    return df_normalized, scaler

train_normalized, sensor_scaler = normalize_data(train_df, useful_sensors)

# Calculate sensor correlations to identify related sensors
print("Calculating sensor correlations...")
sensor_correlation = train_normalized[useful_sensors].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(sensor_correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Sensor Correlation Matrix')
plt.tight_layout()
plt.savefig('phase1_plots/sensor_correlation.png')
plt.close()

# Identify most important sensors for degradation monitoring
# We'll use correlation with RUL as a simple metric
rul_correlation = train_normalized[useful_sensors + ['RUL']].corr()['RUL'].abs().sort_values(ascending=False)
top_sensors = rul_correlation.index[:8].tolist()  # Select top 8 sensors most correlated with RUL
print(f"Top sensors correlated with RUL: {top_sensors}")

# Visualize these top sensors for a few engines
print("\nGenerating plots for top sensors...")
if 'RUL' in top_sensors:  # Remove RUL if it's in top_sensors by accident
    top_sensors.remove('RUL')
    
sensors_to_plot = top_sensors[:6]  # Get top 6 sensors or fewer
print(f"Top sensors to plot: {sensors_to_plot}")

for engine_id in engine_ids_to_plot:
    plot_sensor_trends(engine_id=engine_id, sensors=sensors_to_plot)

# Calculate advanced features for each engine
print("Creating advanced features...")
def create_advanced_features(df, sensor_cols):
    df_enhanced = df.copy()
    
    # For each engine, calculate features over the entire lifecycle
    for engine_id in df['engine_id'].unique():
        engine_data = df[df['engine_id'] == engine_id]
        
        for sensor in sensor_cols:
            # Apply Savitzky-Golay filter to smooth sensor readings
            if len(engine_data) > 5:  # Need at least 5 points for filter
                window_length = min(15, len(engine_data) - (len(engine_data) % 2) - 1)  # Must be odd
                if window_length >= 5:
                    df_enhanced.loc[engine_data.index, f'{sensor}_smooth'] = savgol_filter(
                        engine_data[sensor], window_length, 3)
            
            # Calculate rate of change (first derivative) - use .diff() with fillna
            df_enhanced.loc[engine_data.index, f'{sensor}_rate'] = (
                engine_data[sensor].diff() / engine_data['cycle'].diff()).fillna(0)
            
            # Calculate acceleration (second derivative) - for detecting sudden changes
            # Fill NaN values immediately to prevent propagation
            df_enhanced.loc[engine_data.index, f'{sensor}_accel'] = (
                df_enhanced.loc[engine_data.index, f'{sensor}_rate'].diff() / engine_data['cycle'].diff()).fillna(0)
            
            # Calculate moving averages
            for window in [5, 10, 20]:
                if len(engine_data) >= window:
                    df_enhanced.loc[engine_data.index, f'{sensor}_ma_{window}'] = (
                        engine_data[sensor].rolling(window=window).mean().bfill().ffill())
            
            # Calculate moving standard deviation (for volatility)
            for window in [5, 10, 20]:
                if len(engine_data) >= window:
                    df_enhanced.loc[engine_data.index, f'{sensor}_std_{window}'] = (
                        engine_data[sensor].rolling(window=window).std().bfill().ffill())
    
    # Replace any remaining infinities or NaNs with 0
    df_enhanced = df_enhanced.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return df_enhanced

# Create advanced features based on top sensors
train_enhanced = create_advanced_features(train_normalized, top_sensors)

# Double-check for any remaining infinities or NaNs
print("Checking for infinities or NaNs...")
inf_counts = np.isinf(train_enhanced.select_dtypes(include=[np.number])).sum().sum()
nan_counts = np.isnan(train_enhanced.select_dtypes(include=[np.number])).sum().sum()
print(f"Infinities found: {inf_counts}")
print(f"NaNs found: {nan_counts}")

# Replace any remaining infinities or NaNs with 0
train_enhanced = train_enhanced.replace([np.inf, -np.inf], np.nan).fillna(0)

# Create feature subset for clustering
# We'll use a mix of raw sensors, smoothed values, rates and moving averages
feature_cols = top_sensors + [f'{sensor}_smooth' for sensor in top_sensors if f'{sensor}_smooth' in train_enhanced.columns]
feature_cols += [f'{sensor}_rate' for sensor in top_sensors]
feature_cols += [f'{sensor}_ma_10' for sensor in top_sensors if f'{sensor}_ma_10' in train_enhanced.columns]

# Make sure all feature columns exist in the dataframe
valid_feature_cols = [col for col in feature_cols if col in train_enhanced.columns]
print(f"Using {len(valid_feature_cols)} features for clustering")

# Create data frame with engineered features for clustering
# We'll use samples from different parts of the lifecycle
print("Preparing data for clustering...")
def sample_engine_lifecycle(df, feature_cols, n_samples=5):
    """Sample points from each engine's lifecycle to represent different stages"""
    samples = []
    
    for engine_id in df['engine_id'].unique():
        engine_data = df[df['engine_id'] == engine_id].copy()
        
        # Sort by cycle to ensure we sample along the lifecycle
        engine_data = engine_data.sort_values('cycle')
        
        # To ensure we capture the end-of-life, always include the last 2 cycles
        last_points = engine_data.tail(2)
        
        # Sample n_samples-2 points from the rest of the lifecycle
        if len(engine_data) > 2:
            remaining_data = engine_data.iloc[:-2]
            
            # Determine breakpoints - ensure we sample from different parts of lifecycle
            if len(remaining_data) >= (n_samples-2):
                # Create roughly equal segments
                idx = np.linspace(0, len(remaining_data)-1, n_samples-2, dtype=int)
                sampled_points = remaining_data.iloc[idx]
                samples.append(pd.concat([sampled_points, last_points]))
            else:
                # If not enough points, just take what we have
                samples.append(pd.concat([remaining_data, last_points]))
        else:
            # If very short lifecycle, just take all points
            samples.append(engine_data)
    
    return pd.concat(samples).reset_index(drop=True)

# Sample points from each engine's lifecycle
lifecycle_samples = sample_engine_lifecycle(train_enhanced, valid_feature_cols, n_samples=10)

# Final check for any remaining infinities or NaNs in clustering data
X_cluster = lifecycle_samples[valid_feature_cols].values
inf_mask = np.isinf(X_cluster)
if inf_mask.any():
    print(f"Warning: Found {inf_mask.sum()} infinity values in clustering data. Replacing with 0.")
    X_cluster[inf_mask] = 0

nan_mask = np.isnan(X_cluster)
if nan_mask.any():
    print(f"Warning: Found {nan_mask.sum()} NaN values in clustering data. Replacing with 0.")
    X_cluster[nan_mask] = 0

# Scale features for clustering
print("Scaling features for clustering...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# Determine optimal number of clusters
print("Determining optimal number of clusters...")
silhouette_scores = []
inertia_values = []
max_clusters = 10  # Try up to 10 clusters

for n_clusters in range(2, max_clusters + 1):
    # Apply KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    
    # Calculate inertia (within-cluster sum of squares)
    inertia_values.append(kmeans.inertia_)
    
    print(f"For n_clusters = {n_clusters}, silhouette score = {silhouette_avg:.3f}")

# Plot silhouette scores and inertia (elbow method)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(2, max_clusters + 1), silhouette_scores, 'b-o')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score Method')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(2, max_clusters + 1), inertia_values, 'r-o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.grid(True)

plt.tight_layout()
plt.savefig('phase1_plots/cluster_optimization.png')
plt.close()

# Define number of clusters
n_clusters = 5

# Apply KMeans for the final clustering
print(f"Performing final K-means clustering with {n_clusters} clusters...")
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
lifecycle_samples['cluster'] = kmeans.fit_predict(X_scaled)

# Apply PCA to visualize clusters
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Add PCA components and cluster to dataframe
lifecycle_samples['pca_1'] = X_pca[:, 0]
lifecycle_samples['pca_2'] = X_pca[:, 1]

# Visualize the clusters in PCA space with RUL information
plt.figure(figsize=(12, 10))
scatter = plt.scatter(lifecycle_samples['pca_1'], lifecycle_samples['pca_2'], 
                     c=lifecycle_samples['cluster'], cmap='viridis', 
                     s=50, alpha=0.8)
plt.colorbar(scatter, label='Cluster')
plt.title('PCA Visualization of Engine Degradation Clusters')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.grid(True, alpha=0.3)
plt.savefig('phase1_plots/pca_clusters.png')
plt.close()

# Overlay RUL information on PCA visualization
plt.figure(figsize=(12, 10))
scatter = plt.scatter(lifecycle_samples['pca_1'], lifecycle_samples['pca_2'], 
                     c=lifecycle_samples['RUL'], cmap='coolwarm', 
                     s=50, alpha=0.8)
plt.colorbar(scatter, label='RUL')
plt.title('PCA Visualization with RUL Values')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.grid(True, alpha=0.3)
plt.savefig('phase1_plots/pca_rul.png')
plt.close()

# Analyze the relationship between clusters and RUL
rul_by_cluster = lifecycle_samples.groupby('cluster')['RUL'].agg(['mean', 'median', 'min', 'max', 'count'])
rul_by_cluster = rul_by_cluster.sort_values('mean', ascending=False)
print("\nRUL statistics by cluster:")
print(rul_by_cluster)

# Visualize RUL distribution by cluster
plt.figure(figsize=(14, 8))
sns.boxplot(x='cluster', y='RUL', data=lifecycle_samples, order=rul_by_cluster.index)
plt.title('RUL Distribution by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Remaining Useful Life (cycles)')
plt.savefig('phase1_plots/rul_by_cluster.png')
plt.close()

# Map clusters to degradation stages based on average RUL
# Sorting clusters by average RUL (highest to lowest)
stage_mapping = dict(zip(rul_by_cluster.index, ['Stage 0: Normal', 
                                              'Stage 1: Slightly Degraded', 
                                              'Stage 2: Moderately Degraded', 
                                              'Stage 3: Critical', 
                                              'Stage 4: Failure']))

# Map stages to numerical values for later modeling
stage_to_numerical = {
    'Stage 0: Normal': 0,
    'Stage 1: Slightly Degraded': 1,
    'Stage 2: Moderately Degraded': 2,
    'Stage 3: Critical': 3,
    'Stage 4: Failure': 4
}

# Add stage labels to the samples
lifecycle_samples['stage'] = lifecycle_samples['cluster'].map(stage_mapping)
lifecycle_samples['stage_numeric'] = lifecycle_samples['stage'].map(stage_to_numerical)

print("\nDegradation stage mapping:")
for cluster, stage in stage_mapping.items():
    print(f"Cluster {cluster} → {stage}")

# Visualize stage distribution
plt.figure(figsize=(12, 6))
stage_counts = lifecycle_samples['stage'].value_counts().sort_index()
sns.barplot(x=stage_counts.index, y=stage_counts.values)
plt.title('Distribution of Degradation Stages')
plt.xlabel('Degradation Stage')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('phase1_plots/stage_distribution.png')
plt.close()

# Apply the trained clustering model to all data
print("Applying clustering model to all training data...")
# Prepare the full dataset with the same features
train_features = train_enhanced[valid_feature_cols].copy()

# Check and clean the full dataset before prediction
inf_mask_full = np.isinf(train_features.values)
if inf_mask_full.any():
    print(f"Warning: Found {inf_mask_full.sum()} infinity values in full dataset. Replacing with 0.")
    train_features = train_features.replace([np.inf, -np.inf], 0)

nan_mask_full = np.isnan(train_features.values)
if nan_mask_full.any():
    print(f"Warning: Found {nan_mask_full.sum()} NaN values in full dataset. Replacing with 0.")
    train_features = train_features.fillna(0)

# Scale the features
X_all_scaled = scaler.transform(train_features.values)
# Predict clusters for all data points
train_enhanced['cluster'] = kmeans.predict(X_all_scaled)
# Map to degradation stages
train_enhanced['stage'] = train_enhanced['cluster'].map(stage_mapping)
train_enhanced['stage_numeric'] = train_enhanced['stage'].map(stage_to_numerical)

# Save the mapping between clusters and stages for later use
cluster_stage_mapping = pd.DataFrame({
    'cluster': list(stage_mapping.keys()),
    'stage': list(stage_mapping.values()),
    'stage_numeric': [stage_to_numerical[stage] for stage in stage_mapping.values()],
    'mean_RUL': rul_by_cluster['mean'].values,
    'min_RUL': rul_by_cluster['min'].values,
    'max_RUL': rul_by_cluster['max'].values,
})
cluster_stage_mapping.to_csv('phase1_csv/cluster_stage_mapping.csv', index=False)

# Visualize the degradation progression for a few sample engines
def plot_engine_degradation(engine_ids, df):
    plt.figure(figsize=(15, 10))
    
    for i, engine_id in enumerate(engine_ids):
        engine_data = df[df['engine_id'] == engine_id].sort_values('cycle')
        
        plt.subplot(len(engine_ids), 1, i+1)
        plt.plot(engine_data['cycle'], engine_data['stage_numeric'], marker='o', markersize=4)
        plt.title(f'Degradation Progression for Engine {engine_id}')
        plt.xlabel('Cycle')
        plt.ylabel('Degradation Stage')
        plt.yticks(range(5), ['Normal', 'Slightly Deg', 'Moderately Deg', 'Critical', 'Failure'])
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('phase1_plots/degradation_progression.png')
    plt.close()

# Plot degradation for 5 sample engines
sample_engines = train_enhanced['engine_id'].unique()[:5]
plot_engine_degradation(sample_engines, train_enhanced)

# Calculate time-to-next-stage for each data point
print("Calculating time to next stage...")
def calculate_time_to_next_stage(df):
    df = df.copy()
    df['time_to_next_stage'] = np.nan
    
    for engine_id in df['engine_id'].unique():
        engine_data = df[df['engine_id'] == engine_id].sort_values('cycle')
        
        for i in range(len(engine_data)-1):
            current_stage = engine_data.iloc[i]['stage_numeric']
            next_stages = engine_data.iloc[i+1:]['stage_numeric']
            
            # Find the first occurrence of a higher stage
            next_stage_indices = next_stages[next_stages > current_stage].index
            
            if len(next_stage_indices) > 0:
                next_stage_idx = next_stage_indices[0]
                next_stage_cycle = df.loc[next_stage_idx, 'cycle']
                current_cycle = engine_data.iloc[i]['cycle']
                df.loc[engine_data.iloc[i].name, 'time_to_next_stage'] = next_stage_cycle - current_cycle
    
    return df

train_enhanced = calculate_time_to_next_stage(train_enhanced)

# Visualization of time to next stage by current stage
plt.figure(figsize=(12, 8))
sns.boxplot(x='stage', y='time_to_next_stage', data=train_enhanced.dropna(subset=['time_to_next_stage']))
plt.title('Time to Next Degradation Stage')
plt.xlabel('Current Stage')
plt.ylabel('Cycles until Next Stage')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('phase1_plots/time_to_next_stage.png')
plt.close()

# Save the processed data for later phases
print("Saving processed data...")
train_enhanced.to_csv('phase1_csv/train_enhanced_with_stages.csv', index=False)

# Print summary statistics
print("\nSummary statistics for degradation stages:")
stage_stats = train_enhanced.groupby('stage_numeric')['RUL'].agg(['count', 'mean', 'min', 'max'])
print(stage_stats)

print("\nSummary statistics for time to next stage:")
time_stats = train_enhanced.dropna(subset=['time_to_next_stage']).groupby('stage_numeric')['time_to_next_stage'].agg(['count', 'mean', 'min', 'max'])
print(time_stats)

print("\nPhase 1 preprocessing complete. Enhanced data saved for model development.")