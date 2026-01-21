"""
MLP Regression Model Optimization
Predicting 0-100 km/h Acceleration Time for Cars
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

warnings.filterwarnings('ignore')

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# ============================================================================
# 0. SETUP PATHS
# ============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
input_file_path = os.path.join(script_dir, 'cars_data_preprocessed_no_missing.csv')

print("=" * 70)
print("MLP REGRESSION MODEL OPTIMIZATION")
print(f"Working Directory: {script_dir}")
print("=" * 70)

# ============================================================================
# 1. LOAD DATA AND PREPROCESSING
# ============================================================================

print("\n1. LOADING DATA...")
try:
    df = pd.read_csv(input_file_path)
    print(f"   Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
except FileNotFoundError:
    print(f"   ERROR: File not found at {input_file_path}")
    print("   Please run 'data_preprocessing.py' first.")
    exit()

# Fuel_Type_Code adjustment
df['Fuel_Type_Code'] = df['Fuel_Type_Code'].replace({16: 15})

# Data preview
print("\n2. DATA PREVIEW:")
print("-" * 40)
print(df.head())
print(f"\n   Columns: {list(df.columns)}")

# ============================================================================
# 2. DATA ANALYSIS & TARGET VARIABLE
# ============================================================================

print("\n3. TARGET VARIABLE ANALYSIS:")
print("-" * 40)

print(f"   Target Variable (Acceleration_sec) Statistics:")
print(f"   - Mean: {df['Acceleration_sec'].mean():.3f} sec")
print(f"   - Std Dev: {df['Acceleration_sec'].std():.3f} sec")
print(f"   - Min: {df['Acceleration_sec'].min():.3f} sec")
print(f"   - Max: {df['Acceleration_sec'].max():.3f} sec")

# Target Variable Distribution Plot
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(df['Acceleration_sec'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
plt.title('0-100 km/h Time Distribution', fontsize=12, fontweight='bold')
plt.xlabel('Seconds')
plt.ylabel('Count')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.boxplot(df['Acceleration_sec'], patch_artist=True, 
            boxprops=dict(facecolor='lightcoral'))
plt.title('0-100 km/h Time Boxplot', fontsize=12, fontweight='bold')
plt.ylabel('Seconds')
plt.grid(True, alpha=0.3)

plt.tight_layout()
print("   > Displaying target variable plot... (Close window to continue)")
plt.show() 

# ============================================================================
# 3. DATA PREPARATION
# ============================================================================

print("\n4. DATA PREPARATION:")
print("-" * 40)

# Drop Car_Name column
df_ml = df.drop('Car_Name', axis=1)

# Split features (X) and target (y)
X = df_ml.drop('Acceleration_sec', axis=1)
y = df_ml['Acceleration_sec']

print(f"   Features (X): {X.shape}")
print(f"   Target (y): {y.shape}")

# ============================================================================
# 4. SPLIT AND SCALE
# ============================================================================

print("\n5. SPLITTING AND SCALING DATA:")
print("-" * 40)

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"   Training Set: {X_train.shape[0]} samples")
print(f"   Test Set: {X_test.shape[0]} samples")

# Scaling (StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("   Data scaling (StandardScaler) completed.")

# ============================================================================
# 5. MLP MODEL OPTIMIZATION (GRID SEARCH)
# ============================================================================

print("\n6. MLP PARAMETER OPTIMIZATION (GRID SEARCH):")
print("-" * 40)

# Initialize MLP model
mlp = MLPRegressor(
    random_state=42,
    max_iter=5000,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    verbose=False
)

# Optimized Parameter Grid
param_grid = {
    'hidden_layer_sizes': [(100,), (100, 50), (150, 75)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam'],
    'alpha': [0.0001, 0.001],
    'learning_rate': ['adaptive'],
    'learning_rate_init': [0.001, 0.01],
    'batch_size': [32, 64]
}

# Calculate total combinations
total_combinations = (len(param_grid['hidden_layer_sizes']) * len(param_grid['activation']) * len(param_grid['solver']) * len(param_grid['alpha']) * len(param_grid['learning_rate']) * len(param_grid['learning_rate_init']) * len(param_grid['batch_size']))

print(f"   Total combinations to test: {total_combinations}")
print(f"   Total training runs (with 5-fold CV): {total_combinations * 5}")
print("\n   Starting Grid Search... Please wait.")

# Initialize GridSearchCV
grid_search = GridSearchCV(
    mlp,
    param_grid,
    cv=5,
    scoring=['r2', 'neg_mean_squared_error', 'neg_mean_absolute_percentage_error'],
    refit='r2',
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)

# Run GridSearch
grid_search.fit(X_train_scaled, y_train)

print("\n   Grid Search completed.")

# ============================================================================
# 6. PERFORMANCE TABLE
# ============================================================================

print("\n7. PERFORMANCE TABLE (ALL COMBINATIONS):")
print("-" * 40)

# Convert results to DataFrame
cv_results = pd.DataFrame(grid_search.cv_results_)

# Create readable performance table
performance_list = []

for idx in range(len(cv_results)):
    params = cv_results.loc[idx, 'params']
    
    mean_r2 = cv_results.loc[idx, 'mean_test_r2']
    mean_neg_mse = cv_results.loc[idx, 'mean_test_neg_mean_squared_error']
    mean_mse = -mean_neg_mse
    mean_rmse = np.sqrt(mean_mse)
    
    mean_neg_mape = cv_results.loc[idx, 'mean_test_neg_mean_absolute_percentage_error']
    mean_mape = -mean_neg_mape * 100
    
    performance_list.append({
        'Rank': int(cv_results.loc[idx, 'rank_test_r2']),
        'Hidden_Layers': str(params['hidden_layer_sizes']),
        'Activation': params['activation'],
        'Solver': params['solver'],
        'Alpha': params['alpha'],
        'LR': params['learning_rate'],
        'Init_LR': params['learning_rate_init'],
        'Batch_Size': params['batch_size'],
        'R2': f"{mean_r2:.4f}",
        'MSE': f"{mean_mse:.4f}",
        'RMSE': f"{mean_rmse:.4f}",
        'MAPE_%': f"{mean_mape:.2f}",
        'R2_Std': f"{cv_results.loc[idx, 'std_test_r2']:.4f}"
    })

performance_df = pd.DataFrame(performance_list)
performance_df.sort_values('Rank', inplace=True)
performance_df['Best'] = performance_df['Rank'].apply(lambda x: 'YES' if x == 1 else '')

# Display top 10
print("\n   TOP 10 PARAMETER COMBINATIONS:")
print("   " + "-" * 90)

display_cols = ['Rank', 'Hidden_Layers', 'Activation', 'Alpha', 'Init_LR', 
                'Batch_Size', 'R2', 'RMSE', 'MAPE_%', 'Best']

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)
print(performance_df[display_cols].head(10).to_string(index=False))

# Save table to CSV (Tables are still saved as they are data, not plots)
csv_filename = os.path.join(script_dir, f"mlp_performance_table_{timestamp}.csv")
performance_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
print(f"\n   Performance table saved: {csv_filename}")

# Display best params
best_params = grid_search.best_params_
print(f"\n   BEST MODEL PARAMETERS (Rank 1):")
print(f"   - Hidden Layers: {best_params['hidden_layer_sizes']}")
print(f"   - Activation: {best_params['activation']}")
print(f"   - Solver: {best_params['solver']}")
print(f"   - Alpha: {best_params['alpha']}")
print(f"   - Learning Rate: {best_params['learning_rate']}")
print(f"   - Init Learning Rate: {best_params['learning_rate_init']}")
print(f"   - Batch Size: {best_params['batch_size']}")
print(f"   - Best CV R2 Score: {grid_search.best_score_:.4f}")

# ============================================================================
# 7. TEST SET PERFORMANCE
# ============================================================================

print("\n8. BEST MODEL PERFORMANCE ON TEST SET:")
print("-" * 40)

# Get best model
best_model = grid_search.best_estimator_

# Predict on test set
y_pred = best_model.predict(X_test_scaled)

# Calculate metrics
test_r2 = r2_score(y_test, y_pred)
test_mse = mean_squared_error(y_test, y_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_pred)
test_mape = mean_absolute_percentage_error(y_test, y_pred) * 100

print(f"   Test Set Metrics:")
print(f"   - R2 Score: {test_r2:.4f}")
print(f"   - MSE: {test_mse:.4f}")
print(f"   - RMSE: {test_rmse:.4f} sec")
print(f"   - MAE: {test_mae:.4f} sec")
print(f"   - MAPE: {test_mape:.2f}%")

# Save test results
test_results_df = pd.DataFrame({
    'Metric': ['R2', 'MSE', 'RMSE', 'MAE', 'MAPE_%'],
    'Value': [test_r2, test_mse, test_rmse, test_mae, test_mape],
    'Unit': ['-', 'sec^2', 'sec', 'sec', '%']
})
test_csv_filename = os.path.join(script_dir, f"test_performance_{timestamp}.csv")
test_results_df.to_csv(test_csv_filename, index=False, encoding='utf-8-sig')
print(f"\n   Test metrics saved: {test_csv_filename}")

# ============================================================================
# 8. VISUALIZATIONS
# ============================================================================

print("\n9. GENERATING PLOTS:")
print("-" * 40)

# Create a figure with subplots
plt.figure(figsize=(14, 10))

# 1. Actual vs Predicted Scatter Plot
plt.subplot(2, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='w', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values (sec)', fontsize=11)
plt.ylabel('Predicted Values (sec)', fontsize=11)
plt.title(f'Actual vs Predicted (Test R2 = {test_r2:.3f})', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)

# 2. Error Distribution Histogram
plt.subplot(2, 2, 2)
errors = y_test - y_pred
plt.hist(errors, bins=30, edgecolor='black', alpha=0.7, color='lightgreen')
plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
plt.xlabel('Error (Actual - Predicted)', fontsize=11)
plt.ylabel('Frequency', fontsize=11)
plt.title('Error Distribution Histogram', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)

# 3. Feature Correlations
plt.subplot(2, 2, 3)
correlations = X.corrwith(y).abs().sort_values(ascending=True)
colors = ['skyblue' if x < 0.5 else 'lightcoral' for x in correlations.values]
plt.barh(range(len(correlations)), correlations.values, color=colors, edgecolor='black')
plt.yticks(range(len(correlations)), correlations.index)
plt.xlabel('Absolute Correlation Coefficient', fontsize=11)
plt.title('Feature Correlation with Acceleration', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')

# 4. Loss Curve
plt.subplot(2, 2, 4)
if hasattr(best_model, 'loss_curve_'):
    plt.plot(best_model.loss_curve_, color='purple', linewidth=2)
    plt.xlabel('Iterations', fontsize=11)
    plt.ylabel('Loss', fontsize=11)
    plt.title('Training Loss Curve', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
else:
    # Fallback: Residual plot
    plt.scatter(y_pred, errors, alpha=0.6, color='orange')
    plt.axhline(y=0, color='black', linestyle='--', linewidth=2)
    plt.xlabel('Predicted Values', fontsize=11)
    plt.ylabel('Residuals', fontsize=11)
    plt.title('Residual Plot', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
print("   > Displaying final result plots... (Close window to finish)")
plt.show()

# ============================================================================
# 9. PROJECT SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("PROJECT SUMMARY")
print("=" * 70)

print(f"\nDATASET:")
print(f"  - Total samples: {df.shape[0]}")
print(f"  - Features: {X.shape[1]}")
print(f"  - Target: Acceleration_sec (0-100 km/h)")

print(f"\nMODEL OPTIMIZATION:")
print(f"  - Total combinations tested: {total_combinations}")
print(f"  - Best CV R2: {grid_search.best_score_:.4f}")
print(f"  - Test R2: {test_r2:.4f}")

print(f"\nBEST MODEL METRICS:")
print(f"  - RMSE: {test_rmse:.3f} sec")
print(f"  - MAPE: {test_mape:.2f}%")

print(f"\nOUTPUTS:")
print(f"  1. {csv_filename} (Performance Table)")
print(f"  2. {test_csv_filename} (Test Metrics)")
print(f"  3. Plots shown in pop-up windows.")

print("\n" + "=" * 70)
print("PROCESS COMPLETED")
print("=" * 70)