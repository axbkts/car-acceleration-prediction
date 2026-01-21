import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore') 

# ============================================
# 1. LOAD DATASET
# ============================================
print("CAR DATASET PREPARATION - REMOVING MISSING VALUES")
print("=" * 50)

try:
    data = pd.read_csv('Cars Datasets 2025.csv', encoding='cp1252')
    print("Dataset loaded successfully.")
    print(f"Initial row count: {data.shape[0]}")
    print(f"Column count: {data.shape[1]}")
except Exception as e:
    print(f"Error: {e}")
    exit()

# ============================================
# 2. MERGE COMPANY AND CAR NAMES
# ============================================
print("\n2. MERGING COMPANY AND CAR NAMES")
print("-" * 50)

if 'Company Names' in data.columns and 'Cars Names' in data.columns:
    data['Car_Name'] = data['Company Names'] + ' ' + data['Cars Names']
    print("Car names merged: 'Car_Name' column created.")
    print(f"Examples: {data['Car_Name'].iloc[0]}, {data['Car_Name'].iloc[1]}")

# ============================================
# 3. REMOVE ROWS WITH RANGE VALUES
# ============================================
print("\n3. DETECTING AND REMOVING ROWS WITH RANGE VALUES")
print("-" * 50)

range_patterns = [
    r'\d+\s*-\s*\d+',
    r'\d+\s*–\s*\d+',
    r'\$\d+,\d+\s*-\s*\$\d+,\d+',
]

rows_with_ranges = []
for idx, row in data.iterrows():
    row_has_range = False
    
    numeric_cols_to_check = ['HorsePower', 'Total Speed', 'Performance(0 - 100 )KM/H', 
                             'Cars Prices', 'Torque', 'CC/Battery Capacity', 'Seats']
    
    for col in numeric_cols_to_check:
        if col in data.columns and isinstance(row[col], str):
            cell_value = str(row[col])
            for pattern in range_patterns:
                if re.search(pattern, cell_value):
                    rows_with_ranges.append(idx)
                    row_has_range = True
                    break
        if row_has_range:
            break

rows_with_ranges = list(set(rows_with_ranges))

print(f"Number of rows containing range expressions: {len(rows_with_ranges)}")

if len(rows_with_ranges) > 0:
    data_clean = data.drop(rows_with_ranges).reset_index(drop=True)
    print(f"{len(rows_with_ranges)} rows removed (range expressions).")
else:
    data_clean = data.copy()
    print("No range expressions found.")

print(f"Row count after removing ranges: {data_clean.shape[0]}")

# ============================================
# 4. REMOVE ENGINE COLUMN
# ============================================
print("\n4. REMOVING ENGINE COLUMN")
print("-" * 50)

if 'Engines' in data_clean.columns:
    data_clean = data_clean.drop(columns=['Engines'])
    print("'Engines' column removed.")
else:
    print("'Engines' column not found.")

# ============================================
# 5. CLEAN NUMERIC VALUES
# ============================================
print("\n5. CLEANING NUMERIC VALUES")
print("-" * 50)

def extract_numeric(value):
    if pd.isna(value):
        return np.nan
    
    if isinstance(value, (int, float)):
        return float(value)
    
    value_str = str(value).strip()
    value_str = value_str.replace(',', '')
    value_str = value_str.replace('$', '').replace('€', '').replace('£', '')
    
    units = ['hp', 'km/h', 'sec', 'Nm', 'cc', 'km', 'h', 'kg', 'l', 'L', 'hp,', 'cc,', 'Nm,', 'sec,', 'km/h,']
    for unit in units:
        value_str = value_str.replace(unit, '')
    
    match = re.search(r'(\d+\.?\d*)', value_str)
    if match:
        try:
            return float(match.group(1))
        except:
            return np.nan
    
    return np.nan

numeric_columns = ['HorsePower', 'Total Speed', 'Performance(0 - 100 )KM/H', 
                   'Cars Prices', 'Torque', 'CC/Battery Capacity', 'Seats']

for col in numeric_columns:
    if col in data_clean.columns:
        data_clean[f'{col}_clean'] = data_clean[col].apply(extract_numeric)
        valid_count = data_clean[f'{col}_clean'].notna().sum()
        print(f"{col:30s} -> {valid_count:4d} valid values")

# ============================================
# 6. ENCODE FUEL TYPES
# ============================================
print("\n6. ENCODING FUEL TYPES")
print("-" * 50)

if 'Fuel Types' in data_clean.columns:
    le_fuel = LabelEncoder()
    data_clean['Fuel_Code'] = le_fuel.fit_transform(data_clean['Fuel Types'].fillna('Unknown'))
    
    print("Fuel Type Encoding:")
    for code, fuel_type in enumerate(le_fuel.classes_):
        count = (data_clean['Fuel Types'] == fuel_type).sum()
        print(f"  Code {code:2d} = {fuel_type:20s} : {count:3d} cars")
else:
    print("'Fuel Types' column not found.")

# ============================================
# 7. HANDLE MISSING VALUES
# ============================================
print("\n7. MISSING VALUE ANALYSIS AND REMOVAL")
print("-" * 50)

clean_columns = [col for col in data_clean.columns if col.endswith('_clean') or col == 'Fuel_Code']

# Missing value analysis
missing_data = data_clean[clean_columns].isnull().sum()
total_missing_before = missing_data.sum()

print("Missing data distribution (before removal):")
for col in clean_columns:
    if missing_data[col] > 0:
        col_name = col.replace('_clean', '')
        print(f"{col_name:30s} : {missing_data[col]:4d} missing")

print(f"\nTotal missing values: {total_missing_before}")
print(f"Row count before removal: {data_clean.shape[0]}")

# Find rows with missing values
rows_with_missing = data_clean[data_clean[clean_columns].isnull().any(axis=1)]

if len(rows_with_missing) > 0:
    print(f"\nNumber of rows with missing data: {len(rows_with_missing)}")
    
    # Remove rows with missing values
    data_clean = data_clean.dropna(subset=clean_columns).reset_index(drop=True)
    print(f"\n{len(rows_with_missing)} rows removed (contained missing values).")
else:
    print("\nNo rows with missing data found.")

print(f"Row count after removal: {data_clean.shape[0]}")

# ============================================
# 8. CREATE FINAL DATASET
# ============================================
print("\n8. CREATING FINAL DATASET")
print("-" * 50)

final_data = pd.DataFrame()
final_data['Car_Name'] = data_clean['Car_Name']

column_mapping = {
    'HorsePower_clean': 'Horsepower',
    'Total Speed_clean': 'TopSpeed_kmh',
    'Performance(0 - 100 )KM/H_clean': 'Acceleration_sec',
    'Cars Prices_clean': 'Price_USD',
    'Torque_clean': 'Torque_Nm',
    'CC/Battery Capacity_clean': 'Engine_CC',
    'Seats_clean': 'Seats',
    'Fuel_Code': 'Fuel_Type_Code'
}

for old_col, new_col in column_mapping.items():
    if old_col in data_clean.columns:
        final_data[new_col] = data_clean[old_col]

print(f"Final dataset created: {final_data.shape[0]} rows, {final_data.shape[1]} columns")

# Basic statistics
print("\nBasic Statistics (First 5 features):")
numeric_cols = [col for col in final_data.columns if col != 'Car_Name']
for col in numeric_cols[:5]:
    print(f"{col:20s}: Min={final_data[col].min():.2f}, Max={final_data[col].max():.2f}, "
          f"Mean={final_data[col].mean():.2f}")

# ============================================
# 9. SAVE DATA
# ============================================
print("\n9. SAVING DATA")
print("-" * 50)

final_data.to_csv('cars_data_preprocessed_no_missing.csv', index=False)
print("Dataset saved as 'cars_data_preprocessed_no_missing.csv'")

numeric_only = final_data.drop('Car_Name', axis=1)
numeric_only.to_csv('cars_data_numeric_no_missing.csv', index=False)
print("Numeric-only data saved as 'cars_data_numeric_no_missing.csv'")

if 'le_fuel' in locals():
    fuel_mapping = pd.DataFrame({
        'Fuel_Code': range(len(le_fuel.classes_)),
        'Fuel_Type': le_fuel.classes_
    })
    fuel_mapping.to_csv('fuel_type_codes.csv', index=False)
    print("Fuel type codes saved as 'fuel_type_codes.csv'")

print("\n" + "=" * 50)
print("PROCESS COMPLETED SUCCESSFULLY")
print("=" * 50)