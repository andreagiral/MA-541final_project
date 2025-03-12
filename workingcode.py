import pandas as pd
import numpy as np

df = pd.read_csv('surv_variants_updated.csv')
summary_stats = df.describe()
print(summary_stats)
print ("Initial Data Info:")
print(df.info())
print("\nInitial Data Head:")
print(df.head())

# 2. Remove duplicate rows
num_duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {num_duplicates}")
if num_duplicates > 0:
    df = df.drop_duplicates()
    print("Duplicates have been removed.")

# 3. Check for missing values and display count per column
missing = df.isnull().sum()
print("\nMissing values per column:")
print(missing)

# 4. Handle missing values:
# For numeric columns, fill missing values with the median.
# For categorical columns, fill missing values with the mode.
for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)
    else:
        # Use the mode; if mode() returns an empty series, use a default placeholder.
        mode_value = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
        df[col] = df[col].fillna(mode_value)
print("Missing values have been handled.")

# 5. Outlier detection and removal using the IQR method for numeric columns
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
outlier_mask = pd.Series(False, index=df.index)
print("\nOutlier removal process:")
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Mark as True if the value is an outlier in this column
    outlier_mask |= (df[col] < lower_bound) | (df[col] > upper_bound)
    print(f"Column '{col}': Lower bound = {lower_bound}, Upper bound = {upper_bound}")
# 6. Filter the dataframe once
df_cleaned = df[~outlier_mask]
print("\nCleaned data shape:", df_cleaned.shape)

# 7. Fix data formatting issues:
# Example: trim whitespace from string columns
str_cols = df.select_dtypes(include=['object']).columns
for col in str_cols:
    df[col] = df[col].str.strip()

# Optionally, if a date column exists (e.g., named 'date'), convert it to datetime
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    print("Converted 'date' column to datetime format.")

# 8. Final inspection of the cleaned data
print("\nCleaned Data Info:")
print(df.info())
print("\nCleaned Data Head:")
print(df.head())
print("\nCleaned Data Tail:")
print(df.tail())

# 9. Save the cleaned DataFrame to a new CSV file
cleaned_file_path = 'surv_variants_cleaned.csv'
df.to_csv(cleaned_file_path, index=False)
print(f"\nCleaned data saved to {cleaned_file_path}")