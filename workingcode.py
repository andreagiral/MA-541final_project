import pandas as pd
import numpy as np

# 1. Load the dataset
df = pd.read_csv('surv_variants_updated.csv')
df.columns = df.columns.str.strip()  # Remove any leading/trailing whitespace
print("Initial Data Info:")
print(df.info())
print("\nInitial Data Head:")
print(df.head())
print("\nSummary Statistics:")
print(df.describe())
print("--------------------------------------------------\n")

# 2. Remove duplicate rows
num_duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {num_duplicates}")
if num_duplicates > 0:
    df = df.drop_duplicates()
    print("Duplicates have been removed.")
print("--------------------------------------------------\n")

# 3. Check for missing values (NaN) and zeros for specific columns
missing = df.isnull().sum()
print("\nMissing values per column (NaN):")
print(missing)
print("--------------------------------------------------\n")

# 4. Handle missing values (NaN):
# For numeric columns, fill missing values with the median.
# For categorical columns, fill missing values with the mode.
for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
print("Missing (NaN) values have been handled.")
print("--------------------------------------------------\n")

# 5. Specific handling for zeros representing missing values in key columns.
# We'll assume that missing data in total_cases, total_deaths, and mortality_rate are recorded as 0.
# We'll group by 'country' and 'variant' if available.
group_cols = []
if 'Country' in df.columns:
    group_cols.append('Country')
if 'variant' in df.columns:
    group_cols.append('variant')
print("Group columns:", group_cols)

# For total_cases and total_deaths: replace zeros with NaN and impute.
for col in ['total_cases', 'total_deaths']:
    if col in df.columns:
        print(f"Processing group-based imputation for {col}")
        df[col] = df[col].replace(0, np.nan)
        if group_cols:
            df[col] = df.groupby(group_cols)[col].transform(lambda x: x.fillna(x.median()))
        df[col] = df[col].fillna(df[col].median())
    else:
        print(f"Column {col} not found.")

# For mortality_rate: replace zeros with NaN and impute.
if 'mortality_rate' in df.columns:
    print("Processing mortality_rate adjustments")
    df['mortality_rate'] = df['mortality_rate'].replace(0, np.nan)
    if group_cols:
        df['mortality_rate'] = df.groupby(group_cols)['mortality_rate'].transform(lambda x: x.fillna(x.median()))
    df['mortality_rate'] = df['mortality_rate'].fillna(df['mortality_rate'].median())
else:
    print("Column 'mortality_rate' not found.")

# Ensure internal consistency:
# For rows where total_cases is nonzero:
mask = df['total_cases'].notna() & (df['total_cases'] != 0)
# 1. If mortality_rate is missing or zero, recalculate it as total_deaths / total_cases.
df.loc[mask & (df['mortality_rate'].isna() | (df['mortality_rate'] == 0)),
       'mortality_rate'] = df['total_deaths'] / df['total_cases']
# 2. If total_deaths is missing or zero but mortality_rate is valid, infer total_deaths.
df.loc[mask & ((df['total_deaths'].isna()) | (df['total_deaths'] == 0)) &
       (df['mortality_rate'].notna()) & (df['mortality_rate'] != 0),
       'total_deaths'] = df['total_cases'] * df['mortality_rate']
# 3. Recalculate mortality_rate for all rows with nonzero total_cases for final consistency.
df.loc[mask, 'mortality_rate'] = df['total_deaths'] / df['total_cases']

print("Checkpoint 5: Specific handling for zeros and consistency adjustments complete.")
print("--------------------------------------------------\n")

# 6. Date conversion and duration calculation:
if 'first_seq' in df.columns and 'last_seq' in df.columns:
    print("Processing date conversion and duration calculation using 'first_seq' and 'last_seq'...")
    df['first_seq'] = pd.to_datetime(df['first_seq'], errors='coerce')
    df['last_seq'] = pd.to_datetime(df['last_seq'], errors='coerce')
    # Calculate duration as the difference in days plus 1 (ensuring a minimum of 1)
    df['duration'] = (df['last_seq'] - df['first_seq']).dt.days + 1
    # Explicitly set any duration < 1 to 1
    df.loc[df['duration'] < 1, 'duration'] = 1
    print("After .loc adjustment, min duration:", df['duration'].min())
    print("Duration column calculated based on 'first_seq' and 'last_seq', with minimum value set to 1.")
else:
    print("Either 'first_seq' or 'last_seq' column is missing; skipping duration calculation.")
print("--------------------------------------------------\n")

# 7. Outlier detection and removal using the IQR method for numeric columns
IQR_multiplier = 3     # Increased from 1.5 to 3 to reduce removal of legitimate data
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
outlier_mask = pd.Series(False, index=df.index)
print("\nOutlier removal process:")
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - IQR_multiplier * IQR
    upper_bound = Q3 + IQR_multiplier * IQR
    # Mark rows as outliers if any numeric column falls outside its bounds
    outlier_mask |= (df[col] < lower_bound) | (df[col] > upper_bound)
    print(f"Column '{col}': Lower bound = {lower_bound}, Upper bound = {upper_bound}")
# Filter the dataframe to remove outliers
df = df[~outlier_mask]
print("\nCleaned data shape after outlier removal:", df.shape)
print("--------------------------------------------------\n")

# 8. Fix data formatting issues:
# Trim whitespace from string columns
str_cols = df.select_dtypes(include=['object']).columns
for col in str_cols:
    df[col] = df[col].str.strip()
if 'date' in df.columns: # Optionally, convert a generic 'date' column to datetime if it exists
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
print("Converted 'date' column to datetime format.")
print("--------------------------------------------------\n")

# 9. Final inspection of the cleaned data
print("\nCleaned Data Info:")
print(df.info())
print("\nCleaned Data Head:")
print(df.head())
print("\nCleaned Data Tail:")
print(df.tail())
print("--------------------------------------------------\n")

#10. Grouping by Continent
if 'Continent' in df.columns:
    print("Grouping data by Continent:")
    continents = df['Continent'].unique()
    for cont in sorted(continents):
        cont_df = df[df['Continent'] == cont]
        # Optionally, sort by country if available
        if 'country' in cont_df.columns:
            cont_df = cont_df.sort_values(by='country')
        print(f"Continent: {cont}, Number of records: {len(cont_df)}")
        # Save each continent's data to a separate CSV file
        cont_file = f"surv_variants_cleaned_{cont}.csv"
        cont_df.to_csv(cont_file, index=False)
        print(f"Saved data for {cont} to {cont_file}")
else:
    print("No 'Continent' column found. Skipping grouping by continent.")
print("--------------------------------------------------\n")

# 11. Save the cleaned DataFrame to a new CSV file
cleaned_file_path = 'surv_variants_cleaned.csv'
df.to_csv(cleaned_file_path, index=False)
print(f"\nCleaned data saved to {cleaned_file_path}")