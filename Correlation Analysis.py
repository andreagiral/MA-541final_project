import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# 1. Load the dataset
df = pd.read_csv('surv_variants_cleaned.csv')
df.columns = df.columns.str.strip()  
print("Initial Data Info:")
print(df.info())

#############################################
# 12. Correlation Analysis
#############################################

# Compute correlation using only numeric columns
numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()

# Print the correlation matrix
print("\nCorrelation Matrix:")
print(corr_matrix)

# Visualize the correlation matrix using a heatmap.
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Numerical Variables")
plt.show()

# Optionally, display a pairplot for an exploratory view of pairwise relationships.
# This will include scatter plots and distributions for each numeric variable.
numeric_cols = df.select_dtypes(include=[np.number]).columns
sns.pairplot(df[numeric_cols], diag_kind="kde")
plt.suptitle("Pairwise Relationships of Numerical Variables", y=1.02)
plt.show()

# Another option: Calculate and print the Pearson correlation coefficients directly for key variables.
# For example, to see the correlation between total_cases and total_deaths:
pearson_corr = df['total_cases'].corr(df['total_deaths'])
print(f"\nPearson correlation between total_cases and total_deaths: {pearson_corr:.2f}")

# Similarly, check correlations between duration, mortality_rate, and other metrics:
corr_duration_mort = df['duration'].corr(df['mortality_rate'])
print(f"Pearson correlation between duration and mortality_rate: {corr_duration_mort:.2f}")

corr_growth_tc = None
if 'growth_rate' in df.columns:
    corr_growth_tc = df['growth_rate'].corr(df['total_cases'])
    print(f"Pearson correlation between growth_rate and total_cases: {corr_growth_tc:.2f}")
