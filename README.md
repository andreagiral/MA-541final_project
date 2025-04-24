
# MA-541final_project
# 🧬 COVID-19 Variant Impact Analysis
<small>
This project explores how different COVID-19 variants behaved across various countries and continents using data-driven methods. We analyzed trends in **mortality**, **duration**, **growth rate**, and **total cases**, while applying statistical inference and predictive modeling techniques. Our goal was to understand the relationship between spread rate, geography, and severity of each variant.
---
## 📁 Repository Structure

| File/Folder | Description |
|-------------|-------------|
| `surv_variants_updated.csv` | Original dataset containing raw COVID-19 variant data |
| `surv_variants_cleaned.csv` | Cleaned version of the dataset after preprocessing |
| `surv_variants_cleaned_<Continent>.csv` | Cleaned datasets broken down by continent (Africa, Asia, Europe, etc.) |
| `Data Cleaning.py` | Script for comprehensive data cleaning (missing values, outliers, formatting, consistency) |
| `summary_statistics and Hypothesis Tests.py` | Summary statistics (means, counts, unique values) and hypothesis testing (t-tests, Mann–Whitney U) |
| `Types of Distribution and Estimates Parameters.py` | Distribution fitting (normal vs. lognormal), Q-Q plots, and AIC/K-S test evaluations |
| `Correlation Analysis.py` | Correlation heatmaps, Pearson coefficients, and pairwise variable analysis |
| `Building Models.py` | Regression (linear, ridge, lasso, PCR) and classification models (logistic, decision tree) |
| `Final_code.py` | Combined script with all sections (cleaning, statistics, hypothesis, distributions, correlations, models/classification) |
| `CODE+VISUALS.ipynb` |	Full code with visual graphs for presentation and exploration |
| `ANOVA Tests.xlsx` | ANOVA test results for comparing mortality rates across continents |
---
## 🔍 Key Analyses
### ✅ Data Cleaning
- Removal of duplicates and outliers using IQR
- Imputation of missing or zero values with medians and group-based methods
- Feature engineering: calculated **duration** from sequencing dates
- Outputs: `surv_variants_cleaned.csv` and per-continent CSVs

### 📊 Summary Statistics & Hypothesis Testing
- Descriptive statistics by variant and region
- Top 10 countries by total cases and deaths
- Two-sample *t*-test and Mann–Whitney U test to compare mortality rates between groups

### 🔁 Distribution Fitting
- Fitted **normal** and **lognormal** distributions to total cases, duration, and mortality rate
- Lognormal provided a better fit in most cases (AIC and K-S tests)

### 📈 Correlation Analysis
- Heatmap and Pearson correlation between all numerical features
- Found strong correlation between total cases and total deaths
- Weak correlation between duration, growth rate, and mortality rate

### 🔮 Predictive Modeling
- Linear, Ridge, Lasso, and Principal Components Regression to predict **total_cases**
- Random Forest and Support Vector Regression explored for non-linear modeling
- All regression models showed limited predictive performance due to data complexity

### 🧠 Classification Modeling
- Classified variants as **high** or **low mortality** using logistic regression and decision tree classifiers
- Models had limited accuracy (~50–58%), suggesting that mortality rate depends on factors outside available features
---
## 🧾 Conclusion
Our analysis reveals valuable descriptive insights but limited predictive power due to the complexity of COVID-19 variant behavior. Mortality rates vary significantly across regions and growth rates, but prediction requires additional data (e.g., healthcare infrastructure, vaccination coverage). While we modeled distributions and correlations effectively, accurate forecasting remains a challenge with this dataset alone.

---
<small>
