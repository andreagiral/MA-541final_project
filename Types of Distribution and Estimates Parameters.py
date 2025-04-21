import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 1. Load the dataset
df = pd.read_csv('surv_variants_cleaned.csv')
df.columns = df.columns.str.strip()  
print("Initial Data Info:")
print(df.info())

#############################################
# Analysis for "Total Cases"
#############################################
data_tc = df['total_cases'].dropna()

# 1. Histogram & KDE for Total Cases
plt.figure(figsize=(10, 6))
sns.histplot(data_tc, bins=30, kde=True, color="skyblue", edgecolor="black")
plt.title("Histogram with KDE of Total Cases")
plt.xlabel("Total Cases")
plt.ylabel("Frequency")
plt.show()

# 2. Q–Q Plot for Total Cases (Normality Check)
plt.figure(figsize=(10, 6))
stats.probplot(data_tc, dist="norm", plot=plt)
plt.title("Normal Q–Q Plot for Total Cases")
plt.show()

# 3. Fit a Normal Distribution to Total Cases: # This will estimate the mean and standard deviation (mu and sigma) using maximum likelihood estimation.
mu_tc, sigma_tc = stats.norm.fit(data_tc)
print(f"[Total Cases] Fitted Normal: mu = {mu_tc:.2f}, sigma = {sigma_tc:.2f}")

# Calculate log likelihood and AIC for Normal (Total Cases)
log_likelihood_norm_tc = np.sum(np.log(stats.norm.pdf(data_tc, mu_tc, sigma_tc)))
k_norm = 2  # Two estimated parameters (mu and sigma)
AIC_norm_tc = -2 * log_likelihood_norm_tc + 2 * k_norm
ks_stat_norm_tc, p_value_norm_tc = stats.kstest(data_tc, 'norm', args=(mu_tc, sigma_tc))

# 4. Fit a Lognormal Distribution to Total Cases
ln_shape_tc, ln_loc_tc, ln_scale_tc = stats.lognorm.fit(data_tc, floc=0)
print(f"[Total Cases] Fitted Lognormal: shape = {ln_shape_tc:.2f}, loc = {ln_loc_tc:.2f}, scale = {ln_scale_tc:.2f}")

# Calculate log likelihood and AIC for Lognormal (Total Cases)
log_likelihood_lnorm_tc = np.sum(np.log(stats.lognorm.pdf(data_tc, ln_shape_tc, ln_loc_tc, ln_scale_tc)))
k_lnorm = 2  # We assume loc fixed at 0, so two free parameters: shape and scale.
AIC_lnorm_tc = -2 * log_likelihood_lnorm_tc + 2 * k_lnorm
ks_stat_lnorm_tc, p_value_lnorm_tc = stats.kstest(data_tc, 'lognorm', args=(ln_shape_tc, ln_loc_tc, ln_scale_tc))

# Plot fitted PDFs for Total Cases
x_tc = np.linspace(data_tc.min(), data_tc.max(), 100)
pdf_norm_tc = stats.norm.pdf(x_tc, mu_tc, sigma_tc)
pdf_lnorm_tc = stats.lognorm.pdf(x_tc, ln_shape_tc, ln_loc_tc, ln_scale_tc)

plt.figure(figsize=(10, 6))
sns.histplot(data_tc, bins=30, stat="density", color="skyblue", edgecolor="black", alpha=0.7)
plt.plot(x_tc, pdf_norm_tc, 'r-', lw=2, label="Fitted Normal PDF")
plt.plot(x_tc, pdf_lnorm_tc, 'g-', lw=2, label="Fitted Lognormal PDF")
plt.title("Total Cases with Fitted Distributions")
plt.xlabel("Total Cases")
plt.ylabel("Density")
plt.legend()
plt.show()
# Print Comparison Metrics for Total case
print("\n--- Total Cases Distribution Comparison ---")
print(f"Normal Distribution: AIC = {AIC_norm_tc:.2f}, K-S p-value = {p_value_norm_tc:.4f}")
print(f"Lognormal Distribution: AIC = {AIC_lnorm_tc:.2f}, K-S p-value = {p_value_lnorm_tc:.4f}")

#############################################
# Analysis for "Duration"
#############################################
data_duration = df['duration'].dropna()
# Visualization: Histogram, KDE, & Q-Q plot
plt.figure(figsize=(10, 6))
sns.histplot(data_duration, bins=30, kde=True, color="skyblue", edgecolor="black")
plt.title("Histogram with KDE of Duration")
plt.xlabel("Duration (days)")
plt.ylabel("Frequency")
plt.show()

# 2. Q–Q Plot for Duration (Normality Check)
plt.figure(figsize=(10, 6))
stats.probplot(data_duration, dist="norm", plot=plt)
plt.title("Normal Q–Q Plot for Duration")
plt.show()

# 3. Fit a Normal Distribution to Durantion
mu_dur, sigma_dur = stats.norm.fit(data_duration)
print(f"[Duration] Fitted Normal: mu = {mu_dur:.2f}, sigma = {sigma_dur:.2f}")

# Calculate log likelihood and AIC for Normal (Duration)
log_likelihood_norm_dur = np.sum(np.log(stats.norm.pdf(data_duration, mu_dur, sigma_dur)))
k_norm = 2  # Two estimated parameters (mu and sigma)
AIC_norm_dur = -2 * log_likelihood_norm_dur + 2 * k_norm
ks_stat_norm_dur, p_value_norm_dur = stats.kstest(data_duration, 'norm', args=(mu_dur, sigma_dur))

# 4. Fit a Lognormal Distribution to Duration
ln_shape_dur, ln_loc_dur, ln_scale_dur = stats.lognorm.fit(data_duration, floc=0)
print(f"[Duration] Fitted Lognormal: shape = {ln_shape_dur:.2f}, loc = {ln_loc_dur:.2f}, scale = {ln_scale_dur:.2f}")

# Calculate log likelihood and AIC for Lognormal (Duration)
log_likelihood_lnorm_dur = np.sum(np.log(stats.lognorm.pdf(data_duration, ln_shape_dur, ln_loc_dur, ln_scale_dur)))
k_lnorm = 2  # We assume loc fixed at 0, so two free parameters: shape and scale.
AIC_lnorm_dur = -2 * log_likelihood_lnorm_dur + 2 * k_lnorm
ks_stat_lnorm_dur, p_value_lnorm_dur = stats.kstest(data_duration, 'lognorm', args=(ln_shape_dur, ln_loc_dur, ln_scale_dur))

# Plot fitted PDFs for Duration
x_dur = np.linspace(data_duration.min(), data_duration.max(), 100)
pdf_norm_dur = stats.norm.pdf(x_dur, mu_dur, sigma_dur)
pdf_lnorm_dur = stats.lognorm.pdf(x_dur, ln_shape_dur, ln_loc_dur, ln_scale_dur)

plt.figure(figsize=(10, 6))
sns.histplot(data_duration, bins=30, stat="density", color="skyblue", edgecolor="black", alpha=0.7)
plt.plot(x_dur, pdf_norm_dur, 'r-', lw=2, label="Fitted Normal PDF")
plt.plot(x_dur, pdf_lnorm_dur, 'g-', lw=2, label="Fitted Lognormal PDF")
plt.title("Duration with Fitted Distributions")
plt.xlabel("Duration (days)")
plt.ylabel("Density")
plt.legend()
plt.show()

# Print Comparison Metrics for Duration
print("\n--- Duration Distribution Comparison ---")
print(f"Normal Distribution: AIC = {AIC_norm_dur:.2f}, K-S p-value = {p_value_norm_dur:.4f}")
print(f"Lognormal Distribution: AIC = {AIC_lnorm_dur:.2f}, K-S p-value = {p_value_lnorm_dur:.4f}")

#############################################
# Analysis for "Mortality Rate"
#############################################
data_mort = df['mortality_rate'].dropna()
# Visualization: Histogram, KDE, & Q-Q plot
plt.figure(figsize=(10, 6))
sns.histplot(data_mort, bins=30, kde=True, color="skyblue", edgecolor="black")
plt.title("Histogram with KDE of Mortality Rate")
plt.xlabel("Mortality Rate")
plt.ylabel("Frequency")
plt.show()

# 2. Q–Q Plot for Mortality Rate (Normality Check)
plt.figure(figsize=(10, 6))
stats.probplot(data_mort, dist="norm", plot=plt)
plt.title("Normal Q–Q Plot for Mortality Rate")
plt.show()

# 3. Fit a Normal Distribution to Mortality Rate
mu_mort, sigma_mort = stats.norm.fit(data_mort)
print(f"[Mortality Rate] Fitted Normal: mu = {mu_mort:.4f}, sigma = {sigma_mort:.4f}")

# Calculate log likelihood and AIC for Normal (Mortality Rate)
log_likelihood_norm_mort = np.sum(np.log(stats.norm.pdf(data_mort, mu_mort, sigma_mort)))
k_norm = 2  # Two estimated parameters (mu and sigma)
AIC_norm_mort = -2 * log_likelihood_norm_mort + 2 * k_norm
ks_stat_norm_mort, p_value_norm_mort = stats.kstest(data_mort, 'norm', args=(mu_mort, sigma_mort))

# 4. Fit a Lognormal Distribution to Mortality Rate
ln_shape_mort, ln_loc_mort, ln_scale_mort = stats.lognorm.fit(data_mort, floc=0)
print(f"[Mortality Rate] Fitted Lognormal: shape = {ln_shape_mort:.4f}, loc = {ln_loc_mort:.4f}, scale = {ln_scale_mort:.4f}")

# Calculate log likelihood and AIC for Lognormal (Mortality Rate)
log_likelihood_lnorm_mort = np.sum(np.log(stats.lognorm.pdf(data_mort, ln_shape_mort, ln_loc_mort, ln_scale_mort)))
k_lnorm = 2  # We assume loc fixed at 0, so two free parameters: shape and scale.
AIC_lnorm_mort = -2 * log_likelihood_lnorm_mort + 2 * k_lnorm
ks_stat_lnorm_mort, p_value_lnorm_mort = stats.kstest(data_mort, 'lognorm', args=(ln_shape_mort, ln_loc_mort, ln_scale_mort))

# Plot fitted PDFs for Mortality Rate
x_mort = np.linspace(data_mort.min(), data_mort.max(), 100)
pdf_norm_mort = stats.norm.pdf(x_mort, mu_mort, sigma_mort)
pdf_lnorm_mort = stats.lognorm.pdf(x_mort, ln_shape_mort, ln_loc_mort, ln_scale_mort)

plt.figure(figsize=(10, 6))
sns.histplot(data_mort, bins=30, stat="density", color="skyblue", edgecolor="black", alpha=0.7)
plt.plot(x_mort, pdf_norm_mort, 'r-', lw=2, label="Fitted Normal PDF")
plt.plot(x_mort, pdf_lnorm_mort, 'g-', lw=2, label="Fitted Lognormal PDF")
plt.title("Mortality Rate with Fitted Distributions")
plt.xlabel("Mortality Rate")
plt.ylabel("Density")
plt.legend()
plt.show()

# Print Comparison Metrics for Mortality Rate
print("\n--- Mortality Rate Distribution Comparison ---")
print(f"Normal Distribution: AIC = {AIC_norm_mort:.2f}, K-S p-value = {p_value_norm_mort:.4f}")
print(f"Lognormal Distribution: AIC = {AIC_lnorm_mort:.2f}, K-S p-value = {p_value_lnorm_mort:.4f}")

