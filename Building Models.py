import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Regression imports
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Classification imports
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler

#############################################
# Regression Modeling: Predicting total_cases
#############################################

# Load the cleaned dataset and ensure column names are stripped
df = pd.read_csv('surv_variants_cleaned.csv')
df.columns = df.columns.str.strip()
print("Initial Data Info:")
print(df.info())

# Select predictors and response for regression.
predictor_candidates = ['duration', 'mortality_rate', 'growth_rate']
predictors = [col for col in predictor_candidates if col in df.columns]
print("Using regression predictors:", predictors)

# Remove rows with missing values for the predictors and response.
df_model = df.dropna(subset=predictors + ['total_cases'])
X = df_model[predictors]
y = df_model['total_cases']

# Split the dataset into training and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

### 1. Baseline Linear Regression ###
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("\n--- Linear Regression ---")
print("R2 Score:", r2_score(y_test, y_pred_lr))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))

# Plot Actual vs Predicted for Linear Regression
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_lr, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual total_cases")
plt.ylabel("Predicted total_cases")
plt.title("Linear Regression: Actual vs. Predicted")
plt.show()

### 2. Ridge Regression ###
ridge = Ridge()
param_grid = {'alpha': np.logspace(-3, 3, 7)}
ridge_cv = GridSearchCV(ridge, param_grid, scoring='neg_mean_squared_error', cv=5)
ridge_cv.fit(X_train, y_train)
print("\n--- Ridge Regression ---")
print("Best alpha:", ridge_cv.best_params_['alpha'])
y_pred_ridge = ridge_cv.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred_ridge))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_ridge)))

# Plot Actual vs Predicted for Ridge Regression
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_ridge, color='green', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual total_cases")
plt.ylabel("Predicted total_cases")
plt.title("Ridge Regression: Actual vs. Predicted")
plt.show()

### 3. Lasso Regression ###
lasso = Lasso(max_iter=10000)
param_grid = {'alpha': np.logspace(-3, 3, 7)}
lasso_cv = GridSearchCV(lasso, param_grid, scoring='neg_mean_squared_error', cv=5)
lasso_cv.fit(X_train, y_train)
print("\n--- Lasso Regression ---")
print("Best alpha:", lasso_cv.best_params_['alpha'])
y_pred_lasso = lasso_cv.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred_lasso))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lasso)))

# Plot Actual vs Predicted for Lasso Regression
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_lasso, color='purple', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual total_cases")
plt.ylabel("Predicted total_cases")
plt.title("Lasso Regression: Actual vs. Predicted")
plt.show()

### 4. Principal Components Regression (PCR) ###
n_components = min(len(predictors), 2)
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
lr_pca = LinearRegression()
lr_pca.fit(X_train_pca, y_train)
y_pred_pca = lr_pca.predict(X_test_pca)
print("\n--- Principal Components Regression ---")
print(f"Number of components used: {n_components}")
print("R2 Score:", r2_score(y_test, y_pred_pca))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_pca)))

# Plot Actual vs Predicted for PCR
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_pca, color='orange', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual total_cases")
plt.ylabel("Predicted total_cases")
plt.title("PCR: Actual vs. Predicted")
plt.show()

#############################################
# Nonlinear Relationships
#############################################

# ----------------------------------
# 1. Random Forest Regressor
# ----------------------------------
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

# ----------------------------------
# 2. Support Vector Regressor (with scaling)
# ----------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
svr.fit(X_train_scaled, y_train)
y_pred_svr = svr.predict(X_test_scaled)
r2_svr = r2_score(y_test, y_pred_svr)
rmse_svr = np.sqrt(mean_squared_error(y_test, y_pred_svr))

# Plot: Random Forest Predictions
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_rf, color='darkorange', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual total_cases")
plt.ylabel("Predicted total_cases")
plt.title("Random Forest: Actual vs. Predicted")
plt.grid(True)
plt.show()

# Plot: SVR Predictions
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_svr, color='teal', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual total_cases")
plt.ylabel("Predicted total_cases")
plt.title("SVR (RBF Kernel): Actual vs. Predicted")
plt.grid(True)
plt.show()

print("\n--- Random Forest Prediction ---")
print ("R2 Score:", r2_rf)
print("RMSE:", rmse_rf) 
    
print("\n--- SVR Prediction ---")
print ("R2 Score:", r2_svr)
print ("RMSE:", rmse_svr)

#############################################
# Classification Modeling: Predicting High Mortality
#############################################

# Create a binary target variable 'high_mortality'
median_mort = df['mortality_rate'].median()
df['high_mortality'] = (df['mortality_rate'] > median_mort).astype(int)
print("\nMedian Mortality Rate:", median_mort)

# Choose classification predictors.
predictor_candidates_class = []
if 'total_cases' in df.columns:
    predictor_candidates_class.append('total_cases')
if 'duration' in df.columns:
    predictor_candidates_class.append('duration')
if 'growth_rate' in df.columns:
    predictor_candidates_class.append('growth_rate')
print("Classification predictors:", predictor_candidates_class)

X_class = df[predictor_candidates_class].dropna()
y_class = df.loc[X_class.index, 'high_mortality']

Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_class, y_class, test_size=0.20, random_state=42)

# Standardize predictors for logistic regression.
scaler = StandardScaler()
Xc_train_scaled = scaler.fit_transform(Xc_train)
Xc_test_scaled = scaler.transform(Xc_test)

### 5. Logistic Regression Classifier ###
logreg = LogisticRegression()
logreg.fit(Xc_train_scaled, yc_train)
yc_pred_logreg = logreg.predict(Xc_test_scaled)
print("\n--- Logistic Regression Classification ---")
print("Accuracy:", accuracy_score(yc_test, yc_pred_logreg))
print("Classification Report:")
print(classification_report(yc_test, yc_pred_logreg))
print("Confusion Matrix:")
print(confusion_matrix(yc_test, yc_pred_logreg))

# Plot Confusion Matrix for Logistic Regression
cm_lr = confusion_matrix(yc_test, yc_pred_logreg)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap="Blues")
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve for Logistic Regression
yc_pred_prob_logreg = logreg.predict_proba(Xc_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(yc_test, yc_pred_prob_logreg)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Logistic Regression")
plt.legend(loc="lower right")
plt.show()

### 6. Decision Tree Classifier ###
dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(Xc_train, yc_train)  # Decision trees do not require scaling.
yc_pred_dtree = dtree.predict(Xc_test)
print("\n--- Decision Tree Classification ---")
print("Accuracy:", accuracy_score(yc_test, yc_pred_dtree))
print("Classification Report:")
print(classification_report(yc_test, yc_pred_dtree))
print("Confusion Matrix:")
print(confusion_matrix(yc_test, yc_pred_dtree))

# Plot Confusion Matrix for Decision Tree
cm_dt = confusion_matrix(yc_test, yc_pred_dtree)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_dt, annot=True, fmt='d', cmap="Reds")
plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

