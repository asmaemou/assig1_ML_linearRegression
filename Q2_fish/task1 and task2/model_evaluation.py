import pickle
import numpy as np
from sklearn.linear_model import Ridge, Lasso  

# Load trained models and preprocessed data
with open('trained_models.pkl', 'rb') as f:
    models = pickle.load(f)

with open('preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)

X, y = data['X'], data['y']
scaler = models['scaler']

X_scaled = scaler.transform(X)

# Retrieve trained models
ols_model = models['ols_model']
ridge_model = models['ridge_model']
lasso_model = models['lasso_model']

# Evaluate OLS Model

print("\n OLS Regression Results:")
print(ols_model.summary())

print("\n OLS Coefficients (Least Squares Estimates):")
print(ols_model.params)

print("\nStandard Errors of OLS Coefficients:")
print(ols_model.bse)

print("\n OLS R-squared Value:")
print(ols_model.rsquared)


# Evaluation for  Ridge & Lasso Models
ridge_r2 = ridge_model.score(X_scaled, y)
lasso_r2 = lasso_model.score(X_scaled, y)

print("\n Ridge Regression Results:")
print("R-squared:", ridge_r2)
print("Coefficients:", ridge_model.coef_)

print("\nLasso Regression Results:")
print("R-squared:", lasso_r2)
print("Coefficients:", lasso_model.coef_)


# Bootstrapping for Standard Errors
def bootstrap_coef_std(model, X, y, n_bootstraps=1000):
    """Estimate standard errors of coefficients using bootstrapping."""
    coef_samples = []
    n = len(y)
    for _ in range(n_bootstraps):
        sample_indices = np.random.choice(n, size=n, replace=True)
        X_sample, y_sample = X[sample_indices], y.iloc[sample_indices]
        model.fit(X_sample, y_sample)
        coef_samples.append(model.coef_)
    coef_samples = np.array(coef_samples)
    return np.std(coef_samples, axis=0)

ridge_std_errors = bootstrap_coef_std(Ridge(alpha=1.0), X_scaled, y)
lasso_std_errors = bootstrap_coef_std(Lasso(alpha=0.1, max_iter=10000), X_scaled, y)

print("\n Bootstrap Estimated Standard Errors:")
print("Ridge Coefficients Standard Errors:", ridge_std_errors)
print("Lasso Coefficients Standard Errors:", lasso_std_errors)
