import pickle
import statsmodels.api as sm
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Load preprocessed data
with open('preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)

X, y = data['X'], data['y']

# Add a constant for the intercept (for OLS)
X_const = sm.add_constant(X)

# Train OLS regression
ols_model = sm.OLS(y, X_const).fit()

# Standardize the features (required for Ridge & Lasso regularization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# GridSearchCV for Ridge Regression
ridge_param_grid = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
ridge_grid = GridSearchCV(Ridge(), ridge_param_grid, scoring='neg_mean_squared_error', cv=5)
ridge_grid.fit(X_scaled, y)
best_ridge = ridge_grid.best_estimator_
print("Best alpha for Ridge:", ridge_grid.best_params_)

# GridSearchCV for Lasso Regression
lasso_param_grid = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]}
lasso_grid = GridSearchCV(Lasso(max_iter=10000), lasso_param_grid, scoring='neg_mean_squared_error', cv=5)
lasso_grid.fit(X_scaled, y)
best_lasso = lasso_grid.best_estimator_
print("Best alpha for Lasso:", lasso_grid.best_params_)

# Save all models (OLS, best Ridge, best Lasso) along with the scaler
with open('trained_models.pkl', 'wb') as f:
    pickle.dump({
        'ols_model': ols_model,
        'ridge_model': best_ridge,
        'lasso_model': best_lasso,
        'scaler': scaler
    }, f)

print("OLS, Ridge (tuned), and Lasso (tuned) models trained and saved to 'trained_models.pkl'.")