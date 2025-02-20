import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import statsmodels.api as sm
import joblib

def load_processed_data():
    df = pd.read_csv('data/processed_data.csv')
    return df

def split_data(df):
    """
    Split the dataset into training and testing sets.
    """
    X = df.drop(columns=['median_house_value'])
    y = df['median_house_value']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_simple_linear_regression(X_train, y_train):
    """
    Train a simple linear regression model using only the 'median_income' feature.
    """
    X_train_simple = X_train[['median_income']]
    simple_lr = LinearRegression()
    simple_lr.fit(X_train_simple, y_train)
    return simple_lr

def train_multiple_linear_regression(X_train, y_train):
    """
    Train a multiple linear regression model using all features.
    """
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    return lr

def tune_ridge_regression(X_train, y_train):
    """
    Tune Ridge regression using GridSearchCV to find the best alpha.
    """
    ridge = Ridge()
    param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
    ridge_grid = GridSearchCV(ridge, param_grid, scoring='neg_mean_squared_error', cv=5)
    ridge_grid.fit(X_train, y_train)
    print("Best alpha for Ridge:", ridge_grid.best_params_)
    return ridge_grid.best_estimator_

def tune_lasso_regression(X_train, y_train):
    """
    Tune Lasso regression using GridSearchCV to find the best alpha.
    """
    lasso = Lasso(max_iter=10000)
    param_grid = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]}
    lasso_grid = GridSearchCV(lasso, param_grid, scoring='neg_mean_squared_error', cv=5)
    lasso_grid.fit(X_train, y_train)
    print("Best alpha for Lasso:", lasso_grid.best_params_)
    return lasso_grid.best_estimator_

def train_ols_regression(X_train, y_train):
    """
    Train an OLS regression model using statsmodels to obtain a detailed summary.
    """
    X_train_sm = sm.add_constant(X_train)
    ols_model = sm.OLS(y_train, X_train_sm).fit()
    return ols_model

def main():
    df = load_processed_data()
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Train simple linear regression using 'median_income'
    simple_lr_model = train_simple_linear_regression(X_train, y_train)
    print("Simple Linear Regression (using 'median_income') trained.")
    
    # Train multiple linear regression using all features
    lr_model = train_multiple_linear_regression(X_train, y_train)
    print("Multiple Linear Regression trained.")
    
    # Tune and train Ridge regression
    ridge_model = tune_ridge_regression(X_train, y_train)
    print("Tuned Ridge Regression trained.")
    
    # Tune and train Lasso regression
    lasso_model = tune_lasso_regression(X_train, y_train)
    print("Tuned Lasso Regression trained.")
    
    # 5) OLS Regression
    ols_model = train_ols_regression(X_train, y_train)
    print("OLS Regression trained. Summary:")
    print(ols_model.summary())
    
    # Ensure models directory exists
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Save the trained models
    joblib.dump(simple_lr_model, 'models/simple_linear_model.pkl')
    joblib.dump(lr_model, 'models/linear_model.pkl')
    joblib.dump(ridge_model, 'models/ridge_model.pkl')
    joblib.dump(lasso_model, 'models/lasso_model.pkl')
    ols_model.save('models/ols_model.pickle')
    print("OLS model saved as models/ols_model.pickle")


if __name__ == "__main__":
    main()
