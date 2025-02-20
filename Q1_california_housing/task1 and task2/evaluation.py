import pandas as pd
import joblib
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm

def load_data():
    df = pd.read_csv('data/processed_data.csv')
    X = df.drop(columns=['median_house_value'])
    y = df['median_house_value']
    return X, y

def load_models():

    simple_lr_model = joblib.load('models/simple_linear_model.pkl')
    lr_model = joblib.load('models/linear_model.pkl')
    ridge_model = joblib.load('models/ridge_model.pkl')
    lasso_model = joblib.load('models/lasso_model.pkl')
    ols_model = sm.load('models/ols_model.pickle')

    return simple_lr_model, lr_model, ridge_model, lasso_model,ols_model

def evaluate_model(model, X, y, model_name="Model"):

    predictions = model.predict(X)
    r2 = r2_score(y, predictions)
    mse = mean_squared_error(y, predictions)
    print(f"{model_name} - R² Score: {r2:.4f}, MSE: {mse:.4f}")
def evaluate_ols_model(ols_model, X, y):

    X_sm = sm.add_constant(X, has_constant='add')
    predictions = ols_model.predict(X_sm)
    r2 = r2_score(y, predictions)
    mse = mean_squared_error(y, predictions)
    print(f"OLS Regression - R² Score: {r2:.4f}, MSE: {mse:.4f}")

def main():
    X, y = load_data()
    simple_lr_model, lr_model, ridge_model, lasso_model, ols_model = load_models()
    
    # I evaluated Simple Linear Regression (using only 'median_income')
    print("Evaluating Simple Linear Regression (using 'median_income'):")
    X_simple = X[['median_income']]
    evaluate_model(simple_lr_model, X_simple, y, "Simple Linear Regression")
    
    # I evaluated Multiple Linear Regression
    print("\nEvaluating Multiple Linear Regression:")
    evaluate_model(lr_model, X, y, "Multiple Linear Regression")
    
    # I evaluated Ridge Regression
    print("\nEvaluating Ridge Regression:")
    evaluate_model(ridge_model, X, y, "Ridge Regression")
    
    # I evaluated Lasso Regression
    print("\nEvaluating Lasso Regression:")
    evaluate_model(lasso_model, X, y, "Lasso Regression")

    # I evaluated OLS Regression
    print("\nEvaluating OLS Regression:")
    evaluate_ols_model(ols_model, X, y)
if __name__ == "__main__":
    main()
