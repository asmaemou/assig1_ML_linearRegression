import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """
    Load the fish dataset
    """
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully!")
    print(df.head())
    return df

def perform_feature_engineering(df):
    """
    I computed the average of Length1, Length2, and Length3.
    Then select 'Length_avg', 'Height', and 'Width' as features (X) and 'Weight' as the target (y).
    """
    df['Length_avg'] = df[['Length1', 'Length2', 'Length3']].mean(axis=1)
    
    # Select the features and target
    X = df[['Length_avg', 'Height', 'Width']]
    y = df['Weight']
    
    return X, y

def apply_log_transformation(df):
    """
    I applied log transformation to handle outliers and reduce skewness.
    """
    log_features = ['Length_avg', 'Height', 'Width', 'Weight']
    
    for feature in log_features:
        df[f'log_{feature}'] = np.log1p(df[feature])  # log(1 + X) to handle zero values

    return df

def perform_eda(df, log_transformed=False):
    """
    Perform exploratory data analysis (EDA) by printing dataset overview and generating plots.
    Histograms, box plots, and scatter plots are saved to a 'figures' directory.
    """
    folder = 'figures_log' if log_transformed else 'figures'
    
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    print("Data Information:")
    print(df.info(), "\n")

    print("Missing values:")
    print(df.isnull().sum(), "\n")

    print("Number of columns and rows:")
    print(df.shape)

    print("Summary Statistics:")
    print(df.describe().round(2), "\n")
    
    plot_features = ['Length_avg', 'Height', 'Width', 'Weight']
    if log_transformed:
        plot_features = [f'log_{feature}' for feature in plot_features]
    
    for col in plot_features:
        plt.figure(figsize=(8,6))
        sns.histplot(df[col], kde=False, bins=20)
        plt.title(f'Histogram of {col}')
        plt.savefig(f'{folder}/hist_{col}.png')
        plt.close()
    
    for col in plot_features:
        plt.figure(figsize=(8,6))
        sns.boxplot(x=df[col])
        plt.title(f'Box Plot of {col}')
        plt.savefig(f'{folder}/box_{col}.png')
        plt.close()
    
    for feature in ['Length_avg', 'Height', 'Width']:
        plt.figure(figsize=(8,6))
        sns.scatterplot(x=df[f'log_{feature}'], y=df['log_Weight'])
        plt.title(f'Scatter Plot: log(Weight) vs log({feature})')
        plt.savefig(f'{folder}/scatter_log_Weight_{feature}.png')
        plt.close()

def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def main():
    file_path = '../dataset/Fish.csv'
    df = load_data(file_path)
    
    X, y = perform_feature_engineering(df)
    
    df = apply_log_transformation(df)
    
    print("\n Performing EDA on Original Data:")
    perform_eda(df, log_transformed=False)

    print("\n Performing EDA on Log-Transformed Data:")
    perform_eda(df, log_transformed=True)
    
    log_features = ['log_Length_avg', 'log_Height', 'log_Width']
    X_log = df[log_features]
    y_log = df['log_Weight']
    
    X_scaled, scaler = scale_features(X_log)
    
    preprocessed_data = {'X': X_log, 'X_scaled': X_scaled, 'y': y_log, 'scaler': scaler}
    with open('preprocessed_data.pkl', 'wb') as f:
        pickle.dump(preprocessed_data, f)
    
    print("Data preprocessing complete. Preprocessed data saved to 'preprocessed_data.pkl'.")

if __name__ == "__main__":
    main()
