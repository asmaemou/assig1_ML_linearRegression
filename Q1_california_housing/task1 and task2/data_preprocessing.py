"""
This module loads the California Housing dataset, performs exploratory data analysis (EDA),
applies transformations (log + polynomial), removes outliers, applies feature scaling,
and saves the processed data for subsequent modeling.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

file_path = "../dataset/housing.csv"  # Ensure this path is correct

def load_data(file_path):
    """
    Load the California Housing dataset from a CSV file.
    """
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully!")
    print(df.head(10))  # Show the first few rows for verification
    return df

def transform_features(df):
    """
    1. Log-transform the target (median_house_value) and skewed features (total_rooms, population).
    2. Add a polynomial feature for median_income (degree=2).
    """
    # 1) Log-transform the target
    df["median_house_value"] = np.log1p(df["median_house_value"])  # log(1 + x)

    # 2) Log-transform skewed numerical features
    skewed_cols = ["total_rooms", "population"]
    for col in skewed_cols:
        df[col] = np.log1p(df[col])

    return df

def remove_outliers(df):
    """
    Remove outliers from numerical columns using the IQR method.
    I do this AFTER log transformations so the data is more normalized.
    """
    numerical_features = ["total_rooms", "total_bedrooms", "population", "median_income"]
    
    for feature in numerical_features:
        if feature in df.columns:  
            Q1 = df[feature].quantile(0.25)
            Q3 = df[feature].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            df = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]
    
    return df

def perform_eda(df):
    """
    Perform exploratory data analysis (EDA)
    """
    # Ensure the 'figures' directory exists
    if not os.path.exists('figures'):
        os.makedirs('figures')

    # Basic Dataset Overview
    print("Overview of the Dataset:")
    print(df.head(), "\n")

    print("Checking for missing values and data types:")
    print(df.info(), "\n")

    print("Summary Statistics:")
    print(df.describe().round(2), "\n")

    print("Missing values:")
    print(df.isnull().sum(), "\n")

    # Plot 1: Histogram of *Log-Transformed* Median House Value
    plt.figure(figsize=(14, 6))
    sns.histplot(df['median_house_value'], bins=30, kde=True, color='blue')
    plt.title('Distribution of Log(1 + Median House Value)', fontsize=14)
    plt.xlabel('Log(1 + Median House Value)', fontsize=12, labelpad=10)
    plt.ylabel('Frequency', fontsize=12, labelpad=10)
    plt.xticks(fontsize=10, rotation=30, ha='right')
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('figures/log_median_house_val_distribution.png', bbox_inches='tight', dpi=300)
    plt.close()

    # Plot 2: Box Plot of (Log) Features
    plt.figure(figsize=(10, 4))
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove('median_house_value')  # remove target from boxplot
    sns.boxplot(data=df[numeric_cols])
    plt.title('Boxplot of (Log/Transformed) Features')
    plt.xticks(rotation=30, ha='right', fontsize=8)
    plt.xlabel("Features")
    plt.savefig('figures/transformed_features_boxplot.png', bbox_inches='tight', dpi=300)
    plt.close()

    # Plot 3: Correlation Heatmap
    df_encoded = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True)
    plt.figure(figsize=(12, 8))
    corr_matrix = df_encoded.corr()
    sns.heatmap(
        corr_matrix, annot=True, fmt=".2f", cmap='coolwarm',
        linewidths=0.5, xticklabels=True, yticklabels=True
    )
    plt.title("Feature Correlation Heatmap (After Transformations)", fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.savefig('figures/transformed_correlation_heatmap.png', bbox_inches='tight', dpi=300)
    plt.close()

    # Plot 4: Scatter Plot - `median_income` vs. `median_house_value`
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df['median_income'], y=df['median_house_value'], alpha=0.5)
    plt.title('Median Income vs. Log(1 + Median House Value)', fontsize=14)
    plt.xlabel('Median Income', fontsize=12)
    plt.ylabel('Log(1 + Median House Value)', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('figures/income_vs_log_median_house_value.png', bbox_inches='tight', dpi=300)
    plt.close()

    print("EDA completed; plots have been saved in the 'figures' folder.")


def feature_scaling(df):
    """
    1) Transform features (log + polynomial).
    2) Remove outliers.
    3) Impute missing values.
    4) One-Hot Encode.
    5) Scale numeric features.
    """
    # 1) Transform features (log + polynomial)
    df = transform_features(df)

    # 2) Remove outliers AFTER transformations
    df = remove_outliers(df)

    # 3) Impute missing values
    imputer = SimpleImputer(strategy='median')
    if 'total_bedrooms' in df.columns:
        df[['total_bedrooms']] = imputer.fit_transform(df[['total_bedrooms']])

    # 4) One-Hot Encoding
    if 'ocean_proximity' in df.columns:
        df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True)

    # 5) Feature Scaling
    scaler = StandardScaler()
    features = df.drop(columns=['median_house_value'])
    scaled_features = scaler.fit_transform(features)

    scaled_df = pd.DataFrame(scaled_features, columns=features.columns)

    # I kept the (log-transformed) target in the final dataset
    scaled_df['median_house_value'] = df['median_house_value'].values
    
    return scaled_df, scaler

def main():
    df = load_data(file_path)
    print("Data loaded successfully.")
    print(df.shape)

    df_transformed = transform_features(df.copy())
    perform_eda(df_transformed)
    print("EDA completed; figures saved in the 'figures' folder.")
    
    scaled_df, scaler = feature_scaling(df)
    print("Missing values after applying imputation:")
    print(scaled_df.isnull().sum())

    if not os.path.exists('data'):
        os.makedirs('data')
    scaled_df.to_csv('data/processed_data.csv', index=False)
    print("Feature scaling applied; processed data saved in 'data/processed_data.csv'.")

if __name__ == "__main__":
    main()
