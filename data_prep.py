import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_clean_data(filepath='WA_Fn-UseC_-Telco-Customer-Churn.csv'):
    """
    Loads raw Telco Churn CSV, performs data cleaning, converts types,
    and encodes categorical variables for modeling.
    """
    # Load dataset
    df = pd.read_csv(filepath)

    # 1. Convert TotalCharges from string object to numeric float
    # Blank strings ' ' occur when tenure == 0; coercing errors yields NaN, filled with 0.
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].str.strip(), errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)

    # 2. Target Encoding
    df['Churn_Numeric'] = df['Churn'].map({'Yes': 1, 'No': 0})

    return df

def prepare_encoded_data(df):
    """
    Encodes features into dummy variables and splits into train/test sets.
    """
    # Drop customerID (unique identifier) and raw target string
    df_model = df.drop(columns=['customerID', 'Churn'])

    # Convert binary/categorical features to numeric dummies automatically
    df_encoded = pd.get_dummies(df_model, drop_first=True)

    X = df_encoded.drop(columns=['Churn_Numeric'])
    y = df_encoded['Churn_Numeric']

    # 80/20 Train-Test split stratified on Churn rate
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, X, y

if __name__ == "__main__":
    df = load_and_clean_data()
    X_train, X_test, y_train, y_test, X, y = prepare_encoded_data(df)
    print(f"Data Prep Complete: X_train {X_train.shape}, X_test {X_test.shape}")
