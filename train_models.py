import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import shap
from data_prep import load_and_clean_data, prepare_encoded_data

def train_and_evaluate():
    # 1. Load clean data
    df = load_and_clean_data()
    X_train, X_test, y_train, y_test, X, y = prepare_encoded_data(df)

    # 2. Scale features for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Model 1: Logistic Regression ---
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_scaled, y_train)

    y_pred_lr = lr_model.predict(X_test_scaled)
    y_prob_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

    # --- Model 2: XGBoost Classifier ---
    xgb_model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)

    y_pred_xgb = xgb_model.predict(X_test)
    y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

    # Save models and scaler
    joblib.dump(lr_model, 'lr_model.joblib')
    joblib.dump(xgb_model, 'xgb_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    print("Models and scaler saved successfully!")

    # Calculate metrics
    metrics_summary = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
        'Logistic Regression': [
            accuracy_score(y_test, y_pred_lr),
            precision_score(y_test, y_pred_lr),
            recall_score(y_test, y_pred_lr),
            f1_score(y_test, y_pred_lr),
            roc_auc_score(y_test, y_prob_lr)
        ],
        'XGBoost': [
            accuracy_score(y_test, y_pred_xgb),
            precision_score(y_test, y_pred_xgb),
            recall_score(y_test, y_pred_xgb),
            f1_score(y_test, y_pred_xgb),
            roc_auc_score(y_test, y_prob_xgb)
        ]
    })

    return lr_model, xgb_model, scaler, metrics_summary, X_test, y_test

if __name__ == "__main__":
    lr_model, xgb_model, scaler, metrics, X_test, y_test = train_and_evaluate()
    print("\n=== MODEL PERFORMANCE COMPARISON ===")
    print(metrics.to_string(index=False))
