# Telco Customer Churn Intelligence & Retention Strategy

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.0-F7931E.svg)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-006600.svg)](https://xgboost.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Summary

An end-to-end machine learning platform that predicts telecom customer churn, quantifies annualised revenue at risk, and simulates targeted retention campaign ROI -- all inside an interactive Streamlit dashboard. Raw IBM Telco data (7,043 customers, 19 model features) flows through a scikit-learn / XGBoost pipeline, SHAP explainability surfaces the top churn drivers, and a built-in financial simulator lets business users test custom offer costs and retention rates without touching code.

---

## Key Results

| Metric | Value |
|---|---|
| Test Set ROC-AUC (XGBoost) | **0.844** |
| Top-20% risk targeting | captures **~50% of all churners** |
| Baseline churn rate | 26.54% (1,869 / 7,043 customers) |
| Simulated net annual savings | **$29.7K** (at $50 offer, 25% retention rate) |
| Simulated campaign ROI | **211.7%** *(best-case, using actual churners -- see Limitations)* |

> **Note on metrics**: Logistic Regression achieves slightly higher accuracy (80.7% vs 80.1%) and recall than XGBoost. XGBoost was chosen for its superior ROC-AUC (0.8441 vs 0.8418), non-linear feature interaction capture, and native SHAP TreeExplainer support. Both models have a recall of ~53%, meaning roughly half of true churners are missed at the default 0.5 threshold -- a known limitation.

---

## Tech Stack

| Layer | Tools |
|---|---|
| ML & Modelling | scikit-learn 1.9.0, XGBoost >= 2.0, SHAP |
| Data Processing | Pandas >= 2.2, NumPy |
| Visualisation | Plotly, Matplotlib, Seaborn |
| Dashboard | Streamlit >= 1.40 |
| Serialisation | Joblib |

---

## Project Structure

`
customer churn/
|-- .streamlit/
|   +-- config.toml              # Streamlit theme configuration
|-- scratch/
|   |-- generate_eda_plot.py     # Script to regenerate EDA PNGs (dark)
|   |-- generate_eda_plot_light.py
|   +-- generate_shap_plot.py    # Script to regenerate SHAP summary PNG
|-- app.py                       # Main Streamlit dashboard
|-- data_prep.py                 # Data loading, cleaning & encoding
|-- train_models.py              # Model training & serialisation
|-- business_impact.py           # Headless ROI report script
|-- executive_summary_report.md  # Executive / portfolio write-up
|-- requirements.txt             # Runtime dependencies
|-- requirements-dev.txt         # Dev/training dependencies
|-- .gitignore
|-- WA_Fn-UseC_-Telco-Customer-Churn.csv
|-- xgb_model.joblib             # Trained XGBoost model
|-- lr_model.joblib              # Trained Logistic Regression model
|-- scaler.joblib                # StandardScaler (for LR)
|-- eda_churn_drivers.png        # EDA plot -- dark theme
|-- eda_churn_drivers_light.png  # EDA plot -- light theme
+-- shap_summary.png             # SHAP global feature impact
`

---

## Run Locally

### 1. Clone the repository
`ash
git clone https://github.com/<your-username>/telco-churn-dashboard.git
cd telco-churn-dashboard
`

### 2. Create & activate a virtual environment (recommended)
`ash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
`

### 3. Install runtime dependencies
`ash
pip install -r requirements.txt
`

### 4. Launch the dashboard
`ash
streamlit run app.py
`

The app opens at http://localhost:8501 by default.

---

## Retrain Models

> Models are pre-trained and committed to the repo. Only run this if you change the data or hyperparameters.

`ash
# Install dev dependencies (includes SHAP, Matplotlib, Seaborn)
pip install -r requirements-dev.txt

# Retrain XGBoost + Logistic Regression and overwrite .joblib files
python train_models.py

# Regenerate SHAP summary plot
python scratch/generate_shap_plot.py

# Regenerate EDA plots
python scratch/generate_eda_plot.py
python scratch/generate_eda_plot_light.py

# Print headless ROI report
python business_impact.py
`

---

## Live Demo

> **Coming soon** -- will be deployed on Streamlit Community Cloud (https://share.streamlit.io/).
> Placeholder link: https://share.streamlit.io/<your-username>/telco-churn-dashboard/main/app.py

---

## Screenshots

> *(Screenshots will be added after first deployment.)*

| Dashboard Overview | Model Benchmarking | ROI Simulator |
|---|---|---|
| *(coming soon)* | *(coming soon)* | *(coming soon)* |

---

## Limitations

1. **Recall ~53%**: At the default 0.5 probability threshold, approximately half of true churners are not flagged. Lowering the threshold captures more churners but increases false positives.
2. **In-sample risk scores**: The dashboard scores all 7,043 customers, including the 5,634 used for training. Risk scores for training-set customers are in-sample; reported model metrics (ROC-AUC 0.844) are from the held-out 20% test set only.
3. **Correlation, not causation**: All EDA insights (e.g., month-to-month churn rates) are observational correlations. Campaign impact projections are estimates, not guarantees.
4. **ROI best-case estimate**: The 211.7% ROI figure applies the 25% success rate to customers who *actually* churned (identified in hindsight). A probability-weighted expected-value estimate would be more conservative.
5. **Static dataset**: The model was trained on a fixed IBM/Kaggle snapshot. Real-world performance will degrade as customer behaviour evolves; periodic retraining is recommended.

---

## Future Scope

- Deploy to Streamlit Community Cloud with live demo link
- Add threshold tuning slider to balance precision vs. recall
- Integrate real-time CRM data feed via REST API
- Implement customer lifetime value (CLV) model alongside churn risk
- Add multi-model ensemble (LR + XGBoost probability averaging)
- Email / Slack alert integration for newly identified high-risk customers
- A/B test tracking module for measuring actual retention campaign outcomes

---

*Built with Python, XGBoost, SHAP & Streamlit.*
