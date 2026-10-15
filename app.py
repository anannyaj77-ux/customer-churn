import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from data_prep import load_and_clean_data, prepare_encoded_data

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Customer Churn Intelligence & Retention Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- THEME MANAGEMENT VIA SESSION STATE (DEFAULT: DARK MODE) ---
if 'theme_mode' not in st.session_state:
    st.session_state.theme_mode = "Dark Mode"

st.sidebar.header("Appearance")
theme_selection = st.sidebar.radio(
    "Theme Mode",
    options=["Dark Mode", "Light Mode"],
    key="theme_mode"
)

is_dark = (st.session_state.theme_mode == "Dark Mode")

# --- DYNAMIC COLOR PALETTE TOKENS ---
if is_dark:
    bg_main = '#08071A'
    bg_card = '#15112B'
    bg_sidebar = '#1D1738'
    text_main = '#F5F3FF'
    text_sub = '#B8B2CC'
    border_clr = '#3A2C58'
    title_clr = '#C084FC'
    accent_clr = '#7C3AED'
    purple_clr = '#A855F7'
    gold_clr = '#F5B942'
else:
    bg_main = '#F8F7FC'
    bg_card = '#FFFFFF'
    bg_sidebar = '#F1EFF8'
    text_main = '#17152B'
    text_sub = '#5F5A73'
    border_clr = '#DDD8EA'
    title_clr = '#7C3AED'
    accent_clr = '#7C3AED'
    purple_clr = '#A855F7'
    gold_clr = '#F5B942'

color_yes = '#FF4FA3'  # Churn Yes -> Hot Pink / Magenta
color_no = '#16C7D9'   # Churn No -> Bright Cyan / Turquoise

# --- THEME STYLING INJECTION ---
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    /* 1. Global Reset & Base Canvas */
    html, body, [class*="css"], .stApp, [data-theme="light"], [data-theme="dark"] {{
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: {bg_main} !important;
        color: {text_main} !important;
    }}

    /* 2. Top Header & Toolbar */
    header[data-testid="stHeader"], [data-testid="stHeader"] {{
        background-color: {bg_main} !important;
        color: {text_main} !important;
    }}

    /* 3. Sidebar Container & Components */
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div {{
        background-color: {bg_sidebar} !important;
        border-right: 1px solid {border_clr} !important;
    }}

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {{
        color: {title_clr} !important;
        font-weight: 600;
    }}

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {{
        color: {text_main} !important;
        font-weight: 700;
    }}

    /* Radio buttons styling */
    div[role="radiogroup"] label span {{
        color: {text_main} !important;
        font-weight: 600;
    }}

    /* 4. Form Selectboxes, Inputs & Controls */
    div[data-baseweb="select"],
    div[data-baseweb="select"] > div,
    div[data-baseweb="select"] *,
    div[data-baseweb="input"],
    div[data-baseweb="input"] > div,
    div[data-baseweb="input"] *,
    input, textarea {{
        background-color: {bg_card} !important;
        color: {text_main} !important;
    }}

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {{
        border: 1px solid {border_clr} !important;
        border-radius: 10px !important;
    }}

    div[data-baseweb="select"] > div:hover,
    div[data-baseweb="input"] > div:hover {{
        border-color: {accent_clr} !important;
        box-shadow: 0 0 10px {'rgba(124, 108, 255, 0.25)' if is_dark else 'rgba(124, 58, 237, 0.15)'} !important;
    }}

    div[data-baseweb="select"] span,
    div[data-baseweb="select"] svg {{
        color: {text_main} !important;
        fill: {text_main} !important;
    }}

    ::placeholder {{
        color: {text_sub} !important;
        opacity: 1;
    }}

    /* 5. Dropdown Options Popup Menu */
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] *,
    div[data-baseweb="menu"],
    div[data-baseweb="menu"] *,
    ul[role="listbox"],
    li[role="option"] {{
        background-color: {bg_card} !important;
        color: {text_main} !important;
        border-color: {border_clr} !important;
    }}

    li[role="option"]:hover,
    li[aria-selected="true"] {{
        background-color: {bg_sidebar} !important;
        color: {title_clr} !important;
    }}

    /* 6. Sliders Styling */
    .stSlider label, .stSlider span {{
        color: {title_clr} !important;
        font-weight: 600;
    }}

    div[data-baseweb="slider"] div {{
        color: {text_main} !important;
    }}

    /* 7. Main Headers & Typography */
    .main-header {{
        font-size: 2.3rem;
        font-weight: 800;
        color: {text_main};
        margin-bottom: 0.3rem;
        background: linear-gradient(135deg, {title_clr} 0%, {color_yes} 50%, {color_no} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
    }}

    .sub-header {{
        font-size: 1.05rem;
        color: {text_sub};
        margin-bottom: 1.5rem;
        line-height: 1.6;
    }}

    h1, h2, h3 {{
        color: {text_main} !important;
        font-weight: 700;
    }}

    p, span, label, li {{
        color: {text_main} !important;
    }}

    .stMarkdown p, .stMarkdown li {{
        color: {text_main} !important;
        line-height: 1.6;
    }}

    .stMarkdown b, .stMarkdown strong {{
        color: {text_main} !important;
        font-weight: 700 !important;
    }}

    .stMarkdown h2 {{
        color: {title_clr} !important;
    }}

    .stMarkdown h3 {{
        color: {purple_clr} !important;
    }}

    /* 8. Styled Metric Cards */
    .metric-card {{
        background-color: {bg_card};
        border: 1px solid {border_clr};
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        box-shadow: {'0 4px 20px rgba(8, 7, 26, 0.4), 0 0 12px rgba(168, 85, 247, 0.08)' if is_dark else '0 4px 15px rgba(23, 21, 43, 0.04)'};
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
    }}
    .metric-card:hover {{
        border-color: {accent_clr};
        transform: translateY(-3px);
        box-shadow: {'0 8px 25px rgba(124, 108, 255, 0.25)' if is_dark else '0 8px 20px rgba(124, 58, 237, 0.12)'};
    }}

    .metric-value {{
        font-size: 1.9rem;
        font-weight: 800;
        color: {text_main};
        margin-top: 6px;
    }}

    .metric-label {{
        font-size: 0.85rem;
        color: {text_sub} !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }}

    /* 9. Callout Boxes */
    .info-box {{
        background-color: {bg_card};
        border-left: 4px solid {accent_clr};
        border-top: 1px solid {border_clr};
        border-right: 1px solid {border_clr};
        border-bottom: 1px solid {border_clr};
        padding: 18px 22px;
        border-radius: 10px;
        margin-bottom: 20px;
        color: {text_main};
        box-shadow: {'0 4px 15px rgba(124, 108, 255, 0.1)' if is_dark else '0 2px 10px rgba(23, 21, 43, 0.03)'};
    }}

    .warning-box {{
        background-color: {bg_card};
        border-left: 4px solid {gold_clr};
        border-top: 1px solid {border_clr};
        border-right: 1px solid {border_clr};
        border-bottom: 1px solid {border_clr};
        padding: 18px 22px;
        border-radius: 10px;
        margin-bottom: 20px;
        color: {text_main};
        box-shadow: {'0 4px 15px rgba(245, 185, 66, 0.12)' if is_dark else '0 2px 10px rgba(23, 21, 43, 0.03)'};
    }}

    .success-box {{
        background-color: {bg_card};
        border-left: 4px solid {color_no};
        border-top: 1px solid {border_clr};
        border-right: 1px solid {border_clr};
        border-bottom: 1px solid {border_clr};
        padding: 18px 22px;
        border-radius: 10px;
        margin-bottom: 20px;
        color: {text_main};
        box-shadow: {'0 4px 15px rgba(22, 199, 217, 0.12)' if is_dark else '0 2px 10px rgba(23, 21, 43, 0.03)'};
    }}

    /* 10. Strategy Cards */
    .strategy-card {{
        background-color: {bg_card};
        border: 1px solid {border_clr};
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: {'0 4px 15px rgba(8, 7, 26, 0.3)' if is_dark else '0 2px 10px rgba(23, 21, 43, 0.03)'};
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
    }}
    .strategy-card:hover {{
        border-color: {accent_clr};
        transform: translateY(-2px);
        box-shadow: {'0 6px 20px rgba(168, 85, 247, 0.2)' if is_dark else '0 6px 15px rgba(124, 58, 237, 0.1)'};
    }}
    .strategy-title {{
        font-weight: 700;
        color: {title_clr};
        font-size: 1.15rem;
        margin-bottom: 8px;
    }}
    .strategy-card p {{
        color: {text_sub} !important;
        margin-bottom: 6px;
    }}
    .strategy-card b {{
        color: {text_main} !important;
    }}

    /* 11. Navigation Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: {bg_sidebar} !important;
        padding: 8px;
        border-radius: 14px;
        border: 1px solid {border_clr};
    }}

    .stTabs [data-baseweb="tab"] {{
        height: 44px;
        white-space: pre;
        border-radius: 10px;
        color: {text_sub} !important;
        font-weight: 600;
        border: none;
        padding: 0px 18px;
        transition: all 0.2s ease;
        background-color: transparent !important;
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        color: {text_main} !important;
        background-color: {'rgba(168, 85, 247, 0.15)' if is_dark else 'rgba(124, 58, 237, 0.08)'} !important;
    }}

    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, #7C3AED 0%, #A855F7 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.3);
    }}

    /* 12. Buttons Styling */
    .stButton > button, .stDownloadButton > button {{
        background: linear-gradient(135deg, #7C3AED 0%, #6D28D9 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid #7C3AED !important;
        border-radius: 10px !important;
        padding: 10px 24px !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.25) !important;
        transition: all 0.3s ease !important;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        background: linear-gradient(135deg, #6D28D9 0%, #5B21B6 100%) !important;
        border-color: #6D28D9 !important;
        box-shadow: 0 6px 20px rgba(109, 40, 217, 0.35) !important;
        transform: translateY(-2px) !important;
        color: #FFFFFF !important;
    }}

    /* 13. Native Metric Cards Override */
    [data-testid="stMetric"] {{
        background-color: {bg_card} !important;
        border: 1px solid {border_clr} !important;
        border-radius: 12px !important;
        padding: 16px !important;
        box-shadow: {'0 4px 15px rgba(8, 7, 26, 0.3)' if is_dark else '0 2px 10px rgba(23, 21, 43, 0.03)'} !important;
    }}
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {{
        color: {text_sub} !important;
        font-weight: 600 !important;
    }}
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] * {{
        color: {text_main} !important;
        font-weight: 800 !important;
    }}

    /* 14. DataFrame & Data Tables Full Dark/Light Styling */
    [data-testid="stDataFrame"],
    [data-testid="stTable"],
    [data-testid="stDataFrame"] > div,
    [data-testid="stDataFrame"] iframe,
    .dsv-grid,
    .glideDataEditor,
    .gdg-header-clip,
    .gdg-canvas {{
        border: 1px solid {border_clr} !important;
        border-radius: 12px !important;
        background-color: {bg_card} !important;
        color: {text_main} !important;
    }}

    /* 15. Expanders & Containers */
    [data-testid="stExpander"] {{
        background-color: {bg_card} !important;
        border: 1px solid {border_clr} !important;
        border-radius: 12px !important;
        color: {text_main} !important;
    }}

    /* 16. Custom Scrollbars */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
        background-color: {bg_card} !important;
    }}
    ::-webkit-scrollbar-track {{
        background: {bg_card} !important;
    }}
    ::-webkit-scrollbar-thumb {{
        background: {border_clr} !important;
        border-radius: 4px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: {accent_clr} !important;
    }}

    /* 17. Captions & Dividers */
    .stCaption, figcaption {{
        color: {text_sub} !important;
    }}
    hr {{
        border-color: {border_clr} !important;
        margin: 2rem 0 !important;
    }}
    </style>
""", unsafe_allow_html=True)

# --- DATA & MODEL LOADING (CACHED) ---
@st.cache_data
def load_all_assets():
    df = load_and_clean_data()
    X_train, X_test, y_train, y_test, X, y = prepare_encoded_data(df)
    
    # Load models & scaler
    xgb_model = joblib.load('xgb_model.joblib')
    lr_model = joblib.load('lr_model.joblib')
    scaler = joblib.load('scaler.joblib')
    
    # Generate predictions on full dataset
    df_pred = df.copy()
    df_pred['Churn_Probability'] = xgb_model.predict_proba(X)[:, 1]
    
    def assign_risk(prob):
        if prob >= 0.60:
            return 'High Risk (>=60%)'
        elif prob >= 0.30:
            return 'Medium Risk (30-59%)'
        else:
            return 'Low Risk (<30%)'
            
    df_pred['Risk_Category'] = df_pred['Churn_Probability'].apply(assign_risk)
    
    # Evaluate models on Test Set for exact metrics
    X_test_scaled = scaler.transform(X_test)
    
    y_pred_lr = lr_model.predict(X_test_scaled)
    y_prob_lr = lr_model.predict_proba(X_test_scaled)[:, 1]
    
    y_pred_xgb = xgb_model.predict(X_test)
    y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
    
    metrics_summary = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
        'Logistic Regression': [
            accuracy_score(y_test, y_pred_lr),
            precision_score(y_test, y_pred_lr),
            recall_score(y_test, y_pred_lr),
            f1_score(y_test, y_pred_lr),
            roc_auc_score(y_test, y_prob_lr)
        ],
        'XGBoost Classifier': [
            accuracy_score(y_test, y_pred_xgb),
            precision_score(y_test, y_pred_xgb),
            recall_score(y_test, y_pred_xgb),
            f1_score(y_test, y_pred_xgb),
            roc_auc_score(y_test, y_prob_xgb)
        ]
    })
    
    return df_pred, xgb_model, lr_model, scaler, metrics_summary, X.columns, X_test, y_test

df_data, xgb_model, lr_model, scaler, metrics_df, feature_names, X_test, y_test = load_all_assets()

# --- HEADER SECTION ---
st.markdown('<div class="main-header">🎯 Customer Churn Intelligence & Retention Strategy</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Production analytics decision-support platform for churn risk prediction, model benchmarking, and ROI-driven retention campaigns.</div>', unsafe_allow_html=True)

# --- SIDEBAR FILTERS ---
st.sidebar.header("🔍 Customer Filters & Segmentation")

risk_options = ['All Risk Levels', 'High Risk (>=60%)', 'Medium Risk (30-59%)', 'Low Risk (<30%)']
selected_risk = st.sidebar.selectbox("Customer Risk Tier", risk_options)

contract_options = ['All Contracts'] + list(df_data['Contract'].unique())
selected_contract = st.sidebar.selectbox("Contract Type", contract_options)

internet_options = ['All Internet Services'] + list(df_data['InternetService'].unique())
selected_internet = st.sidebar.selectbox("Internet Service Type", internet_options)

payment_options = ['All Payment Methods'] + list(df_data['PaymentMethod'].unique())
selected_payment = st.sidebar.selectbox("Payment Method", payment_options)

# Filter Data Logic
filtered_df = df_data.copy()

if selected_risk != 'All Risk Levels':
    filtered_df = filtered_df[filtered_df['Risk_Category'] == selected_risk]
if selected_contract != 'All Contracts':
    filtered_df = filtered_df[filtered_df['Contract'] == selected_contract]
if selected_internet != 'All Internet Services':
    filtered_df = filtered_df[filtered_df['InternetService'] == selected_internet]
if selected_payment != 'All Payment Methods':
    filtered_df = filtered_df[filtered_df['PaymentMethod'] == selected_payment]

# --- TOP SUMMARY KPI CARDS ---
total_cust = len(filtered_df)
avg_churn_prob = filtered_df['Churn_Probability'].mean() * 100 if total_cust > 0 else 0
monthly_rev_at_risk = filtered_df['MonthlyCharges'].sum() if total_cust > 0 else 0
annual_rev_at_risk = monthly_rev_at_risk * 12

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Filtered Customers</div>
        <div class="metric-value" style="color: {color_no};">{total_cust:,}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Avg Churn Risk</div>
        <div class="metric-value" style="color: {color_yes if avg_churn_prob > 40 else gold_clr if avg_churn_prob > 25 else color_no};">{avg_churn_prob:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Monthly Revenue at Risk</div>
        <div class="metric-value" style="color: {text_main};">$ {monthly_rev_at_risk:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Annual Revenue at Risk</div>
        <div class="metric-value" style="color: {color_yes};">$ {annual_rev_at_risk:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- NAVIGATION TABS ---
tab_overview, tab_explorer, tab_models, tab_explain, tab_roi = st.tabs([
    "📌 Project Overview", 
    "📋 Customer Risk Explorer", 
    "⚖️ Model Performance", 
    "📈 ML Explainability & Drivers", 
    "💰 Retention ROI & Strategy"
])

# ==========================================
# TAB 1: PROJECT OVERVIEW
# ==========================================
with tab_overview:
    st.header("📌 Project Executive Overview")
    
    st.markdown("""
    <div class="info-box">
    <b>Executive Summary:</b> This analytics solution bridges machine learning risk prediction with financial strategy. 
    By ranking telecom customers by churn probability, identifying underlying friction points, and simulating targeted retention campaigns, 
    the system converts raw data into actionable revenue protection.
    </div>
    """, unsafe_allow_html=True)
    
    col_ov1, col_ov2 = st.columns(2)
    
    with col_ov1:
        st.markdown("### 🎯 Problem Statement & Context")
        st.markdown("""
        - **Business Challenge**: Customer churn directly erodes Monthly Recurring Revenue (MRR) and Customer Lifetime Value (CLV).
        - **Acquisition vs. Retention Cost**: Acquiring a new telecom customer costs **5–7x more** than retaining an existing subscriber.
        - **Objective**: Build a predictive machine learning engine to accurately rank customers by churn risk, quantify annual revenue at risk, and calculate net campaign ROI from targeted retention offers.
        """)
        
        st.markdown("### 📊 Dataset Overview")
        st.markdown("""
        - **Source**: Telco Customer Churn Dataset (IBM / Kaggle).
        - **Records & Features**: **7,043 customer accounts** evaluated across **19 features (plus customer ID and churn label)** (contract types, services, payment methods, billing charges, and tenure).
        - **Baseline Churn Rate**: **26.54%** (1,869 churned customers vs. 5,174 retained customers).
        """)
        
    with col_ov2:
        st.markdown("### 🔍 Key Exploratory Data Insights")
        st.markdown("""
        1. **Contract Type Lock-in**: Month-to-month subscribers are associated with a churn rate of **42.71%**, compared to just **2.83%** for 2-year contracts (**15x risk multiplier**).
        2. **First-Year Attrition**: Churned customers leave early in their tenure (**median 10 months**).
        3. **Fiber Optic & Bill Size Friction**: Fiber optic subscribers average **$91.50/mo** (all subscribers) and are associated with a **41.89% churn rate** — a correlation, not a guaranteed campaign impact.
        4. **Payment Method Friction**: Manual Electronic Check users are associated with churn of **45.29%**, compared to ~15% for automated payments.
        """)
        
        st.markdown("### 🛠️ Technology Stack")
        st.markdown("""
        - **Core Stack**: Python (Pandas, NumPy, Scikit-Learn, XGBoost, SHAP).
        - **Visualization & UI**: Plotly Express, Seaborn, Matplotlib, Streamlit.
        - **Deployment & Serialization**: Joblib, Streamlit Web Server.
        """)

    st.markdown("---")
    st.markdown("### 📊 Initial Exploratory Data Visualizations")
    eda_img = "eda_churn_drivers.png" if is_dark else "eda_churn_drivers_light.png"
    if not os.path.exists(eda_img):
        eda_img = "eda_churn_drivers.png"
    if os.path.exists(eda_img):
        st.image(eda_img, caption="Exploratory Data Analysis - Key Churn Segment Breakdown", width="stretch")

# ==========================================
# TAB 2: CUSTOMER RISK EXPLORER
# ==========================================
with tab_explorer:
    st.header("📋 Customer-Level Risk Explorer & Export")
    st.write(f"Showing **{total_cust:,}** active customer records based on currently applied sidebar filters.")
    st.caption(
        "⚠️ **Data scope note**: Risk scores shown here cover all 7,043 customers, "
        "including ~5,634 used during model training. Scores for training-set customers "
        "are **in-sample** and will appear better-calibrated than true out-of-sample performance. "
        "All reported model metrics (ROC-AUC 0.844, Accuracy 80.1%) come from the "
        "held-out **20% test set** (N=1,409) only."
    )
    
    display_cols = [
        'customerID', 'Risk_Category', 'Churn_Probability', 
        'MonthlyCharges', 'TotalCharges', 'tenure', 
        'Contract', 'InternetService', 'PaymentMethod', 'Churn'
    ]
    
    display_df = filtered_df[display_cols].copy()
    display_df['Churn_Probability'] = (display_df['Churn_Probability'] * 100).map('{:.1f}%'.format)
    display_df['MonthlyCharges'] = display_df['MonthlyCharges'].map('${:.2f}'.format)
    display_df['TotalCharges'] = display_df['TotalCharges'].map('${:.2f}'.format)
    
    st.dataframe(display_df, use_container_width=True, height=380)
    
    # Download Button
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered High-Risk Target List (CSV)",
        data=csv,
        file_name="filtered_high_risk_customers.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.subheader("🔎 Individual Customer Deep-Dive Search")
    selected_id = st.selectbox("Select Customer ID to Inspect", options=df_data['customerID'].tolist())
    
    cust_row = df_data[df_data['customerID'] == selected_id].iloc[0]
    
    c_info1, c_info2, c_info3, c_info4 = st.columns(4)
    c_info1.metric("Selected Customer ID", cust_row['customerID'])
    c_info2.metric("Predicted Churn Risk", f"{cust_row['Churn_Probability']*100:.1f}%", delta=cust_row['Risk_Category'])
    c_info3.metric("Monthly Billing", f"${cust_row['MonthlyCharges']:.2f}")
    c_info4.metric("Tenure", f"{cust_row['tenure']} months")

# ==========================================
# TAB 3: MODEL PERFORMANCE
# ==========================================
with tab_models:
    st.header("⚖️ Machine Learning Model Benchmarking")
    st.markdown("Evaluated on an independent **20% Test Set** ($N=1,409$ unseen customer accounts).")
    
    # Performance KPI Cards
    col_m1, col_m2 = st.columns(2)
    
    lr_row = metrics_df.set_index('Metric')['Logistic Regression']
    xgb_row = metrics_df.set_index('Metric')['XGBoost Classifier']
    
    with col_m1:
        st.markdown("### 🔹 Logistic Regression (Linear Benchmark)")
        st.markdown(f"""
        <div class="metric-card">
            <p>Accuracy: <b>{lr_row['Accuracy']*100:.2f}%</b> | Precision: <b>{lr_row['Precision']*100:.2f}%</b></p>
            <p>Recall: <b>{lr_row['Recall']*100:.2f}%</b> | F1 Score: <b>{lr_row['F1 Score']:.4f}</b></p>
            <h4 style="color: {title_clr}; margin-top: 10px;">ROC-AUC: {lr_row['ROC AUC']:.4f}</h4>
        </div>
        """, unsafe_allow_html=True)
        
    with col_m2:
        st.markdown("### 🏆 XGBoost Classifier (Selected Model)")
        st.markdown(f"""
        <div class="metric-card" style="border-color: {color_no}; box-shadow: {'0 0 18px rgba(22, 199, 217, 0.2)' if is_dark else '0 0 12px rgba(22, 199, 217, 0.15)'};">
            <p>Accuracy: <b>{xgb_row['Accuracy']*100:.2f}%</b> | Precision: <b>{xgb_row['Precision']*100:.2f}%</b></p>
            <p>Recall: <b>{xgb_row['Recall']*100:.2f}%</b> | F1 Score: <b>{xgb_row['F1 Score']:.4f}</b></p>
            <h4 style="color: {color_no}; margin-top: 10px;">ROC-AUC: {xgb_row['ROC AUC']:.4f}</h4>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Plotly Metrics Comparison Bar Chart
    metrics_melted = pd.melt(metrics_df, id_vars=['Metric'], var_name='Model', value_name='Score')
    fig_metrics = px.bar(
        metrics_melted, x='Metric', y='Score', color='Model', barmode='group',
        title="Model Performance Metric Comparison (Test Set)",
        color_discrete_map={'Logistic Regression': title_clr, 'XGBoost Classifier': color_no},
        text_auto='.3f'
    )
    fig_metrics.update_layout(
        title=dict(text="Model Performance Metric Comparison (Test Set)", font=dict(color=title_clr, size=18, family='Plus Jakarta Sans')),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=text_main, family='Plus Jakarta Sans'),
        xaxis=dict(gridcolor=border_clr, zerolinecolor=border_clr, title_font=dict(color=text_sub)),
        yaxis=dict(range=[0.4, 1.0], gridcolor=border_clr, zerolinecolor=border_clr, title_font=dict(color=text_sub)),
        legend=dict(font=dict(color=text_main)),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig_metrics, width="stretch")

    st.markdown(f"""
    <div class="success-box">
    <b>Model Selection Rationale:</b><br>
    The two models perform <b>comparably</b> overall. Logistic Regression has slightly higher accuracy (80.7% vs 80.1%), precision, and recall. XGBoost was selected for the following reasons:<br><br>
    1. <b>Better Probability Ranking (ROC-AUC 0.8441 vs 0.8418)</b>: XGBoost produces a marginally better risk-ordered list, ensuring higher-probability churners surface at the top of marketing target lists.<br>
    2. <b>Non-Linear Feature Interaction Capture</b>: Captures complex interactions between high monthly charges, short tenure, fiber optic plans, and contract length that linear models approximate less well.<br>
    3. <b>Native SHAP Integration</b>: Integrates directly with SHAP TreeExplainer for granular, explainable AI attribution without approximation.<br><br>
    <b>Known limitation</b>: Recall is ~53% at the default 0.5 threshold — roughly half of true churners are not flagged. Threshold tuning can improve recall at the cost of more false positives.
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# TAB 4: ML EXPLAINABILITY & DRIVERS
# ==========================================
with tab_explain:
    st.header("📈 Machine Learning Explainability & Churn Drivers")
    
    exp_col1, exp_col2 = st.columns(2)
    
    with exp_col1:
        st.markdown("### Top Feature Importances (XGBoost)")
        importances = xgb_model.feature_importances_
        fi_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=True).tail(10)
        
        color_scale = [[0.0, border_clr], [0.4, purple_clr], [0.7, title_clr], [1.0, color_no]]
        fig_fi = px.bar(
            fi_df, x='Importance', y='Feature', orientation='h',
            title="Top 10 Feature Weights in XGBoost Model",
            color='Importance',
            color_continuous_scale=color_scale
        )
        fig_fi.update_layout(
            showlegend=False,
            title=dict(text="Top 10 Feature Weights in XGBoost Model", font=dict(color=purple_clr, size=18, family='Plus Jakarta Sans')),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=text_main, family='Plus Jakarta Sans'),
            xaxis=dict(gridcolor=border_clr, zerolinecolor=border_clr, title_font=dict(color=text_sub)),
            yaxis=dict(gridcolor=border_clr, zerolinecolor=border_clr, title_font=dict(color=text_sub)),
            coloraxis_showscale=False,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_fi, width="stretch")
        
    with exp_col2:
        st.markdown("### SHAP Feature Impact Summary")
        if os.path.exists("shap_summary.png"):
            st.image("shap_summary.png", caption="SHAP TreeExplainer Global Feature Impact Plot", width="stretch")

    st.markdown("---")
    st.markdown("### 💡 Business Interpretation of Top Churn Drivers")
    
    st.markdown("""
    - **Customer Tenure (`tenure`)**: Customers in their first 12 months are at the highest risk. As tenure increases past 2 years, brand loyalty and switching costs are associated with dramatically lower churn probability.
    - **Contract Length (`Contract_Two year`, `Contract_One year`)**: Month-to-month contracts are associated with the highest churn rate (**42.7%**). Upgrading to 1-year or 2-year contracts is correlated with churn rates of **2.8%–11.2%** (correlation, not a guaranteed campaign impact).
    - **Internet Service Type (`InternetService_Fiber optic`)**: Fiber Optic subscribers average **$91.50/mo** and are associated with a higher churn rate (41.9%) — likely reflecting high-bill dissatisfaction and service friction.
    - **Payment Method (`PaymentMethod_Electronic check`)**: Electronic Check payments require manual monthly action, repeatedly reminding customers of their bill and creating monthly cancellation touchpoints.
    - **Add-on Services (`OnlineSecurity_Yes`, `TechSupport_Yes`)**: Customers enrolled in active tech support or security services are associated with significantly lower churn, suggesting that service integration increases product stickiness.
    """)

# ==========================================
# TAB 5: RETENTION ROI & STRATEGY
# ==========================================
with tab_roi:
    st.header("💰 Retention Campaign ROI Simulator & Business Strategy")
    
    st.markdown("""
    <div class="warning-box">
    <b>📌 SCENARIO ASSUMPTIONS NOTE:</b> The calculations below represent <b>interactive financial simulations</b> based on user-configurable campaign costs ($/customer) and retention conversion rates (%). They demonstrate financial impact under customizable parameters, not actual historical company financials.
    </div>
    """, unsafe_allow_html=True)
    
    # Interactive Simulator Sliders
    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        campaign_cost = st.slider("Retention Offer Cost per Targeted Customer ($)", min_value=10, max_value=200, value=50, step=5)
    with sim_col2:
        success_rate = st.slider("Targeted Retention Success Rate (%)", min_value=5, max_value=50, value=25, step=5) / 100.0

    # Calculate ROI for current filtered group using Expected-Value method:
    # expected_revenue_saved = sum over each targeted customer of
    #   (success_rate x churn_probability x annual_revenue)
    # This avoids the hindsight bias of applying the success rate only to known churners.
    high_risk_group = filtered_df[filtered_df['Churn_Probability'] >= 0.50]
    num_targeted = len(high_risk_group)
    
    if num_targeted > 0:
        total_cost = num_targeted * campaign_cost
        # Expected-value method: weight revenue recovery by predicted churn probability
        expected_revenue_per_customer = (
            high_risk_group['Churn_Probability'] * success_rate * high_risk_group['MonthlyCharges'] * 12
        )
        annual_saved_revenue = expected_revenue_per_customer.sum()
        net_financial_benefit = annual_saved_revenue - total_cost
        roi = (net_financial_benefit / total_cost) * 100 if total_cost > 0 else 0
        
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Targeted High-Risk Cohort", f"{num_targeted:,} users")
        r2.metric("Total Campaign Cost", f"${total_cost:,.2f}")
        r3.metric("Expected Annual Revenue Saved", f"${annual_saved_revenue:,.2f}")
        r4.metric("Net Financial Benefit", f"${net_financial_benefit:,.2f}", delta=f"{roi:.1f}% ROI")

        st.caption(
            "💡 **Method**: Revenue saved is calculated as an expected value — "
            "for each targeted customer: `success_rate × churn_probability × annual_revenue`. "
            "This is a probability-weighted estimate and will be more conservative than "
            "applying the success rate only to customers who actually churned (hindsight best-case)."
        )
        
        # Plotly Gauge Chart
        step1_bg = '#3A2C58' if is_dark else '#DDD8EA'
        step2_bg = '#1D1738' if is_dark else '#F1EFF8'
        step3_bg = '#15112B' if is_dark else '#FFFFFF'

        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = roi,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Simulated Campaign ROI (%)", 'font': {'color': color_no, 'size': 18, 'family': 'Plus Jakarta Sans'}},
            number = {'font': {'color': text_main, 'family': 'Plus Jakarta Sans'}, 'suffix': "%"},
            gauge = {
                'axis': {'range': [-100, 400], 'tickcolor': text_sub, 'tickfont': {'color': text_sub}},
                'bar': {'color': color_no if roi > 0 else color_yes},
                'steps': [
                    {'range': [-100, 0], 'color': step1_bg},
                    {'range': [0, 100], 'color': step2_bg},
                    {'range': [100, 400], 'color': step3_bg}
                ],
                'threshold': {
                    'line': {'color': gold_clr, 'width': 3},
                    'thickness': 0.75,
                    'value': roi
                }
            }
        ))
        fig_gauge.update_layout(
            height=280,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=text_main, family='Plus Jakarta Sans')
        )
        st.plotly_chart(fig_gauge, width="stretch")
    else:
        st.warning("No high-risk customers (>=50% churn probability) match the active sidebar filters. Adjust sidebar filters to view campaign simulations.")

    st.markdown("---")
    st.subheader("🎯 Segment-Specific Retention Action Plan")
    
    st_c1, st_c2 = st.columns(2)
    
    with st_c1:
        st.markdown("""
        <div class="strategy-card">
            <div class="strategy-title">1. Contract Extension Incentives</div>
            <p><b>Target Segment</b>: Month-to-Month customers with >60% churn probability.</p>
            <p><b>Action</b>: Offer a $10/month bill credit in exchange for upgrading to a 12-month or 24-month contract commitment.</p>
            <p><b>Impact</b>: Correlates with churn rates of <b>2.8%–11.2%</b> vs. 42.7% for month-to-month (correlation, not a guaranteed campaign impact).</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="strategy-card">
            <div class="strategy-title">2. Auto-Pay Conversion Campaign</div>
            <p><b>Target Segment</b>: Customers paying via Electronic Check.</p>
            <p><b>Action</b>: Offer a one-time $15 account credit upon setting up automatic Bank Transfer or Credit Card payments.</p>
            <p><b>Impact</b>: Auto-pay customers are associated with a churn rate of ~15%, vs. 45.3% for Electronic Check payers (correlation, not a guaranteed campaign impact).</p>
        </div>
        """, unsafe_allow_html=True)

    with st_c2:
        st.markdown("""
        <div class="strategy-card">
            <div class="strategy-title">3. Fiber Optic Loyalty & Service Assurance</div>
            <p><b>Target Segment</b>: Fiber Optic subscribers (avg. $91.50/mo).</p>
            <p><b>Action</b>: Provide complimentary 3-month Online Security or Tech Support add-on packages paired with a proactive technical service check.</p>
            <p><b>Impact</b>: Increases perceived service value and product stickiness for high-ARPU subscribers.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="strategy-card">
            <div class="strategy-title">4. First-Year Onboarding Retention</div>
            <p><b>Target Segment</b>: New customers with tenure < 12 months.</p>
            <p><b>Action</b>: Trigger automated 30-day and 90-day onboarding satisfaction calls and feature usage tips.</p>
            <p><b>Impact</b>: Targets the high-risk first 10 months of the customer lifecycle, where churn rates are highest.</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.caption("Telco Customer Churn Analytics Platform | Built with Streamlit, Scikit-Learn & XGBoost")
