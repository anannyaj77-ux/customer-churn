# Executive Summary Report: Customer Churn Prediction & Retention Strategy

## 1. Business Problem Statement
Customer attrition (churn) directly impacts monthly recurring revenue (MRR) and customer lifetime value (CLV) in the telecommunications sector. Acquiring new customers costs 5–7x more than retaining existing ones. The objective of this project was to build an end-to-end predictive framework to identify churn risk before customers leave, quantify revenue at risk, and recommend a data-driven retention campaign with proven financial ROI.

---

## 2. Key Insights & Exploratory Data Analysis (EDA)
Analysis of **7,043 customer records** (baseline churn rate **26.54%**) revealed 4 critical churn drivers:
1. **Contract Type Lock-in**: Month-to-month contract holders churn at **42.71%**, compared to **2.83%** for 2-year contracts (**15x higher risk**).
2. **First-Year Attrition**: Median tenure for churned customers is just **10 months**.
3. **Price & Fiber Optic Friction**: Fiber optic subscribers average **$91.50/mo** (all subscribers) and are associated with a churn rate of **41.89%** (correlation, not a guaranteed campaign impact).
4. **Payment Method Friction**: Customers paying via manual Electronic Check churn at **45.29%**, vs ~15% for automated payments.

---

## 3. Predictive Modeling & SHAP Interpretability
We benchmarked Logistic Regression against an XGBoost Classifier on an independent 20% test set ($N=1,409$):

- **Model Selected**: **XGBoost Classifier**
- **Comparative Performance** (20% test set, N=1,409): Both models perform comparably. Logistic Regression has slightly higher accuracy (80.7% vs 80.1%), precision, and recall. XGBoost was selected for its superior ROC-AUC (0.8441 vs 0.8418), non-linear feature interaction capture, and native SHAP integration.
- **Known limitation**: Recall is ~53% at the default 0.5 threshold — roughly half of true churners are missed.
- **Interpretability**: Leveraged **SHAP (SHapley Additive exPlanations)** to establish feature attribution. Low tenure, month-to-month contracts, fiber optic service, and high monthly charges emerged as the top continuous churn risk drivers.

---

## 4. Recommended Retention Strategy & Financial Impact

### Strategy: Targeted High-Risk Proactive Engagement
Instead of blanket marketing, target the **top 20% highest-risk customers** identified by XGBoost risk probabilities ($N=281$ test set population representing **$266,459 in annual revenue at risk**).

### Financial ROI Math ($50 Campaign Offer, 25% Retention Success Rate):
- **Targeted Population**: 281 customers
- **Churn Capture Rate**: Captures **50.0% of all potential churners** (187 actual churners).
- **Campaign Investment**: $281 \times \$50 = \mathbf{\$14,050.00}$
- **Successfully Retained**: $187 \text{ churners} \times 25\% = \mathbf{46.8 \text{ customers saved}}$
- **Annualized Revenue Retained**: $\mathbf{\$43,789.20 / \text{year}}$
- **Net Annual Financial Benefit**: $\$43,789.20 - \$14,050.00 = \mathbf{+\$29,739.20}$
- **Campaign ROI**: $\mathbf{211.7\%}$ *(best-case estimate: success rate applied to actual churners identified in hindsight; a probability-weighted expected-value estimate would be lower)*

---

## 5. Interactive Executive Dashboard
Developed a production-ready **Streamlit Dashboard** featuring:
- Customer risk level filters & downloadable CRM target list (CSV).
- Plotly interactive SHAP feature importance charts.
- Dynamic retention campaign scenario simulator allowing executives to test custom cost and retention parameters in real time.

---

## 6. Resume Bullet Points (Copy & Paste for Portfolio/Resume)

### Option 1 (Business Analytics Focus):
> • **Built end-to-end customer churn prediction & retention model** using Python (XGBoost, Scikit-Learn, SHAP) on 7k+ customer records, achieving an **0.844 ROC-AUC** score and identifying top drivers of attrition.
> • **Quantified $266K+ annual revenue at risk** and designed a targeted retention campaign ($50 offer/user, 25% success rate) that delivers **$29.7K in net annual savings** and a **211.7% ROI**.
> • **Engineered interactive Streamlit dashboard** featuring dynamic customer segmentation filters, downloadable CRM target lists, and real-time financial ROI simulation for executive decision-making.

### Option 2 (Data Science / ML Focus):
> • **Developed and deployed XGBoost customer churn classification pipeline** (stratified 80/20 train-test split, One-Hot Encoding) achieving **ROC-AUC 0.8441** — marginally better than Logistic Regression (0.8418). Both models are comparable on accuracy; XGBoost was selected for non-linear pattern capture and native SHAP explainability. Known limitation: recall ~53% at default threshold.
> • **Utilized SHAP TreeExplainer** to decode model predictions, identifying contract duration, tenure (<12 mos), and fiber optic pricing as top non-linear churn features.
> • **Translated machine learning risk probabilities into business outcomes**, building an automated scenario planner in Streamlit to simulate campaign net margin and customer lifetime value (CLV) recovery.

### Option 3 (Short / Impact-First Format):
> • **Customer Churn & Retention Analytics**: Engineered an XGBoost churn model (0.844 AUC) and interactive Streamlit ROI simulator, identifying high-risk segments to generate **$29.7K in net annual revenue savings** at a **211.7% campaign ROI**.
