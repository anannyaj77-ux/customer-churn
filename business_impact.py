import pandas as pd
import numpy as np
import joblib
from data_prep import load_and_clean_data, prepare_encoded_data

def calculate_business_impact(top_pct=0.20, cost_per_customer=50.0, success_rate=0.25):
    # 1. Load data & model
    df = load_and_clean_data()
    X_train, X_test, y_train, y_test, X, y = prepare_encoded_data(df)
    xgb_model = joblib.load('xgb_model.joblib')

    # 2. Predict churn probability
    test_df = df.iloc[X_test.index].copy()
    test_df['churn_prob'] = xgb_model.predict_proba(X_test)[:, 1]
    test_df['actual_churn'] = y_test.values

    # 3. Filter top high-risk cohort
    target_count = int(len(test_df) * top_pct)
    high_risk_df = test_df.sort_values(by='churn_prob', ascending=False).head(target_count)

    # 4. Revenue calculations
    avg_monthly_rev = high_risk_df['MonthlyCharges'].mean()
    avg_annual_rev = avg_monthly_rev * 12

    total_monthly_at_risk = high_risk_df['MonthlyCharges'].sum()
    total_annual_at_risk = total_monthly_at_risk * 12

    actual_churners = high_risk_df['actual_churn'].sum()
    total_test_churners = y_test.sum()
    capture_rate = (actual_churners / total_test_churners) * 100

    # 5. Campaign Financial Math
    total_campaign_cost = target_count * cost_per_customer
    actual_churners_df = high_risk_df[high_risk_df['actual_churn'] == 1]
    
    retained_count = actual_churners * success_rate
    retained_annual_revenue = actual_churners_df['MonthlyCharges'].sum() * 12 * success_rate

    # NOTE: This uses a best-case "hindsight" method:
    # The success_rate is applied only to customers who ACTUALLY churned (identified in hindsight).
    # A more conservative probability-weighted expected-value estimate would multiply
    # success_rate x churn_probability x annual_revenue for each targeted customer.
    net_benefit = retained_annual_revenue - total_campaign_cost
    roi_pct = (net_benefit / total_campaign_cost) * 100

    results = {
        'test_size': len(test_df),
        'target_pct': top_pct * 100,
        'target_count': target_count,
        'actual_churners': actual_churners,
        'total_test_churners': total_test_churners,
        'capture_rate': capture_rate,
        'avg_monthly_rev': avg_monthly_rev,
        'avg_annual_rev': avg_annual_rev,
        'total_monthly_at_risk': total_monthly_at_risk,
        'total_annual_at_risk': total_annual_at_risk,
        'cost_per_customer': cost_per_customer,
        'success_rate_pct': success_rate * 100,
        'total_campaign_cost': total_campaign_cost,
        'retained_count': retained_count,
        'retained_annual_revenue': retained_annual_revenue,
        'net_benefit': net_benefit,
        'roi_pct': roi_pct
    }
    return results

if __name__ == '__main__':
    res = calculate_business_impact()
    print("=== BUSINESS IMPACT & ROI REPORT ===")
    print(f"Test Set Population: {res['test_size']} customers")
    print(f"Targeted High-Risk Cohort (Top {res['target_pct']:.0f}%): {res['target_count']} customers")
    print(f"Captured Churners in Target: {res['actual_churners']} out of {res['total_test_churners']} ({res['capture_rate']:.1f}%)")
    print(f"Average Monthly Revenue per Targeted Customer: ${res['avg_monthly_rev']:.2f}")
    print(f"Average Annual Revenue per Targeted Customer: ${res['avg_annual_rev']:.2f}")
    print(f"Total Monthly Revenue at Risk: ${res['total_monthly_at_risk']:,.2f}")
    print(f"Total Annual Revenue at Risk: ${res['total_annual_at_risk']:,.2f}")
    print("\n--- RETENTION CAMPAIGN FINANCIAL MATH ---")
    print(f"Campaign Cost per Customer: ${res['cost_per_customer']:.2f}")
    print(f"Total Campaign Investment ({res['target_count']} x ${res['cost_per_customer']:.2f}): ${res['total_campaign_cost']:,.2f}")
    print(f"Targeted Customer Retention Success Rate: {res['success_rate_pct']:.0f}%")
    print(f"Successfully Retained Customers ({res['actual_churners']} churners x {res['success_rate_pct']:.0f}%): {res['retained_count']:.1f} customers")
    print(f"Annualized Retained Revenue Saved: ${res['retained_annual_revenue']:,.2f}")
    print(f"NET FINANCIAL BENEFIT (SAVINGS - COST): ${res['net_benefit']:,.2f}")
    print(f"CAMPAIGN ROI (BEST-CASE ESTIMATE*): {res['roi_pct']:.1f}%")
    print("* Best-case: success rate applied to actual churners identified in hindsight.")
    print("  A probability-weighted (expected-value) estimate would be more conservative.")
