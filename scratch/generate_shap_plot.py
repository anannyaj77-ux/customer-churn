import sys
sys.path.append('.')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import joblib
import shap
from data_prep import load_and_clean_data, prepare_encoded_data

# Set dark background theme
plt.style.use('dark_background')

# Load data & model
df = load_and_clean_data()
X_train, X_test, y_train, y_test, X, y = prepare_encoded_data(df)
xgb_model = joblib.load('xgb_model.joblib')

# Compute SHAP values
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer(X_test)

# Color theme constants
BG_COLOR = '#15112B'
TEXT_COLOR = '#F5F3FF'
TITLE_COLOR = '#C084FC'
MUTED_TEXT = '#B8B2CC'
BORDER_COLOR = '#3A2C58'

# Custom colormap: Low (#16C7D9 Cyan) to High (#FF4FA3 Hot Pink)
cmap = mcolors.LinearSegmentedColormap.from_list('custom_cyan_pink', ['#16C7D9', '#7C6CFF', '#A855F7', '#FF4FA3'])

fig, ax = plt.subplots(figsize=(10, 8), facecolor=BG_COLOR)
ax.set_facecolor(BG_COLOR)

# Generate SHAP summary plot
shap.summary_plot(
    shap_values.values, 
    X_test, 
    show=False, 
    color_bar=True,
    plot_size=None,
    cmap=cmap
)

# Customize figure styling
fig = plt.gcf()
fig.patch.set_facecolor(BG_COLOR)
ax = plt.gca()
ax.set_facecolor(BG_COLOR)

ax.set_title("SHAP Feature Importance (XGBoost Churn Model)", fontsize=16, fontweight='bold', color=TITLE_COLOR, pad=15)
ax.set_xlabel("SHAP value (impact on model output)", fontsize=12, color=TEXT_COLOR)

ax.tick_params(colors=TEXT_COLOR, labelsize=11)
for spine in ax.spines.values():
    spine.set_color(BORDER_COLOR)

# Customize colorbar if present
for child in fig.get_children():
    if isinstance(child, plt.Axes) and child != ax:
        child.set_facecolor(BG_COLOR)
        child.tick_params(colors=TEXT_COLOR)
        for spine in child.spines.values():
            spine.set_color(BORDER_COLOR)

plt.tight_layout()
plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight', facecolor=BG_COLOR)
print("shap_summary.png successfully generated with dark purple palette!")
