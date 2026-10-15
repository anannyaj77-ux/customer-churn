import sys
sys.path.append('.')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_prep import load_and_clean_data

# Set light background theme
plt.style.use('default')

# Load data
df = load_and_clean_data()

# Light Palette Colors
BG_COLOR = '#FFFFFF'
CARD_BG = '#F1F5F9'
TEXT_COLOR = '#0F172A'
MUTED_TEXT = '#475569'
BORDER_COLOR = '#CBD5E1'
GRID_COLOR = '#E2E8F0'
TITLE_COLOR = '#4F46E5'

COLOR_YES = '#DB2777'  # Vibrant Rose/Magenta
COLOR_NO = '#0891B2'   # Bright Cyan/Teal

palette = {'Yes': COLOR_YES, 'No': COLOR_NO}

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=BG_COLOR)
fig.suptitle("Telco Customer Churn - Key EDA Insights", fontsize=18, fontweight='bold', color=TEXT_COLOR, y=0.96)

# 1. Churn Rate by Contract Type (%)
ax1 = axes[0, 0]
ax1.set_facecolor(BG_COLOR)
contract_churn = df.groupby('Contract')['Churn'].value_counts(normalize=True).unstack() * 100
contract_churn = contract_churn.reindex(['Month-to-month', 'One year', 'Two year'])[['Yes', 'No']]
contract_df = contract_churn.reset_index().melt(id_vars='Contract', value_name='Percentage', var_name='Churn')

sns.barplot(data=contract_df, x='Contract', y='Percentage', hue='Churn', palette=palette, ax=ax1)
ax1.set_title("Churn Rate by Contract Type (%)", fontsize=13, fontweight='bold', color=TITLE_COLOR, pad=12)
ax1.set_xlabel("", color=TEXT_COLOR)
ax1.set_ylabel("Percentage (%)", fontsize=11, color=MUTED_TEXT)
ax1.set_ylim(0, 105)
ax1.grid(True, linestyle='--', alpha=0.6, color=GRID_COLOR)

# 2. Tenure Distribution by Churn Status
ax2 = axes[0, 1]
ax2.set_facecolor(BG_COLOR)
sns.kdeplot(data=df, x='tenure', hue='Churn', common_norm=False, fill=True, palette=palette, alpha=0.35, linewidth=2, ax=ax2)
ax2.set_title("Tenure Distribution by Churn Status", fontsize=13, fontweight='bold', color=TITLE_COLOR, pad=12)
ax2.set_xlabel("Tenure (Months)", fontsize=11, color=MUTED_TEXT)
ax2.set_ylabel("Density", fontsize=11, color=MUTED_TEXT)
ax2.grid(True, linestyle='--', alpha=0.6, color=GRID_COLOR)

# 3. Churn Rate by Internet Service (%)
ax3 = axes[1, 0]
ax3.set_facecolor(BG_COLOR)
net_churn = df.groupby('InternetService')['Churn'].value_counts(normalize=True).unstack() * 100
net_churn = net_churn.reindex(['DSL', 'Fiber optic', 'No'])[['Yes', 'No']]
net_df = net_churn.reset_index().melt(id_vars='InternetService', value_name='Percentage', var_name='Churn')

sns.barplot(data=net_df, x='InternetService', y='Percentage', hue='Churn', palette=palette, ax=ax3)
ax3.set_title("Churn Rate by Internet Service (%)", fontsize=13, fontweight='bold', color=TITLE_COLOR, pad=12)
ax3.set_xlabel("", color=TEXT_COLOR)
ax3.set_ylabel("Percentage (%)", fontsize=11, color=MUTED_TEXT)
ax3.set_ylim(0, 105)
ax3.grid(True, linestyle='--', alpha=0.6, color=GRID_COLOR)

# 4. Churn Rate by Payment Method (%)
ax4 = axes[1, 1]
ax4.set_facecolor(BG_COLOR)
pay_churn = df.groupby('PaymentMethod')['Churn'].value_counts(normalize=True).unstack() * 100
pay_order = ['Mailed check', 'Electronic check', 'Credit card (automatic)', 'Bank transfer (automatic)']
pay_churn = pay_churn.reindex(pay_order)[['Yes', 'No']]
pay_df = pay_churn.reset_index().melt(id_vars='PaymentMethod', value_name='Percentage', var_name='Churn')

sns.barplot(data=pay_df, y='PaymentMethod', x='Percentage', hue='Churn', palette=palette, ax=ax4)
ax4.set_title("Churn Rate by Payment Method (%)", fontsize=13, fontweight='bold', color=TITLE_COLOR, pad=12)
ax4.set_ylabel("", color=TEXT_COLOR)
ax4.set_xlabel("Percentage (%)", fontsize=11, color=MUTED_TEXT)
ax4.set_xlim(0, 105)
ax4.grid(True, linestyle='--', alpha=0.6, color=GRID_COLOR)

# Apply common styling across subplots
for ax in [ax1, ax2, ax3, ax4]:
    ax.tick_params(colors=TEXT_COLOR, labelsize=10)
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)
        spine.set_linewidth(1.2)
    
    # Legend styling
    leg = ax.get_legend()
    if leg:
        leg.set_title("Churn")
        leg.get_title().set_color(TEXT_COLOR)
        leg.get_title().set_fontsize(10)
        leg.get_title().set_fontweight('bold')
        frame = leg.get_frame()
        frame.set_facecolor(CARD_BG)
        frame.set_edgecolor(BORDER_COLOR)
        for text in leg.get_texts():
            text.set_color(TEXT_COLOR)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig('eda_churn_drivers_light.png', dpi=300, bbox_inches='tight', facecolor=BG_COLOR)
print("eda_churn_drivers_light.png successfully generated!")
