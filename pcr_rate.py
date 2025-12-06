import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# 1. DATA LOADING & CLEANING
# =============================================================================
file_path = r'D:\Neeraj\UoN\Machine Learning\Assignment 2\TrainDataset2025 (1).csv'
df = pd.read_csv(file_path, header=0)

# Standard Cleaning
df.rename(columns={df.columns[0]: 'ID', df.columns[1]: 'pCR', df.columns[2]:'RFS'}, inplace=True)

# THE CRITICAL STEP: Convert 'Hidden Nulls' (999) to actual NaNs for analysis
df.replace([999, 999.0], np.nan, inplace=True)

target = 'pCR'
# Select all features except ID and Target
features_to_check = [c for c in df.columns if c not in ['ID', 'pCR', 'RFS', '(outcome)']]

# =============================================================================
# 2. CALCULATE RISK METRICS (Automated Loop)
# =============================================================================
print("Calculating pCR risk differentials for all features...")
results = []

for col in features_to_check:
    # Masks for missing vs known
    missing_mask = df[col].isna()
    known_mask = df[col].notna()
    
    missing_count = missing_mask.sum()
    
    # We only analyze features that actually have missing data
    if missing_count > 0:
        # Calculate pCR rates
        rate_missing = df.loc[missing_mask, target].mean()
        rate_known = df.loc[known_mask, target].mean()
        
        # Calculate risk difference
        diff = rate_missing - rate_known
        
        results.append({
            'Feature': col,
            'Missing_Count': missing_count,
            'pCR_Rate_Missing': rate_missing,
            'pCR_Rate_Known': rate_known,
            'Difference': diff
        })

# Create a DataFrame from the results
results_df = pd.DataFrame(results)

# Sort by the "Difference" to highlight the strongest signals (High Risk Missingness)
results_df = results_df.sort_values(by='Difference', ascending=False)

# Display the data table
print("\n--- Calculated Risk Table ---")
print(results_df[['Feature', 'Missing_Count', 'pCR_Rate_Missing', 'pCR_Rate_Known', 'Difference']].to_string(index=False))

# =============================================================================
# 3. GENERATE VISUALIZATIONS
# =============================================================================
print("\nGenerating Plots...")

# Set up the figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
plt.style.use('ggplot')

# --- Plot 1: Missing Value Counts ---
sns.barplot(x='Missing_Count', y='Feature', data=results_df, ax=axes[0], color='skyblue')
axes[0].set_title('Count of Missing Values (N)', fontsize=14)
axes[0].set_xlabel('Count', fontsize=12)
axes[0].set_ylabel('Feature', fontsize=12)
# Add text labels
for i, v in enumerate(results_df['Missing_Count']):
    axes[0].text(v, i, f" {v}", color='black', va='center', fontweight='bold')

# --- Plot 2: Risk Comparison (Missing vs Known) ---
# Melt data for side-by-side bar plot
melted_df = results_df.melt(id_vars='Feature', value_vars=['pCR_Rate_Missing', 'pCR_Rate_Known'], 
                            var_name='Group', value_name='pCR_Rate')
melted_df['Group'] = melted_df['Group'].replace({'pCR_Rate_Missing': 'Missing Data', 'pCR_Rate_Known': 'Known Data'})

sns.barplot(x='pCR_Rate', y='Feature', hue='Group', data=melted_df, ax=axes[1], palette=['#E24A33', '#348ABD'])
axes[1].set_title('pCR Probability: Missing vs. Known', fontsize=14)
axes[1].set_xlabel('Probability of pCR', fontsize=12)
axes[1].set_ylabel('')
axes[1].legend(title='Patient Group')

# --- Plot 3: Risk Differential (The "Signal Strength") ---
# Color code: Green = Missing indicates High Risk, Red = Missing indicates Low Risk
colors = ['green' if x > 0 else 'red' for x in results_df['Difference']]
sns.barplot(x='Difference', y='Feature', data=results_df, ax=axes[2], palette=colors)
axes[2].set_title('Risk Differential (Signal Strength)', fontsize=14)
axes[2].set_xlabel('Added Risk when Data is Missing', fontsize=12)
axes[2].set_ylabel('')
axes[2].axvline(0, color='black', linestyle='--', linewidth=1)

# Final Layout Adjustments
plt.tight_layout()
plt.show()

print("Plot saved as 'missingness_analysis_full.png'")