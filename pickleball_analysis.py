import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import chi2_contingency

# 1. Load Data
df = pd.read_csv("Copy of Observations - Sheet1.csv")

# 2. Clean Data
# Standardize point_winner
df['point_winner'] = df['point_winner'].str.lower().str.strip()

# Standardize return_direction
df['return_direction'] = df['return_direction'].str.lower().str.strip()
df['return_direction'] = df['return_direction'].replace({
    'line': 'down the line', 
    'down the line': 'down the line',
    'cross court': 'crosscourt',
    'crosscourt': 'crosscourt'
})

# Standardize kitchen_rush
df['kitchen_rush'] = df['kitchen_rush'].str.lower().str.strip()
df['kitchen_rush'] = df['kitchen_rush'].replace({'retuner': 'returner'})

# Standardize point_end_type
df['point_end_type'] = df['point_end_type'].str.lower().str.strip()
df['point_end_type'] = df['point_end_type'].replace({
    'unforced-error': 'unforced error', 
    'forced-error': 'forced error',
    'forcer error': 'forced error'
})

# Standardize point_end_location
df['point_end_location'] = df['point_end_location'].str.lower().str.strip()
df['point_end_location'] = df['point_end_location'].replace({'middle': 'mid-court'})

# Drop rows with N/A in key columns 
df_clean = df.dropna(subset=['return_direction', 'kitchen_rush', 'point_winner']).copy()

# 3. Summary Statistics
print("--- CATEGORICAL SUMMARY ---")
cats = ['return_direction', 'point_winner', 'point_end_type', 'point_end_location', 'kitchen_rush']
for c in cats:
    print(f"\n{c.upper()}")
    vc = df_clean[c].value_counts()
    for val, count in vc.items():
        print(f"{val}: {count} ({count/len(df_clean)*100:.1f}%)")

print("\n--- NUMERIC SUMMARY ---")
print(df_clean['shot_count'].describe())

# 4. Statistical Testing
print("\n--- STATISTICAL TESTS ---")
# ANOVA: Kitchen Rush vs Shot Count
groups = [group['shot_count'].values for name, group in df_clean.groupby('kitchen_rush')]
f_stat, p_val_anova = stats.f_oneway(*groups)
print(f"ANOVA (Kitchen Rush vs Shot Count): p-value = {p_val_anova:.6f}")

# Chi-Square: Kitchen Rush vs Point End Location
contingency_table = pd.crosstab(df_clean['kitchen_rush'], df_clean['point_end_location'])
chi2, p_val_chi, dof, expected = chi2_contingency(contingency_table)
print(f"Chi-Square (Kitchen Rush vs Point End Location): p-value = {p_val_chi:.6f}")

# 5. Visualizations
sns.set_theme(style="whitegrid")

# Figure 1: Average Shot Count by Kitchen Rush with CI
plt.figure(figsize=(8, 6))
sns.barplot(data=df_clean, x='kitchen_rush', y='shot_count', errorbar=('ci', 95), 
            order=['none', 'returner', 'server', 'both'], capsize=0.1, palette='Blues_d')
plt.title("Average Shot Count by Kitchen Rush Status\n(with 95% Confidence Intervals)")
plt.xlabel("Kitchen Rush Status")
plt.ylabel("Average Shot Count")
plt.tight_layout()
plt.savefig("figure1_shot_count.png")
print("\nSaved 'figure1_shot_count.png'")

# Figure 2: Stacked Bar Chart of Point End Location by Kitchen Rush
plt.figure(figsize=(8, 6))
crosstab_pct = pd.crosstab(df_clean['kitchen_rush'], df_clean['point_end_location'], normalize='index') * 100
crosstab_pct = crosstab_pct.reindex(['none', 'returner', 'server', 'both'])
crosstab_pct.plot(kind='bar', stacked=True, figsize=(8, 6), colormap='viridis')
plt.title("Point End Location Distribution by Kitchen Rush Status")
plt.xlabel("Kitchen Rush Status")
plt.ylabel("Percentage of Points (%)")
plt.legend(title='Point End Location')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("figure2_end_location.png")
print("Saved 'figure2_end_location.png'")