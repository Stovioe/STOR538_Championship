import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import scipy.stats as stats
from scipy.stats import chi2_contingency, ttest_ind, f_oneway
import warnings
warnings.filterwarnings('ignore')

# ── Load & Clean ──────────────────────────────────────────────────────────────
df = pd.read_csv("Copy of Observations - Sheet1.csv")

df['return_direction']  = df['return_direction'].str.lower().str.strip()
df['point_winner']      = df['point_winner'].str.lower().str.strip()
df['point_end_type']    = df['point_end_type'].str.lower().str.strip()
df['point_end_location']= df['point_end_location'].str.lower().str.strip()
df['kitchen_rush']      = df['kitchen_rush'].str.lower().str.strip()

df['return_direction'] = df['return_direction'].replace({
    'line': 'down the line', 'down the line': 'down the line',
    'cross court': 'crosscourt', 'crosscourt': 'crosscourt',
    'middle': 'middle'
})
df['kitchen_rush'] = df['kitchen_rush'].replace({'retuner': 'returner'})
df['point_end_type'] = df['point_end_type'].replace({
    'unforced-error': 'unforced error', 'forced-error': 'forced error',
    'forcer error': 'forced error'
})
df['point_end_location'] = df['point_end_location'].replace({'middle': 'mid-court'})

# All observations — kitchen_rush NaN → "none" (observers left blank when nobody rushed)
df['kitchen_rush'] = df['kitchen_rush'].fillna('none')

# Full dataset (all 267 rows)
df_all = df.copy()
n_all = len(df_all)
print(f"Total observations: {n_all}")

# Subset that has a valid return direction (excludes unreturned serves)
df_clean = df_all[df_all['return_direction'].notna()].copy()
n = len(df_clean)
print(f"Observations with recorded return direction: {n}")

# ── Summary Statistics ────────────────────────────────────────────────────────
print("\n=== CATEGORICAL SUMMARIES (n=267 except return_direction) ===")
# return_direction uses df_clean (has valid return); all others use df_all
cat_sources = {
    'return_direction': (df_clean, n),
    'point_winner':     (df_all, n_all),
    'point_end_type':   (df_all, n_all),
    'point_end_location': (df_all, n_all),
    'kitchen_rush':     (df_all, n_all),
}
for c, (src, nn) in cat_sources.items():
    print(f"\n{c.upper()} (n={nn})")
    vc = src[c].value_counts()
    for val, count in vc.items():
        pct = count / nn * 100
        ci_lo = count/nn - 1.96*np.sqrt((count/nn)*(1-count/nn)/nn)
        ci_hi = count/nn + 1.96*np.sqrt((count/nn)*(1-count/nn)/nn)
        print(f"  {val}: {count} ({pct:.1f}%) [95% CI: {ci_lo*100:.1f}%–{ci_hi*100:.1f}%]")

print("\n=== NUMERIC SUMMARY: shot_count (n=267) ===")
sc = df_all['shot_count']
print(f"  n={sc.count()}, mean={sc.mean():.2f}, SD={sc.std():.2f}, "
      f"min={sc.min()}, Q1={sc.quantile(.25)}, median={sc.median()}, "
      f"Q3={sc.quantile(.75)}, max={sc.max()}")
ci_lo = sc.mean() - 1.96*sc.std()/np.sqrt(sc.count())
ci_hi = sc.mean() + 1.96*sc.std()/np.sqrt(sc.count())
print(f"  95% CI for mean: [{ci_lo:.2f}, {ci_hi:.2f}]")

# ── Statistical Tests ─────────────────────────────────────────────────────────
print("\n=== STATISTICAL TESTS ===")

# 1. ANOVA: kitchen_rush vs shot_count  (use df_all — all 267 obs)
groups_kr = [g['shot_count'].values for _, g in df_all.groupby('kitchen_rush')]
f_stat, p_anova = f_oneway(*groups_kr)
print(f"\n[1] ANOVA — kitchen rush vs shot count (n=267)")
print(f"    F={f_stat:.3f}, p={p_anova:.6f}")
for name, g in df_all.groupby('kitchen_rush'):
    m, s, cnt = g['shot_count'].mean(), g['shot_count'].std(), len(g)
    ci_lo = m - 1.96*s/np.sqrt(cnt)
    ci_hi = m + 1.96*s/np.sqrt(cnt)
    print(f"    {name}: mean={m:.2f}, SD={s:.2f}, n={cnt}, 95% CI [{ci_lo:.2f}, {ci_hi:.2f}]")

# 2. Chi-Square: kitchen_rush vs point_end_location  (df_all)
ct_kl = pd.crosstab(df_all['kitchen_rush'], df_all['point_end_location'])
chi2_kl, p_kl, dof_kl, _ = chi2_contingency(ct_kl)
print(f"\n[2] Chi-Square — kitchen rush vs point end location (n=267)")
print(f"    chi2={chi2_kl:.3f}, dof={dof_kl}, p={p_kl:.6f}")

# 3. Chi-Square: return_direction vs point_winner  (df_clean — needs valid return dir)
ct_rw = pd.crosstab(df_clean['return_direction'], df_clean['point_winner'])
chi2_rw, p_rw, dof_rw, _ = chi2_contingency(ct_rw)
print(f"\n[3] Chi-Square — return direction vs point winner (n={n})")
print(f"    chi2={chi2_rw:.3f}, dof={dof_rw}, p={p_rw:.6f}")
print(ct_rw)
ct_rw_pct = ct_rw.div(ct_rw.sum(axis=1), axis=0) * 100
print("  Row %:")
print(ct_rw_pct.round(1))

# 4. Chi-Square: return_direction vs kitchen_rush  (df_clean)
ct_rk = pd.crosstab(df_clean['return_direction'], df_clean['kitchen_rush'])
chi2_rk, p_rk, dof_rk, _ = chi2_contingency(ct_rk)
print(f"\n[4] Chi-Square — return direction vs kitchen rush (n={n})")
print(f"    chi2={chi2_rk:.3f}, dof={dof_rk}, p={p_rk:.6f}")

# 5. t-test: shot_count for server-won vs returner-won (df_all)
srv_shots = df_all[df_all['point_winner']=='server']['shot_count']
ret_shots = df_all[df_all['point_winner']=='returner']['shot_count']
t_stat, p_ttest = ttest_ind(srv_shots, ret_shots)
print(f"\n[5] t-test — shot count: server-won vs returner-won (n=267)")
print(f"    server mean={srv_shots.mean():.2f}, returner mean={ret_shots.mean():.2f}")
print(f"    t={t_stat:.3f}, p={p_ttest:.4f}")

# 6. Kitchen rush winner rates (df_all)
print(f"\n[6] Point winner rate by kitchen rush (n=267)")
winner_by_rush = df_all.groupby('kitchen_rush')['point_winner'].value_counts(normalize=True).unstack() * 100
print(winner_by_rush.round(1))

# ── Colour palette ────────────────────────────────────────────────────────────
BLUE   = '#2C5F8A'
ORANGE = '#E07B39'
GREEN  = '#4A9B6F'
PURPLE = '#7B5EA7'
GREY   = '#B0B8C1'
PALETTE = [BLUE, ORANGE, GREEN, PURPLE]
sns.set_theme(style='whitegrid', font_scale=1.15)

rush_order = ['none', 'returner', 'server', 'both']
loc_order  = ['baseline', 'mid-court', 'kitchen']
dir_order  = ['crosscourt', 'down the line', 'middle']

# ── Figure 1: Shot Count by Kitchen Rush (barplot w/ CI) ─────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
means, cis, labels = [], [], rush_order
for rk in rush_order:
    sub = df_all[df_all['kitchen_rush']==rk]['shot_count']
    m = sub.mean(); se = sub.std()/np.sqrt(len(sub))
    means.append(m); cis.append(1.96*se)
bars = ax.bar(labels, means, color=PALETTE, edgecolor='white', linewidth=0.8, width=0.55)
ax.errorbar(labels, means, yerr=cis, fmt='none', color='black', capsize=6, linewidth=1.5)
ax.set_title("Average Rally Length by Kitchen Rush Status\n(with 95% Confidence Intervals)", fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel("Kitchen Rush Status", fontsize=12)
ax.set_ylabel("Average Shot Count (shots per rally)", fontsize=12)
for bar, m, ci in zip(bars, means, cis):
    ax.text(bar.get_x()+bar.get_width()/2, m+ci+0.15, f'{m:.1f}', ha='center', va='bottom', fontsize=10.5, fontweight='bold')
ax.set_ylim(0, 10)
ax.yaxis.grid(True, linestyle='--', alpha=0.6)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig("figure1_shot_count.png", dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved figure1_shot_count.png")

# ── Figure 2: Point End Location by Kitchen Rush (stacked %) ─────────────────
fig, ax = plt.subplots(figsize=(9, 6))
ct_pct = pd.crosstab(df_all['kitchen_rush'], df_all['point_end_location'], normalize='index')*100
ct_pct = ct_pct.reindex(rush_order)[loc_order]
bottom = np.zeros(len(rush_order))
loc_colors = [BLUE, ORANGE, GREEN]
for col, color in zip(loc_order, loc_colors):
    vals = ct_pct[col].values
    ax.bar(rush_order, vals, bottom=bottom, color=color, label=col.title(), edgecolor='white', linewidth=0.8, width=0.55)
    for i, (v, b) in enumerate(zip(vals, bottom)):
        if v > 5:
            ax.text(i, b + v/2, f'{v:.0f}%', ha='center', va='center', fontsize=9.5, color='white', fontweight='bold')
    bottom += vals
ax.set_title("Point Ending Location by Kitchen Rush Status", fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel("Kitchen Rush Status", fontsize=12)
ax.set_ylabel("Percentage of Points (%)", fontsize=12)
ax.legend(title='Point End Location', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=10)
ax.set_ylim(0, 105)
ax.yaxis.grid(False)
plt.tight_layout()
plt.savefig("figure2_end_location.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure2_end_location.png")

# ── Figure 3: Return Direction vs Point Winner (grouped bar) ──────────────────
fig, ax = plt.subplots(figsize=(9, 6))
ct_rw_pct2 = pd.crosstab(df_clean['return_direction'], df_clean['point_winner'], normalize='index')*100
ct_rw_pct2 = ct_rw_pct2.reindex(dir_order)
x = np.arange(len(dir_order)); w = 0.35
b1 = ax.bar(x - w/2, ct_rw_pct2['returner'], width=w, color=ORANGE, label='Returner Wins', edgecolor='white')
b2 = ax.bar(x + w/2, ct_rw_pct2['server'],   width=w, color=BLUE,   label='Server Wins',   edgecolor='white')
for bar in list(b1)+list(b2):
    h = bar.get_height()
    ax.text(bar.get_x()+bar.get_width()/2, h+0.8, f'{h:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.axhline(50, color='black', linewidth=1.2, linestyle='--', alpha=0.5, label='50% line')
ax.set_xticks(x)
ax.set_xticklabels([d.title() for d in dir_order], fontsize=11)
ax.set_title("Point Win Rate by Return Direction", fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel("Return Direction", fontsize=12)
ax.set_ylabel("Win Rate (%)", fontsize=12)
ax.set_ylim(0, 75)
ax.legend(fontsize=10)
ax.yaxis.grid(True, linestyle='--', alpha=0.6)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig("figure3_return_winner.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure3_return_winner.png")

# ── Figure 4: Shot Count Distribution by Point End Type (violin) ──────────────
fig, ax = plt.subplots(figsize=(9, 6))
end_order = ['winner', 'forced error', 'unforced error']
end_colors = [GREEN, BLUE, ORANGE]
parts = ax.violinplot([df_all[df_all['point_end_type']==e]['shot_count'].values for e in end_order],
                       positions=[1,2,3], showmedians=True, showextrema=True)
for pc, color in zip(parts['bodies'], end_colors):
    pc.set_facecolor(color); pc.set_alpha(0.75)
parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(2)
parts['cmins'].set_color('black'); parts['cmaxes'].set_color('black')
parts['cbars'].set_color('black')
ax.set_xticks([1,2,3])
ax.set_xticklabels([e.title() for e in end_order], fontsize=11)
ax.set_title("Rally Length Distribution by How the Point Ended", fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel("Point End Type", fontsize=12)
ax.set_ylabel("Shot Count (shots per rally)", fontsize=12)
for i, e in enumerate(end_order):
    m = df_all[df_all['point_end_type']==e]['shot_count'].median()
    ax.text(i+1, m+0.3, f'Median={m:.0f}', ha='center', va='bottom', fontsize=9.5, color='black')
ax.yaxis.grid(True, linestyle='--', alpha=0.6)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig("figure4_shot_dist.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure4_shot_dist.png")

# ── Figure 5: Heatmap — Return Direction × Kitchen Rush (% within direction) ──
fig, ax = plt.subplots(figsize=(9, 6))
ct_rk_pct = pd.crosstab(df_clean['return_direction'], df_clean['kitchen_rush'], normalize='index')*100
ct_rk_pct = ct_rk_pct.reindex(dir_order)[rush_order]
sns.heatmap(ct_rk_pct, annot=True, fmt='.1f', cmap='Blues', linewidths=0.5,
            linecolor='white', ax=ax, cbar_kws={'label': '% of Points (within direction)'})
ax.set_title("Kitchen Rush Behavior by Return Direction\n(row % within each return direction)", fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel("Kitchen Rush Status", fontsize=11)
ax.set_ylabel("Return Direction", fontsize=11)
ax.set_xticklabels([r.title() for r in rush_order], fontsize=10)
ax.set_yticklabels([d.title() for d in dir_order], fontsize=10, rotation=0)
plt.tight_layout()
plt.savefig("figure5_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved figure5_heatmap.png")

# ── Print data preview (first 8 clean rows, key cols) ─────────────────────────
preview_cols = ['return_direction','point_winner','point_end_type','point_end_location','kitchen_rush','shot_count']
print("\n=== DATA PREVIEW (first 8 rows) ===")
print(df_all[preview_cols].head(8).to_string(index=False))

print("\nDone.")
