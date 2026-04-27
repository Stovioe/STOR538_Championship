import matplotlib.pyplot as plt
import numpy as np

HEADER   = '#2C5F8A'
ROW_EVEN = '#EBF2FA'
ROW_ODD  = '#FFFFFF'
CAPTION  = dict(ha='center', fontsize=9.5, color='#444444', style='italic')
FONT     = 13

def style_table(tbl, group_rows=None, header_font=13, body_font=13):
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#CCCCCC')
        cell.set_linewidth(0.5)
        cell.PAD = 0.04
        if r == 0:
            cell.set_facecolor(HEADER)
            cell.set_text_props(color='white', fontweight='bold', fontsize=header_font)
        else:
            cell.set_facecolor(ROW_EVEN if (r - 1) % 2 == 0 else ROW_ODD)
            cell.set_text_props(color='#222222', fontsize=body_font)
    if group_rows:
        ncols = max(c for (r, c) in tbl.get_celld()) + 1
        for r in group_rows:
            for c in range(ncols):
                if (r, c) in tbl.get_celld():
                    tbl[r, c].set_facecolor('#D6E8F7')

# ── Table 1 ───────────────────────────────────────────────────────────────────
rows1 = [
    ['Middle',        'Server',   'Winner',        'Mid-Court', 'Both',     '5'],
    ['Crosscourt',    'Returner', 'Winner',        'Kitchen',   'Returner', '4'],
    ['Middle',        'Server',   'Winner',        'Mid-Court', 'Both',     '3'],
    ['Crosscourt',    'Server',   'Unforced Error','Baseline',  'Returner', '2'],
    ['Crosscourt',    'Server',   'Winner',        'Kitchen',   'Both',     '5'],
    ['Down the Line', 'Returner', 'Unforced Error','Baseline',  'Returner', '4'],
    ['Middle',        'Returner', 'Winner',        'Kitchen',   'Returner', '4'],
    ['Crosscourt',    'Server',   'Unforced Error','Baseline',  'Returner', '2'],
]
cols1 = ['Return\nDirection (RD)', 'Point\nWinner (PW)', 'Point End\nType (PET)',
         'Point End\nLocation (PEL)', 'Kitchen\nRush (KR)', 'Shot\nCount (SC)']

# column widths sized to content
col_widths1 = [0.21, 0.16, 0.20, 0.20, 0.16, 0.10]

fig, ax = plt.subplots(figsize=(12, 4.5))
ax.axis('off')
tbl = ax.table(cellText=rows1, colLabels=cols1, loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(FONT)
tbl.scale(1, 2.0)
style_table(tbl)
for c, w in enumerate(col_widths1):
    for r in range(len(rows1) + 1):
        tbl[r, c].set_width(w)
fig.text(0.5, 0.01,
    'Table 1. First eight observations from the pickleball dataset. '
    'RD = Return Direction, PW = Point Winner, PET = Point End Type, '
    'PEL = Point End Location, KR = Kitchen Rush, SC = Shot Count.',
    **CAPTION)
plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.savefig('table1_preview.png', dpi=180, bbox_inches='tight')
plt.close()
print('Saved table1_preview.png')

# ── Table 2 ───────────────────────────────────────────────────────────────────
rows2 = [
    ['Return Direction',  'Crosscourt',     '110', '43.7%', '[37.5%, 49.8%]'],
    ['',                  'Down the Line',   '77', '30.6%', '[24.9%, 36.2%]'],
    ['',                  'Middle',          '65', '25.8%', '[20.4%, 31.2%]'],
    ['Point Winner',      'Server',         '144', '53.9%', '[48.0%, 59.9%]'],
    ['',                  'Returner',       '123', '46.1%', '[40.1%, 52.0%]'],
    ['Point End Type',    'Unforced Error',  '95', '35.6%', '[29.8%, 41.3%]'],
    ['',                  'Winner',          '92', '34.5%', '[28.8%, 40.2%]'],
    ['',                  'Forced Error',    '80', '30.0%', '[24.5%, 35.5%]'],
    ['Point End Location','Baseline',       '127', '47.6%', '[41.6%, 53.6%]'],
    ['',                  'Mid-Court',       '95', '35.6%', '[29.8%, 41.3%]'],
    ['',                  'Kitchen',         '45', '16.9%', '[12.4%, 21.3%]'],
    ['Kitchen Rush',      'Returner',       '108', '40.4%', '[34.6%, 46.3%]'],
    ['',                  'None',            '65', '24.3%', '[19.2%, 29.5%]'],
    ['',                  'Server',          '61', '22.8%', '[17.8%, 27.9%]'],
    ['',                  'Both',            '33', '12.4%',  '[8.4%, 16.3%]'],
]
cols2 = ['Variable', 'Value', 'Count', '%', '95% CI']

# Variable col needs room; Value moderate; Count/% narrow; CI moderate
col_widths2 = [0.26, 0.20, 0.10, 0.10, 0.20]

fig, ax = plt.subplots(figsize=(10, 7.5))
ax.axis('off')
tbl = ax.table(cellText=rows2, colLabels=cols2, loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(FONT)
tbl.scale(1, 1.7)
style_table(tbl, group_rows=[1, 4, 6, 9, 12])
for c, w in enumerate(col_widths2):
    for r in range(len(rows2) + 1):
        tbl[r, c].set_width(w)
fig.text(0.5, 0.01,
    'Table 2. Frequency summary of categorical variables (n = 267) with 95% confidence intervals. '
    '*Return Direction based on n = 252 (15 points had no return).',
    **CAPTION)
plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig('table2_categorical.png', dpi=180, bbox_inches='tight')
plt.close()
print('Saved table2_categorical.png')

# ── Table 3 ───────────────────────────────────────────────────────────────────
rows3 = [
    ['Minimum',              '1'],
    ['Q1 (25th percentile)', '3'],
    ['Median',               '5'],
    ['Mean',                 '5.28'],
    ['Q3 (75th percentile)', '7'],
    ['Maximum',              '18'],
    ['Standard Deviation',   '2.81'],
    ['95% CI for Mean',      '[4.94, 5.61]'],
]
cols3 = ['Statistic', 'Value']

col_widths3 = [0.48, 0.28]

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.axis('off')
tbl = ax.table(cellText=rows3, colLabels=cols3, loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(FONT)
tbl.scale(1, 1.9)
style_table(tbl)
for c, w in enumerate(col_widths3):
    for r in range(len(rows3) + 1):
        tbl[r, c].set_width(w)
fig.text(0.5, 0.01,
    'Table 3. Numeric summary of Shot Count (SC) with 95% confidence interval for the mean (n = 267).',
    **CAPTION)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig('table3_numeric.png', dpi=180, bbox_inches='tight')
plt.close()
print('Saved table3_numeric.png')

# ── Table 4 ───────────────────────────────────────────────────────────────────
rows4 = [
    ['None',          '35.4%', '64.6%'],
    ['Returner Only', '50.0%', '50.0%'],
    ['Server Only',   '44.3%', '55.7%'],
    ['Both',          '57.6%', '42.4%'],
]
cols4 = ['Kitchen Rush', 'Returner Win %', 'Server Win %']

col_widths4 = [0.28, 0.22, 0.22]

fig, ax = plt.subplots(figsize=(6, 3.2))
ax.axis('off')
tbl = ax.table(cellText=rows4, colLabels=cols4, loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(FONT)
tbl.scale(1, 2.1)
style_table(tbl)
for c, w in enumerate(col_widths4):
    for r in range(len(rows4) + 1):
        tbl[r, c].set_width(w)
tbl[1, 2].set_facecolor('#FDECEA')
tbl[1, 2].set_text_props(fontweight='bold', color='#B71C1C', fontsize=FONT)
tbl[4, 1].set_facecolor('#E8F5E9')
tbl[4, 1].set_text_props(fontweight='bold', color='#1B5E20', fontsize=FONT)
fig.text(0.5, 0.01,
    'Table 4. Point win rate by kitchen rush status (n = 267). '
    'Red = largest server advantage; green = largest returner advantage.',
    **CAPTION)
plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig('table4_winrate.png', dpi=180, bbox_inches='tight')
plt.close()
print('Saved table4_winrate.png')

print('\nAll tables saved.')
