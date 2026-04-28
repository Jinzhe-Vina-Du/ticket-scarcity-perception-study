"""
Scarcity Perception and Decision-Making Among University Students
Independent Research Project — Python Analysis
Author: Vina
Dataset: Survey on university students' attitudes and behaviors toward ticket scarcity (n=125)

Research focus: How does perceived ticket scarcity affect purchase urgency,
decision-making behavior, and the psychological gap between urgency and action?
The β inertia coefficient concept captures this urgency-action gap.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
})
PALETTE = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']

# ── 1. Load & clean data ──────────────────────────────────────────────────────
FILE = "286187274_按文本_大学生对门票稀缺的态度与行为调查_A_survey_about_university_students39_attitudes_and_behaviors_to_tickets_scarcity_125_125.xlsx"

df_raw = pd.read_excel(FILE, header=0)

# Rename key columns to short English names
col_map = {
    df_raw.columns[7]:  'gender',
    df_raw.columns[8]:  'age_group',
    df_raw.columns[9]:  'has_bought_ticket',
    df_raw.columns[10]: 'has_experienced_scarcity',
    df_raw.columns[11]: 'budget',
    df_raw.columns[12]: 'scarcity_feeling',
    df_raw.columns[13]: 'purchase_intent_impact',   # Q7 (scale 1-5 text)
    df_raw.columns[14]: 'purchase_possibility',      # Q8 (scale 1-5 text)
    df_raw.columns[15]: 'perceived_value',           # Q9
    df_raw.columns[16]: 'urgency',                   # Q10
    df_raw.columns[17]: 'urgency_to_action',         # Q11
    df_raw.columns[18]: 'fear_of_loss',              # Q12a
    df_raw.columns[19]: 'love_for_event',            # Q12b
    df_raw.columns[20]: 'friends_opinion',           # Q12c
    df_raw.columns[21]: 'peer_competition',          # Q12d
    df_raw.columns[22]: 'event_popularity',          # Q12e
    df_raw.columns[23]: 'promotion',                 # Q12f
}
df = df_raw.rename(columns=col_map)

# ── Encode Likert scales ───────────────────────────────────────────────────────
likert5_map = {
    'Strongly disagree': 1, 'Strongly Disagree': 1,
    '2)Disagree': 2, 'Disagree': 2,
    '3)No opinion': 3, 'No opinion': 3,
    '4)Agree': 4, 'Agree': 4,
    'Strongly agree': 5, 'Strongly Agree': 5,
}
importance_map = {
    'the least important': 1,
    'Not important': 2,
    'Neutral': 3,
    'Important': 4,
    'Very important': 5,
}
possibility_map = {
    'Not possible at all': 1,
    'Unlikely': 2,
    'Uncertain': 3,
    'Likely': 4,
    'Definitely': 5,
}

for col in ['perceived_value', 'urgency', 'urgency_to_action']:
    df[col + '_num'] = df[col].map(likert5_map)

for col in ['fear_of_loss', 'love_for_event', 'friends_opinion',
            'peer_competition', 'event_popularity', 'promotion']:
    df[col + '_num'] = df[col].map(importance_map)

df['purchase_possibility_num'] = df['purchase_possibility'].map(possibility_map)

# β inertia coefficient: captures urgency-action gap (higher = more inertia)
df['beta_inertia'] = df['urgency_num'] - df['urgency_to_action_num']

# Gender cleanup
df['gender_clean'] = df['gender'].apply(
    lambda x: 'Female' if 'Female' in str(x) else ('Male' if 'Male' in str(x) else 'Other'))

# Age group cleanup
age_map = {
    '18到20岁 from 18 to 20': '18–20',
    '21到23岁 from 21 to 23': '21–23',
    '23岁以上 over 23': '23+',
}
df['age_clean'] = df['age_group'].map(age_map).fillna('Other')

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Total responses: {len(df)}")
print(f"\nGender distribution:\n{df['gender_clean'].value_counts()}")
print(f"\nAge distribution:\n{df['age_clean'].value_counts()}")
print(f"\nHas bought ticket:\n{df['has_bought_ticket'].value_counts()}")
print(f"\nHas experienced scarcity:\n{df['has_experienced_scarcity'].value_counts()}")

# ── 2. Descriptive statistics ─────────────────────────────────────────────────
scale_vars = {
    'Perceived Value (Q9)':      'perceived_value_num',
    'Urgency (Q10)':             'urgency_num',
    'Urgency→Action (Q11)':      'urgency_to_action_num',
    'Fear of Loss (Q12a)':       'fear_of_loss_num',
    'Love for Event (Q12b)':     'love_for_event_num',
    'Friends\' Opinion (Q12c)':  'friends_opinion_num',
    'Peer Competition (Q12d)':   'peer_competition_num',
    'Event Popularity (Q12e)':   'event_popularity_num',
    'Promotion (Q12f)':          'promotion_num',
    'β Inertia Coefficient':     'beta_inertia',
}

print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)
desc_data = []
for label, col in scale_vars.items():
    s = df[col].dropna()
    desc_data.append({
        'Variable': label,
        'N': len(s),
        'Mean': round(s.mean(), 2),
        'SD': round(s.std(), 2),
        'Min': s.min(),
        'Max': s.max(),
    })
desc_df = pd.DataFrame(desc_data)
print(desc_df.to_string(index=False))

# ── 3. Correlation matrix ─────────────────────────────────────────────────────
corr_cols = [
    'urgency_num', 'urgency_to_action_num', 'fear_of_loss_num',
    'friends_opinion_num', 'event_popularity_num', 'perceived_value_num',
]
corr_labels = [
    'Urgency\n(Q10)', 'Urgency→\nAction (Q11)', 'Fear of\nLoss',
    'Friends\'\nOpinion', 'Event\nPopularity', 'Perceived\nValue',
]

corr_matrix = df[corr_cols].corr()

# Compute p-values
n = len(df[corr_cols].dropna())
p_matrix = pd.DataFrame(np.ones_like(corr_matrix), columns=corr_cols, index=corr_cols)
for i, c1 in enumerate(corr_cols):
    for j, c2 in enumerate(corr_cols):
        if i != j:
            valid = df[[c1, c2]].dropna()
            r, p = stats.pearsonr(valid[c1], valid[c2])
            p_matrix.loc[c1, c2] = p

print("\n" + "=" * 60)
print("CORRELATION MATRIX (Pearson r)")
print("=" * 60)
corr_display = corr_matrix.copy()
corr_display.columns = corr_labels
corr_display.index = corr_labels
print(corr_display.round(2).to_string())

# ── 4. FIGURES ─────────────────────────────────────────────────────────────────

# Figure 1: β Inertia Coefficient distribution
fig, ax = plt.subplots(figsize=(8, 5))
beta_vals = df['beta_inertia'].dropna()
counts = beta_vals.value_counts().sort_index()
bars = ax.bar(counts.index, counts.values, color=PALETTE[0], edgecolor='white', linewidth=0.8)
ax.axvline(0, color='#C44E52', linestyle='--', linewidth=1.5, label='No inertia (β=0)')
ax.axvline(beta_vals.mean(), color='#DD8452', linestyle='-', linewidth=1.5,
           label=f'Mean β = {beta_vals.mean():.2f}')
ax.set_xlabel('β Inertia Coefficient (Urgency − Urgency-to-Action)', fontsize=11)
ax.set_ylabel('Number of Respondents', fontsize=11)
ax.set_title('Distribution of the β Inertia Coefficient\namong University Students (n=125)', fontsize=13, pad=12)
ax.legend(frameon=False, fontsize=10)
pct_pos = (beta_vals > 0).mean() * 100
ax.text(0.98, 0.95, f'{pct_pos:.0f}% show positive inertia\n(feel urgency but hesitate to act)',
        transform=ax.transAxes, ha='right', va='top', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.4', fc='#f5f5f5', ec='none'))
plt.tight_layout()
plt.savefig('fig1_beta_inertia_distribution.png', bbox_inches='tight')
plt.close()
print("\n[Saved] fig1_beta_inertia_distribution.png")

# Figure 2: Correlation heatmap
fig, ax = plt.subplots(figsize=(8, 6))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
annot_matrix = corr_matrix.copy().round(2).astype(str)
for i, c1 in enumerate(corr_cols):
    for j, c2 in enumerate(corr_cols):
        if i != j:
            r_val = corr_matrix.loc[c1, c2]
            p_val = p_matrix.loc[c1, c2]
            star = '**' if p_val < 0.01 else ('*' if p_val < 0.05 else '')
            annot_matrix.loc[c1, c2] = f'{r_val:.2f}{star}'

sns.heatmap(
    corr_matrix, mask=mask, annot=annot_matrix.values, fmt='',
    cmap='Blues', vmin=0, vmax=1,
    xticklabels=corr_labels, yticklabels=corr_labels,
    linewidths=0.5, linecolor='white', ax=ax,
    cbar_kws={'label': 'Pearson r'}
)
ax.set_title('Correlation Matrix: Scarcity Perception Variables\n(* p<0.05, ** p<0.01)',
             fontsize=12, pad=12)
plt.tight_layout()
plt.savefig('fig2_correlation_heatmap.png', bbox_inches='tight')
plt.close()
print("[Saved] fig2_correlation_heatmap.png")

# Figure 3: Mean scores of key variables (bar chart)
key_vars = {
    'Perceived\nValue': 'perceived_value_num',
    'Urgency\n(Q10)': 'urgency_num',
    'Urgency→\nAction (Q11)': 'urgency_to_action_num',
    'Fear of\nLoss': 'fear_of_loss_num',
    'Event\nPopularity': 'event_popularity_num',
    'Friends\'\nOpinion': 'friends_opinion_num',
}
means = [df[v].mean() for v in key_vars.values()]
sems  = [df[v].sem()  for v in key_vars.values()]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(list(key_vars.keys()), means, yerr=sems,
              color=PALETTE[:len(key_vars)], edgecolor='white',
              capsize=4, linewidth=0.8)
ax.axhline(3, color='gray', linestyle='--', linewidth=1, alpha=0.6, label='Neutral midpoint (3)')
ax.set_ylim(1, 5.3)
ax.set_ylabel('Mean Score (1–5 scale)', fontsize=11)
ax.set_title('Mean Scores of Key Scarcity Perception Variables\n(error bars = ±1 SE, n=125)',
             fontsize=12, pad=10)
ax.legend(frameon=False, fontsize=9)
for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width() / 2, m + 0.12,
            f'{m:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig('fig3_mean_scores.png', bbox_inches='tight')
plt.close()
print("[Saved] fig3_mean_scores.png")

# Figure 4: β inertia by scarcity experience
fig, ax = plt.subplots(figsize=(7, 5))
exp_map = {
    '是的，我经历过  Yes': 'Experienced\nScarcity',
    '没有 No': 'No Scarcity\nExperience',
}
df['scarcity_exp_clean'] = df['has_experienced_scarcity'].map(exp_map)
groups = df.groupby('scarcity_exp_clean')['beta_inertia'].agg(['mean', 'sem', 'count'])
colors = [PALETTE[0], PALETTE[1]]
bars = ax.bar(groups.index, groups['mean'], yerr=groups['sem'],
              color=colors, edgecolor='white', capsize=5, width=0.5)
ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
ax.set_ylabel('Mean β Inertia Coefficient', fontsize=11)
ax.set_title('β Inertia Coefficient by Ticket Scarcity Experience\n(error bars = ±1 SE)',
             fontsize=12, pad=10)
for bar, (idx, row) in zip(bars, groups.iterrows()):
    ax.text(bar.get_x() + bar.get_width() / 2, row['mean'] + 0.05,
            f'{row["mean"]:.2f}\n(n={int(row["count"])})',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# t-test
g1 = df[df['scarcity_exp_clean'] == 'Experienced\nScarcity']['beta_inertia'].dropna()
g2 = df[df['scarcity_exp_clean'] == 'No Scarcity\nExperience']['beta_inertia'].dropna()
t_stat, p_val = stats.ttest_ind(g1, g2)
sig_label = f't = {t_stat:.2f}, p = {p_val:.3f}' + (' *' if p_val < 0.05 else ' (n.s.)')
ax.text(0.5, 0.05, sig_label, transform=ax.transAxes, ha='center', fontsize=9, color='gray')
plt.tight_layout()
plt.savefig('fig4_beta_by_experience.png', bbox_inches='tight')
plt.close()
print("[Saved] fig4_beta_by_experience.png")

# Figure 5: Scarcity emotional response breakdown (Q6)
fig, ax = plt.subplots(figsize=(9, 5))
feeling_raw = df['scarcity_feeling'].value_counts()
feeling_labels = {
    '我更想得到门票了 I want to get the ticket more.': 'Want ticket more',
    '（潜在）的门票稀缺不会影响我的购票决策 (Potential) ticket shortage does not affect my purchase decision.': 'Not affected',
    '我更不想去得到门票了 It makes me less likely to buy the ticket.': 'Less likely to buy',
}
feeling_clean = feeling_raw.rename(index=feeling_labels)
colors_feeling = [PALETTE[0], PALETTE[2], PALETTE[3]]
wedges, texts, autotexts = ax.pie(
    feeling_clean.values, labels=feeling_clean.index,
    autopct='%1.1f%%', colors=colors_feeling,
    startangle=140, pctdistance=0.82,
    wedgeprops={'edgecolor': 'white', 'linewidth': 2}
)
for at in autotexts:
    at.set_fontsize(10)
    at.set_fontweight('bold')
ax.set_title('Emotional Response to Ticket Scarcity (Q6)\nn=125', fontsize=12, pad=15)
plt.tight_layout()
plt.savefig('fig5_scarcity_feeling_pie.png', bbox_inches='tight')
plt.close()
print("[Saved] fig5_scarcity_feeling_pie.png")

print("\n" + "=" * 60)
print("ALL ANALYSES COMPLETE")
print("Output files: fig1 – fig5 (PNG)")
print("=" * 60)
