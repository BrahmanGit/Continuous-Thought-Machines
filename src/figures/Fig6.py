# -*- coding: utf-8 -*-
"""
Fig6: ECDF and boxplot comparison of MAGI, CNN, LSTM, and centroid-based classifier.
Reads all model results from unified_evaluation_results.csv.

MAGI and centroid use experiment_type='fingers', condition=all-fingers.
CNN uses experiment_type='nn_cnn', LSTM uses experiment_type='nn_lstm'.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "experiments"))
import utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

target_names = utils.object_list
target_map = dict(zip(range(8), target_names))

# All-fingers condition string
lists, strings = utils.generate_finger_subsets()
ALL_FINGERS_CONDITION = strings[-1]

# ============================================================================
# Load unified dataframe
# ============================================================================

# print(f"Rows: {len(all_fingers)}")
# print(f"Participants: {sorted(all_fingers['participant'].unique())}")

unified_path = '../results/latency/unified_evaluation_results.csv'
all_df = pd.read_csv(unified_path)
all_df['condition'] = all_df['condition'].astype(str)
all_df[(all_df['participant'] == 1) & (all_df['experiment_type'] == 'fingers') & (all_df['condition'] == '01234')]
# ============================================================================
# Configuration
# ============================================================================
REFERENCE_WIDTH = 8
REFERENCE_FONTSIZE = 15
fig_width = 8
FS = REFERENCE_FONTSIZE * (fig_width / REFERENCE_WIDTH)

# Model definitions: (label, experiment_type filter, condition filter, latency_column)
# For MAGI/CNN/LSTM: latency = max(t_nl1 - t_lift, t_hand - t_lift)
# For centroid: latency = max(t_nc - t_lift, t_hand - t_lift)
model_configs = [
    ('MAGI',    'fingers',  '01234', 'nl1'),
    ('CNN',     'nn_cnn',   'all_fingers',         'nl1'),
    ('LSTM',    'nn_lstm',  'all_fingers',         'nl1'),
    ('Centroid-\nbased\nclassifier', 'fingers', '01234', 'nc'),
]
# model_configs = model_configs[:1]

label_list = [m[0] for m in model_configs]
clist = ['#5f9cc7', '#6fb744', '#00959e', '#b9348b']
linestyles = ['-', '--', '--', '-']

# ============================================================================
# Build per-model data
# ============================================================================
boxplot_data = []

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_width, 8),
                                gridspec_kw={'height_ratios': [1, 0.5]},
                                sharex=True)

bin_size = 15
quantile_steps = 0.2

for model_idx, (label, exp_type, condition, latency_col) in enumerate(model_configs):
    if model_idx in [0,3]:
        finger_path = '../results/latency/evaluation_fingers.csv'
        finger_df = pd.read_csv(finger_path)
        finger_df = pd.read_csv('../results/latency/evaluation_fingers.csv', dtype={'condition': str})
        all_df = finger_df[finger_df['condition'] == '01234']
    else:
        all_df = pd.read_csv(unified_path)
        all_df['condition'] = all_df['condition'].astype(str)
        all_df[(all_df['participant'] == 1) & (all_df['experiment_type'] == 'fingers') & (all_df['condition'] == '01234')]
        
        
    subset = all_df[
        (all_df['experiment_type'] == exp_type) &
        (all_df['condition'] == condition)
    ].copy()
    print(subset.shape)
    # Only correct trials
    correct = subset[subset['correct'] == 1].copy()

    if len(correct) == 0:
        # print(f"  {label}: no correct trials found")
        boxplot_data.append(np.array([]))
        continue

    # Compute latency
    if latency_col == 'nl1':
        correct['nl'] = np.maximum(
            correct['t_nl1'] - correct['t_lift'],
            correct['t_hand'] - correct['t_lift']
        )
    elif latency_col == 'nc':
        correct['nl'] = np.maximum(
            correct['t_nc'] - correct['t_lift'],
            correct['t_hand'] - correct['t_lift']
        )

    total_trials = len(subset)
    accuracy = len(correct) / total_trials if total_trials > 0 else 0
    negative_pct = (correct['nl'] < 0).sum() / len(correct) if len(correct) > 0 else 0

    print(f"{label}:")
    print(f"  Accuracy: {accuracy:.2f}")
    print(f"  r (fraction negative): {negative_pct:.4f}")
    print(f"  S (stability): {correct['stability_nc'].mean():.3f}")
    print(f"  Mean latency: {correct['nl'].mean():.3f}")

    data = correct['nl'].to_numpy()
    boxplot_data.append(data)

    # ECDF
    if len(data) > 0:
        bins = np.arange(np.min(data), np.max(data) + bin_size, bin_size)
        counts, bin_edges = np.histogram(data, bins=bins, density=False)
        cdf_values = np.cumsum(counts) / np.sum(counts)

        ax1.step(
            bin_edges[1:], cdf_values, where='post',
            color=clist[model_idx], linewidth=2, alpha=0.8,
            linestyle=linestyles[model_idx], label=label
        )

# ============================================================================
# Configure ECDF plot
# ============================================================================
ax1.grid(True, zorder=-1)
ax1.set_ylabel('Trials anticipated', fontsize=FS)
xmin, xmax = -2000, 1500
ax1.set_xlim(xmin, xmax)

quantiles = np.arange(0, 1 + quantile_steps, quantile_steps)
ax1.set_ylim(0, 1)
ax1.set_yticks(np.round(quantiles, 1))
ax1.set_yticklabels(np.round(quantiles, 1), fontsize=FS)

ax1.axvline(x=0, color='grey', linewidth=3, zorder=-1)
ax1.legend(loc='lower right', fontsize=FS, title_fontsize=FS, ncol=1)

# ============================================================================
# Horizontal boxplot
# ============================================================================
sns.set(style='whitegrid')

positions = np.arange(len(boxplot_data))
# Filter out empty arrays for boxplot
valid_data = [d if len(d) > 0 else np.array([0]) for d in boxplot_data]

bp = ax2.boxplot(valid_data, vert=False, positions=positions, widths=0.4,
                 patch_artist=True, showfliers=True,
                 flierprops=dict(marker='o', markersize=3, alpha=0.3))

for patch, color in zip(bp['boxes'], clist):
    patch.set_facecolor(color)
    patch.set_alpha(1)

for median in bp['medians']:
    median.set_color('black')
    median.set_linewidth(2)

ax2.axvline(x=0, color='grey', linewidth=3, zorder=-1)
ax2.grid(True, axis='x', zorder=-1)
ax2.set_yticks(positions)
ax2.set_yticklabels(label_list, fontsize=FS)
ax2.set_xlabel(r'Latency [ms]', fontsize=FS)
ax2.invert_yaxis()

plt.tight_layout()
plt.savefig('Fig6_rev.pdf', bbox_inches='tight')

# ============================================================================
# Paired t-test: MAGI vs centroid-based classifier
# ============================================================================
from scipy import stats

# Load finger data once
finger_df = pd.read_csv('../results/latency/evaluation_fingers.csv', dtype={'condition': str})
finger_all = finger_df[
    (finger_df['condition'] == '01234') &
    (finger_df['experiment_type'] == 'fingers')
].copy()

# Only correct trials
correct_all = finger_all[finger_all['correct'] == 1].copy()

# Compute latencies
correct_all['nl_magi'] = np.maximum(
    correct_all['t_nl1'] - correct_all['t_lift'],
    correct_all['t_hand'] - correct_all['t_lift']
)
correct_all['nl_centroid'] = np.maximum(
    correct_all['t_nc'] - correct_all['t_lift'],
    correct_all['t_hand'] - correct_all['t_lift']
)

# Both are computed on the same rows, so they're naturally paired
magi_nl = correct_all['nl_magi'].values
centroid_nl = correct_all['nl_centroid'].values

t_stat, p_value = stats.ttest_rel(magi_nl, centroid_nl)
differences = magi_nl - centroid_nl
cohens_d = np.mean(differences) / np.std(differences, ddof=1)

print("\n" + "=" * 80)
print("PAIRED T-TEST: MAGI vs Centroid-based classifier")
print("=" * 80)
print(f"  N paired samples:     {len(magi_nl)}")
print(f"  MAGI mean latency:    {np.mean(magi_nl):.3f} ms (std={np.std(magi_nl, ddof=1):.3f})")
print(f"  Centroid mean latency:{np.mean(centroid_nl):.3f} ms (std={np.std(centroid_nl, ddof=1):.3f})")
print(f"  Mean difference:      {np.mean(differences):.3f} ms")
print(f"  t({len(magi_nl)-1}) = {t_stat:.2f}, p = {p_value:.6f}")
print(f"  Cohen's d:            {cohens_d:.4f}")
print(f"  Significant (p<0.05): {'YES' if p_value < 0.05 else 'NO'}")
print("=" * 80)
