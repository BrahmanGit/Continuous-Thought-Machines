# -*- coding: utf-8 -*-
"""
Creates a dual-panel figure showing:
- Top: ECDF of movement completion ratio at prediction time
- Bottom: Boxplots of completion ratio per target and k value
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "experiments"))
import utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ============================================================================
# Setup
# ============================================================================
target_names = utils.object_list[:-1]
targets = target_names
target_map = dict(zip(range(8), target_names))

lists, strings = utils.generate_finger_subsets()
ALL_FINGERS_CONDITION = strings[-1]  # '01234'

# ============================================================================
# Load unified dataframe
# ============================================================================
unified_path = '../results/latency/evaluation_fingers.csv'
all_df = pd.read_csv(unified_path, dtype={'condition': str})

finger_df = all_df[
    (all_df['experiment_type'] == 'fingers') &
    (all_df['condition'] == ALL_FINGERS_CONDITION)
].copy()

print(f"Loaded {len(finger_df)} trials (condition='{ALL_FINGERS_CONDITION}')")

# ============================================================================
# Build records
# ============================================================================
records = []

for _, row in finger_df.iterrows():
    if row['correct'] == 1:
        target = int(row['lifted_target'])

        nl1 = row['travelled_dist_lift_in_percent_nl1']
        nl2 = row['travelled_dist_lift_in_percent_nl2']
        nl3 = row['travelled_dist_lift_in_percent_nl3']

        nlc1 = max(0, row['travelled_dist_nc_in_percent_nl1'])
        nlc2 = max(0, row['travelled_dist_nc_in_percent_nl2'])
        nlc3 = max(0, row['travelled_dist_nc_in_percent_nl3'])

        base = {
            'lifted_target': target_map[target],
            'block': row['block'],
            'participant': row['participant'],
            'fingers': row['condition'],
        }

        records.append({**base, 'nl': nl1, 'nl_c': nlc1, 'type': 'k=1'})
        records.append({**base, 'nl': nl2, 'nl_c': nlc2, 'type': 'k=2'})
        records.append({**base, 'nl': nl3, 'nl_c': nlc3, 'type': 'k=3'})

plot_df = pd.DataFrame(records)
plot_df = plot_df.dropna(subset=['nl'])

# Print percentage of rows with non-negative 'nl_c' for k=1
k1_df = plot_df[plot_df['type'] == 'k=1']
percent_positive = (k1_df['nl_c'] >= 0).sum() / len(k1_df) * 100
print(f"{percent_positive:.2f}% of k=1 rows have nl_c >= 0")

# ============================================================================
# Plotting
# ============================================================================
REFERENCE_WIDTH = 8
REFERENCE_FONTSIZE = 15
fig_width = 8
FS = REFERENCE_FONTSIZE * (fig_width / REFERENCE_WIDTH)
font_size = FS

plt.rcParams.update({
    'font.size': font_size,
    'axes.titlesize': font_size,
    'axes.labelsize': font_size,
    'xtick.labelsize': font_size,
    'ytick.labelsize': font_size,
    'legend.fontsize': font_size,
    'legend.title_fontsize': font_size
})

fig, axes = plt.subplots(
    2, 1,
    figsize=(fig_width, 10),
    sharex=True,
    gridspec_kw={'height_ratios': [1.2, 2.1]}
)

# --- TOP: CDF PLOTS ---
colors = ['#2bad70', '#416897', '#afc410', '#84cfed', '#EDA985', '#96368b', '#d9328a', 'black', 'teal', 'pink']
linestyles = {'k=1': '-', 'k=2': '--', 'k=3': ':'}

for color, target in zip(colors, targets):
    for k in ['k=1']:
        data = plot_df[(plot_df['lifted_target'] == target) & (plot_df['type'] == k)]['nl'].values
        if len(data) > 0:
            sorted_data = np.sort(data)
            cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            axes[0].plot(sorted_data, cdf, color=color, linestyle=linestyles[k],
                        label=f"{target}, {k}")

target_handles = [Line2D([0], [0], color=color, linewidth=2, label=target)
                 for color, target in zip(colors, targets)]
k_handles = [Line2D([0], [0], color='black', linestyle=linestyles[k], linewidth=2, label=k)
            for k in ['k=1']]

legend1 = axes[0].legend(handles=target_handles, title="", loc='upper left',
                        fontsize=FS, title_fontsize=font_size, ncol=1, frameon=True)
axes[0].add_artist(legend1)
legend1.get_frame().set_alpha(None)
axes[0].set_ylabel("Trials anticipated", fontsize=FS)

# --- BOTTOM: BOXPLOTS ---
group_data = []
group_positions = []
position = 0
width = 0.25
ks = ['k=1', 'k=2', 'k=3'][::-1]

for t_idx, target in enumerate(targets):
    for k_idx, k in enumerate(ks):
        group_data.append(plot_df[(plot_df['lifted_target'] == target) & (plot_df['type'] == k)]['nl'])
        group_positions.append(position + k_idx * width)
    position += len(ks) * width + 0.25

bplot = axes[1].boxplot(group_data, positions=group_positions[::-1], vert=False,
                        widths=0.2, patch_artist=True, showfliers=False, zorder=10)

yticks_left = [np.mean(group_positions[i*len(ks):(i+1)*len(ks)]) for i in range(len(targets))]
axes[1].set_yticks(yticks_left[::-1])
axes[1].set_yticklabels(targets, fontsize=FS)

ax2 = axes[1].twinx()
yticks_right = group_positions
yticklabels_right = [k[-1] for _ in targets for k in ks[::-1]]
ax2.set_yticks(group_positions)
ax2.set_yticklabels(yticklabels_right, fontsize=FS)
ax2.set_ylabel("k value", fontsize=FS)
ax2.yaxis.set_ticks_position('right')
ax2.yaxis.set_label_position('right')
ax2.set_ylim(axes[1].get_ylim())

for y in yticks_right:
    axes[1].axhline(y=y, color='lightgrey', linewidth=1, zorder=-1, linestyle='-')

for patch_idx, patch in enumerate(bplot['boxes']):
    k_idx = patch_idx % len(ks)
    target_idx = patch_idx // len(ks)
    facecolor = colors[target_idx]
    patch.set_facecolor(facecolor)
    patch.set_alpha(1.0)
    for element in ['whiskers', 'caps', 'medians']:
        for line in bplot[element][2*patch_idx:2*patch_idx+2]:
            line.set_color('black')

axes[1].set_xlim([-0.01, 1.01])
axes[1].set_xlabel('Completion [ratio]', fontsize=FS)
axes[1].grid(False)
axes[0].grid(False)

axes[0].tick_params(axis='both', which='major', labelsize=FS)
axes[1].tick_params(axis='both', which='major', labelsize=FS)
ax2.tick_params(axis='both', which='major', labelsize=FS)

plt.tight_layout()
plt.savefig('Fig8_rev.pdf', bbox_inches='tight')
