# -*- coding: utf-8 -*-
"""
Combined analysis script for metrics across different numbers of objects and noise levels.
Computes accuracy, prediction rate, stability, and average latency.
Generates a dual-subplot figure with constant offsets and error bars.
Second subplot shows SNR with reference noise level.

Reads from unified_evaluation_results.csv produced by run_all_experiments.py.

NOTE: Object-count analysis is restricted to conditions that include object 0.
"""

import os
import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "experiments"))
import utils
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

target_names = utils.object_list
target_map = dict(zip(range(8), target_names))

# ============================================================================
# Load unified evaluation results
# ============================================================================
unified_path = '../results/latency/unified_evaluation_results.csv'
all_df = pd.read_csv(unified_path)

# ========== FIRST ANALYSIS: NUMBER OF OBJECTS ==========
objects_df = all_df[all_df['experiment_type'] == 'objects'].copy()

# Only keep conditions that include object 0
objects_df = objects_df[objects_df['condition'].apply(
    lambda x: '8' in str(x).split('_')
)]

# Derive n_objects from condition string if not present
if 'n_objects' not in objects_df.columns or objects_df['n_objects'].isna().all():
    objects_df['n_objects'] = objects_df['condition'].apply(lambda x: len(str(x).split('_')))

stats_objects = []

for num_objects in sorted(objects_df['n_objects'].unique()):
    subset = objects_df[objects_df['n_objects'] == num_objects]
    counter = len(subset)
    print(subset.shape)
    if counter == 0:
        continue

    all_correct = subset['correct'].values
    correct_subset = subset[subset['correct'] == 1].copy()

    correct_subset['nl'] = np.maximum(
        correct_subset['t_nl1'] - correct_subset['t_lift'],
        correct_subset['t_hand'] - correct_subset['t_lift']
    )
    
    accuracy = len(correct_subset) / counter
    total_samples = len(correct_subset)

    if total_samples > 0:
        negative_nl_samples = (correct_subset['nl'] < 0).sum()
        percentage_negative = negative_nl_samples / total_samples

        acc_se = stats.sem(all_correct) if len(all_correct) > 0 else 0
        r_array = (correct_subset['nl'] < 0).astype(int).values
        r_se = stats.sem(r_array) if len(r_array) > 0 else 0
        stab_se = stats.sem(correct_subset['stability_pred']) if len(correct_subset) > 0 else 0
        nl_se = stats.sem(correct_subset['nl']) if len(correct_subset) > 0 else 0

        stats_objects.append({
            'num_objects': num_objects - 1,
            'Acc': accuracy,
            'Acc_se': acc_se,
            'r': percentage_negative,
            'r_se': r_se,
            'e_nl': correct_subset['nl'].mean(),
            'e_nl_se': nl_se,
            'stab': correct_subset['stability_pred'].mean(),
            'stab_se': stab_se,
        })

df_objects = pd.DataFrame(stats_objects)

# ========== SECOND ANALYSIS: NOISE LEVELS ==========
signal = []
grasp_data_path = "../results/grasping_data/"
participant_data_path = "../../data/dataframes/"

for name_c, name in enumerate(['participant_1', 'participant_2', 'participant_3',
                                'participant_4', 'participant_5']):
    grasp_df = utils.load_dataframe_from_csv(grasp_data_path + name + '_Xy.csv')
    X = grasp_df[utils.glove_data_columns].to_numpy()
    y = grasp_df['class'].to_numpy()

    lists, strings = utils.generate_finger_subsets()

    Xf = utils.preprocess_data_finger_config(glove_data_subset=X, finger_config=[0, 1, 2, 3, 4])
    df_filename = participant_data_path + 'processed_' + name + '.csv'
    df = utils.load_dataframe_from_csv(df_filename)

    for phase in range(1, df['block'].max() + 1):
        for trial_number in np.arange(df[df['block'] == phase]['trial_number'].max() + 1):
            for rep_counter in range(df[(df['block'] == phase) &
                                        (df['trial_number'] == trial_number)]['rep_number'].max() + 1):
                df_att = df[(df['block'] == phase) &
                            (df['trial_number'] == trial_number) &
                            (df['rep_number'] == rep_counter)]

                glove_time = df_att['glove_timestamps'].to_numpy()
                glove_data = df_att[utils.glove_data_columns].to_numpy()

                board_data = df_att[utils.board_data_columns].to_numpy()
                board_data = utils.replace_with_highest(arr=board_data, target=8191.0)
                board_data = utils.replace_with_highest(arr=board_data, target=8190.0)

                palm_data = df_att['palm_data'].to_numpy()
                palm_data = utils.replace_with_highest(arr=palm_data, target=8191.0)
                palm_data = utils.replace_with_highest(arr=palm_data, target=8190.0)
                palm_data = palm_data - palm_data[0]

                start = 0
                end = -1

                board_ts = board_data[start:end, int(trial_number)]
                board_ts = board_ts - board_ts[0]

                t_lift = utils.first_change_index(time_series=board_ts, threshold=board_ts.max() * 0.1)
                hand_lift = utils.first_change_index(time_series=palm_data, threshold=palm_data.max() * 0.1)

                board_ts_flip = board_ts[::-1] - board_ts[end]
                t_down = len(board_ts) - utils.first_change_index(time_series=board_ts_flip, threshold=board_ts.max() * 0.1)
                if t_down <= t_lift:
                    t_down = len(board_ts) - 1

                glove_segment = glove_data[hand_lift:t_lift]
                glove_segment = utils.preprocess_data_finger_config(
                    glove_data_subset=glove_segment, finger_config=[0, 1, 2, 3, 4])
                for x in glove_segment:
                    signal.append(x)

signal = np.array(signal)
reference_noise = np.mean(np.std(signal, axis=0))
print(f"Reference signal std: {reference_noise}")

noise_df = all_df[all_df['experiment_type'] == 'noise'].copy()
noise_df['noise_lvl'] = noise_df['condition'].astype(float)
noise_lvls = sorted([n for n in noise_df['noise_lvl'].unique() if n > 0])

stats_noise = []

for noise_lvl in noise_lvls:
    subset = noise_df[noise_df['noise_lvl'] == noise_lvl]
    counter = len(subset)

    if counter == 0:
        continue

    all_correct = subset['correct'].values
    correct_subset = subset[subset['correct'] == 1].copy()

    correct_subset['nl'] = np.maximum(
        correct_subset['t_nl1'] - correct_subset['t_lift'],
        correct_subset['t_hand'] - correct_subset['t_lift']
    )

    accuracy = len(correct_subset) / counter
    total_samples = len(correct_subset)

    if total_samples > 0:
        negative_nl_samples = (correct_subset['nl'] < 0).sum()
        percentage_negative = negative_nl_samples / total_samples

        acc_se = stats.sem(all_correct) if len(all_correct) > 0 else 0
        r_array = (correct_subset['nl'] < 0).astype(int).values
        r_se = stats.sem(r_array) if len(r_array) > 0 else 0
        stab_se = stats.sem(correct_subset['stability_pred']) if len(correct_subset) > 0 else 0
        nl_se = stats.sem(correct_subset['nl']) if len(correct_subset) > 0 else 0

        snr_db = 10 * np.log10(reference_noise / noise_lvl) if noise_lvl > 0 else np.inf

        stats_noise.append({
            'noise_lvl': noise_lvl,
            'snr_db': snr_db,
            'Acc': accuracy,
            'Acc_se': acc_se,
            'r': percentage_negative,
            'r_se': r_se,
            'e_nl': correct_subset['nl'].mean(),
            'e_nl_se': nl_se,
            'stab': correct_subset['stability_pred'].mean(),
            'stab_se': stab_se,
        })

df_noise = pd.DataFrame(stats_noise)

# ========== CREATE COMBINED VISUALIZATION ==========
REFERENCE_WIDTH = 8
REFERENCE_FONTSIZE = 15
fig_width = 8
FS = REFERENCE_FONTSIZE * (fig_width / REFERENCE_WIDTH)
sns.set(style='whitegrid')

fig, (ax1, ax3) = plt.subplots(2, figsize=(fig_width, 7))

marker_symbol = '.'
ms = 40

offset_acc = -0.3
offset_r = -0.1
offset_stab = 0.1
offset_latency = 0.3

c1 = '#FAA43F'
c2 = '#AE1DC4'
c3 = '#0A099E'
c4 = '#353B43'

# ========== TOP SUBPLOT: NUMBER OF OBJECTS ==========
x_objects = df_objects['num_objects'].values

ax1.errorbar(x_objects + offset_acc, df_objects['Acc'], yerr=df_objects['Acc_se'],
             fmt=marker_symbol, markersize=np.sqrt(ms), label=r'$Acc$', color=c1,
             capsize=5, capthick=2, zorder=3)
ax1.errorbar(x_objects + offset_r, df_objects['r'], yerr=df_objects['r_se'],
             fmt=marker_symbol, markersize=np.sqrt(ms), label=r'$r$', color=c2,
             capsize=5, capthick=2, zorder=3)
ax1.errorbar(x_objects + offset_stab, df_objects['stab'], yerr=df_objects['stab_se'],
             fmt=marker_symbol, markersize=np.sqrt(ms * 1.3), label=r'$S$', color=c3,
             capsize=5, capthick=2, zorder=3)

ax1.set_xticks(df_objects['num_objects'])
ax1.set_xticklabels([f"{int(x)}" for x in df_objects['num_objects']], fontsize=FS)
ax1.set_xlabel('Number of Objects', fontsize=FS)
ax1.set_ylabel(r'$\text{Acc}, r, S$', fontsize=FS)

ax2 = ax1.twinx()
ax2.errorbar(x_objects + offset_latency, df_objects['e_nl'], yerr=df_objects['e_nl_se'],
             fmt=marker_symbol, markersize=np.sqrt(ms), color=c4, label='Latency',
             capsize=5, capthick=2, zorder=3)
ax2.set_ylabel('Average latency [ms]', fontsize=FS)

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax3.legend(lines_1 + lines_2, labels_1 + labels_2,
           bbox_to_anchor=(0.5, -0.55), loc='lower center', ncol=4, fontsize=FS)

ax1.tick_params(axis='y', labelsize=FS)
ax2.tick_params(axis='y', labelsize=FS)
ax1.grid(False)
ax2.grid(False)

for num_obj in df_objects['num_objects']:
    ax1.axvline(x=num_obj, zorder=-1, color='lightgrey', linewidth=0.5)

# ========== BOTTOM SUBPLOT: SNR ==========
x_snr = df_noise['snr_db'].values
finite_mask = np.isfinite(x_snr)
df_noise_finite = df_noise[finite_mask].copy()
x_snr_finite = x_snr[finite_mask]
n_points = len(df_noise_finite)

ax3.set_ylabel(r'$\text{Acc}, r, S$', fontsize=FS)
ax3.errorbar(np.arange(n_points) + offset_acc, df_noise_finite['Acc'], yerr=df_noise_finite['Acc_se'],
             fmt=marker_symbol, markersize=np.sqrt(ms), label=r'$Acc$', color=c1,
             capsize=5, capthick=2, zorder=3)
ax3.errorbar(np.arange(n_points) + offset_r, df_noise_finite['r'], yerr=df_noise_finite['r_se'],
             fmt=marker_symbol, markersize=np.sqrt(ms), label=r'$r$', color=c2,
             capsize=5, capthick=2, zorder=3)
ax3.errorbar(np.arange(n_points) + offset_stab, df_noise_finite['stab'], yerr=df_noise_finite['stab_se'],
             fmt=marker_symbol, markersize=np.sqrt(ms * 1.3), label=r'$S$', color=c3,
             capsize=5, capthick=2, zorder=3)

ax3.set_xticks(np.arange(n_points))
ax3.set_xlabel('SNR [dB]', fontsize=FS)
ax3.invert_xaxis()

ax4 = ax3.twinx()
ax4.errorbar(np.arange(n_points) + offset_latency, df_noise_finite['e_nl'], yerr=df_noise_finite['e_nl_se'],
             fmt=marker_symbol, markersize=np.sqrt(ms), color=c4, label='Latency',
             capsize=5, capthick=2, zorder=3)
ax4.set_ylabel('Average latency [ms]', fontsize=FS)

ax3.tick_params(axis='y', labelsize=FS)
ax4.tick_params(axis='y', labelsize=FS)
ax3.grid(False)
ax4.grid(False)

for xtick in range(n_points):
    ax3.axvline(x=xtick, zorder=-1, color='lightgrey', linewidth=0.5)

ax3.set_xticks(np.arange(n_points))
ax3.set_xticklabels([f"{int(x)}" for x in x_snr_finite], fontsize=FS)

plt.tight_layout()
print("\nSNR Values:")
print(df_noise[['noise_lvl', 'snr_db']])
plt.savefig('Fig11_rev.pdf', bbox_inches="tight")