# -*- coding: utf-8 -*-
"""
@author: jonas

Script for analyzing model accuracy, F1 score, and negative latency (nl) 
as a function of distance to target gesture.

The results will be stored in a dataframe and saved to store_results_path.
Relies on utilities defined in `utils.py`.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "experiments"))
import utils
import models

# ============================================================================
# ADJUSTABLE PARAMETERS
# ============================================================================
MIN_SAMPLES_PER_BIN = 5
MAX_DISTANCE = 0.15  # Set to a number to limit x-axis, or None for no limit
N_BINS = 30
N_BOOTSTRAP = 2000
S_BOOTSTRAP = 2000
F1_AVERAGE = "binary"

# ============================================================================
# Setup
# ============================================================================
mean_data_columns = [f'mean_pred_{i}' for i in range(9)]
var_data_columns = [f'var_pred_{i}' for i in range(9)]
nc_pred_data_columns = [f'nc_pred_{i}' for i in range(9)]

lists, strings = utils.generate_finger_subsets()
ALL_FINGERS_CONDITION = strings[-1]  # '01234'

grasp_data_path = "../results/grasping_data/"
pred_fingers_path = '../results/predictions/fingers/'
participant_data_path = "../../data/dataframes/"
store_results_path = '../results/distance_accuracy/'
latency_results_path = '../results/latency/'

target_names = utils.object_list[:-1]
target_map = dict(zip(range(8), target_names))

# ============================================================================
# Load latency data from unified evaluation CSV
# ============================================================================
unified_path = latency_results_path + 'evaluation_fingers.csv'
latency_all_df = pd.read_csv(unified_path, dtype={'condition': str})

latency_all_df = latency_all_df[
    (latency_all_df['experiment_type'] == 'fingers') &
    (latency_all_df['condition'] == ALL_FINGERS_CONDITION)
].copy()

print(f"Loaded {len(latency_all_df)} latency records (condition='{ALL_FINGERS_CONDITION}')")

PARTICIPANT_NAMES = [
    'participant_1', 'participant_2', 'participant_3',
    'participant_4', 'participant_5'
]
mov_dur = [68.6875, 81.07936507936508, 78.44444444444444, 100.40625, 91.078125]

# ============================================================================
# Main processing loop
# ============================================================================
all_data = []

for name_c, name in enumerate(PARTICIPANT_NAMES):
    print(f"Processing participant: {name}")

    finger_combination = strings[-1]
    finger_config = lists[-1]

    df = utils.load_dataframe_from_csv(participant_data_path + "processed_" + name + ".csv")

    grasp_df = utils.load_dataframe_from_csv(grasp_data_path + name + '_Xy_movement.csv')
    static_grasp_df = utils.load_dataframe_from_csv(grasp_data_path + name + '_Xy.csv')
    X_static = static_grasp_df[utils.glove_data_columns].to_numpy()
    X_static = utils.preprocess_data_finger_config(X_static, finger_config=finger_config)
    y_static = static_grasp_df['class'].to_numpy()

    nc = models.NearestCentroidClassifier()
    nc.integration_window = 10
    nc.fit(X_static, y_static)

    pred_df = utils.load_dataframe_from_csv(
        pred_fingers_path + f'{name}_fingers_{finger_combination}.csv'
    )

    participant_latency = latency_all_df[latency_all_df['participant'] == name_c]

    for phase in np.arange(1, pred_df['block'].max() + 1):
        for trial_number in np.arange(pred_df[pred_df['block'] == phase]['trial_number'].max() + 1):
            for rep_counter in np.arange(pred_df[(pred_df['block'] == phase) &
                                                  (pred_df['trial_number'] == trial_number)]['rep_number'].max() + 1):

                pred_df_att = pred_df[(pred_df['block'] == phase) &
                                      (pred_df['trial_number'] == trial_number) &
                                      (pred_df['rep_number'] == rep_counter)]

                df_att = df[(df['block'] == phase) &
                            (df['trial_number'] == trial_number) &
                            (df['rep_number'] == rep_counter) &
                            (df['trial_success'] == 1)]

                if len(df_att) == 0:
                    continue

                glove_data = df_att[utils.glove_data_columns].to_numpy()
                glove_time = df_att['glove_timestamps'].to_numpy()
                board_data = df_att[utils.board_data_columns].to_numpy()
                board_data = utils.replace_with_highest(arr=board_data, target=8191.0)
                board_data = utils.replace_with_highest(arr=board_data, target=8190.0)

                palm_data = df_att['palm_data'].to_numpy()
                palm_data = utils.replace_with_highest(arr=palm_data, target=8191.0)
                palm_data = utils.replace_with_highest(arr=palm_data, target=8190.0)
                palm_data = palm_data - palm_data[0]

                y_true = int(df_att['target_object'].to_numpy()[-1])
                means = pred_df_att[mean_data_columns].to_numpy()
                nc_pred = pred_df_att[nc_pred_data_columns].to_numpy()

                start = 0
                end = np.argmax(board_data[:, int(trial_number)])
                board_ts = board_data[start:end, int(trial_number)]
                board_ts = board_ts - board_ts[0]
                t_lift = utils.first_change_index(time_series=board_ts, threshold=board_ts.max() * 0.1)
                hand_lift = utils.first_change_index(time_series=palm_data, threshold=palm_data.max() * 0.1)

                mov_data = utils.preprocess_data_reduction(glove_data, 5)

                latency_trial = participant_latency[
                    (participant_latency['block'] == phase) &
                    (participant_latency['trial_number'] == trial_number) &
                    (participant_latency['rep_number'] == rep_counter)
                ]

                if len(latency_trial) > 0:
                    nl1_trial = latency_trial.iloc[0]['t_nl1']
                else:
                    nl1_trial = np.nan

                for t in range(hand_lift, t_lift + 0):
                    idx = np.argmax(means[t])
                    if idx != 8:
                        dist = utils.compute_distance_travelled(mov_data[t:t+1], target=nc.centroids[idx])[0]
                        if glove_time[t] < nl1_trial:
                            neglat1 = np.nan
                        else:
                            neglat1 = glove_time[t] - glove_time[t_lift]
                        one_hot = np.zeros_like(means[t])
                        one_hot[idx] = 1
                        one_hot_true = np.zeros_like(means[t])
                        one_hot_true[y_true] = 1
                        pred_val = 1 if one_hot[idx] == 1 and one_hot_true[idx] == 1 else \
                                  (0 if one_hot[idx] == 0 and one_hot_true[idx] == 1 else \
                                  (1 if one_hot[idx] == 1 and one_hot_true[idx] == 0 else 0))
                        true_val = 1 if one_hot_true[idx] == 1 else 0
                        all_data.append([dist, pred_val, true_val, neglat1])

data = np.array(all_data)

# ============================================================================
# Apply distance limit
# ============================================================================
distances = data[:, 0]
y_pred = data[:, 1]
y_true = data[:, 2]
nl1_data = data[:, 3]

if MAX_DISTANCE is not None:
    mask = distances <= MAX_DISTANCE
    distances = distances[mask]
    y_pred = y_pred[mask]
    y_true = y_true[mask]
    nl1_data = nl1_data[mask]
    print(f"Distance limit applied: {MAX_DISTANCE}, {mask.sum()}/{len(mask)} samples retained")

# ============================================================================
# Binning and bootstrap
# ============================================================================
bin_edges = np.linspace(distances.min(), distances.max(), N_BINS + 1)
bin_indices = np.digitize(distances, bin_edges) - 1
bin_indices = np.clip(bin_indices, 0, N_BINS - 1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

bin_f1 = np.full(N_BINS, np.nan)
bin_acc = np.full(N_BINS, np.nan)
bin_nl1 = np.full(N_BINS, np.nan)
bin_f1_std = np.full(N_BINS, np.nan)
bin_acc_std = np.full(N_BINS, np.nan)
bin_nl1_std = np.full(N_BINS, np.nan)
bin_counts = np.zeros(N_BINS, dtype=int)

for b in range(N_BINS):
    idx = bin_indices == b
    bin_counts[b] = np.sum(idx)

    # Skip bins with fewer than minimum samples
    if bin_counts[b] < MIN_SAMPLES_PER_BIN:
        continue

    y_true_bin = y_true[idx]
    y_pred_bin = y_pred[idx]
    nl1_bin = nl1_data[idx]

    if not np.isnan(nl1_bin).all():
        # Bootstrap for F1
        f1_scores = []
        for _ in range(N_BOOTSTRAP):
            y_true_resampled, y_pred_resampled = resample(y_true_bin, y_pred_bin,
                                                           random_state=np.random.randint(1, S_BOOTSTRAP))
            f1_scores.append(f1_score(y_true_resampled, y_pred_resampled, average=F1_AVERAGE))
        bin_f1[b] = np.mean(f1_scores)
        bin_f1_std[b] = np.std(f1_scores)

        # Bootstrap for accuracy
        acc_scores = []
        for _ in range(N_BOOTSTRAP):
            y_true_resampled, y_pred_resampled = resample(y_true_bin, y_pred_bin,
                                                           random_state=np.random.randint(1, S_BOOTSTRAP))
            acc_scores.append(accuracy_score(y_true_resampled, y_pred_resampled))
        bin_acc[b] = np.mean(acc_scores)
        bin_acc_std[b] = np.std(acc_scores)

        # Bootstrap for negative latency
        valid_nl1 = nl1_bin[~np.isnan(nl1_bin)]
        if len(valid_nl1) >= MIN_SAMPLES_PER_BIN:
            nl1_samples = []
            for _ in range(N_BOOTSTRAP):
                nl1_resampled = resample(valid_nl1, random_state=np.random.randint(1, S_BOOTSTRAP))
                nl1_samples.append(np.mean(nl1_resampled))
            bin_nl1[b] = np.mean(nl1_samples)
            bin_nl1_std[b] = np.std(nl1_samples)

# ============================================================================
# Save results
# ============================================================================
os.makedirs(store_results_path, exist_ok=True)

df_distance_metrics = pd.DataFrame({
    "bin_center": bin_centers,
    "bin_start": bin_edges[:-1],
    "bin_end": bin_edges[1:],
    "f1_score": bin_f1,
    "f1_std": bin_f1_std,
    "accuracy": bin_acc,
    "accuracy_std": bin_acc_std,
    "nl_k1": bin_nl1,
    "nl_k1_std": bin_nl1_std,
    "n_samples": bin_counts
})

df_distance_metrics.to_csv(store_results_path + 'distance_metrics_with_latency.csv', index=False)
print(f"Results saved to {store_results_path}distance_metrics_with_latency.csv")

# ============================================================================
# Plotting — only show bins with enough samples
# ============================================================================
fig_width = 8
REFERENCE_WIDTH = fig_width
REFERENCE_FONTSIZE = 15
FS = REFERENCE_FONTSIZE * (fig_width / REFERENCE_WIDTH)

# Mask for valid bins
valid_mask = bin_counts >= MIN_SAMPLES_PER_BIN

fig, (ax_metrics, ax_hist) = plt.subplots(
    2, 1,
    figsize=(fig_width, 5),
    sharex=True,
    gridspec_kw={"height_ratios": [3.3, 1]}
)

ax_metrics.errorbar(
    bin_centers[valid_mask], bin_f1[valid_mask],
    yerr=bin_f1_std[valid_mask],
    marker='.', label='F1', capsize=3, alpha=0.8, color='C0'
)
ax_metrics.errorbar(
    bin_centers[valid_mask], bin_acc[valid_mask],
    yerr=bin_acc_std[valid_mask],
    marker='.', label=r'$\text{Acc}$', capsize=3, alpha=0.8, color='C1'
)

ax_metrics.set_ylabel("Score", fontsize=FS)
ax_metrics.set_ylim(0, 1.05)
ax_metrics.grid(True)

ax_nl = ax_metrics.twinx()

# Only plot latency for bins with valid latency AND enough samples
nl_valid = valid_mask & ~np.isnan(bin_nl1)
ax_nl.errorbar(
    bin_centers[nl_valid], bin_nl1[nl_valid],
    yerr=bin_nl1_std[nl_valid],
    marker='.', label='Latency [ms]', capsize=3, alpha=0.8, color='C2'
)
ax_nl.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=1)
ax_nl.set_ylabel("Latency [ms]", fontsize=FS)

lines1, labels1 = ax_metrics.get_legend_handles_labels()
lines2, labels2 = ax_nl.get_legend_handles_labels()
ax_metrics.legend(lines1 + lines2, labels1 + labels2, fontsize=FS, loc='lower left')

ax_hist.bar(
    bin_centers,
    bin_counts,
    width=np.diff(bin_edges),
    align="center",
    alpha=1.0
)

# Draw minimum sample threshold line
ax_hist.axhline(y=MIN_SAMPLES_PER_BIN, color='red', linestyle='--', alpha=0.5,
                linewidth=1, label=f'min = {MIN_SAMPLES_PER_BIN}')

ax_hist.set_xlabel("Geometric distance [cm]", fontsize=FS)
ax_hist.set_ylabel("Samples", fontsize=FS)
ax_hist.grid(True)

for ax in [ax_metrics, ax_nl, ax_hist]:
    for tick in ax.get_yticklabels():
        tick.set_fontsize(FS)
    for tick in ax.get_xticklabels():
        tick.set_fontsize(FS)

# Auto-scale histogram y-ticks
max_count = bin_counts.max()
if max_count > 5000:
    ax_hist.set_yticks([0, 2000, 4000, 6000])
    ax_hist.set_yticklabels(['0', '2k', '4k', '6k'])
elif max_count > 2000:
    ax_hist.set_yticks([0, 2000, 4000])
    ax_hist.set_yticklabels(['0', '2k', '4k'])
else:
    ax_hist.set_yticks([0, 1000, 2000])
    ax_hist.set_yticklabels(['0', '1k', '2k'])

if MAX_DISTANCE is not None:
    ax_hist.set_xlim(0, MAX_DISTANCE)

plt.tight_layout()
plt.savefig('Fig9_rev.pdf', bbox_inches="tight")
print(f"Figure saved to Fig9_rev.pdf")
print(f"Bins with >= {MIN_SAMPLES_PER_BIN} samples: {valid_mask.sum()} / {N_BINS}")