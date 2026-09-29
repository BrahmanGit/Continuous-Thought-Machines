# -*- coding: utf-8 -*-
"""
Creates stacked horizontal bar charts at different normalized time points (0.3, 0.6, 0.9, 1.0)
to visualize how prediction distributions for each true object class evolve during movement.

Each bar represents a true object class, with stacked segments showing the proportion
of predictions made for each class (colored according to predicted class).
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "experiments"))
import utils
import models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# ============================================================================
# ADJUSTABLE PARAMETER: Change this value to control all font sizes in the figure
# ============================================================================
FONT_SIZE = 15

# Define column names for prediction data
mean_data_columns = [f'mean_pred_{i}' for i in range(9)]
var_data_columns = [f'var_pred_{i}' for i in range(9)]
nc_pred_data_columns = [f'nc_pred_{i}' for i in range(9)]

# Generate finger subset configurations
lists, strings = utils.generate_finger_subsets()

# Define file paths
grasp_data_path = "../results/grasping_data/"
pred_fingers_path = '../results/predictions/fingers/'
participant_data_path = "../../data/dataframes/"

# Load target names from utils, excluding the last element
target_names = utils.object_list[:-1]
target_map = dict(zip(range(8), target_names))

PARTICIPANT_NAMES = [
    'participant_1', 'participant_2', 'participant_3',
    'participant_4', 'participant_5'
]

# Define normalized time points to analyze (as ratios from hand_lift to t_lift)
time_points = [0.3, 0.6, 0.9, 1.0]

# Collect confusion matrix data for each timepoint
confusion_data = {tp: {'y_true': [], 'y_pred': []} for tp in time_points}

# Main processing loop
for name_c, name in enumerate(PARTICIPANT_NAMES):
    print(f"Processing participant: {name}")

    finger_combination = strings[-1]  # '01234'
    finger_config = lists[-1]         # [0, 1, 2, 3, 4]

    # Load raw participant data
    df = utils.load_dataframe_from_csv(participant_data_path + "processed_" + name + ".csv")

    # Load grasp data for nearest centroid classifier
    grasp_df = utils.load_dataframe_from_csv(grasp_data_path + name + '_Xy_movement.csv')
    static_grasp_df = utils.load_dataframe_from_csv(grasp_data_path + name + '_Xy.csv')
    X_static = static_grasp_df[utils.glove_data_columns].to_numpy()
    X_static = utils.preprocess_data_finger_config(X_static, finger_config=finger_config)
    y_static = static_grasp_df['class'].to_numpy()

    # Fit nearest centroid classifier
    nc = models.NearestCentroidClassifier()
    nc.integration_window = 10
    nc.fit(X_static, y_static)

    # Load prediction file (unified pipeline format)
    pred_df = utils.load_dataframe_from_csv(
        pred_fingers_path + f'{name}_fingers_{finger_combination}.csv'
    )

    # Loop through experimental blocks/phases
    for phase in np.arange(1, pred_df['block'].max() + 1):
        for trial_number in np.arange(pred_df[pred_df['block'] == phase]['trial_number'].max() + 1):
            for rep_counter in np.arange(pred_df[(pred_df['block'] == phase) &
                                                  (pred_df['trial_number'] == trial_number)]['rep_number'].max() + 1):

                # Filter prediction data for current trial
                pred_df_att = pred_df[(pred_df['block'] == phase) &
                                      (pred_df['trial_number'] == trial_number) &
                                      (pred_df['rep_number'] == rep_counter)]

                # Filter raw data for current trial (only successful trials)
                df_att = df[(df['block'] == phase) &
                            (df['trial_number'] == trial_number) &
                            (df['rep_number'] == rep_counter) &
                            (df['trial_success'] == 1)]

                if len(df_att) == 0:
                    continue

                # Extract data
                glove_data = df_att[utils.glove_data_columns].to_numpy()
                board_data = df_att[utils.board_data_columns].to_numpy()
                board_data = utils.replace_with_highest(arr=board_data, target=8191.0)
                board_data = utils.replace_with_highest(arr=board_data, target=8190.0)

                # Extract palm sensor data
                palm_data = df_att['palm_data'].to_numpy()
                palm_data = utils.replace_with_highest(arr=palm_data, target=8191.0)
                palm_data = utils.replace_with_highest(arr=palm_data, target=8190.0)
                palm_data = palm_data - palm_data[0]

                # Get true target object
                y_true = int(df_att['target_object'].to_numpy()[-1])

                # Extract predictions (mean probabilities for each class)
                means = pred_df_att[mean_data_columns].to_numpy()

                # Find movement start and end points
                start = 0
                end = np.argmax(board_data[:, int(trial_number)])
                board_ts = board_data[start:end, int(trial_number)]
                board_ts = board_ts - board_ts[0]
                t_lift = utils.first_change_index(time_series=board_ts,
                                                  threshold=board_ts.max() * 0.1)
                hand_lift = utils.first_change_index(time_series=palm_data,
                                                    threshold=palm_data.max() * 0.1)

                # Calculate movement duration
                movement_duration = t_lift - hand_lift

                if movement_duration <= 0:
                    continue

                # For each normalized time point
                for tp in time_points:
                    # Calculate actual time index
                    t_normalized = hand_lift + int(tp * movement_duration)

                    # Make sure we don't exceed bounds
                    if t_normalized >= len(means) or t_normalized >= t_lift:
                        t_normalized = min(len(means) - 1, t_lift - 1)

                    if t_normalized < hand_lift:
                        continue

                    # Get prediction at this time point (argmax of mean predictions)
                    y_pred = np.argmax(means[t_normalized])

                    # Only include if prediction is not the "no object" class (8)
                    if y_pred != 8:
                        confusion_data[tp]['y_true'].append(y_true)
                        confusion_data[tp]['y_pred'].append(y_pred)

print("\nData collection complete. Creating stacked bar charts...")

# Define colors for each object class
colors = utils.colors

# Create figure with subplots for each timepoint (all side-by-side)
fig, axes = plt.subplots(4, figsize=(6, 9), sharey=False, sharex=True)

# Plot stacked bar chart for each timepoint
for idx, tp in enumerate(time_points):
    ax = axes[idx]

    y_true = np.array(confusion_data[tp]['y_true'])
    y_pred = np.array(confusion_data[tp]['y_pred'])

    if len(y_true) > 0:
        # Create confusion matrix to get counts
        cm = confusion_matrix(y_true, y_pred, labels=range(8))

        # Normalize by row (true class) to get proportions
        cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)

        y_positions = np.arange(8) * -1

        # Plot each predicted class as a segment in the stacked bar
        left = np.zeros(8)

        for pred_class in range(8):
            widths = cm_normalized[:, pred_class]
            ax.barh(y_positions, widths, left=left,
                   color=colors[pred_class],
                   label=target_names[pred_class],
                   edgecolor='white', linewidth=0.5)
            left += widths
            
        # Add per-object accuracy at the end of each bar
        for obj_idx in range(8):
            row_total = cm.sum(axis=1)[obj_idx]
            if row_total > 0:
                obj_accuracy = cm[obj_idx, obj_idx] / row_total
                ax.text(1.02, y_positions[obj_idx], f'{obj_accuracy:.2f}',
                        va='center', ha='left', fontsize=FONT_SIZE - 3)
                
        # Customize plot
        ax.set_yticks(y_positions)
        ax.set_yticklabels(target_names, fontsize=FONT_SIZE)
        # ax.set_xlabel('Proportion', fontsize=FONT_SIZE)
        ax.set_xlim([0, 1])
        ax.set_title(f'Completion: {tp}', fontsize=FONT_SIZE)

        # Add grid for easier reading
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        # Calculate and display overall accuracy
        accuracy = np.trace(cm) / np.sum(cm)

        # Only show legend on the rightmost plot
        # if idx == 3:
        #     ax.legend(title='Predicted Class', bbox_to_anchor=(1.05, 1),
        #              loc='upper left', fontsize=FONT_SIZE-2,
        #              title_fontsize=FONT_SIZE-1, framealpha=0.9)

        # Set tick label sizes
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE-1)
        ax.set_ylabel('True Object Class', fontsize=FONT_SIZE)
    else:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
               transform=ax.transAxes, fontsize=FONT_SIZE)
        ax.set_title(f'Completion: {tp}%', fontsize=FONT_SIZE)

axes[-1].set_xlabel('Proportion', fontsize=FONT_SIZE)
# Set ylabel only for leftmost plot
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=FONT_SIZE - 2,
           bbox_to_anchor=(0.5, -0.08), title='Predicted Class', title_fontsize=FONT_SIZE - 1)
plt.tight_layout()
plt.savefig('Fig_confusion.pdf', bbox_inches='tight', dpi=300)

print("\nStacked bar charts saved!")

# Print statistics for each timepoint
print("\nSample counts and accuracies at each timepoint:")
for tp in time_points:
    n = len(confusion_data[tp]['y_true'])
    if n > 0:
        y_true = np.array(confusion_data[tp]['y_true'])
        y_pred = np.array(confusion_data[tp]['y_pred'])
        accuracy = np.mean(y_true == y_pred)
        print(f"  Completion {int(tp*100):3d}%: {n:5d} samples, Accuracy: {accuracy:.3f}")
    else:
        print(f"  Completion {int(tp*100):3d}%: {n:5d} samples")