# -*- coding: utf-8 -*-
"""
Hyperparameter search analysis.

Loads per-participant hyperparameter search results, finds the optimal
parameter combination (lowest average negative latency across all participants),
and writes it to a JSON config file that downstream scripts can load.

Config file written: ../results/parameter_space/best_model_config.json

@author: jonas
"""
import os
import json
import numpy as np
import utils
import pandas as pd

pd.set_option('display.float_format', lambda x: '%.9f' % x)

# =============================================================================
# Lower bound filters — set to None to disable a filter
# =============================================================================
PARAM_LOWER_BOUNDS = {
    "noise_process": 1e-7,   # e.g. 1e-5 to exclude values below that
    "noise_obs":     1e-7,   # e.g. 1e-4
}

PARAM_UPPER_BOUNDS = {
    "integration_window_length": 20,
    "reset_interval": 20,# e.g. 50 to exclude values above that
}
# =============================================================================

target_names = utils.object_list
target_map = dict(zip(range(8), target_names))

# Define data paths for input and output files
grasp_data_path = "../results/grasping_data/"
participant_data_path = "../../data/dataframes/"
results_data_path = "../results/parameter_space/"

# Time step size in milliseconds (or relevant time unit)
dt = 0.015
lists, strings = utils.generate_finger_subsets()

param_columns = ['beta', 'noise_process', 'noise_obs', 'integration_window_length', 'reset_interval']
participants = range(1, 6)

# Mean movement durations for each participant (needed to back-compute multiplier from beta)
mov_dur_list = [68.68, 81.07, 78.44, 100.4, 91.07]

# =============================================================================
# Original per-participant analysis
# =============================================================================
print("=" * 80)
print("PER-PARTICIPANT BEST PARAMETERS")
print("=" * 80)
for i in participants:
    metrics_file_path = results_data_path + f"hyperparameter_search_metrics_participant_{i}.csv"
    df = pd.read_csv(metrics_file_path)
    best_nl_row = df.loc[df['nl'].idxmin()]
    print(f"\nParticipant {i} - Best NL:")
    print(best_nl_row[['beta', 'noise_process', 'noise_obs', 'integration_window_length', 'reset_interval', 'nl']])

nl_rows = []
r_rows = []
s_rows = []
acc_rows = []
for i in participants:
    metrics_file_path = results_data_path + f"hyperparameter_search_metrics_participant_{i}.csv"
    df = pd.read_csv(metrics_file_path)
    nl_rows.append(df.loc[df["nl"].idxmin()]['nl'])
    r_rows.append(df.loc[df["r"].idxmax()]['r'])
    s_rows.append(df.loc[df["s"].idxmax()]['s'])
    acc_rows.append(df.loc[df["accuracy"].idxmax()]['accuracy'])

print("\n" + "=" * 80)
print("AVERAGE BEST METRICS (per-participant optima)")
print("=" * 80)
print(f"Average best NL:       {np.mean(nl_rows):.6f}")
print(f"Average best R:        {np.mean(r_rows):.6f}")
print(f"Average best S:        {np.mean(s_rows):.6f}")
print(f"Average best Accuracy: {np.mean(acc_rows):.6f}")

# =============================================================================
# Load and concatenate all participant dataframes
# =============================================================================
all_dfs = []
for i in participants:
    metrics_file_path = results_data_path + f"hyperparameter_search_metrics_participant_{i}.csv"
    df = pd.read_csv(metrics_file_path)
    df['participant'] = i
    all_dfs.append(df)

combined_df = pd.concat(all_dfs, ignore_index=True)

# =============================================================================
# Apply lower bound filters
# =============================================================================
active_lower = {k: v for k, v in PARAM_LOWER_BOUNDS.items() if v is not None}
active_upper = {k: v for k, v in PARAM_UPPER_BOUNDS.items() if v is not None}
if active_lower or active_upper:
    print("\n" + "=" * 80)
    print("APPLYING FILTERS")
    print("=" * 80)
    n_before = len(combined_df)
    for param, lb in active_lower.items():
        combined_df = combined_df[combined_df[param] >= lb]
        print(f"  {param} >= {lb:.2e}")
    for param, ub in active_upper.items():
        combined_df = combined_df[combined_df[param] <= ub]
        print(f"  {param} <= {ub:.2e}")
    n_after = len(combined_df)
    print(f"  Rows retained: {n_after} / {n_before} ({n_before - n_after} removed)")
else:
    print("\nNo filters active.")

# =============================================================================
# Only consider parameter combinations present in ALL participants
# =============================================================================
param_counts = combined_df.groupby(param_columns)['participant'].nunique().reset_index()
param_counts.columns = param_columns + ['participant_count']
full_coverage = param_counts[param_counts['participant_count'] == len(participants)][param_columns]

n_total = len(param_counts)
n_full = len(full_coverage)
print(f"\nParameter combinations total:                   {n_total}")
print(f"Parameter combinations present in all participants: {n_full}")
print(f"Excluded (incomplete coverage):                 {n_total - n_full}")

# Group by parameter combinations and calculate average metrics
avg_metrics_all = combined_df.groupby(param_columns).agg(
    nl_mean=('nl', 'mean'),
    nl_std=('nl', 'std'),
    r_mean=('r', 'mean'),
    r_std=('r', 'std'),
    s_mean=('s', 'mean'),
    s_std=('s', 'std'),
    accuracy_mean=('accuracy', 'mean'),
    accuracy_std=('accuracy', 'std'),
).reset_index()

# Filter to full-coverage combinations only
avg_metrics = avg_metrics_all.merge(full_coverage, on=param_columns)


# =============================================================================
# Helper to print a best-row summary with per-participant breakdown
# =============================================================================
def print_best_row(row, label):
    print(f"\nParameter combination with {label}:")
    print("-" * 80)
    print(f"Beta:                        {row['beta']:.6f}")
    print(f"Noise Process:               {row['noise_process']:.2e}")
    print(f"Noise Observation:           {row['noise_obs']:.2e}")
    print(f"Integration Window Length:   {row['integration_window_length']:.0f}")
    print(f"Reset Interval:              {row['reset_interval']:.0f}")
    print(f"\nResulting Metrics (averaged over all {len(participants)} participants):")
    print(f"  Average NL:       {row['nl_mean']:.6f} ± {row['nl_std']:.6f}")
    print(f"  Average R:        {row['r_mean']:.6f} ± {row['r_std']:.6f}")
    print(f"  Average S:        {row['s_mean']:.6f} ± {row['s_std']:.6f}")
    print(f"  Average Accuracy: {row['accuracy_mean']:.6f} ± {row['accuracy_std']:.6f}")

    # Per-participant breakdown using robust float comparison
    print("\n  Per-participant breakdown:")
    optimal_params = row[param_columns].to_dict()
    for i in participants:
        participant_df = combined_df[combined_df['participant'] == i]
        mask = np.ones(len(participant_df), dtype=bool)
        for col in param_columns:
            mask &= np.isclose(participant_df[col], optimal_params[col])
        matching_row = participant_df[mask]
        if not matching_row.empty:
            print(f"    Participant {i}: nl={matching_row['nl'].values[0]:.6f}, "
                  f"r={matching_row['r'].values[0]:.6f}, "
                  f"s={matching_row['s'].values[0]:.6f}, "
                  f"accuracy={matching_row['accuracy'].values[0]:.6f}")
        else:
            print(f"    Participant {i}: No matching configuration found")


# =============================================================================
# Report optimal parameters
# =============================================================================
print("\n" + "=" * 80)
print("OPTIMAL PARAMETERS ACROSS ALL PARTICIPANTS (full coverage only)")
print("=" * 80)

best_nl_row = avg_metrics.loc[avg_metrics['nl_mean'].idxmin()]
print_best_row(best_nl_row, "LOWEST AVERAGE NEGATIVE LATENCY")
print_best_row(avg_metrics.loc[avg_metrics['r_mean'].idxmax()], "HIGHEST AVERAGE R")
print_best_row(avg_metrics.loc[avg_metrics['s_mean'].idxmax()], "HIGHEST AVERAGE S")
print_best_row(avg_metrics.loc[avg_metrics['accuracy_mean'].idxmax()], "HIGHEST AVERAGE ACCURACY")

# =============================================================================
# Save averaged metrics (full coverage only) to file
# =============================================================================
avg_metrics_output_path = results_data_path + "hyperparameter_search_averaged_metrics.csv"
avg_metrics.to_csv(avg_metrics_output_path, index=False)
print(f"\nAveraged metrics saved to: {avg_metrics_output_path}")

# =============================================================================
# Write best config to JSON
# =============================================================================
# Extract the best parameters (lowest average negative latency)
best_beta = float(best_nl_row['beta'])
best_noise_process = float(best_nl_row['noise_process'])
best_noise_obs = float(best_nl_row['noise_obs'])
best_integration_window = int(best_nl_row['integration_window_length'])
best_reset_interval = int(best_nl_row['reset_interval'])

# Back-compute the gamma multiplier per participant from beta = 1 / (mov_dur * multiplier * dt)
# => multiplier = 1 / (beta * mov_dur * dt)
per_participant_multipliers = []
for mov_dur in mov_dur_list:
    multiplier = 1.0 / (best_beta * mov_dur * dt)
    per_participant_multipliers.append(round(multiplier, 6))

config = {
    "_description": "Best model parametrization (lowest average negative latency across all participants). "
                    "Written automatically by analyze_hyperparameter_search.py.",
    "model_parameters": {
        "beta": best_beta,
        "noise_process": best_noise_process,
        "noise_obs": best_noise_obs,
        "integration_window_length": best_integration_window,
        "reset_interval": best_reset_interval,
        "dt": dt,
    },
    "metrics": {
        "nl_mean": float(best_nl_row['nl_mean']),
        "nl_std": float(best_nl_row['nl_std']),
        "r_mean": float(best_nl_row['r_mean']),
        "accuracy_mean": float(best_nl_row['accuracy_mean']),
    },
    "filters_applied": {
        "lower_bounds": {k: v for k, v in PARAM_LOWER_BOUNDS.items() if v is not None},
        "upper_bounds": {k: v for k, v in PARAM_UPPER_BOUNDS.items() if v is not None},
    },
}

config_path = os.path.join(results_data_path, "best_model_config.json")
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print(f"\nBest model config written to: {config_path}")
print(f"  beta                     = {best_beta}")
print(f"  noise_process            = {best_noise_process}")
print(f"  noise_obs                = {best_noise_obs}")
print(f"  integration_window_length = {best_integration_window}")
print(f"  reset_interval           = {best_reset_interval}")