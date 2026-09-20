# -*- coding: utf-8 -*-
"""
Visualization script for parameter space exploration results.
Generates a single figure with 5 subplots showing how metrics (Acc, r, S, and average latency)
vary with each parameter while keeping others fixed at the best values from config.
Reads best_model_config.json to determine which points to highlight in green.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

pd.set_option('display.float_format', lambda x: '%.9f' % x)

# ============================================================================
# Plot range filters — set to None to disable
# These mirror the filters in analyze_hyperparameter_search.py and restrict
# which parameter values are shown in the plots.
# ============================================================================
PARAM_LOWER_BOUNDS = {
    "noise_process": 1e-7,   # e.g. 1e-5 to exclude values below that
    "noise_obs":     1e-7,   # e.g. 1e-4
}

PARAM_UPPER_BOUNDS = {
    "integration_window_length": 20,
    "reset_interval": 20,# e.g. 50 to exclude values above that
}
# ============================================================================

# ============================================================================
# Load best config from JSON
# ============================================================================
CONFIG_PATH = "../results/parameter_space/best_model_config.json"
with open(CONFIG_PATH, 'r') as f:
    best_config = json.load(f)

mp = best_config['model_parameters']
print(f"Loaded best config from {CONFIG_PATH}:")
print(f"  beta                      = {mp['beta']}")
print(f"  noise_process             = {mp['noise_process']}")
print(f"  noise_obs                 = {mp['noise_obs']}")
print(f"  integration_window_length = {mp['integration_window_length']}")
print(f"  reset_interval            = {mp['reset_interval']}")

# Build FIXED_PARAMS from the config (these are the best values)
FIXED_PARAMS = {
    'beta': mp['beta'],
    'noise_process': mp['noise_process'],
    'noise_obs': mp['noise_obs'],
    'integration_window_length': mp['integration_window_length'],
    'reset_interval': mp['reset_interval'],
}

# ============================================================================
# Load data
# ============================================================================
results_data_path = "../results/parameter_space/"
participants = range(1, 6)
all_participant_dfs = []

for i in participants:
    metrics_file_path = results_data_path + f"hyperparameter_search_metrics_participant_{i}.csv"
    df_p = pd.read_csv(metrics_file_path)
    df_p['participant'] = i
    all_participant_dfs.append(df_p)

df = pd.concat(all_participant_dfs, ignore_index=True)

# ============================================================================
# Apply plot range filters
# ============================================================================
active_lower = {k: v for k, v in PARAM_LOWER_BOUNDS.items() if v is not None}
active_upper = {k: v for k, v in PARAM_UPPER_BOUNDS.items() if v is not None}
if active_lower or active_upper:
    print("\nApplying plot range filters:")
    n_before = len(df)
    for param, lb in active_lower.items():
        df = df[df[param] >= lb]
        print(f"  {param} >= {lb:.2e}")
    for param, ub in active_upper.items():
        df = df[df[param] <= ub]
        print(f"  {param} <= {ub:.2e}")
    print(f"  Rows retained: {len(df)} / {n_before} ({n_before - len(df)} removed)")
else:
    print("\nNo plot range filters active.")

# ============================================================================
# Configuration
# ============================================================================
parameters = ['beta', 'noise_process', 'noise_obs', 'integration_window_length', 'reset_interval']
x_label_list = [r'$\beta$', r'$\sigma_r$', r'$\sigma_z$', '$w$', '$R$']

REFERENCE_WIDTH = 8
REFERENCE_FONTSIZE = 15
fig_width = 8
FS = REFERENCE_FONTSIZE * (fig_width / REFERENCE_WIDTH)
sns.set(style='whitegrid')
green = '#2bad70'

fig, axes = plt.subplots(5, figsize=(fig_width, 16))
axes = axes.flatten()

# ============================================================================
# Plot each parameter
# ============================================================================
for idx, param in enumerate(parameters):
    print(f"\nProcessing parameter: {param}")

    if idx == 0:   X_OFFSETS = {'accuracy': -0.02, 'r': -0.01, 's': 0.01, 'nl': 0.02}
    elif idx == 1: X_OFFSETS = {'accuracy': -0.03, 'r': -0.01, 's': 0.01, 'nl': 0.03}
    elif idx == 2: X_OFFSETS = {'accuracy': -0.03, 'r': -0.01, 's': 0.01, 'nl': 0.03}
    elif idx == 3: X_OFFSETS = {'accuracy': -0.02, 'r': -0.01, 's': 0.01, 'nl': 0.02}
    elif idx == 4: X_OFFSETS = {'accuracy': -0.02, 'r': -0.01, 's': 0.01, 'nl': 0.02}

    # Filter data: keep other parameters at their best (fixed) values
    filtered_df = df.copy()
    for other_param, fixed_value in FIXED_PARAMS.items():
        if other_param != param:
            filtered_df = filtered_df[np.isclose(filtered_df[other_param], fixed_value, rtol=1e-5)]

    # Group by the current parameter
    param_stats = filtered_df.groupby(param).agg({
        'nl': ['mean', 'std'],
        'r': ['mean', 'std'],
        's': ['mean', 'std'],
        'accuracy': ['mean', 'std']
    }).reset_index()

    param_stats.columns = [param, 'nl_mean', 'nl_std', 'r_mean', 'r_std',
                           's_mean', 's_std', 'accuracy_mean', 'accuracy_std']
    param_stats = param_stats.sort_values(param)

    ax1 = axes[idx]
    ms = 60

    x_vals = param_stats[param].values

    # Use integer x-axis for discrete parameters
    use_int_axis = idx in [3, 4]
    if use_int_axis:
        x_vals_plot = np.arange(len(x_vals))
    else:
        x_vals_plot = x_vals

    # Find the index of the best value for this parameter (from config)
    best_value = FIXED_PARAMS[param]
    best_idx = None
    for i, v in enumerate(x_vals):
        if np.isclose(v, best_value, rtol=1e-5):
            best_idx = i
            break

    # Compute x offsets
    if param in ['noise_process', 'noise_obs']:
        x_offset_scale = np.exp(np.log(x_vals_plot.max() / x_vals_plot.min()) * 0.015)
        x_accuracy = x_vals_plot * (x_offset_scale ** (X_OFFSETS['accuracy'] / 0.015))
        x_r        = x_vals_plot * (x_offset_scale ** (X_OFFSETS['r']        / 0.015))
        x_s        = x_vals_plot * (x_offset_scale ** (X_OFFSETS['s']        / 0.015))
        x_nl       = x_vals_plot * (x_offset_scale ** (X_OFFSETS['nl']       / 0.015))
    else:
        x_range = x_vals_plot.max() - x_vals_plot.min()
        x_offset_scale = x_range * 0.015 if x_range > 0 else 0.015
        x_accuracy = x_vals_plot + x_offset_scale * X_OFFSETS['accuracy'] / 0.015
        x_r        = x_vals_plot + x_offset_scale * X_OFFSETS['r']        / 0.015
        x_s        = x_vals_plot + x_offset_scale * X_OFFSETS['s']        / 0.015
        x_nl       = x_vals_plot + x_offset_scale * X_OFFSETS['nl']       / 0.015

    # Clip error bars
    accuracy_mean = param_stats['accuracy_mean'] * 100
    accuracy_std  = param_stats['accuracy_std']  * 100
    accuracy_yerr = np.array([np.minimum(accuracy_std, accuracy_mean),
                               np.minimum(accuracy_std, 100 - accuracy_mean)])

    r_mean = param_stats['r_mean']
    r_std  = param_stats['r_std']
    r_yerr = np.array([np.minimum(r_std, r_mean),
                        np.minimum(r_std, 100 - r_mean)])

    s_mean = param_stats['s_mean'] * 100
    s_std  = param_stats['s_std']  * 100
    s_yerr = np.array([np.minimum(s_std, s_mean),
                        np.minimum(s_std, 100 - s_mean)])

    # Plot metrics with green highlighting for the best config value
    for i in range(len(x_accuracy)):
        c = green if i == best_idx else 'black'
        ax1.errorbar(x_accuracy[i], accuracy_mean.iloc[i],
                     yerr=accuracy_yerr[:, i].reshape((2, 1)),
                     fmt='o', markersize=np.sqrt(ms / 3),
                     color=c, capsize=3, capthick=1, zorder=3)
        ax1.errorbar(x_r[i], r_mean.iloc[i],
                     yerr=r_yerr[:, i].reshape((2, 1)),
                     fmt='s', markersize=np.sqrt(ms / 3),
                     color=c, capsize=3, capthick=1, zorder=3)
        ax1.errorbar(x_s[i], s_mean.iloc[i],
                     yerr=s_yerr[:, i].reshape((2, 1)),
                     fmt='x', markersize=np.sqrt(ms / 3),
                     color=c, capsize=3, capthick=1, zorder=3)

    # Configure x-axis
    if param in ['noise_process', 'noise_obs']:
        ax1.set_xscale('log')
        ax1.set_xticks(x_vals)
        ax1.set_xticklabels([f"{x:.0e}" for x in x_vals], fontsize=FS - 2)
    else:
        ax1.set_xscale('linear')
        if use_int_axis:
            ax1.set_xticks(np.arange(len(x_vals)))
            ax1.set_xticklabels([f"{int(x)}" for x in x_vals], fontsize=FS)
        else:
            ax1.set_xticks(x_vals)
            if x_vals.max() < 1:
                # For beta: show enough decimals to distinguish values
                ax1.set_xticklabels([f"{x:.3f}" for x in x_vals], fontsize=FS - 2,
                                    rotation=45, ha='right')
            else:
                ax1.set_xticklabels([f"{x:.0f}" for x in x_vals], fontsize=FS)

    ax1.set_xlabel(x_label_list[idx], fontsize=FS)
    ax1.set_ylabel(r'Acc, $r$, $S$', fontsize=FS)

    # Second y-axis for latency
    ax2 = ax1.twinx()
    for i in range(len(x_nl)):
        c = green if i == best_idx else 'black'
        ax2.errorbar(x_nl[i], param_stats['nl_mean'].iloc[i],
                     yerr=param_stats['nl_std'].iloc[i],
                     fmt='^', color=c, markersize=np.sqrt(ms / 3),
                     capsize=3, capthick=1, zorder=3)
    ax2.set_ylabel('Average latency [ms]', fontsize=FS)

    # Legend on first subplot only
    if idx == 0:
        legend_elements = [
            Line2D([0], [0], marker='o', color='black', linestyle='None', label='Acc'),
            Line2D([0], [0], marker='s', color='black', linestyle='None', label=r'$r$'),
            Line2D([0], [0], marker='x', color='black', linestyle='None', label=r'$S$'),
            Line2D([0], [0], marker='^', color='black', linestyle='None', label='Latency'),
            Patch(facecolor=green, edgecolor=green, label='Best config')
        ]
        axes[-1].legend(
            handles=legend_elements,
            bbox_to_anchor=(0.5, -0.7),
            loc='lower center',
            ncol=3,
            fontsize=FS
        )

    ax1.tick_params(axis='y', labelsize=FS)
    ax2.tick_params(axis='y', labelsize=FS)
    ax1.grid(False)
    ax2.grid(False)

    # Vertical grid lines
    if use_int_axis:
        for value in np.arange(len(x_vals)):
            ax1.axvline(x=value, zorder=-1, color='lightgrey', linewidth=0.5)
    else:
        for value in x_vals:
            ax1.axvline(x=value, zorder=-1, color='lightgrey', linewidth=0.5)

plt.tight_layout()
plt.savefig('Fig5_rev.pdf', bbox_inches='tight')
print("\nSaved Fig5_rev.pdf")

# ============================================================================
# Report stats for best configuration
# ============================================================================
cfg_df = df.copy()
for k, v in FIXED_PARAMS.items():
    cfg_df = cfg_df[np.isclose(cfg_df[k], v, rtol=1e-6)]

print("\nBest configuration rows:")
print(cfg_df[['participant', 'nl']])

nl_mean = cfg_df['nl'].mean()
nl_std  = cfg_df['nl'].std(ddof=1)

print(f"\n=== Negative latency statistics (best config) ===")
print(f"  beta={mp['beta']}, σr={mp['noise_process']}, σz={mp['noise_obs']}, "
      f"w={mp['integration_window_length']}, reset={mp['reset_interval']}")
print(f"  Mean negative latency: {nl_mean:.3f} ms")
print(f"  Std  negative latency: {nl_std:.3f} ms")
print("===================================================")