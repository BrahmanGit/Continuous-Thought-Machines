# -*- coding: utf-8 -*-
"""
Fig7: Finger configuration analysis.
Reads from the unified evaluation dataframe (unified_evaluation_results.csv).
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "experiments"))
import utils
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
from statsmodels.stats.multitest import multipletests

# ============================================================================
# Setup
# ============================================================================
features = ["thumb", "index", "middle", "ring", "little"]

def fingers_to_binary(fingers_str):
    return [1 if str(i) in str(fingers_str) else 0 for i in range(len(features))]

target_names = utils.object_list
target_map = dict(zip(range(8), target_names))

# ============================================================================
# Load unified dataframe and filter to finger experiments
# ============================================================================
unified_path = '../results/latency/evaluation_fingers.csv'
all_df = pd.read_csv(unified_path, dtype={'condition': str})

# Filter to finger experiments only
finger_df = all_df[all_df['experiment_type'] == 'fingers'].copy()

# ============================================================================
# Build records for plotting
# ============================================================================
records = []
corr_records = []

for _, row in finger_df.iterrows():
    finger_cond = row['condition']

    # Correctness record (all trials)
    corr_records.append({
        "fingers": finger_cond,
        "correct": row['correct']
    })

    # Latency record (correct trials only)
    if row['correct'] == 1:
        nl1 = row['t_nl1'] - row['t_lift']
        records.append({
            'lifted_target': target_map.get(int(row['lifted_target']), row['lifted_target']),
            'nl': nl1,
            'trial': row['trial_number'],
            'rep': row['rep_number'],
            'stab': row['stability_pred'],
            'type': 'k=1',
            'block': row['block'],
            'participant': row['participant'],
            'fingers': finger_cond,
        })

corr_df = pd.DataFrame(corr_records)
plot_df = pd.DataFrame(records)

# ============================================================================
# Expand finger combinations to binary columns
# ============================================================================
binary_cols = plot_df['fingers'].apply(fingers_to_binary)
binary_df = pd.DataFrame(binary_cols.tolist(), columns=features, index=plot_df.index)

df = pd.concat([plot_df[['nl', 'fingers']], binary_df], axis=1)

# Group nl values by unique finger combinations
grouped = df.groupby(features)["nl"].apply(list)

# Sort combinations by mean nl
grouped_sorted = grouped.apply(np.mean).sort_values()
combinations = grouped_sorted.index.tolist()
boxplot_data = [grouped[combo] for combo in combinations]

# Correctness per combination
binary_cols_corr = corr_df['fingers'].apply(fingers_to_binary)
binary_corr = pd.DataFrame(binary_cols_corr.tolist(), columns=features, index=corr_df.index)
corr_expanded = pd.concat([corr_df[['fingers', 'correct']], binary_corr], axis=1)
correct_grouped = corr_expanded.groupby(features)["correct"].mean()
correct_means = [correct_grouped[combo] for combo in combinations]

# Stability per combination
stab_grouped = df.groupby(features)["fingers"].apply(
    lambda idx: plot_df.loc[idx.index, "stab"].mean()
)
stab_means = [stab_grouped[combo] for combo in combinations]

# ============================================================================
# Paired T-Tests vs all-fingers combination
# ============================================================================
all_fingers_combo = (1, 1, 1, 1, 1)
all_fingers_mask = (binary_df == all_fingers_combo).all(axis=1)

alpha = 0.01
combo_p = {}

for combo in combinations:
    combo_key = tuple(combo)
    print(combo)

    if combo_key == all_fingers_combo:
        combo_p[combo_key] = 0
        print(f"  Skipping all-fingers combination (reference group)")
        continue

    combo_mask = (binary_df == combo).all(axis=1)
    combo_indices = plot_df[combo_mask].index

    combo_present = []
    combo_absent = []

    for idx in combo_indices:
        block_val = plot_df.loc[idx, 'block']
        trial_val = plot_df.loc[idx, 'trial']
        rep_val = plot_df.loc[idx, 'rep']
        participant_val = plot_df.loc[idx, 'participant']

        combo_present.append(plot_df.loc[idx, 'nl'])

        matching_all_fingers = (
            (plot_df['block'] == block_val) &
            (plot_df['trial'] == trial_val) &
            (plot_df['rep'] == rep_val) &
            (plot_df['participant'] == participant_val) &
            all_fingers_mask
        )

        matching_data = plot_df.loc[matching_all_fingers, 'nl'].values
        if len(matching_data) > 0:
            combo_absent.append(matching_data[0])
        else:
            combo_present.pop()

    combo_present = np.array(combo_present)
    combo_absent = np.array(combo_absent)

    print(f"  n_matched_pairs={len(combo_present)}")

    if len(combo_present) > 0 and len(combo_absent) > 0 and len(combo_present) == len(combo_absent):
        t_stat, p_value = stats.ttest_rel(combo_present, combo_absent)
        mean_present = np.mean(combo_present)
        mean_absent = np.mean(combo_absent)
        std_present = np.std(combo_present, ddof=1)
        std_absent = np.std(combo_absent, ddof=1)
        differences = combo_present - combo_absent
        cohens_d = np.mean(differences) / np.std(differences, ddof=1) if len(differences) > 0 else 0

        finger_names = [features[i] for i, bit in enumerate(combo) if bit == 1]
        finger_label = "+".join(finger_names) if finger_names else "No fingers"

        if p_value < 0.001: sig_marker = "***"
        elif p_value < 0.01: sig_marker = "**"
        elif p_value < 0.05: sig_marker = "*"
        else: sig_marker = ""

        combo_p[combo_key] = 1 if p_value <= alpha else 0

        print(f"  {finger_label}: n={len(combo_present)}, mean_pres={mean_present:.2f} ms, mean_abs={mean_absent:.2f} ms, std_pres={std_present:.2f} ms, std_abs={std_absent:.2f} ms")
        print(f"  Difference: {mean_present - mean_absent:.2f} ms, p={p_value:.4e} {sig_marker}")
    else:
        finger_names = [features[i] for i, bit in enumerate(combo) if bit == 1]
        finger_label = "+".join(finger_names) if finger_names else "No fingers"
        combo_p[combo_key] = 0
        print(f"  Skipping {finger_label}: insufficient matched data (present={len(combo_present)}, absent={len(combo_absent)})")

# ============================================================================
# Plotting
# ============================================================================
FS = 15
fig = plt.figure(figsize=(18, 6))
gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])

ax_box = fig.add_subplot(gs[0])
ax_upset = fig.add_subplot(gs[1], sharex=ax_box)

# --- Top: Boxplots with color based on significance ---
bp = ax_box.boxplot(boxplot_data, vert=True, patch_artist=True, showfliers=False)

for i, (patch, combo) in enumerate(zip(bp['boxes'], combinations)):
    combo_key = tuple(combo)
    if combo_key != (1, 1, 1, 1, 1):
        if combo_p.get(combo_key, 0) == 1:
            patch.set_facecolor('#6fb744')
        else:
            patch.set_facecolor('#5f9cc7')
    else:
        patch.set_facecolor('white')

ax_box.set_xticks(range(1, len(combinations) + 1))
ax_box.set_xticklabels([])
ax_box.set_xlabel("")
ax_box.tick_params(axis='both', which='major', labelsize=FS)
ax_box.set_ylabel("latency (ms)", fontsize=FS)
ax_box.set_ylim([-1400, 1000])
ax_box.grid(True)

legend_elements = [
    Patch(facecolor='#6fb744', label='Significant (p > 0.01)'),
    Patch(facecolor='#5f9cc7', label='Not Significant')
]
ax_box.legend(handles=legend_elements, loc='upper right', fontsize=FS - 2)

# Prediction rate (fraction of negative latency)
g0 = []
for u in range(len(boxplot_data)):
    g0.append(len(np.where(np.array(boxplot_data[u]) < 0)[0]) / np.array(boxplot_data[u]).shape[0])

# --- Bottom: Binary matrix (UpSet-style) ---
ms = 12
fs = 15
for i, combo in enumerate(combinations):
    coords_x = []
    coords_y = []
    for j, bit in enumerate(combo):
        x = i + 1
        y = -(j + 1)
        if bit == 1:
            ax_upset.plot(x, y, "o", color="black", markersize=ms)
            coords_x.append(x)
            coords_y.append(y)
        else:
            ax_upset.plot(x, y, "o", color="lightgray", markersize=ms)
    if len(coords_x) > 1:
        ax_upset.plot(coords_x, coords_y, color="black", linewidth=1.5)

    ax_upset.text(i + 1, -len(features) - 1.8,
                  f"{stab_means[i]:.2f}",
                  ha="center", va="top", fontsize=fs)

    ax_upset.text(i + 1, -len(features) - 0.8,
                  f"{g0[i]:.2f}",
                  ha="center", va="top", fontsize=fs)

ax_upset.text(-0.0, -len(features) - 0.8, r"$r$", ha="center", va="top", fontsize=fs)
ax_upset.text(-0.0, -len(features) - 1.8, r"$S$", ha="center", va="top", fontsize=fs)

ax_upset.set_yticks([-(j + 1) for j in range(len(features))])
ax_upset.set_yticklabels(features, fontsize=fs)
ax_upset.set_ylim(-len(features) - 0.5, -0.5)
ax_upset.set_xticks([])
ax_upset.set_xticklabels([])

plt.tight_layout()
plt.savefig('Fig10_rev.pdf')

# --- Summary statistics ---
print("\n" + "=" * 60)
print("Average latency for single-finger combinations")
print("=" * 60)

means = []
for i, finger in enumerate(features):
    combo = tuple(1 if j == i else 0 for j in range(len(features)))
    combo_mask = (binary_df == list(combo)).all(axis=1)
    nl_values = plot_df[combo_mask]['nl'].values
    if len(nl_values) > 0:
        means.append(np.mean(nl_values))
        print(f"  {finger} only: n={len(nl_values)}, mean latency={np.mean(nl_values):.2f} ms, std={np.std(nl_values, ddof=1):.2f} ms")
    else:
        print(f"  {finger} only: no data found")

print('single-finger combos', np.mean(means))

# ============================================================================
# T-Test Summary Table (with FDR Correction)
# ============================================================================
print("\n" + "=" * 70)
print("Paired T-Test Summary: each finger combo vs. all-fingers (FDR Adjusted)")
print("=" * 70)

cohens_d_label = "Cohen's d"
header = f"{'Combination':<30} {'t-stat':>10} {'df':>6} {'mean1(ms)':>12} {'sd1(ms)':>10} {'mean2(ms)':>12} {'sd2(ms)':>10} {'p-adj':>12} {cohens_d_label:>10}"
print(header)
print("-" * len(header))

# Step A: Collect all test results and raw p-values
test_results = []
raw_p_values = []

for combo in combinations:
    combo_key = tuple(combo)
    if combo_key == all_fingers_combo:
        continue

    combo_mask = (binary_df == combo).all(axis=1)
    combo_indices = plot_df[combo_mask].index

    combo_present = []
    combo_absent = []

    for idx in combo_indices:
        block_val     = plot_df.loc[idx, 'block']
        trial_val     = plot_df.loc[idx, 'trial']
        rep_val       = plot_df.loc[idx, 'rep']
        participant_val = plot_df.loc[idx, 'participant']

        combo_present.append(plot_df.loc[idx, 'nl'])

        matching_all_fingers = (
            (plot_df['block']       == block_val) &
            (plot_df['trial']       == trial_val) &
            (plot_df['rep']         == rep_val) &
            (plot_df['participant'] == participant_val) &
            all_fingers_mask
        )
        matching_data = plot_df.loc[matching_all_fingers, 'nl'].values
        if len(matching_data) > 0:
            combo_absent.append(matching_data[0])
        else:
            combo_present.pop()

    combo_present = np.array(combo_present)
    combo_absent  = np.array(combo_absent)

    finger_names = [features[i] for i, bit in enumerate(combo) if bit == 1]
    finger_label = "+".join(finger_names) if finger_names else "No fingers"

    if len(combo_present) < 2 or len(combo_present) != len(combo_absent):
        print(f"{finger_label:<30} {'insufficient data':>10}")
        continue

    # Perform the test
    t_stat, p_value = stats.ttest_rel(combo_present, combo_absent)
    
    n              = len(combo_present)
    mean1          = np.mean(combo_present)
    sd1            = np.std(combo_present, ddof=1)
    mean2          = np.mean(combo_absent)
    sd2            = np.std(combo_absent, ddof=1)
    differences    = combo_present - combo_absent
    cohens_d       = np.mean(differences) / np.std(differences, ddof=1) if np.std(differences, ddof=1) > 0 else 0

    # Store for correction
    raw_p_values.append(p_value)
    test_results.append({
        'label': finger_label,
        't_stat': t_stat,
        'n': n,
        'mean1': mean1, 'sd1': sd1,
        'mean2': mean2, 'sd2': sd2,
        'cohens_d': cohens_d
    })

# Step B: Perform FDR Correction (Benjamini-Hochberg)
if raw_p_values:
    # returns: reject_array, pvals_corrected, alphacSidak, alphacBonf
    _, pvals_corrected, _, _ = multipletests(raw_p_values, alpha=0.05, method='fdr_bh')
else:
    pvals_corrected = []

# Step C: Print the results using the ADJUSTED p-values
for res, p_adj in zip(test_results, pvals_corrected):
    
    # Assess significance using the adjusted p-value
    if p_adj < 0.001:   sig = "***"
    elif p_adj < 0.01:  sig = "**"
    elif p_adj < 0.05:  sig = "*"
    else:               sig = "ns"

    print(
        f"{res['label']:<30} "
        f"t({res['n']-1:4d})={res['t_stat']:+7.3f}  "
        f"{res['mean1']:>10.1f}  "
        f"{res['sd1']:>10.1f}  "
        f"{res['mean2']:>10.1f}  "
        f"{res['sd2']:>10.1f}  "
        f"p={p_adj:>10.6f} {sig:3s}  "
        f"d={res['cohens_d']:+.3f}"
    )

print("=" * len(header))
print("Note: p-values are FDR-adjusted (Benjamini-Hochberg).")
print("Note: mean1/sd1 = finger combo; mean2/sd2 = all-fingers reference")
