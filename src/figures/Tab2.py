# -*- coding: utf-8 -*-
"""
Statistical analysis of negative latency across target objects.
Reads from the unified evaluation dataframe (unified_evaluation_results.csv).
Filters to all-fingers configuration only.

Computes:
  - Percentage of trials with negative latency per target and k value
  - Bootstrap confidence intervals
  - Kruskal-Wallis test across targets
  - Post-hoc Dunn's test with Bonferroni correction
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "experiments"))
import utils
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import kruskal

# ============================================================================
# Setup
# ============================================================================
font_size = 15
target_names = utils.object_list[:-1]
targets = target_names
target_map = dict(zip(range(8), target_names))

# All-fingers condition string (last entry from utils.generate_finger_subsets())
lists, strings = utils.generate_finger_subsets()
ALL_FINGERS_CONDITION = strings[-1]  # e.g. "01234"

# ============================================================================
# Load unified dataframe, filter to fingers experiment with all fingers only
# ============================================================================
# unified_path = '../results/latency/unified_evaluation_results.csv'
# all_df = pd.read_csv(unified_path, dtype={'condition': str})

unified_path = '../results/latency/evaluation_fingers.csv'
all_df = pd.read_csv(unified_path, dtype={'condition': str})

finger_df = all_df[
    (all_df['experiment_type'] == 'fingers') &
    (all_df['condition'] == ALL_FINGERS_CONDITION)
].copy()

print(f"Loaded {len(finger_df)} trials (all-fingers condition: '{ALL_FINGERS_CONDITION}')")

# ============================================================================
# Build records with k=1, k=2, k=3
# ============================================================================
records = []

for _, row in finger_df.iterrows():
    if row['correct'] == 1:
        target = int(row['lifted_target'])

        nl1 = max(row['t_nl1'] - row['t_lift'], row['t_hand'] - row['t_lift'])
        nl2 = max(row['t_nl2'] - row['t_lift'], row['t_hand'] - row['t_lift'])
        nl3 = max(row['t_nl3'] - row['t_lift'], row['t_hand'] - row['t_lift'])

        base = {
            'lifted_target': target_map[target],
            'block': row['block'],
            'participant': row['participant'],
            'fingers': row['condition'],
        }

        records.append({**base, 'nl': nl1, 'nl_c': row['t_nl1'] - row['t_nc'], 'type': 'k=1'})
        records.append({**base, 'nl': nl2, 'nl_c': row['t_nl2'] - row['t_nc'], 'type': 'k=2'})
        records.append({**base, 'nl': nl3, 'nl_c': row['t_nl3'] - row['t_nc'], 'type': 'k=3'})

plot_df = pd.DataFrame(records)

# ============================================================================
# Percentage table with bootstrap CIs
# ============================================================================
def calculate_negative_percentage(df, target, k_value):
    subset = df[(df['lifted_target'] == target) & (df['type'] == f'k={k_value}')]
    print(subset.shape)
    if len(subset) == 0:
        return 0.0
    return (subset['nl'] < 0).sum() / len(subset)


def bootstrap_ci(df, target, k_value, n_bootstrap=10000, ci_level=0.95):
    subset = df[(df['lifted_target'] == target) & (df['type'] == f'k={k_value}')]
    if len(subset) == 0:
        return (0.0, 0.0)

    data = (subset['nl'] < 0).astype(int).values
    n = len(data)

    np.random.seed(42)
    bootstrap_percentages = []
    for _ in range(n_bootstrap):
        resample = np.random.choice(data, size=n, replace=True)
        bootstrap_percentages.append(resample.mean())

    alpha = 1 - ci_level
    lower = np.percentile(bootstrap_percentages, 100 * alpha / 2)
    upper = np.percentile(bootstrap_percentages, 100 * (1 - alpha / 2))
    return (lower, upper)


# Overall k=1 negative percentage
subset = plot_df[plot_df['type'] == 'k=1']
negative_count = (subset['nl'] < 0).sum()
total_count = len(subset)
print(f"\nOverall k=1 negative latency rate: {negative_count / total_count:.4f}")

# Build tables
percentage_data = []
percentage_data_with_ci = []

for k in [1, 2, 3]:
    row_data = {'k': f'k={k}'}
    row_data_ci = {'k': f'k={k}'}

    all_percentages = []

    for target in target_names:
        percentage = calculate_negative_percentage(plot_df, target, k)
        ci_lower, ci_upper = bootstrap_ci(plot_df, target, k)

        row_data[target] = percentage
        row_data_ci[target] = f"{percentage:.2f} [{ci_lower:.2f}, {ci_upper:.2f}]"
        all_percentages.append(percentage)

    # Overall average and CI
    overall_avg = np.mean(all_percentages)

    all_data_for_k = plot_df[plot_df['type'] == f'k={k}']
    if len(all_data_for_k) > 0:
        data = (all_data_for_k['nl'] < 0).astype(int).values
        n = len(data)
        np.random.seed(42)
        bootstrap_percentages = []
        for _ in range(10000):
            resample = np.random.choice(data, size=n, replace=True)
            bootstrap_percentages.append(resample.mean())
        ci_lower_all = np.percentile(bootstrap_percentages, 2.5)
        ci_upper_all = np.percentile(bootstrap_percentages, 97.5)
    else:
        ci_lower_all, ci_upper_all = 0.0, 0.0

    row_data['all'] = overall_avg
    row_data_ci['all'] = f"{overall_avg:.3f} [{ci_lower_all:.3f}, {ci_upper_all:.3f}]"

    percentage_data.append(row_data)
    percentage_data_with_ci.append(row_data_ci)

percentage_table = pd.DataFrame(percentage_data)
percentage_table_ci = pd.DataFrame(percentage_data_with_ci)

print("\nPercentage of samples with nl < 0 for each k value and target:")
print("=" * 100)
print(percentage_table.to_string(index=False, float_format='%.2f'))

print("\n\nPercentage of samples with nl < 0 with 95% Bootstrap CIs:")
print("=" * 100)
print(percentage_table_ci.to_string(index=False))

# Generate LaTeX table
print("\n\n" + "=" * 100)
print("LATEX TABLE WITH BOOTSTRAP CONFIDENCE INTERVALS")
print("=" * 100)
print()
print("\\begin{table}[ht]")
print("\\centering")
print("\\begin{tabularx}{\\columnwidth}{lXXXXXXXXX}")
print("\\hline")
print("\\textbf{k} & \\textbf{stick} & \\textbf{ball} & \\textbf{screw} & \\textbf{box} & \\textbf{knife} & \\textbf{glass} & \\textbf{assem.} & \\textbf{disk} & \\textbf{all} \\\\")
print("\\hline")

for _, row in percentage_table_ci.iterrows():
    k_val = row['k'].replace('k=', '')
    line = f"{k_val}"
    for target in target_names + ['all']:
        line += f" & {row[target]}"
    line += " \\\\"
    print(line)

print("\\hline")
print("\\end{tabularx}")
print("\\caption{Percentage of samples with $nl < 0$ for each target object and $k$ value, shown with 95\\% bootstrap confidence intervals.}")
print("\\label{tab:nl_negative_percentage}")
print("\\end{table}")
print()


# ============================================================================
# Generalized Linear Model (Logistic Regression) & Deviance Chi-Square Test (k=1)
# ============================================================================
'example: https://advstats.psychstat.org/python/logistic/index.php'
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2

print("\n\n" + "=" * 100)
print("LOGISTIC REGRESSION & DEVIANCE (CHI-SQUARE) TEST (k=1)")
print("=" * 100)

# 1. Prepare data for k=1
k1_df = plot_df[plot_df['type'] == 'k=1'].copy()

# Create the binary outcome variable: 1 if nl < 0 (negative latency), else 0
k1_df['is_negative'] = (k1_df['nl'] < 0).astype(int)

try:
    # 2. Fit the Null Model (Intercept only)
    print("Fitting Null Model (Intercept only)...")
    null_model = smf.glm(formula="is_negative ~ 1", 
                         data=k1_df, 
                         family=sm.families.Binomial()).fit()

    # 3. Fit the Full Model (with 'lifted_target' as a predictor)
    print("Fitting Full Model (with 'lifted_target')...")
    full_model = smf.glm(formula="is_negative ~ 1 + C(lifted_target)", 
                         data=k1_df, 
                         family=sm.families.Binomial()).fit()

    print("\n--- Full Model Summary ---")
    print(full_model.summary())

    # 4. Compare the two models (Analysis of Deviance)
    # The difference in deviance follows a chi-square distribution
    dev_diff = null_model.deviance - full_model.deviance
    
    # Degrees of freedom is the difference in residual degrees of freedom
    df_diff = null_model.df_resid - full_model.df_resid

    # Calculate the p-value using the chi-square cumulative distribution function (CDF)
    # Note: 1 - chi2.cdf(dev_diff, df_diff) is mathematically equivalent to chi2.sf(dev_diff, df_diff)
    p_value = 1 - chi2.cdf(dev_diff, df_diff)

    print("\n\n--- Analysis of Deviance (Likelihood Ratio / Chi-Square Test) ---")
    print(f"Null Model Deviance: {null_model.deviance:.4f} (df_resid={null_model.df_resid})")
    print(f"Full Model Deviance: {full_model.deviance:.4f} (df_resid={full_model.df_resid})")
    print(f"Deviance Difference (Chi-Square Statistic): {dev_diff:.4f}")
    print(f"Degrees of Freedom Difference: {df_diff}")
    print(f"P-value: {p_value:.6e}")

    if p_value < 0.05:
        print("\n→ CONCLUSION: Including the target object significantly improves the model fit compared to the null model (p < 0.05).")
    else:
        print("\n→ CONCLUSION: The target object does NOT significantly improve the model fit compared to the null model (p >= 0.05).")

except Exception as e:
    print(f"An error occurred while running the statistical models: {e}")

    
  












# # ============================================================================
# # Statistical tests (k=1 only)
# # ============================================================================
# k1_data = plot_df[plot_df['type'] == 'k=1'].copy()

# # 1. KRUSKAL-WALLIS TEST
# print("\n1. KRUSKAL-WALLIS TEST: Do negative latency values differ across target objects?")
# print("-" * 100)

# groups = [k1_data[k1_data['lifted_target'] == target]['nl'].values for target in target_names]
# h_stat, kw_p_value = kruskal(*groups)

# print(f"\nH-statistic: {h_stat:.4f}")
# print(f"P-value: {kw_p_value:.6f}")
# print(f"Significance: {'SIGNIFICANT' if kw_p_value < 0.05 else 'NOT SIGNIFICANT'} at α=0.05")

# if kw_p_value < 0.05:
#     print("\n→ CONCLUSION: Negative latency distributions DIFFER significantly across target objects.")
# else:
#     print("\n→ CONCLUSION: Negative latency distributions DO NOT differ significantly across target objects.")

# # 2. POST-HOC DUNN'S TEST
# significant_pairs = []

# if kw_p_value < 0.05:
#     print("\n\n2. POST-HOC DUNN'S TEST with Bonferroni correction")
#     print("-" * 100)

#     from scikit_posthocs import posthoc_dunn

#     dunn_data = k1_data[['lifted_target', 'nl']].copy()
#     dunn_results = posthoc_dunn(dunn_data, val_col='nl', group_col='lifted_target', p_adjust='bonferroni')

#     print("\nDunn's Test P-values (Bonferroni-corrected):")
#     print(dunn_results.to_string(float_format='%.6f'))

#     print("\n\nSignificant pairwise comparisons (p < 0.05):")
#     print("-" * 100)

#     for i in range(len(target_names)):
#         for j in range(i + 1, len(target_names)):
#             target1 = target_names[i]
#             target2 = target_names[j]
#             p_val = dunn_results.loc[target1, target2]

#             if p_val < 0.05:
#                 data1 = k1_data[k1_data['lifted_target'] == target1]['nl']
#                 data2 = k1_data[k1_data['lifted_target'] == target2]['nl']

#                 significant_pairs.append({
#                     'Target_1': target1,
#                     'Target_2': target2,
#                     'P_value': p_val,
#                     'Median_1': data1.median(),
#                     'Median_2': data2.median(),
#                     'Median_Diff': data1.median() - data2.median()
#                 })

#     if significant_pairs:
#         sig_pairs_df = pd.DataFrame(significant_pairs)
#         sig_pairs_df = sig_pairs_df.sort_values('P_value')
#         print(sig_pairs_df.to_string(index=False, float_format='%.6f'))
#         print(f"\nTotal significant pairs: {len(significant_pairs)}")
#     else:
#         print("No significant pairwise differences after Bonferroni correction.")
# else:
#     print("\n\n2. POST-HOC TEST: Skipped (Kruskal-Wallis test was not significant)")

# # 3. SUMMARY
# print("\n\n" + "=" * 100)
# print("SUMMARY OF STATISTICAL FINDINGS")
# print("=" * 100)

# print(f"\n1. Kruskal-Wallis test: {'SIGNIFICANT' if kw_p_value < 0.05 else 'NOT SIGNIFICANT'} (H={h_stat:.4f}, p={kw_p_value:.6f})")
# print(f"   → Negative latency distributions {'DO' if kw_p_value < 0.05 else 'do NOT'} differ significantly across target objects")

# if kw_p_value < 0.05 and len(significant_pairs) > 0:
#     sig_pairs_df = pd.DataFrame(significant_pairs).sort_values('P_value')
#     print(f"\n2. Dunn's post-hoc test identified {len(significant_pairs)} significant pairwise difference(s)")
#     print(f"\n3. Top 3 most significant pairwise differences:")
#     for i, row in sig_pairs_df.head(3).iterrows():
#         print(f"   - {row['Target_1']} vs {row['Target_2']}: p={row['P_value']:.6f} (median diff={row['Median_Diff']:.4f})")

# print("\n" + "=" * 100)