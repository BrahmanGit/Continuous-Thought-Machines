# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 09:39:07 2025

@author: joni_
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "experiments"))
import utils
import numpy as np
import pandas as pd
import time
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.power import FTestAnovaPower

target_names = utils.object_list[:-1]
target_map = dict(zip(range(8), target_names))


records = []
records_abs = []
time = 't_lift'
path_to_all_results = results_path = '../results/latency'

def load_and_process_data():
    for name in ['participant_1', 'participant_2', 'participant_3', 'participant_4', 'participant_5']:
        df = pd.read_csv(path_to_all_results+'/'+name+'_times_data.csv', dtype={'num_fingers': str})
        for config in [0.15]:
            for target in np.arange(8):
                for phase_number in [1, 2, 3, 4]:
                    times = df[(df['lifted_target'] == target) & 
                               (df['block'] == phase_number) &
                               (df['config'] == round(config, 4)) & 
                               (df['num_fingers'] == '01234')]
                    for _, row in times.iterrows():
                        if row['correct'] == 1:
                            nl1 = max(row['t_nl1'] - row[time], row['t_hand']-row[time])    
                            records.append({'lifted_target': target, 'nl': nl1, 'block': phase_number, 
                                          'participant': name, 'target_name': utils.object_list[:-1][target]})
    return pd.DataFrame(records)

def fit_mixed_effects_model(df):
    """Fit mixed-effects model with phase and target interaction"""
    df['lifted_target'] = pd.Categorical(df['lifted_target'])
    df['block'] = pd.Categorical(df['block'])
    
    model = smf.mixedlm("nl ~ C(block) * C(lifted_target)", 
                        data=df, 
                        groups=df["participant"])
    
    result = model.fit()
    return result, df

# Main execution
df = load_and_process_data()
df = load_and_process_data()
# Define the output directory
output_dir = Path("..") / "results" / "mixed_linear_model_data"
output_dir.mkdir(parents=True, exist_ok=True)  # create if it doesn't exist
# Define the output file
output_file = output_dir / "processed_data.csv"
# Save the dataframe
df.to_csv(output_file, index=False)
print(f"Data saved to: {output_file.resolve()}")
result, df = fit_mixed_effects_model(df)
print(result.summary())

