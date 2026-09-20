# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 07:53:15 2025

@author: joni_
"""

import os
import numpy as np
import utils
from functools import partial
import multiprocessing
import NN_utils
import torch
import time
import models
import models_jit
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product

def normalize_config(beta, noise_p, noise_o, reset, window):
    """
    Create a float-safe, deterministic configuration key.
    """
    return (
        round(float(beta), 6),
        round(float(noise_p), 12),
        round(float(noise_o), 12),
        int(reset),
        int(window))

target_names = utils.object_list
target_map = dict(zip(range(8), target_names))

# Define data paths for input and output files
grasp_data_path = "../results/grasping_data/"
participant_data_path = "../../data/dataframes/"
results_data_path = "../results/parameter_space/"

# Time step size in milliseconds (or relevant time unit)
dt = 0.015
lists, strings = utils.generate_finger_subsets()

# Define parameter ranges
betas = np.round(np.arange(0.05, 0.325, 0.025), 3)
noise_process = np.array([1e-7, 1e-6, 1e-5, 1e-4])
noise_obs = np.array([1e-7, 1e-6, 1e-5, 1e-4])
integration_window_lengths = [1, 5, 10, 15, 20]#np.concatenate([np.arange(2, 11), [15, 20, 50]])
reset_intervals = [1, 5, 10, 15, 20]
mov_dur = [68.6875, 81.07936507936508, 78.44444444444444, 100.40625, 91.078125]


def generate_all_parameter_combinations():
    """
    Generate all possible parameter combinations.
    Returns a list of tuples (beta, noise_proc, noise_obs, reset_interval, integration_window_length)
    """
    all_combinations = []
    
    for beta in betas:
        for noise_proc in noise_process:
            for noise_observation in noise_obs:
                for integration_window_length in integration_window_lengths:
                    for reset_interval in reset_intervals:
                        # reset_interval = integration_window_length
                    
                        config = normalize_config(
                            beta,
                            noise_proc,
                            noise_observation,
                            reset_interval,
                            integration_window_length
                        )
                        all_combinations.append(config)
    
    return all_combinations


def get_remaining_configurations(metrics_file_path, all_combinations):
    """
    Load existing metrics file and determine which configurations still need to be computed.
    
    Returns:
        - metrics_records: list of existing metric records
        - remaining_configs: list of configurations that need to be computed
    """
    metrics_records = []
    completed_configs = set()
    
    # Check if metrics file exists
    if os.path.exists(metrics_file_path):
        print(f"Found existing metrics file: {metrics_file_path}")
        existing_metrics_df = pd.read_csv(metrics_file_path)
        print(existing_metrics_df.keys())
        metrics_records = existing_metrics_df.to_dict('records')
        print(f"Loaded {len(metrics_records)} existing configurations")
        
        # Extract completed configurations
        for record in metrics_records:
            config_key = normalize_config(
                record['beta'],
                record['noise_process'],
                record['noise_obs'],
                record['reset_interval'],
                record['integration_window_length']
            )
            completed_configs.add(config_key)
    else:
        print(f"No existing metrics file found. Starting fresh.")
    
    # Calculate remaining configurations
    remaining_configs = [config for config in all_combinations if config not in completed_configs]
    
    print(f"Total configurations: {len(all_combinations)}")
    print(f"Completed configurations: {len(completed_configs)}")
    print(f"Remaining configurations: {len(remaining_configs)}\n")
    
    return metrics_records, remaining_configs


def process_participant(participant_idx, participant_name):
    """
    Process a single participant through all remaining parameter configurations.
    """
    # Create participant-specific metrics file path
    metrics_file_path = results_data_path + f'hyperparameter_search_metrics_{participant_name}.csv'
    
    # Generate all possible parameter combinations
    all_combinations = generate_all_parameter_combinations()
    
    # Get remaining configurations to compute
    metrics_records, remaining_configs = get_remaining_configurations(
        metrics_file_path, all_combinations
    )
    
    if len(remaining_configs) == 0:
        print(f"[{participant_name}] All configurations already completed!")
        return participant_name, len(metrics_records)
    
    # Load participant data
    grasp_df = utils.load_dataframe_from_csv(grasp_data_path + participant_name + '_Xy_movement.csv')
    static_grasp_df = utils.load_dataframe_from_csv(grasp_data_path + participant_name + '_Xy.csv')
    X_static = static_grasp_df[utils.glove_data_columns].to_numpy()
    X_static = utils.preprocess_data_finger_config(X_static, finger_config=lists[-1])
    y_static = static_grasp_df['class'].to_numpy()
    print(np.unique(y_static))
    df = utils.load_dataframe_from_csv(participant_data_path + 'processed_' + participant_name + '.csv')
    
    # Iterate through remaining configurations
    for idx, config in enumerate(remaining_configs):
        beta, noise_proc, noise_observation, reset_interval, integration_window_length = config
        
        config_time = time.time()
        
        print(f"[{participant_name}] [{idx+1}/{len(remaining_configs)}] Executing: "
              f"beta={beta:.2f}, noise_proc={noise_proc:.0e}, noise_obs={noise_observation:.0e}, "
              f"reset={reset_interval}, window={integration_window_length}")
        
        # Initialize classifier
        nc = models.NearestCentroidClassifier()
        nc.integration_window = integration_window_length
        nc.fit(X_static, y_static)
        
        pred_df = utils.create_prediction_dataframe()
        
        # Process all trials
        for phase in range(1, df['block'].max() + 1):
            for trial_number in np.arange(df[df['block'] == phase]['trial_number'].max() + 1):
                for rep_counter in range(df[(df['block'] == phase) &
                                            (df['trial_number'] == trial_number)]['rep_number'].max() + 1):

                    df_att = df[(df['block'] == phase) &
                                (df['trial_number'] == trial_number) &
                                (df['rep_number'] == rep_counter)]
    
                    glove_data = df_att[utils.glove_data_columns].to_numpy()
                    glove_time = df_att['glove_timestamps'].to_numpy()
                    board_data = df_att[utils.board_data_columns].to_numpy()
                    board_data = utils.replace_with_highest(arr=board_data, target=8191.0)
                    board_data = utils.replace_with_highest(arr=board_data, target=8190.0)
                    board_time = df_att['board_timestamps'].to_numpy()
                    
                    palm_data = df_att['palm_data'].to_numpy()
                    palm_data = utils.replace_with_highest(arr=palm_data, target=8191.0)
                    palm_data = utils.replace_with_highest(arr=palm_data, target=8190.0)
                    palm_data = palm_data - palm_data[0]
                    palm_time = df_att['palm_timestamps'].to_numpy()
                    
                    y_true = int(df_att['target_object'].to_numpy()[-1])
                    start = 0
                    lifted_target = int(df_att['target_object'].to_numpy()[-1])
    
                    x0 = utils.preprocess_data_finger_config(glove_data[0], finger_config=lists[-1])
                    obs = utils.preprocess_data_finger_config(glove_data, finger_config=lists[-1])
                    nc_pred = nc.run_sequence(obs)
                
                    grav_model_jitted = models_jit.create_dirichlet_model(
                                                        X=X_static,
                                                        y=y_static,
                                                        noise_process=noise_proc,
                                                        noise_obs=noise_observation,
                                                        x0=x0[0],
                                                        dt=dt,
                                                        integration_window_length=integration_window_length,
                                                        gravitational_const=1/(beta*mov_dur[participant_idx]*dt),
                                                        reset_interval=reset_interval)
                    
                    means, var, ll = grav_model_jitted.run_sequence(observations=obs)
                    
                    pred_df = utils.add_rows_to_prediction_dataframe_batch(
                                                                        df=pred_df,
                                                                        phase=phase,
                                                                        trial=trial_number,
                                                                        repetition=rep_counter,
                                                                        means=means,
                                                                        vars=var,
                                                                        noise_KF=0,
                                                                        noise_obs=0,
                                                                        grav_const=0,
                                                                        likelihoods=ll,
                                                                        nc_preds=nc_pred,
                                                                        window_length=integration_window_length)
        
        # Process results for this configuration
        nl_df = utils.process_participant(df, pred_df)
        print(f"Processed data shape: {nl_df.shape}")
        
        records = []
        for target in np.arange(8):
            for phase_number in [1, 2, 3, 4]:
                times = nl_df[(nl_df['lifted_target'] == target) & (nl_df['block'] == phase_number)]
                
                for _, row in times.iterrows():
                    if row['correct'] == 1:
                        nl1 = max(row['t_nl1'] - row['t_lift'], row['t_hand'] - row['t_lift'])
                        stab = row['stability_pred']
                        
                        records.append({
                            'lifted_target': target_map[target],
                            'nl': nl1,
                            'nl_c': row['t_nc'] - row['t_lift'],
                            'type': 'k=1',
                            'block': phase_number,
                            'participant': participant_name,
                            'correct': row['correct'],
                            'stab': row['stability_pred'],
                            'stab_nc': row['stability_nc']
                        })
        
        # Calculate metrics
        plot_df = pd.DataFrame(records)
        print(f'Result size: {plot_df.shape}')
        
        accuracy = plot_df.shape[0] / 128
        filtered_data = plot_df[(plot_df['type'] == 'k=1')]
        total_samples = len(filtered_data)
        negative_nl_samples = len(filtered_data[filtered_data['nl'] < 0])
        
        if total_samples > 0:
            percentage_negative = (negative_nl_samples / total_samples) * 100
            mean_stab = np.mean(filtered_data['stab_nc'])
            mean_nl = np.mean(filtered_data['nl'])
            
            print(f"[{participant_name}] Accuracy: {accuracy:.2f}, "
                  f"r: {percentage_negative:.3f}%, s: {mean_stab:.3f}, nl: {mean_nl:.3f}, "
                  f"time: {time.time()-config_time:.1f}s")
            
            # Append metrics
            metrics_records.append({
                'participant': participant_name,
                'beta': beta,
                'noise_process': noise_proc,
                'noise_obs': noise_observation,
                'reset_interval': reset_interval,
                'integration_window_length': integration_window_length,
                'accuracy': accuracy,
                'r': percentage_negative,
                's': mean_stab,
                'nl': mean_nl
            })
            
            # Save after each configuration
            try:
                metrics_df = pd.DataFrame(metrics_records)
                os.makedirs(os.path.dirname(metrics_file_path), exist_ok=True)
                metrics_df.to_csv(metrics_file_path, index=False)
                print(f"[{participant_name}] Saved ({len(metrics_records)} configs completed)\n")
            except Exception as e:
                print(f"[{participant_name}] ERROR saving metrics: {e}\n")

    # Final save for this participant
    try:
        metrics_df = pd.DataFrame(metrics_records)
        final_file_path = results_data_path + f'hyperparameter_search_metrics_{participant_name}_FINAL.csv'
        os.makedirs(os.path.dirname(final_file_path), exist_ok=True)
        metrics_df.to_csv(final_file_path, index=False)
        
        print(f"\n{'='*60}")
        print(f"COMPLETED: {participant_name}")
        print(f"Total configurations: {len(metrics_records)}")
        print(f"Final file: {final_file_path}")
        print(f"{'='*60}\n")
    except Exception as e:
        print(f"\n[{participant_name}] ERROR saving final metrics: {e}\n")
    
    return participant_name, len(metrics_records)


if __name__ == '__main__':
    # Define participants
    participants = [
        (0, 'participant_1'),
        (1, 'participant_2'),
        (2, 'participant_3'),
        (3, 'participant_4'),
        (4, 'participant_5')
    ]
    
    print(f"\n{'#'*60}")
    print(f"Starting parallel processing of {len(participants)} participants")
    print(f"{'#'*60}\n")
    
    # Create a pool of workers (one per participant)
    num_processes = min(5, len(participants))
    print(f"Using {num_processes} parallel processes\n")
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Process participants in parallel
        results = pool.starmap(process_participant, participants)
    
    # Print summary
    print(f"\n{'#'*60}")
    print("ALL PARTICIPANTS COMPLETED")
    print(f"{'#'*60}")
    for participant_name, num_configs in results:
        print(f"{participant_name}: {num_configs} configurations")
    print(f"{'#'*60}\n")