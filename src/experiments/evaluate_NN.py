# -*- coding: utf-8 -*-
"""
Combined NN pipeline: train LSTM and CNN, generate predictions, evaluate.

Loads integration_window_length from best_model_config.json so that NNs
use the same window as the Dirichlet model.

Predictions are saved to:
  ../results/predictions/nn/{name}_{LSTM|CNN}_predictions.csv

Evaluation results are appended to:
  ../results/latency/unified_evaluation_results.csv
  with experiment_type = 'nn_lstm' or 'nn_cnn'

Skips prediction files that already exist on disk.
Skips evaluation conditions already present in the unified CSV.

@author: jonas
"""

import os
import json
import numpy as np
import pandas as pd
import multiprocessing
import torch
import utils
import NN_utils
import models

# ============================================================================
# Paths and constants
# ============================================================================
CONFIG_PATH = "../results/parameter_space/best_model_config.json"
grasp_data_path = "../results/grasping_data/"
participant_data_path = "../../data/dataframes/"
PRED_NN_PATH = '../results/predictions/nn/'
EVAL_RESULTS_PATH = '../results/latency/'
UNIFIED_PATH = os.path.join(EVAL_RESULTS_PATH, 'unified_evaluation_results.csv')

PARTICIPANT_NAMES = [
    'participant_1', 'participant_2', 'participant_3',
    'participant_4', 'participant_5'
]

lists, strings = utils.generate_finger_subsets()

mean_data_columns = [f'mean_pred_{i}' for i in range(9)]
var_data_columns = [f'var_pred_{i}' for i in range(9)]
nc_pred_data_columns = [f'nc_pred_{i}' for i in range(9)]


# ============================================================================
# Config
# ============================================================================
def load_config(path=CONFIG_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, 'r') as f:
        cfg = json.load(f)
    mp = cfg['model_parameters']
    print(f"Loaded config: integration_window_length={mp['integration_window_length']}")
    return cfg


# ============================================================================
# Stage 1: Train NN and generate predictions for one participant
# ============================================================================
def train_and_predict(name_c, name, nn_type, integration_window_length):
    """
    Train a CNN or LSTM for one participant and generate predictions.
    Skips if prediction file already exists.
    """
    target_path = os.path.join(PRED_NN_PATH, f"{name}_{nn_type}_predictions.csv")

    if os.path.exists(target_path):
        print(f"  [{nn_type}] Skipping {name} — predictions already exist")
        return

    print(f"  [{nn_type}] Training and predicting for {name}...")

    # Load grasp data
    grasp_df = utils.load_dataframe_from_csv(grasp_data_path + name + '_Xy_movement.csv')
    static_grasp_df = utils.load_dataframe_from_csv(grasp_data_path + name + '_Xy.csv')
    X_static = static_grasp_df[utils.glove_data_columns].to_numpy()
    X_static = utils.preprocess_data_finger_config(X_static, finger_config=lists[-1])
    y_static = static_grasp_df['class'].to_numpy()

    # Train/val split
    arr = np.arange(8)
    shuffled = np.random.permutation(arr)
    split_idx = int(0.7 * len(shuffled))
    arr_70 = shuffled[:split_idx]
    arr_30 = shuffled[split_idx:]

    train_X, train_y = [], []
    for class_ in range(9):
        for rep in arr_70:
            X = grasp_df[(grasp_df['class'] == class_) & (grasp_df['rep'] == rep)]
            X = X[utils.glove_data_columns].to_numpy()
            Xf = utils.preprocess_data_finger_config(glove_data_subset=X, finger_config=lists[-1])
            train_X.append(Xf)
            train_y.append(class_)

    val_X, val_y = [], []
    for class_ in range(9):
        for rep in arr_30:
            X = grasp_df[(grasp_df['class'] == class_) & (grasp_df['rep'] == rep)]
            X = X[utils.glove_data_columns].to_numpy()
            Xf = utils.preprocess_data_finger_config(glove_data_subset=X, finger_config=lists[-1])
            val_X.append(Xf)
            val_y.append(class_)

    train_X_win, train_y_win = NN_utils.create_windows(train_X, train_y, window_size=integration_window_length)
    val_X_win, val_y_win = NN_utils.create_windows(val_X, val_y, window_size=integration_window_length)

    train_dataset = NN_utils.SeqDataset(train_X_win, train_y_win, task="classification")
    val_dataset = NN_utils.SeqDataset(val_X_win, val_y_win, task="classification")

    batch_size = 32
    train_sampler = NN_utils.BalancedClassSampler(train_y_win, shuffle=True)
    val_sampler = NN_utils.BalancedClassSampler(val_y_win, shuffle=True)
    train_loader = NN_utils.DataLoader(train_dataset, batch_size=batch_size,
                                        sampler=train_sampler, collate_fn=NN_utils.collate_fn)
    val_loader = NN_utils.DataLoader(val_dataset, batch_size=batch_size,
                                      sampler=val_sampler, collate_fn=NN_utils.collate_fn)

    input_dim = train_X[0].shape[1]
    hidden_dim = 32
    fc_dim = 16
    output_dim = 9

    if nn_type == 'LSTM':
        model = NN_utils.LSTMNetwork(input_dim, hidden_dim, fc_dim, output_dim)
    elif nn_type == 'CNN':
        model = NN_utils.CNNNetwork(input_dim, hidden_dim, fc_dim, output_dim)
    else:
        raise ValueError(f"Unknown nn_type: {nn_type}")

    trained_model = NN_utils.train_model(model, train_loader, val_loader,
                                          task="classification", epochs=40)

    # Nearest centroid classifier for nc_pred baseline
    nc = models.NearestCentroidClassifier()
    nc.integration_window = integration_window_length
    nc.fit(X_static, y_static)

    # Generate predictions on experimental data
    df = utils.load_dataframe_from_csv(participant_data_path + 'processed_' + name + '.csv')
    pred_df = utils.create_prediction_dataframe()

    for phase in range(1, df['block'].max() + 1):
        for trial_number in np.arange(df[df['block'] == phase]['trial_number'].max() + 1):
            for rep_counter in range(df[(df['block'] == phase) &
                                        (df['trial_number'] == trial_number)]['rep_number'].max() + 1):

                df_att = df[(df['block'] == phase) &
                            (df['trial_number'] == trial_number) &
                            (df['rep_number'] == rep_counter)]

                glove_data = df_att[utils.glove_data_columns].to_numpy()

                x0 = utils.preprocess_data_finger_config(glove_data[0], finger_config=lists[-1])
                obs = utils.preprocess_data_finger_config(glove_data, finger_config=lists[-1])
                nc_pred = nc.run_sequence(obs)

                # Pad beginning with repeated first row for windowing
                first_row = obs[0]
                rows_to_add = np.tile(first_row, (integration_window_length, 1))
                obs_padded = np.vstack((rows_to_add, obs))

                X_new = []
                for start_idx in range(0, len(obs_padded) - integration_window_length + 1):
                    window = obs_padded[start_idx: start_idx + integration_window_length]
                    X_new.append(window)
                X_new = np.array(X_new)

                model.eval()
                with torch.no_grad():
                    X_tensor = torch.tensor(X_new, dtype=torch.float32)
                    lengths = torch.full((X_tensor.size(0),), X_tensor.size(1), dtype=torch.long)
                    means = model(X_tensor, lengths).numpy()

                means = np.array(means)[:, :]
                var = np.zeros(means.shape)
                ll = np.zeros(means.shape)

                for mean, variance, logl, nc_p in zip(means, var, ll, nc_pred):
                    utils.add_row_to_prediction_dataframe(
                        df=pred_df, phase=phase, trial=trial_number,
                        repetition=rep_counter, mean_pred=mean,
                        var_pred=variance, noise_KF=0, noise_obs=0,
                        grav_const=0, likelihoods=logl, nc_pred=nc_p,
                        window_length=integration_window_length)

    utils.save_dataframe_to_csv(pred_df, target_path)
    print(f"  [{nn_type}] {name} predictions saved to {target_path}")


# ============================================================================
# Stage 2: Evaluate NN predictions (same logic as MAGI evaluation)
# ============================================================================
def evaluate_trial(df_att, pred_df_att, trial_number):
    """Compute negative latency metrics for a single trial."""
    if df_att.empty or pred_df_att.empty:
        return None

    glove_time = df_att['glove_timestamps'].to_numpy()
    glove_data = df_att[utils.glove_data_columns].to_numpy()

    board_data = df_att[utils.board_data_columns].to_numpy()
    board_data = utils.replace_with_highest(arr=board_data, target=8191.0)
    board_data = utils.replace_with_highest(arr=board_data, target=8190.0)

    palm_data = df_att['palm_data'].to_numpy()
    palm_data = utils.replace_with_highest(arr=palm_data, target=8191.0)
    palm_data = utils.replace_with_highest(arr=palm_data, target=8190.0)
    palm_data = palm_data - palm_data[0]

    y_true = int(df_att['target_object'].to_numpy()[-1])
    lifted_target = y_true
    start = 0

    means = pred_df_att[mean_data_columns].to_numpy()
    nc_pred = pred_df_att[nc_pred_data_columns].to_numpy()

    speed = utils.transform_data_to_speed_profile(glove_data, glove_time)

    # Stability analysis
    end = -1
    board_ts = board_data[start:end, int(trial_number)]
    board_ts = board_ts - board_ts[0]
    t_lift = utils.first_change_index(time_series=board_ts, threshold=board_ts.max() * 0.1)

    board_ts_flip = board_ts[::-1] - board_ts[end]
    t_down = len(board_ts) - utils.first_change_index(time_series=board_ts_flip, threshold=board_ts.max() * 0.1)
    if t_down <= t_lift:
        t_down = len(board_ts) - 1

    pred_dur_grasp = means.argmax(axis=1)[t_lift:t_down]
    nc_pred_dur_grasp = nc_pred.argmax(axis=1)[t_lift:t_down]

    hand_lift = utils.first_change_index(time_series=palm_data, threshold=palm_data.max() * 0.1)

    stability_pred = np.count_nonzero(pred_dur_grasp == y_true) / pred_dur_grasp.shape[0]
    stability_nc = np.count_nonzero(nc_pred_dur_grasp == y_true) / nc_pred_dur_grasp.shape[0]

    # Negative latency timing
    end = np.argmax(board_data[:, int(trial_number)])
    board_ts = board_data[start:end, int(trial_number)]
    board_ts = board_ts - board_ts[0]
    t_lift = utils.first_change_index(time_series=board_ts, threshold=board_ts.max() * 0.1)

    t_nc_pred = utils.find_first_of_final_sequence(nc_pred[start:end].argmax(axis=1), nc_pred[end].argmax())

    # Top-1
    incrementer = 0
    t_grav_pred_1 = end
    while t_lift + incrementer < end:
        t_grav_pred_1 = utils.find_first_of_final_sequence(
            means.argmax(axis=1)[start:t_lift + incrementer], y_true) + start
        if means[start + t_lift + incrementer].argmax() == y_true:
            break
        else:
            incrementer += 1

    if means.argmax(axis=1)[start + t_nc_pred] == nc_pred[end].argmax():
        t_grav_pred_1 = t_grav_pred_1
    else:
        t_grav_pred_1 = t_nc_pred + start

    cumsum, sets = utils.analyze_probability_distribution(means)
    t_grav_pred_2 = utils.find_first_of_final_sequence(sets[start:start + end, :2], y_true) + start
    t_grav_pred_3 = utils.find_first_of_final_sequence(sets[start:start + end, :3], y_true) + start

    # Assemble result
    d = {}
    d['correct'] = 1 if y_true in means[t_lift:t_down].argmax(axis=1) else 0
    d['nc_correct'] = 1 if y_true in nc_pred[t_lift:t_down].argmax(axis=1) else 0
    d['movement_start'] = glove_time[np.where(df_att['hand_active'] == 1)[0][0]]
    d['lifted_target'] = lifted_target
    d['t_lift'] = glove_time[start:end][t_lift]
    d['t_hand'] = glove_time[start:end][hand_lift]
    d['stability_pred'] = stability_pred
    d['stability_nc'] = stability_nc
    d['t_nc'] = glove_time[start:end][t_nc_pred]
    d['t_nl1'] = glove_time[t_grav_pred_1]
    d['t_nl2'] = glove_time[t_grav_pred_2]
    d['t_nl3'] = glove_time[t_grav_pred_3]

    d['negative_latency_k1_lift'] = max(glove_time[hand_lift] - glove_time[start:end][t_lift],
                                         glove_time[t_grav_pred_1] - glove_time[start:end][t_lift])

    return d


def iterate_trials(df, pred_df):
    for phase in range(1, int(pred_df['block'].max()) + 1):
        phase_mask = pred_df['block'] == phase
        for trial_number in range(int(pred_df[phase_mask]['trial_number'].max()) + 1):
            trial_mask = phase_mask & (pred_df['trial_number'] == trial_number)
            for rep_counter in range(int(pred_df[trial_mask]['rep_number'].max()) + 1):
                rep_mask = trial_mask & (pred_df['rep_number'] == rep_counter)
                pred_df_att = pred_df[rep_mask]
                df_att = df[(df['block'] == phase) &
                            (df['trial_number'] == trial_number) &
                            (df['rep_number'] == rep_counter) &
                            (df['trial_success'] == 1)]
                if df_att.empty or pred_df_att.empty:
                    continue
                result = evaluate_trial(df_att, pred_df_att, trial_number)
                if result is not None:
                    result['block'] = phase
                    result['trial_number'] = trial_number
                    result['rep_number'] = rep_counter
                    yield result


def eval_participant_nn(args):
    """Evaluate NN predictions for one participant and nn_type."""
    name_counter, name, nn_type, completed_keys, model_params_dict = args
    experiment_type = f'nn_{nn_type.lower()}'
    condition = 'all_fingers'

    if (experiment_type, name_counter, condition) in completed_keys:
        return []

    pred_path = os.path.join(PRED_NN_PATH, f'{name}_{nn_type}_predictions.csv')
    if not os.path.exists(pred_path):
        print(f"  [{nn_type}] No predictions for {name}, skipping evaluation")
        return []

    df = utils.load_dataframe_from_csv(participant_data_path + "processed_" + name + ".csv")
    pred_df = utils.load_dataframe_from_csv(pred_path)

    all_times = []
    for d in iterate_trials(df, pred_df):
        d['experiment_type'] = experiment_type
        d['condition'] = condition
        d['participant'] = name_counter
        d['beta'] = model_params_dict.get('beta', np.nan)
        d['noise_process'] = model_params_dict.get('noise_process', np.nan)
        d['noise_obs'] = model_params_dict.get('noise_obs', np.nan)
        d['integration_window_length'] = model_params_dict.get('integration_window_length', np.nan)
        d['reset_interval'] = model_params_dict.get('reset_interval', np.nan)
        all_times.append(d)

    return all_times


# ============================================================================
# Load existing evaluation results
# ============================================================================
def load_existing_results():
    if not os.path.exists(UNIFIED_PATH):
        return None, set()
    try:
        existing_df = pd.read_csv(UNIFIED_PATH)
    except Exception as e:
        print(f"  Warning: could not read existing results ({e})")
        return None, set()
    if existing_df.empty or 'experiment_type' not in existing_df.columns:
        return None, set()
    completed_keys = set(
        zip(
            existing_df['experiment_type'].astype(str),
            existing_df['participant'].astype(int),
            existing_df['condition'].astype(str),
        )
    )
    return existing_df, completed_keys


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    num_cores = multiprocessing.cpu_count()
    print(f"CPU cores: {num_cores}")

    for d in [PRED_NN_PATH, EVAL_RESULTS_PATH]:
        os.makedirs(d, exist_ok=True)

    config = load_config()
    mp = config['model_parameters']
    integration_window_length = mp['integration_window_length']

    # ==================================================================
    # Stage 1: Train and predict (sequential per participant, parallel not
    # used here because GPU training doesn't benefit from CPU multiprocessing)
    # ==================================================================
    for nn_type in ['LSTM', 'CNN']:
        print(f"\n{'=' * 60}")
        print(f"STAGE 1: {nn_type} — Training and generating predictions")
        print(f"{'=' * 60}")
        for name_c, name in enumerate(PARTICIPANT_NAMES):
            train_and_predict(name_c, name, nn_type, integration_window_length)

    # ==================================================================
    # Stage 2: Evaluate predictions
    # ==================================================================
    print(f"\n{'=' * 60}")
    print("STAGE 2: Evaluating NN predictions")
    print(f"{'=' * 60}")

    existing_df, completed_keys = load_existing_results()
    if existing_df is not None:
        print(f"  Existing records: {len(existing_df)}")

    eval_args = []
    for nn_type in ['LSTM', 'CNN']:
        for i, name in enumerate(PARTICIPANT_NAMES):
            eval_args.append((i, name, nn_type, completed_keys, mp))

    new_results = []
    num_workers = min(5, num_cores)
    with multiprocessing.Pool(processes=num_workers) as pool:
        for rows in pool.map(eval_participant_nn, eval_args):
            new_results.extend(rows)

    print(f"  New NN evaluation records: {len(new_results)}")

    # Merge with existing
    df_new = pd.DataFrame(new_results)

    if existing_df is not None and not df_new.empty:
        df_all = pd.concat([existing_df, df_new], ignore_index=True)
        print(f"  Merged: {len(existing_df)} existing + {len(df_new)} new = {len(df_all)} total")
    elif existing_df is not None:
        df_all = existing_df
        print(f"  No new NN evaluations needed.")
    elif not df_new.empty:
        df_all = df_new
        print(f"  Computed {len(df_all)} NN records from scratch.")
    else:
        print("  No results.")
        df_all = pd.DataFrame()

    if not df_all.empty:
        df_all.to_csv(UNIFIED_PATH, index=False)
        print(f"  Saved to {UNIFIED_PATH} ({len(df_all)} total records)")

    print("\n=== NN pipeline complete ===")