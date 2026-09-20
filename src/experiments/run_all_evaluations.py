# -*- coding: utf-8 -*-
"""
Unified MAGI pipeline: prediction generation + evaluation.

Steps (all with skip-if-already-computed):
  1. Generate predictions for finger configurations  -> predictions/fingers/
  2. Generate predictions for noise levels            -> predictions/noise/
  3. Generate predictions for object configurations   -> predictions/objects/
  4. Evaluate all predictions (negative latency etc.) -> latency/unified_evaluation_results.csv

Loads model parameters from best_model_config.json (produced by
analyze_hyperparameter_search.py).

@author: jonas

Relies on utilities defined in `utils.py`, `models.py`, `models_jit.py`.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import multiprocessing
import utils
from functools import partial
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
# ============================================================================
# Paths
# ============================================================================
CONFIG_PATH = "../results/parameter_space/best_model_config.json"
grasp_data_path = "../results/grasping_data/"
participant_data_path = "../../data/dataframes/"

PRED_FINGERS_PATH = '../results/predictions/fingers/'
PRED_NOISE_PATH = '../results/predictions/noise/'
PRED_OBJECTS_PATH = '../results/predictions/objects/'

EVAL_RESULTS_PATH = '../results/latency/'
UNIFIED_PATH = os.path.join(EVAL_RESULTS_PATH, 'unified_evaluation_results.csv')

# ============================================================================
# Shared constants
# ============================================================================
PARTICIPANT_NAMES = ['participant_1', 'participant_2', 'participant_3', 'participant_4', 'participant_5']

# Participant-specific mean movement durations (from gridsearch)
MOV_DUR_LIST = [68.6875, 81.07936507936508, 78.44444444444444, 100.40625, 91.078125]

NOISE_LVLS = [0, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4,
              1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]

mean_data_columns = [f'mean_pred_{i}' for i in range(9)]
var_data_columns = [f'var_pred_{i}' for i in range(9)]
nc_pred_data_columns = [f'nc_pred_{i}' for i in range(9)]

lists, strings = utils.generate_finger_subsets()


# ============================================================================
# Config
# ============================================================================
def load_config(path=CONFIG_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Config not found: {path}\n"
            f"Run analyze_hyperparameter_search.py first."
        )
    with open(path, 'r') as f:
        cfg = json.load(f)
    mp = cfg['model_parameters']
    print(f"Loaded config from: {path}")
    print(f"  beta                      = {mp['beta']}")
    print(f"  noise_process             = {mp['noise_process']}")
    print(f"  noise_obs                 = {mp['noise_obs']}")
    print(f"  integration_window_length = {mp['integration_window_length']}")
    print(f"  reset_interval            = {mp['reset_interval']}")
    print(f"  dt                        = {mp['dt']}")
    return cfg


def compute_gravitational_const(beta, dt, participant_idx):
    """Compute the effective gravitational constant, matching the gridsearch.

    In the gridsearch: gravitational_const = 1 / (beta * mov_dur[idx] * dt)
    This must be applied before passing to create_dirichlet_model.
    """
    return 1.0 / (beta * MOV_DUR_LIST[participant_idx] * dt)


# ============================================================================
# Helpers for consistent condition naming
# ============================================================================
def _normalize_condition(condition_value):
    """Ensure condition is always stored and compared as a string."""
    return str(condition_value)


def _load_unified_csv(path):
    """Load the unified CSV with the condition column forced to string dtype."""
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, dtype={'condition': str}, low_memory=False)
    except Exception as e:
        print(f"  Warning: could not read {path} ({e})")
        return None
    if df.empty or 'experiment_type' not in df.columns:
        return None
    return df


# ############################################################################
#
#  STAGE 1: PREDICTION GENERATION
#
# ############################################################################

# ============================================================================
# 1a. Finger predictions
# ============================================================================
def build_finger_params(config):
    """Build param dicts for finger prediction. Skips if output file exists."""
    mp = config['model_parameters']
    params_list = []

    for name_c, name in enumerate(PARTICIPANT_NAMES):
        grasp_df = utils.load_dataframe_from_csv(grasp_data_path + name + '_Xy.csv')
        X = grasp_df[utils.glove_data_columns].to_numpy()
        y = grasp_df['class'].to_numpy()

        # Compute the correct gravitational constant for this participant
        grav_const = compute_gravitational_const(mp['beta'], mp['dt'], name_c)

        for finger_config_name, finger_config in zip(strings[:], lists[:]):
            target_path = os.path.join(PRED_FINGERS_PATH, f"{name}_fingers_{finger_config_name}.csv")

            # Skip if already computed
            if os.path.exists(target_path):
                continue

            Xf = utils.preprocess_data_finger_config(glove_data_subset=X, finger_config=finger_config)
            preprocessing_fn = partial(utils.preprocess_data_finger_config, finger_config=finger_config)

            params_list.append({
                'df_filename': participant_data_path + 'processed_' + name + '.csv',
                'target_df_filename': target_path,
                'beta': grav_const,  # Already transformed: 1/(beta * mov_dur * dt)
                'noise_KF': mp['noise_process'],
                'noise_obs': mp['noise_obs'],
                'X': Xf,
                'y': y,
                'dt': mp['dt'],
                'integration_window_length': mp['integration_window_length'],
                'preprocessing_function_for_data_reduction': preprocessing_fn,
                'reset_interval': mp['reset_interval'],
            })

    return params_list


# ============================================================================
# 1b. Noise predictions
# ============================================================================
def build_noise_params(config):
    """Build param dicts for noise prediction. Skips if output file exists."""
    mp = config['model_parameters']
    params_list = []

    # Noise experiment uses only the last (full) finger configuration
    finger_config = lists[-1]

    for name_c, name in enumerate(PARTICIPANT_NAMES):
        grasp_df = utils.load_dataframe_from_csv(grasp_data_path + name + '_Xy.csv')
        X = grasp_df[utils.glove_data_columns].to_numpy()
        y = grasp_df['class'].to_numpy()

        # Compute the correct gravitational constant for this participant
        grav_const = compute_gravitational_const(mp['beta'], mp['dt'], name_c)

        for noise_lvl in NOISE_LVLS:
            target_path = os.path.join(PRED_NOISE_PATH, f"{name}_noise_{noise_lvl}.csv")

            # Skip if already computed
            if os.path.exists(target_path):
                continue

            Xf = utils.preprocess_data_finger_config(glove_data_subset=X, finger_config=finger_config)
            noise = np.random.normal(loc=0.0, scale=noise_lvl, size=Xf.shape)
            Xf = Xf + noise

            preprocessing_fn = partial(utils.preprocess_data_finger_config, finger_config=finger_config)

            params_list.append({
                'df_filename': participant_data_path + 'processed_' + name + '.csv',
                'target_df_filename': target_path,
                'beta': grav_const,  # Already transformed: 1/(beta * mov_dur * dt)
                'noise_KF': mp['noise_process'],
                'noise_obs': mp['noise_obs'],
                'X': Xf,
                'y': y,
                'dt': mp['dt'],
                'integration_window_length': mp['integration_window_length'],
                'preprocessing_function_for_data_reduction': preprocessing_fn,
                'noise_lvl': noise_lvl,
                'reset_interval': mp['reset_interval'],
            })

    return params_list


# ============================================================================
# 1c. Object predictions
# ============================================================================
def expand_predictions_to_full_classes(means, var, ll, object_config, total_classes=9):
    n_timesteps = means.shape[0]
    means_expanded = np.full((n_timesteps, total_classes), 0).astype(np.float64)
    var_expanded = np.full((n_timesteps, total_classes), 0).astype(np.float64)
    ll_expanded = np.full((n_timesteps, total_classes), 0).astype(np.float64)
    for i, class_idx in enumerate(object_config):
        means_expanded[:, class_idx] = means[:, i]
        var_expanded[:, class_idx] = var[:, i]
        ll_expanded[:, class_idx] = ll[:, i]
    return means_expanded, var_expanded, ll_expanded


def run_object_predictions_for_participant(args):
    """Generate object-config predictions for one participant. Skips existing."""
    import models
    import models_jit

    name_c, participant_name, config = args
    mp = config['model_parameters']
    noise_proc = mp['noise_process']
    noise_obs = mp['noise_obs']
    integration_window_length = mp['integration_window_length']
    preprocessing_fn = partial(utils.preprocess_data_finger_config, finger_config=lists[-1])
    object_combinations = utils.generate_object_subsets()

    # Compute the correct gravitational constant for this participant
    grav_const = compute_gravitational_const(mp['beta'], mp['dt'], name_c)

    static_grasp_df = utils.load_dataframe_from_csv(grasp_data_path + participant_name + '_Xy.csv')
    X_static = static_grasp_df[utils.glove_data_columns].to_numpy()
    X_static = utils.preprocess_data_finger_config(X_static, lists[-1])
    y_static = static_grasp_df['class'].to_numpy()

    for object_config in object_combinations:
        # Only consider object combinations that include object 8
        if 8 not in object_config:
            continue

        object_config_str = '_'.join(map(str, object_config))
        output_path = os.path.join(PRED_OBJECTS_PATH, f"{participant_name}_objects_{object_config_str}.csv")

        # Skip if already computed
        if os.path.exists(output_path):
            continue

        timer_start = time.time()

        Xf_objects, yf_objects = utils.extract_classes(X_static.copy(), y_static, object_config)

        # NC classifier fitted on all classes (same as fingers/noise)
        nc = models.NearestCentroidClassifier()
        nc.integration_window = integration_window_length
        nc.fit(X_static, y_static)

        pred_df = utils.create_prediction_dataframe()
        df = utils.load_dataframe_from_csv(participant_data_path + 'processed_' + participant_name + '.csv')

        for phase in range(1, df['block'].max() + 1):
            for trial_number in np.arange(df[df['block'] == phase]['trial_number'].max() + 1):
                for rep_counter in range(df[(df['block'] == phase) &
                                            (df['trial_number'] == trial_number)]['rep_number'].max() + 1):

                    df_att = df[(df['block'] == phase) &
                                (df['trial_number'] == trial_number) &
                                (df['rep_number'] == rep_counter)]

                    lifted_target = int(df_att['target_object'].to_numpy()[-1])
                    if lifted_target not in object_config:
                        continue

                    glove_data = df_att[utils.glove_data_columns].to_numpy()

                    print('Running inference on:', output_path)
                    print('Phase', phase, 'Trial', trial_number, 'Rep', rep_counter)

                    x0 = preprocessing_fn(glove_data[0])
                    obs = preprocessing_fn(glove_data)
                    nc_pred = nc.run_sequence(obs)

                    # Dirichlet model fitted on object subset
                    grav_model_jitted = models_jit.create_dirichlet_model(
                        X=Xf_objects,
                        y=yf_objects,
                        noise_process=noise_proc,
                        noise_obs=noise_obs,
                        x0=x0[0],
                        dt=mp['dt'],
                        integration_window_length=integration_window_length,
                        gravitational_const=grav_const,  # Correctly transformed
                        reset_interval=mp['reset_interval'])

                    means, var, ll = grav_model_jitted.run_sequence(observations=obs)

                    # Expand subset predictions to full 9-class layout
                    means_expanded, var_expanded, ll_expanded = expand_predictions_to_full_classes(
                        means, var, ll, object_config, total_classes=9)

                    # Batch save (fast)
                    pred_df = utils.add_rows_to_prediction_dataframe_batch(
                        df=pred_df, phase=phase, trial=trial_number,
                        repetition=rep_counter, means=means_expanded,
                        vars=var_expanded, noise_KF=noise_proc,
                        noise_obs=noise_obs, grav_const=grav_const,
                        likelihoods=ll_expanded, nc_preds=nc_pred,
                        window_length=integration_window_length)

        utils.save_dataframe_to_csv(pred_df, output_path)
        print(f"  [objects] {participant_name} config {object_config_str}, {time.time()-timer_start:.1f}s")


# ############################################################################
#
#  STAGE 2: EVALUATION (negative latency, stability, etc.)
#
# ############################################################################

# ============================================================================
# Load existing evaluation results for skip-if-exists
# ============================================================================
def load_existing_results():
    existing_df = _load_unified_csv(UNIFIED_PATH)
    if existing_df is None:
        return None, set()

    completed_keys = set(
        zip(
            existing_df['experiment_type'].astype(str),
            existing_df['participant'].astype(int),
            existing_df['condition'],  # already str from _load_unified_csv
        )
    )
    print(f"Loaded {len(existing_df)} existing evaluation records from {UNIFIED_PATH}")
    print(f"  Already evaluated combos: {len(completed_keys)}")
    return existing_df, completed_keys


# ============================================================================
# Tag row helper
# ============================================================================
def _tag_row(d, experiment_type, condition, name_counter, model_params_dict):
    d['experiment_type'] = experiment_type
    d['condition'] = _normalize_condition(condition)
    d['participant'] = name_counter
    d['beta'] = model_params_dict['beta']
    d['noise_process'] = model_params_dict['noise_process']
    d['noise_obs'] = model_params_dict['noise_obs']
    d['integration_window_length'] = model_params_dict['integration_window_length']
    d['reset_interval'] = model_params_dict['reset_interval']


# ============================================================================
# Evaluation workers — using utils.process_participant for speed
# ============================================================================
def eval_participant_fingers(args):
    # print('hallo')
    name_counter, name, completed_keys, model_params_dict = args
    all_times = []
    df = utils.load_dataframe_from_csv(participant_data_path + "processed_" + name + ".csv")

    for finger_combination in strings[:]:
        condition = _normalize_condition(finger_combination)
        if ('fingers', name_counter, condition) in completed_keys:
            continue
        pred_path = os.path.join(PRED_FINGERS_PATH, f'{name}_fingers_{finger_combination}.csv')
        # print('hi')
        # print(pred_path)
        # if not os.path.exists(pred_path):
        #     continue

        # try:
        pred_df = utils.load_dataframe_from_csv(pred_path)
        nl_df = utils.process_participant(df, pred_df)
        # print(nl_df)
        for _, row in nl_df.iterrows():
            d = row.to_dict()
            _tag_row(d, 'fingers', condition, name_counter+1, model_params_dict)
            all_times.append(d)
    # print(all_times)
        # except Exception as e:
            # print(f"  [fingers] ERROR {name} condition={condition}: {e}")
            # import traceback; traceback.print_exc()
            # continue

    print(f"  [fingers] {name}: {len(all_times)} records from {len(strings)} conditions")
    return all_times


def eval_participant_noise(args):
    name_counter, name, completed_keys, model_params_dict = args
    all_times = []
    df = utils.load_dataframe_from_csv(participant_data_path + "processed_" + name + ".csv")

    for noise_lvl in NOISE_LVLS:
        condition = _normalize_condition(noise_lvl)
        if ('noise', name_counter, condition) in completed_keys:
            continue
        pred_path = os.path.join(PRED_NOISE_PATH, f'{name}_noise_{noise_lvl}.csv')
        if not os.path.exists(pred_path):
            continue

        try:
            pred_df = utils.load_dataframe_from_csv(pred_path)
            nl_df = utils.process_participant(df, pred_df)
            for _, row in nl_df.iterrows():
                d = row.to_dict()
                _tag_row(d, 'noise', condition, name_counter, model_params_dict)
                all_times.append(d)
        except Exception as e:
            print(f"  [noise] ERROR {name} condition={condition}: {e}")
            import traceback; traceback.print_exc()
            continue

    print(f"  [noise] {name}: {len(all_times)} records from {len(NOISE_LVLS)} noise levels")
    return all_times


def eval_participant_objects(args):
    """Evaluate object configurations using utils.process_participant."""
    name_counter, name, completed_keys, model_params_dict = args
    all_times = []
    object_combinations = utils.generate_object_subsets()

    df = utils.load_dataframe_from_csv(participant_data_path + "processed_" + name + ".csv")
    n_processed = 0

    for object_config in object_combinations:
        object_config_str = '_'.join(map(str, object_config))
        condition = _normalize_condition(object_config_str)
        if ('objects', name_counter, condition) in completed_keys:
            continue
        pred_path = os.path.join(PRED_OBJECTS_PATH, f'{name}_objects_{object_config_str}.csv')
        if not os.path.exists(pred_path):
            continue

        try:
            pred_df = utils.load_dataframe_from_csv(pred_path)
            nl_df = utils.process_participant(df, pred_df)
            n_processed += 1
            for _, row in nl_df.iterrows():
                d = row.to_dict()
                _tag_row(d, 'objects', condition, name_counter, model_params_dict)
                d['object_config'] = list(object_config)
                d['n_objects'] = len(object_config)
                all_times.append(d)
        except Exception as e:
            print(f"  [objects] ERROR {name} condition={condition}: {e}")
            import traceback; traceback.print_exc()
            continue

    print(f"  [objects] {name}: {len(all_times)} records from {n_processed} object configs")
    return all_times


# ############################################################################
#
#  MAIN
#
# ############################################################################

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    num_cores = multiprocessing.cpu_count()
    num_workers = min(7, num_cores)
    print(f"CPU cores: {num_cores}, using {num_workers} workers\n")

    # Create output directories
    for d in [PRED_FINGERS_PATH, PRED_NOISE_PATH, PRED_OBJECTS_PATH, EVAL_RESULTS_PATH]:
        os.makedirs(d, exist_ok=True)

    # Load config once
    config = load_config()
    model_params_dict = config['model_parameters']

    # Print the effective gravitational constants for verification
    print("\n  Effective gravitational constants per participant:")
    for i, name in enumerate(PARTICIPANT_NAMES):
        gc = compute_gravitational_const(model_params_dict['beta'],
                                         model_params_dict['dt'], i)
        print(f"    {name}: 1/({model_params_dict['beta']} * {MOV_DUR_LIST[i]:.4f} * {model_params_dict['dt']}) = {gc:.6f}")

    # ==================================================================
    # STAGE 1: Generate predictions (skips existing files)
    # ==================================================================

    # --- 1a. Finger predictions ---
    print("\n" + "=" * 70)
    print("STAGE 1a: Generating finger predictions")
    print("=" * 70)
    finger_params = build_finger_params(config)
    if finger_params:
        print(f"  {len(finger_params)} configurations to compute (others already exist)")
        with multiprocessing.Pool(processes=num_workers) as pool:
            pool.map(utils.run_inference_MAGI, finger_params)
        print("  Done.")
    else:
        print("  All finger predictions already exist, skipping.")

    # --- 1b. Noise predictions ---
    print("\n" + "=" * 70)
    print("STAGE 1b: Generating noise predictions")
    print("=" * 70)
    noise_params = build_noise_params(config)
    if noise_params:
        print(f"  {len(noise_params)} configurations to compute (others already exist)")
        with multiprocessing.Pool(processes=num_workers) as pool:
            pool.map(utils.run_inference_MAGI_with_noise, noise_params)
        print("  Done.")
    else:
        print("  All noise predictions already exist, skipping.")

    # --- 1c. Object predictions ---
    print("\n" + "=" * 70)
    print("STAGE 1c: Generating object predictions")
    print("=" * 70)
    object_worker_args = [(i, name, config) for i, name in enumerate(PARTICIPANT_NAMES)]
    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.map(run_object_predictions_for_participant, object_worker_args)
    print("  Done.")

    # ==================================================================
    # STAGE 2: Evaluate predictions (skips already-evaluated conditions)
    # ==================================================================

    print("\n" + "=" * 70)
    print("STAGE 2: Evaluating predictions")
    print("=" * 70)

    existing_df, completed_keys = load_existing_results()

    eval_args = [
        (i, name, completed_keys, model_params_dict)
        for i, name in enumerate(PARTICIPANT_NAMES)
    ]

    new_results = []

    # --- Evaluate fingers ---
    print("\n  Evaluating finger configurations...")
    with multiprocessing.Pool(processes=num_workers) as pool:
        for rows in pool.map(eval_participant_fingers, eval_args):
            new_results.extend(rows)
    n_fingers = sum(1 for r in new_results if r['experiment_type'] == 'fingers')
    print(f"    New finger records: {n_fingers}")

    # --- Evaluate noise ---
    print("  Evaluating noise levels...")
    with multiprocessing.Pool(processes=num_workers) as pool:
        for rows in pool.map(eval_participant_noise, eval_args):
            new_results.extend(rows)
    n_noise = sum(1 for r in new_results if r['experiment_type'] == 'noise')
    print(f"    New noise records: {n_noise}")

    # --- Evaluate objects ---
    print("  Evaluating object configurations...")
    with multiprocessing.Pool(processes=num_workers) as pool:
        for rows in pool.map(eval_participant_objects, eval_args):
            new_results.extend(rows)
    n_objects = sum(1 for r in new_results if r['experiment_type'] == 'objects')
    print(f"    New object records: {n_objects}")

    # --- Merge old + new ---
    df_new = pd.DataFrame(new_results)

    if existing_df is not None and not df_new.empty:
        # Deduplicate: drop old rows whose keys appear in new results
        new_keys = set(
            zip(
                df_new['experiment_type'].astype(str),
                df_new['participant'].astype(int),
                df_new['condition'].astype(str),
            )
        )
        keep_mask = ~existing_df.apply(
            lambda row: (
                str(row['experiment_type']),
                int(row['participant']),
                str(row['condition']),
            ) in new_keys,
            axis=1
        )
        existing_kept = existing_df[keep_mask]
        df_all = pd.concat([existing_kept, df_new], ignore_index=True)
        print(f"\n  Kept {len(existing_kept)} existing + {len(df_new)} new = {len(df_all)} total")
        print(f"  (Dropped {len(existing_df) - len(existing_kept)} stale/duplicate rows)")
    elif existing_df is not None:
        df_all = existing_df
        print(f"\n  No new evaluations needed — all {len(df_all)} records already present.")
    elif not df_new.empty:
        df_all = df_new
        print(f"\n  Computed {len(df_all)} evaluation records from scratch.")
    else:
        print("\n  No evaluation results. Check prediction directories.")
        df_all = pd.DataFrame()

    # --- Save ---
    if not df_all.empty:
        # Ensure condition is always string before saving
        df_all['condition'] = df_all['condition'].astype(str)

        df_all.to_csv(UNIFIED_PATH, index=False)
        print(f"\n  Unified results: {UNIFIED_PATH} ({len(df_all)} records)")

        for exp_type in df_all['experiment_type'].unique():
            subset = df_all[df_all['experiment_type'] == exp_type]
            subset_path = os.path.join(EVAL_RESULTS_PATH, f'evaluation_{exp_type}.csv')
            subset.to_csv(subset_path, index=False)
            print(f"    {exp_type}: {len(subset)} records -> {subset_path}")

        config_copy_path = os.path.join(EVAL_RESULTS_PATH, 'config_used.json')
        with open(config_copy_path, 'w') as f:
            json.dump(config, f, indent=2)

    print("\n=== Pipeline complete ===")