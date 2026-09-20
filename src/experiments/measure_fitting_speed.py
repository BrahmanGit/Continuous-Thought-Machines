# -*- coding: utf-8 -*-
"""
Benchmark: Dirichlet gravity model fitting time.

Measures how long models_jit.create_dirichlet_model() takes to fit
(compute centroids + initialize Kalman filter) over 100 repetitions.

Runs two benchmarks:
  1. Dummy data (synthetic, controlled)
  2. Real data (per participant, using actual grasp data)

Loads model parameters from best_model_config.json.

@author: jonas
"""

import os
import json
import numpy as np
import time
import models_jit
import utils

# ============================================================================
# Load config
# ============================================================================
CONFIG_PATH = "../results/parameter_space/best_model_config.json"

with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

mp = config['model_parameters']
beta = mp['beta']
noise_process = mp['noise_process']
noise_obs = mp['noise_obs']
integration_window_length = mp['integration_window_length']
reset_interval = mp['reset_interval']
dt = mp['dt']

print("Model parameters from config:")
print(f"  beta                      = {beta}")
print(f"  noise_process             = {noise_process}")
print(f"  noise_obs                 = {noise_obs}")
print(f"  integration_window_length = {integration_window_length}")
print(f"  reset_interval            = {reset_interval}")
print(f"  dt                        = {dt}")

N_REPS = 100

# ============================================================================
# Benchmark 1: Dummy data
# ============================================================================
print("\n" + "=" * 70)
print(f"BENCHMARK 1: Dummy data ({N_REPS} repetitions)")
print("=" * 70)

n_classes = 9
n_features = 15
samples_per_class = 200

rng = np.random.default_rng(42)
centroids = rng.uniform(-5, 5, size=(n_classes, n_features))
X_list, y_list = [], []
for c in range(n_classes):
    samples = centroids[c] + rng.normal(0, 0.5, size=(samples_per_class, n_features))
    X_list.append(samples)
    y_list.append(np.full(samples_per_class, c))
X_dummy = np.vstack(X_list)
y_dummy = np.concatenate(y_list)
x0_dummy = X_dummy[0]

print(f"  X shape: {X_dummy.shape}, y shape: {y_dummy.shape}")

# Warmup
for _ in range(5):
    _ = models_jit.create_dirichlet_model(
        X=X_dummy, y=y_dummy,
        noise_process=noise_process, noise_obs=noise_obs,
        x0=x0_dummy, dt=dt,
        integration_window_length=integration_window_length,
        gravitational_const=beta, reset_interval=reset_interval)

# Timed runs
fit_times_dummy = []
for i in range(N_REPS):
    start = time.perf_counter()
    _ = models_jit.create_dirichlet_model(
        X=X_dummy, y=y_dummy,
        noise_process=noise_process, noise_obs=noise_obs,
        x0=x0_dummy, dt=dt,
        integration_window_length=integration_window_length,
        gravitational_const=beta, reset_interval=reset_interval)
    elapsed = time.perf_counter() - start
    fit_times_dummy.append(elapsed)

fit_times_dummy = np.array(fit_times_dummy) * 1000  # convert to ms

print(f"\n  Results ({N_REPS} reps):")
print(f"    Mean:   {np.mean(fit_times_dummy):.4f} ms")
print(f"    Std:    {np.std(fit_times_dummy):.4f} ms")
print(f"    Median: {np.median(fit_times_dummy):.4f} ms")
print(f"    Min:    {np.min(fit_times_dummy):.4f} ms")
print(f"    Max:    {np.max(fit_times_dummy):.4f} ms")

# ============================================================================
# Benchmark 2: Real data (per participant)
# ============================================================================
print("\n" + "=" * 70)
print(f"BENCHMARK 2: Real data per participant ({N_REPS} repetitions each)")
print("=" * 70)

grasp_data_path = "../results/grasping_data/"
lists, strings = utils.generate_finger_subsets()

participant_names = [
    'participant_1', 'participant_2', 'participant_3',
    'participant_4', 'participant_5'
]

all_participant_times = []

for name_c, name in enumerate(participant_names):
    grasp_df = utils.load_dataframe_from_csv(grasp_data_path + name + '_Xy.csv')
    X = grasp_df[utils.glove_data_columns].to_numpy()
    X = utils.preprocess_data_finger_config(X, lists[-1])
    y = grasp_df['class'].to_numpy()
    x0 = X[0]

    print(f"\n  {name}: X shape={X.shape}, classes={len(np.unique(y))}")

    # Warmup
    for _ in range(5):
        _ = models_jit.create_dirichlet_model(
            X=X, y=y,
            noise_process=noise_process, noise_obs=noise_obs,
            x0=x0, dt=dt,
            integration_window_length=integration_window_length,
            gravitational_const=beta, reset_interval=reset_interval)

    # Timed runs
    fit_times = []
    for i in range(N_REPS):
        start = time.perf_counter()
        _ = models_jit.create_dirichlet_model(
            X=X, y=y,
            noise_process=noise_process, noise_obs=noise_obs,
            x0=x0, dt=dt,
            integration_window_length=integration_window_length,
            gravitational_const=beta, reset_interval=reset_interval)
        elapsed = time.perf_counter() - start
        fit_times.append(elapsed)

    fit_times = np.array(fit_times) * 1000  # ms
    all_participant_times.append(fit_times)

    print(f"    Mean:   {np.mean(fit_times):.4f} ms")
    print(f"    Std:    {np.std(fit_times):.4f} ms")
    print(f"    Median: {np.median(fit_times):.4f} ms")
    print(f"    Min:    {np.min(fit_times):.4f} ms")
    print(f"    Max:    {np.max(fit_times):.4f} ms")

# Aggregate across participants
all_times = np.concatenate(all_participant_times)
print(f"\n  Aggregate across all participants ({len(all_times)} total fits):")
print(f"    Mean:   {np.mean(all_times):.4f} ms")
print(f"    Std:    {np.std(all_times):.4f} ms")
print(f"    Median: {np.median(all_times):.4f} ms")
print(f"    Min:    {np.min(all_times):.4f} ms")
print(f"    Max:    {np.max(all_times):.4f} ms")

print("\n" + "=" * 70)
print("Benchmark complete.")
print("=" * 70)