from pprint import pprint

from experiment_runner import (
    ExperimentConfig,
    append_result_to_csv,
    run_experiment,
)


config = ExperimentConfig(
    experiment_name="baseline_full_trajectory",
    data_seed=0,
    split_seed=0,
    run_seed=0,
    noise_std=0.05,
    wiener_sigma=0.0,
    full_timesteps=100,
    window_start=0,
    window_length=None,
    epochs=15,
    evaluate_test=False,
)

metrics = run_experiment(
    config,
    verbose=True,
)

append_result_to_csv(
    metrics,
    "results/ctm_runs.csv",
)

summary = {
    "run_id": metrics["run_id"],
    "best_epoch": metrics["best_epoch"],
    "train_accuracy": metrics["train_accuracy_most_certain"],
    "validation_accuracy": metrics["val_accuracy_most_certain"],
    "validation_final_tick_accuracy": (metrics["val_accuracy_final_tick"]),
    "mean_most_certain_tick": (metrics["val_mean_most_certain_tick"]),
    "runtime_seconds": metrics["runtime_seconds"],
}

pprint(summary)
