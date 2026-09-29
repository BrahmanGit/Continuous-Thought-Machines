from experiment_runner import (
    ExperimentConfig,
    run_experiment,
    append_result_to_csv,
)


noise_values = [
    0.05,
    0.5,
    1.0,
    2.0,
    4.0,
]


for noise in noise_values:
    config = ExperimentConfig(
        experiment_name="noise_probe",
        noise_std=noise,
        run_seed=0,
        epochs=15,
        evaluate_test=False,
    )

    print(f"\nRunning noise={noise}")

    metrics = run_experiment(
        config,
        verbose=False,
    )

    append_result_to_csv(
        metrics,
        "results/ctm_runs.csv",
    )

    print(
        f"noise={noise:.2f} | "
        f"val_acc={metrics['val_accuracy_most_certain']:.3f} | "
        f"tick={metrics['val_mean_most_certain_tick']:.1f}"
    )
