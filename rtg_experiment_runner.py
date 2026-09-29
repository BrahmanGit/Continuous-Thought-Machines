from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from synthetic_data import add_wiener_noise

import torch
from torch.utils.data import DataLoader, TensorDataset

from data_loader import (
    DATA_ROOT,
    FINGERTIP_COLUMNS,
    apply_rep0_target_correction,
    build_recording_inventory,
    derive_experimental_mapping,
    extract_raw_metadata,
    first_change_index,
    replace_with_highest,
    resample_trajectory,
    select_successful_recordings,
    build_reach_to_grasp_dataset,
)

from experiment_runner import (
    build_model,
    ctm_loss,
    evaluate_model,
    resolve_device,
    set_run_seed,
)


# ============================================================
# Configuration
# ============================================================


@dataclass(frozen=True)
class RTGExperimentConfig:
    experiment_name: str = "rtg_b1_b4_baseline"

    # Dataset
    data_root: str = str(DATA_ROOT)
    blocks: tuple[int, ...] = (1, 2, 3, 4)

    # Fixed repetition split
    train_reps: tuple[int, ...] = (0, 1)
    val_reps: tuple[int, ...] = (2,)
    test_reps: tuple[int, ...] = (3,)

    # Dataset dimensions
    n_classes: int = 8
    n_channels: int = 15
    full_timesteps: int = 100

    # DataLoader
    batch_size: int = 32

    # CTM architecture
    iterations: int = 25
    d_model: int = 256
    d_input: int = 64
    heads: int = 4

    n_synch_out: int = 32
    n_synch_action: int = 32

    synapse_depth: int = 1
    memory_length: int = 15

    deep_nlms: bool = True
    memory_hidden_dims: int = 16
    do_layernorm_nlm: bool = False

    dropout: float = 0.0

    neuron_select_type: str = "random-pairing"
    n_random_pairing_self: int = 0

    # Optimization
    learning_rate: float = 3e-4
    weight_decay: float = 0.0

    # Numerical safety
    scale_epsilon: float = 1e-8

    # Execution
    run_seed: int = 0
    device: str = "auto"


# ============================================================
# RTG selection
# ============================================================


def select_rtg_blocks(
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    blocks: tuple[int, ...],
):
    """
    Select recordings belonging to the requested experimental blocks.

    X:
        [N, 15, 100]

    y:
        [N]

    metadata:
        one row per recording
    """

    mask = metadata["block"].isin(blocks).to_numpy()

    X_selected = X[mask]
    y_selected = y[mask]
    metadata_selected = metadata.loc[mask].reset_index(drop=True).copy()

    if not (len(X_selected) == len(y_selected) == len(metadata_selected)):
        raise RuntimeError(
            "X, y, and metadata became misaligned during block selection."
        )

    return (
        X_selected,
        y_selected,
        metadata_selected,
    )


# ============================================================
# Fixed repetition split
# ============================================================


def make_repetition_split(
    metadata: pd.DataFrame,
    train_reps=(0, 1),
    val_reps=(2,),
    test_reps=(3,),
):
    """
    Split B1-B4 recordings according to repetition number.

    The first baseline uses:

        rep 0, 1 -> train
        rep 2    -> validation
        rep 3    -> test

    Returns integer row indices into the selected B1-B4 arrays.
    """

    all_reps = set(train_reps) | set(val_reps) | set(test_reps)

    if (
        set(train_reps) & set(val_reps)
        or set(train_reps) & set(test_reps)
        or set(val_reps) & set(test_reps)
    ):
        raise ValueError(
            "Train, validation, and test repetition sets must not overlap."
        )

    observed_reps = set(metadata["rep_number"].astype(int).unique().tolist())

    if not observed_reps.issubset(all_reps):
        raise ValueError(
            f"Some repetitions are not assigned to a split. "
            f"Observed={observed_reps}, assigned={all_reps}"
        )

    rep_values = metadata["rep_number"].to_numpy()

    train_idx = np.flatnonzero(np.isin(rep_values, train_reps))

    val_idx = np.flatnonzero(np.isin(rep_values, val_reps))

    test_idx = np.flatnonzero(np.isin(rep_values, test_reps))

    return train_idx, val_idx, test_idx


# ============================================================
# Training-only feature scaling
# ============================================================


def fit_channel_scaler(
    X_train: np.ndarray,
    epsilon: float = 1e-8,
):
    """
    Fit one mean and standard deviation for each of the
    15 fingertip-coordinate channels.

    Statistics are calculated across:

        training recordings
        AND
        trajectory positions

    but never across channels.

    X_train shape:
        [N_train, channels, time]
    """

    if X_train.ndim != 3:
        raise ValueError(f"Expected [N, C, T], got {X_train.shape}")

    mean = X_train.mean(
        axis=(0, 2),
        keepdims=True,
    )

    std = X_train.std(
        axis=(0, 2),
        keepdims=True,
    )

    std = np.maximum(std, epsilon)

    return {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
    }


def apply_channel_scaler(
    X: np.ndarray,
    scaler: dict[str, np.ndarray],
):
    return ((X - scaler["mean"]) / scaler["std"]).astype(np.float32)


# ============================================================
# PyTorch loaders
# ============================================================


def make_tensor_dataset(
    X: np.ndarray,
    y: np.ndarray,
):
    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
    )

    y_tensor = torch.tensor(
        y,
        dtype=torch.long,
    )

    return TensorDataset(
        X_tensor,
        y_tensor,
    )


def build_rtg_loaders(
    config: RTGExperimentConfig,
):
    """
    Complete loader bridge:

        RTG dataset builder
            ->
        select B1-B4
            ->
        repetition split
            ->
        fit scaler on training data only
            ->
        scale all partitions
            ->
        PyTorch DataLoaders
    """

    # --------------------------------------------------------
    # Build source-defined RTG representation
    # --------------------------------------------------------

    X, y, metadata, excluded = build_reach_to_grasp_dataset(
        data_root=Path(config.data_root),
    )

    # --------------------------------------------------------
    # Select B1-B4
    # --------------------------------------------------------

    X, y, metadata = select_rtg_blocks(
        X=X,
        y=y,
        metadata=metadata,
        blocks=config.blocks,
    )

    # --------------------------------------------------------
    # Fixed repetition split
    # --------------------------------------------------------

    (
        train_idx,
        val_idx,
        test_idx,
    ) = make_repetition_split(
        metadata=metadata,
        train_reps=config.train_reps,
        val_reps=config.val_reps,
        test_reps=config.test_reps,
    )

    X_train = X[train_idx]
    y_train = y[train_idx]

    X_val = X[val_idx]
    y_val = y[val_idx]

    X_test = X[test_idx]
    y_test = y[test_idx]

    # --------------------------------------------------------
    # Fit normalization ONLY using training recordings
    # --------------------------------------------------------

    scaler = fit_channel_scaler(
        X_train,
        epsilon=config.scale_epsilon,
    )

    X_train = apply_channel_scaler(
        X_train,
        scaler,
    )

    X_val = apply_channel_scaler(
        X_val,
        scaler,
    )

    X_test = apply_channel_scaler(
        X_test,
        scaler,
    )

    # --------------------------------------------------------
    # Torch datasets
    # --------------------------------------------------------

    train_dataset = make_tensor_dataset(
        X_train,
        y_train,
    )

    val_dataset = make_tensor_dataset(
        X_val,
        y_val,
    )

    test_dataset = make_tensor_dataset(
        X_test,
        y_test,
    )

    generator = torch.Generator().manual_seed(config.run_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    # --------------------------------------------------------
    # Record split membership explicitly
    # --------------------------------------------------------

    split_metadata = metadata.copy()

    split_metadata["split"] = ""

    split_metadata.loc[
        train_idx,
        "split",
    ] = "train"

    split_metadata.loc[
        val_idx,
        "split",
    ] = "val"

    split_metadata.loc[
        test_idx,
        "split",
    ] = "test"

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "scaler": scaler,
        "metadata": split_metadata,
        "excluded": excluded,
        "train_indices": train_idx,
        "val_indices": val_idx,
        "test_indices": test_idx,
    }


# ============================================================
# Validating RTG Loaders
# ============================================================


def validate_rtg_loaders(
    bundle,
):
    train_loader = bundle["train_loader"]
    val_loader = bundle["val_loader"]
    test_loader = bundle["test_loader"]

    metadata = bundle["metadata"]

    split_counts = metadata["split"].value_counts().to_dict()

    print("Split sizes:")
    print(split_counts)

    print("\nClass counts by split:")
    print(metadata.groupby(["split", "class_label"]).size().unstack(fill_value=0))

    print("\nRepetitions by split:")
    print(metadata.groupby(["split", "rep_number"]).size().unstack(fill_value=0))

    X_train, y_train = next(iter(train_loader))

    X_val, y_val = next(iter(val_loader))

    X_test, y_test = next(iter(test_loader))

    print("\nExample batch shapes:")
    print("train:", X_train.shape, y_train.shape)
    print("val:  ", X_val.shape, y_val.shape)
    print("test: ", X_test.shape, y_test.shape)

    assert split_counts["train"] == 320
    assert split_counts["val"] == 160
    assert split_counts["test"] == 160

    expected_per_class = {
        "train": 40,
        "val": 20,
        "test": 20,
    }

    counts = metadata.groupby(["split", "class_label"]).size()

    for split, expected in expected_per_class.items():
        split_counts_by_class = counts.loc[split]

        assert (split_counts_by_class == expected).all(), (
            f"Unexpected class counts in {split}:\n{split_counts_by_class}"
        )

    assert X_train.shape[1:] == (15, 100)
    assert X_val.shape[1:] == (15, 100)
    assert X_test.shape[1:] == (15, 100)

    print("\nRTG loader validation passed.")

    X_train = train_loader.dataset.tensors[0]
    X_val = val_loader.dataset.tensors[0]
    X_test = test_loader.dataset.tensors[0]

    print("Finite:")
    print("train:", torch.isfinite(X_train).all().item())
    print("val:  ", torch.isfinite(X_val).all().item())
    print("test: ", torch.isfinite(X_test).all().item())

    print("\nTraining channel means:")
    print(X_train.mean(dim=(0, 2)))

    print("\nTraining channel stds:")
    print(X_train.std(dim=(0, 2), correction=0))


def make_balanced_tiny_subset(
    train_loader: DataLoader,
    n_per_class: int = 8,
    seed: int = 0,
):
    """
    Create a small balanced subset from the already-scaled
    RTG training partition.

    This subset is used only as a memorization / integration test.

    For 8 classes and n_per_class=8:
        8 × 8 = 64 recordings
    """

    dataset = train_loader.dataset

    X_train, y_train = dataset.tensors

    rng = np.random.default_rng(seed)

    selected_indices = []

    for class_id in range(8):
        class_indices = torch.where(y_train == class_id)[0].cpu().numpy()

        if len(class_indices) < n_per_class:
            raise ValueError(
                f"Class {class_id} has only {len(class_indices)} examples."
            )

        selected = rng.choice(
            class_indices,
            size=n_per_class,
            replace=False,
        )

        selected_indices.extend(selected.tolist())

    # Randomize ordering after balanced selection.
    rng.shuffle(selected_indices)

    selected_indices = torch.tensor(
        selected_indices,
        dtype=torch.long,
    )

    X_tiny = X_train[selected_indices]
    y_tiny = y_train[selected_indices]

    tiny_dataset = TensorDataset(
        X_tiny,
        y_tiny,
    )

    return tiny_dataset


def run_tiny_overfit_test(
    bundle,
    config: RTGExperimentConfig,
    n_per_class: int = 8,
    max_epochs: int = 100,
    target_accuracy: float = 0.98,
):
    """
    Test whether the CTM can memorize a tiny balanced subset
    of real RTG trajectories.

    This is NOT a generalization experiment.

    Success here means that:
        - RTG tensors are compatible with CTM,
        - gradients flow,
        - the loss works,
        - optimization can fit real trajectories.

    The validation and test partitions are never touched.
    """

    set_run_seed(config.run_seed)

    device = resolve_device(config.device)

    print(f"Device: {device}")

    # --------------------------------------------------------
    # Tiny balanced subset
    # --------------------------------------------------------

    tiny_dataset = make_balanced_tiny_subset(
        train_loader=bundle["train_loader"],
        n_per_class=n_per_class,
        seed=config.run_seed,
    )

    print(f"Tiny subset size: {len(tiny_dataset)}")

    X_tiny, y_tiny = tiny_dataset.tensors

    print(
        "Class counts:",
        torch.bincount(
            y_tiny,
            minlength=config.n_classes,
        ).tolist(),
    )

    # Training loader is shuffled.
    train_generator = torch.Generator().manual_seed(config.run_seed)

    tiny_train_loader = DataLoader(
        tiny_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=train_generator,
    )

    # Separate deterministic loader for evaluation.
    tiny_eval_loader = DataLoader(
        tiny_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    # --------------------------------------------------------
    # Build CTM
    # --------------------------------------------------------

    model = build_model(
        config,
        device,
    )

    # LazyLinear parameters in the CTM need one dry pass.
    dry_X, _ = next(iter(tiny_train_loader))

    dry_X = dry_X.to(device)

    with torch.no_grad():
        predictions, certainties, _ = model(dry_X)

    print(
        "Prediction shape:",
        tuple(predictions.shape),
    )

    print(
        "Certainty shape:",
        tuple(certainties.shape),
    )

    if predictions.shape[1] != config.n_classes:
        raise RuntimeError("Unexpected number of output classes.")

    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Trainable parameters: {parameter_count:,}")

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history = []

    # --------------------------------------------------------
    # Memorization training
    # --------------------------------------------------------

    for epoch in range(
        1,
        max_epochs + 1,
    ):
        model.train()

        epoch_loss = 0.0
        epoch_seen = 0

        for X_batch, y_batch in tiny_train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad(set_to_none=True)

            predictions, certainties, _ = model(X_batch)

            loss, _ = ctm_loss(
                predictions,
                certainties,
                y_batch,
            )

            loss.backward()

            optimizer.step()

            batch_size = y_batch.size(0)

            epoch_loss += loss.item() * batch_size

            epoch_seen += batch_size

        epoch_loss /= epoch_seen

        # Evaluate on exactly the same 64 recordings.
        metrics = evaluate_model(
            model,
            tiny_eval_loader,
            device,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss,
                **metrics,
            }
        )

        print(
            f"epoch {epoch:03d} | "
            f"loss {epoch_loss:.4f} | "
            f"most-certain acc "
            f"{metrics['accuracy_most_certain']:.3f} | "
            f"final-tick acc "
            f"{metrics['accuracy_final_tick']:.3f} | "
            f"mean tick "
            f"{metrics['mean_most_certain_tick']:.2f}"
        )

        if metrics["accuracy_most_certain"] >= target_accuracy:
            print("\nTiny-subset overfit target reached.")
            break

    return {
        "model": model,
        "history": pd.DataFrame(history),
        "tiny_dataset": tiny_dataset,
        "device": device,
        "parameter_count": parameter_count,
    }


def run_rtg_baseline(
    bundle,
    config: RTGExperimentConfig,
    max_epochs: int = 60,
    verbose: bool = True,
):
    """
    Train the first full B1-B4 RTG baseline.

    Training:
        repetitions 0 + 1

    Validation:
        repetition 2

    Test:
        repetition 3, deliberately NOT evaluated here.

    The best model is selected using validation
    most-certain-tick accuracy, with validation loss
    as the tie-breaker.
    """

    import json
    import time

    from dataclasses import asdict
    from datetime import datetime, timezone
    from pathlib import Path

    # --------------------------------------------------------
    # Reproducibility and device
    # --------------------------------------------------------

    set_run_seed(config.run_seed)

    device = resolve_device(config.device)

    print(f"Device: {device}")

    train_loader = rebuild_training_loader(
        bundle["train_loader"],
        seed=config.run_seed,
    )

    val_loader = bundle["val_loader"]

    # --------------------------------------------------------
    # Build CTM
    # --------------------------------------------------------

    model = build_model(
        config,
        device,
    )

    # Materialize lazy layers.
    dry_X, _ = next(iter(train_loader))

    dry_X = dry_X.to(device)

    with torch.no_grad():
        dry_predictions, dry_certainties, _ = model(dry_X)

    print(
        "Prediction shape:",
        tuple(dry_predictions.shape),
    )

    print(
        "Certainty shape:",
        tuple(dry_certainties.shape),
    )

    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Trainable parameters: {parameter_count:,}")

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # --------------------------------------------------------
    # Best-checkpoint tracking
    # --------------------------------------------------------

    best_state = None

    best_epoch = 0
    best_val_accuracy = float("-inf")
    best_val_loss = float("inf")

    history = []

    started_at = time.perf_counter()

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(
        1,
        max_epochs + 1,
    ):
        model.train()

        epoch_loss = 0.0
        epoch_correct = 0
        epoch_seen = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad(set_to_none=True)

            predictions, certainties, _ = model(X_batch)

            loss, idx_max_certainty = ctm_loss(
                predictions,
                certainties,
                y_batch,
            )

            loss.backward()

            optimizer.step()

            batch_size = y_batch.size(0)

            batch_indices = torch.arange(
                batch_size,
                device=device,
            )

            selected_logits = predictions[
                batch_indices,
                :,
                idx_max_certainty,
            ]

            predicted_classes = selected_logits.argmax(dim=1)

            epoch_loss += loss.item() * batch_size

            epoch_correct += (predicted_classes == y_batch).sum().item()

            epoch_seen += batch_size

        train_loss = epoch_loss / epoch_seen

        train_accuracy = epoch_correct / epoch_seen

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        validation = evaluate_model(
            model,
            val_loader,
            device,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": validation["loss"],
                "val_accuracy_most_certain": validation["accuracy_most_certain"],
                "val_accuracy_final_tick": validation["accuracy_final_tick"],
                "val_mean_most_certain_tick": validation["mean_most_certain_tick"],
                "val_mean_selected_certainty": validation["mean_selected_certainty"],
            }
        )

        # ----------------------------------------------------
        # Validation-based model selection
        # ----------------------------------------------------

        is_better = validation["accuracy_most_certain"] > best_val_accuracy or (
            validation["accuracy_most_certain"] == best_val_accuracy
            and validation["loss"] < best_val_loss
        )

        if is_better:
            best_val_accuracy = validation["accuracy_most_certain"]

            best_val_loss = validation["loss"]

            best_epoch = epoch

            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

        if verbose:
            print(
                f"epoch {epoch:03d} | "
                f"train loss {train_loss:.4f} | "
                f"train acc {train_accuracy:.3f} | "
                f"val loss {validation['loss']:.4f} | "
                f"val MC {validation['accuracy_most_certain']:.3f} | "
                f"val final {validation['accuracy_final_tick']:.3f} | "
                f"mean tick {validation['mean_most_certain_tick']:.2f}"
            )

    if best_state is None:
        raise RuntimeError("No checkpoint was selected.")

    # --------------------------------------------------------
    # Restore winning validation checkpoint
    # --------------------------------------------------------

    model.load_state_dict(best_state)

    train_metrics = evaluate_model(
        model,
        train_loader,
        device,
    )

    val_metrics = evaluate_model(
        model,
        val_loader,
        device,
    )

    runtime_seconds = time.perf_counter() - started_at

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    run_id = f"{timestamp}_{config.experiment_name}_seed{config.run_seed}"

    output_dir = Path("results") / "rtg" / run_id

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save exact winning checkpoint
    # --------------------------------------------------------

    checkpoint = {
        "state_dict": best_state,
        "config": asdict(config),
        "best_epoch": best_epoch,
        "parameter_count": parameter_count,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }

    torch.save(
        checkpoint,
        output_dir / "best_checkpoint.pt",
    )

    # --------------------------------------------------------
    # Save scaling parameters
    # --------------------------------------------------------

    np.savez(
        output_dir / "scaler.npz",
        mean=bundle["scaler"]["mean"],
        std=bundle["scaler"]["std"],
    )

    # --------------------------------------------------------
    # Save fixed split assignment
    # --------------------------------------------------------

    bundle["metadata"].to_csv(
        output_dir / "split_metadata.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save configuration
    # --------------------------------------------------------

    with (output_dir / "config.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            asdict(config),
            f,
            indent=2,
            default=str,
        )

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    history_df = pd.DataFrame(history)

    history_df.to_csv(
        output_dir / "training_history.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {
        "run_id": run_id,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "best_val_loss": best_val_loss,
        "train_accuracy_most_certain": train_metrics["accuracy_most_certain"],
        "train_accuracy_final_tick": train_metrics["accuracy_final_tick"],
        "val_accuracy_most_certain": val_metrics["accuracy_most_certain"],
        "val_accuracy_final_tick": val_metrics["accuracy_final_tick"],
        "val_mean_most_certain_tick": val_metrics["mean_most_certain_tick"],
        "runtime_seconds": runtime_seconds,
        # Deliberately absent:
        # test accuracy
    }

    with (output_dir / "summary.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print("\nBest checkpoint")

    print(f"epoch: {best_epoch}")

    print(
        "train MC accuracy:",
        f"{train_metrics['accuracy_most_certain']:.3f}",
    )

    print(
        "validation MC accuracy:",
        f"{val_metrics['accuracy_most_certain']:.3f}",
    )

    print(
        "validation final-tick accuracy:",
        f"{val_metrics['accuracy_final_tick']:.3f}",
    )

    print(
        "validation mean MC tick:",
        f"{val_metrics['mean_most_certain_tick']:.2f}",
    )

    print(
        "saved to:",
        output_dir,
    )

    return {
        "model": model,
        "history": history_df,
        "summary": summary,
        "output_dir": output_dir,
        "device": device,
    }


def collect_split_predictions(
    model,
    loader,
    device,
):
    """
    Collect detailed predictions from a trained CTM.

    This does not train or modify the model.

    Returns:
        targets
        logits for every internal tick
        certainties for every internal tick
        predicted class at every internal tick
        accuracy at every internal tick
        most-certain tick for every recording
        prediction at the most-certain tick
        prediction at the final tick
        certainty at the most-certain tick
    """

    model.eval()

    all_logits = []
    all_certainties = []
    all_targets = []

    with torch.inference_mode():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)

            predictions, certainties, _ = model(X_batch)

            all_logits.append(predictions.cpu())

            all_certainties.append(certainties.cpu())

            all_targets.append(y_batch.cpu())

    logits = torch.cat(
        all_logits,
        dim=0,
    )

    certainties = torch.cat(
        all_certainties,
        dim=0,
    )

    targets = torch.cat(
        all_targets,
        dim=0,
    )

    # logits:
    # [N, classes, internal_ticks]

    # Predicted class at every internal tick:
    # [N, internal_ticks]
    predictions_per_tick = logits.argmax(dim=1)

    # Accuracy separately for each internal tick:
    # [internal_ticks]
    accuracy_per_tick = (predictions_per_tick == targets[:, None]).float().mean(dim=0)

    # Tick with maximum certainty for each recording.
    most_certain_idx = certainties[:, 1, :].argmax(dim=1)

    sample_indices = torch.arange(len(targets))

    # Prediction chosen at each recording's
    # most-certain internal tick.
    most_certain_prediction = logits[
        sample_indices,
        :,
        most_certain_idx,
    ].argmax(dim=1)

    # Prediction at tick 25.
    final_prediction = logits[:, :, -1].argmax(dim=1)

    # Certainty corresponding to the selected tick.
    most_certain_value = certainties[
        sample_indices,
        1,
        most_certain_idx,
    ]

    return {
        "targets": targets.numpy(),
        "logits": logits.numpy(),
        "certainties": certainties.numpy(),
        "predictions_per_tick": predictions_per_tick.numpy(),
        "accuracy_per_tick": accuracy_per_tick.numpy(),
        # +1 because humans count ticks 1..25,
        # while Python indices them 0..24.
        "most_certain_tick": (most_certain_idx + 1).numpy(),
        "most_certain_prediction": most_certain_prediction.numpy(),
        "final_prediction": final_prediction.numpy(),
        "most_certain_value": most_certain_value.numpy(),
    }


# ============================================================
# Partial-prefix evaluation
# ============================================================


def evaluate_partial_prefixes(
    model,
    loader: DataLoader,
    device,
    prefix_lengths=(10, 25, 50, 75, 100),
):
    """
    Evaluate a frozen CTM on progressively longer trajectory prefixes.

    No training or parameter updates are performed.

    The input loader is expected to contain full standardized
    trajectories with shape:

        [N, channels, full_timesteps]

    For each requested prefix length, the trajectories are cropped to:

        X[:, :, :prefix_length]

    and evaluated using the same frozen model.

    Returns
    -------
    pandas.DataFrame

        One row per prefix length containing:

        - observed trajectory length
        - fraction of movement observed
        - loss
        - most-certain-tick accuracy
        - final-tick accuracy
        - mean most-certain tick
        - mean selected certainty
    """

    model.eval()

    if not isinstance(
        loader.dataset,
        TensorDataset,
    ):
        raise TypeError("Expected loader.dataset to be a TensorDataset.")

    X_full, y = loader.dataset.tensors

    if X_full.ndim != 3:
        raise ValueError(f"Expected X with shape [N, C, T], got {X_full.shape}")

    full_timesteps = X_full.shape[-1]

    results = []

    for prefix_length in prefix_lengths:
        prefix_length = int(prefix_length)

        if prefix_length <= 0:
            raise ValueError("Prefix length must be positive.")

        if prefix_length > full_timesteps:
            raise ValueError(
                f"Prefix length {prefix_length} exceeds "
                f"full trajectory length {full_timesteps}."
            )

        # ----------------------------------------------------
        # Keep only the beginning of each trajectory
        # ----------------------------------------------------

        X_prefix = X_full[
            :,
            :,
            :prefix_length,
        ]

        prefix_dataset = TensorDataset(
            X_prefix,
            y,
        )

        prefix_loader = DataLoader(
            prefix_dataset,
            batch_size=loader.batch_size,
            shuffle=False,
        )

        # ----------------------------------------------------
        # Evaluate the same frozen CTM
        # ----------------------------------------------------

        metrics = evaluate_model(
            model,
            prefix_loader,
            device,
        )

        results.append(
            {
                "prefix_length": prefix_length,
                "fraction_observed": prefix_length / full_timesteps,
                "percent_observed": 100.0 * prefix_length / full_timesteps,
                "loss": metrics["loss"],
                "accuracy_most_certain": metrics["accuracy_most_certain"],
                "accuracy_final_tick": metrics["accuracy_final_tick"],
                "mean_most_certain_tick": metrics["mean_most_certain_tick"],
                "mean_selected_certainty": metrics["mean_selected_certainty"],
            }
        )

    return pd.DataFrame(results)


# ============================================================
# Matched partial-observation training
# ============================================================


def build_prefix_loader(
    loader,
    prefix_length,
    shuffle=False,
    seed=0,
):
    """
    Construct a DataLoader containing only the requested
    prefix of each already-standardized trajectory.

    The original labels and recording membership are preserved.
    """

    if not isinstance(
        loader.dataset,
        TensorDataset,
    ):
        raise TypeError("Expected loader.dataset to be a TensorDataset.")

    X, y = loader.dataset.tensors

    if X.ndim != 3:
        raise ValueError(f"Expected [N, C, T], got {X.shape}")

    if prefix_length <= 0:
        raise ValueError("prefix_length must be positive.")

    if prefix_length > X.shape[-1]:
        raise ValueError(
            f"prefix_length={prefix_length} exceeds trajectory length={X.shape[-1]}."
        )

    X_prefix = X[
        :,
        :,
        :prefix_length,
    ]

    dataset = TensorDataset(
        X_prefix,
        y,
    )

    if shuffle:
        generator = torch.Generator().manual_seed(seed)

        return DataLoader(
            dataset,
            batch_size=loader.batch_size,
            shuffle=True,
            generator=generator,
        )

    return DataLoader(
        dataset,
        batch_size=loader.batch_size,
        shuffle=False,
    )


def run_matched_prefix_training(
    bundle,
    config,
    prefix_lengths=(10, 25, 50, 75, 100),
    max_epochs=60,
    verbose=True,
):
    """
    Train a separate CTM for each movement-prefix length.

    Each condition uses:

        train:
            repetitions 0 and 1

        validation:
            repetition 2

        test:
            repetition 3

    Model selection is performed exclusively using validation
    performance.

    The test partition is evaluated only after the best
    validation checkpoint has been selected.

    The already-standardized tensors from the RTG bundle are
    used so that all prefix conditions share the same
    preprocessing transform.
    """

    results = []

    full_timesteps = bundle["train_loader"].dataset.tensors[0].shape[-1]

    for prefix_length in prefix_lengths:
        prefix_length = int(prefix_length)

        print()
        print("=" * 72)

        print(
            f"Matched prefix training: "
            f"{prefix_length}/{full_timesteps} "
            f"({100 * prefix_length / full_timesteps:.0f}%)"
        )

        print("=" * 72)

        # ----------------------------------------------------
        # Reset random state for this condition
        # ----------------------------------------------------

        set_run_seed(config.run_seed)

        device = resolve_device(config.device)

        # ----------------------------------------------------
        # Prefix-specific loaders
        # ----------------------------------------------------

        train_loader = build_prefix_loader(
            loader=bundle["train_loader"],
            prefix_length=prefix_length,
            shuffle=True,
            seed=config.run_seed,
        )

        val_loader = build_prefix_loader(
            loader=bundle["val_loader"],
            prefix_length=prefix_length,
            shuffle=False,
        )

        test_loader = build_prefix_loader(
            loader=bundle["test_loader"],
            prefix_length=prefix_length,
            shuffle=False,
        )

        # ----------------------------------------------------
        # Build a fresh CTM
        # ----------------------------------------------------

        model = build_model(
            config,
            device,
        )

        # Materialize lazy parameters.
        dry_X, _ = next(iter(train_loader))

        dry_X = dry_X.to(device)

        with torch.no_grad():
            dry_predictions, _, _ = model(dry_X)

        if dry_predictions.shape[1] != config.n_classes:
            raise RuntimeError("Unexpected number of output classes.")

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # ----------------------------------------------------
        # Validation checkpoint tracking
        # ----------------------------------------------------

        best_state = None

        best_epoch = 0

        best_val_accuracy = float("-inf")

        best_val_loss = float("inf")

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        for epoch in range(
            1,
            max_epochs + 1,
        ):
            model.train()

            epoch_loss = 0.0
            epoch_seen = 0

            for (
                X_batch,
                y_batch,
            ) in train_loader:
                X_batch = X_batch.to(device)

                y_batch = y_batch.to(device)

                optimizer.zero_grad(set_to_none=True)

                predictions, certainties, _ = model(X_batch)

                loss, _ = ctm_loss(
                    predictions,
                    certainties,
                    y_batch,
                )

                loss.backward()

                optimizer.step()

                batch_size = y_batch.size(0)

                epoch_loss += loss.item() * batch_size

                epoch_seen += batch_size

            train_loss = epoch_loss / epoch_seen

            # ------------------------------------------------
            # Validation
            # ------------------------------------------------

            validation = evaluate_model(
                model,
                val_loader,
                device,
            )

            is_better = validation["accuracy_most_certain"] > best_val_accuracy or (
                validation["accuracy_most_certain"] == best_val_accuracy
                and validation["loss"] < best_val_loss
            )

            if is_better:
                best_val_accuracy = validation["accuracy_most_certain"]

                best_val_loss = validation["loss"]

                best_epoch = epoch

                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }

            if verbose:
                print(
                    f"prefix {prefix_length:3d} | "
                    f"epoch {epoch:03d} | "
                    f"train loss {train_loss:.4f} | "
                    f"val MC "
                    f"{validation['accuracy_most_certain']:.3f} | "
                    f"val final "
                    f"{validation['accuracy_final_tick']:.3f}"
                )

        if best_state is None:
            raise RuntimeError("No best checkpoint was recorded.")

        # ----------------------------------------------------
        # Restore validation-selected model
        # ----------------------------------------------------

        model.load_state_dict(best_state)

        train_metrics = evaluate_model(
            model,
            train_loader,
            device,
        )

        val_metrics = evaluate_model(
            model,
            val_loader,
            device,
        )

        test_metrics = evaluate_model(
            model,
            test_loader,
            device,
        )

        # ----------------------------------------------------
        # Store summary
        # ----------------------------------------------------

        result = {
            "prefix_length": prefix_length,
            "fraction_observed": prefix_length / full_timesteps,
            "percent_observed": 100.0 * prefix_length / full_timesteps,
            "best_epoch": best_epoch,
            "train_accuracy_most_certain": train_metrics["accuracy_most_certain"],
            "val_accuracy_most_certain": val_metrics["accuracy_most_certain"],
            "test_accuracy_most_certain": test_metrics["accuracy_most_certain"],
            "test_accuracy_final_tick": test_metrics["accuracy_final_tick"],
            "test_loss": test_metrics["loss"],
            "test_mean_most_certain_tick": test_metrics["mean_most_certain_tick"],
            "test_mean_selected_certainty": test_metrics["mean_selected_certainty"],
        }

        results.append(result)

        print("\nSelected checkpoint:")

        print(f"epoch: {best_epoch}")

        print(
            "validation MC accuracy:",
            f"{val_metrics['accuracy_most_certain']:.3f}",
        )

        print(
            "test MC accuracy:",
            f"{test_metrics['accuracy_most_certain']:.3f}",
        )

        # ----------------------------------------------------
        # Free device memory before next model
        # ----------------------------------------------------

        del model

        if device.type == "mps" and hasattr(
            torch,
            "mps",
        ):
            torch.mps.empty_cache()

    return pd.DataFrame(results)


# ============================================================
# Internal-tick budget training
# ============================================================


def run_internal_tick_budget_sweep(
    bundle,
    config,
    tick_budgets=(5, 10, 15, 25),
    max_epochs=60,
    verbose=False,
):
    """
    Train separate CTMs with different internal-tick budgets.

    The external input is kept fixed at the complete
    100-position reach-to-grasp trajectory.

    Each tick-budget condition uses the same:

        - B1-B4 dataset
        - repetition split
        - train-only standardization
        - CTM architecture
        - optimizer settings
        - random seed

    Only the number of CTM internal iterations changes.

    Model selection is performed using the validation set.
    The held-out test set is evaluated only after the best
    validation checkpoint has been restored.
    """

    from dataclasses import replace

    results = []

    for tick_budget in tick_budgets:
        tick_budget = int(tick_budget)

        if tick_budget <= 0:
            raise ValueError("tick_budget must be positive.")

        print()
        print("=" * 72)

        print(f"Internal-tick budget: {tick_budget}")

        print("=" * 72)

        # ----------------------------------------------------
        # Create configuration for this tick budget
        # ----------------------------------------------------

        tick_config = replace(
            config,
            experiment_name=(f"b1_b4_tick_budget_{tick_budget}"),
            iterations=tick_budget,
        )

        # ----------------------------------------------------
        # Train using the existing baseline procedure
        # ----------------------------------------------------

        run = run_rtg_baseline(
            bundle=bundle,
            config=tick_config,
            max_epochs=max_epochs,
            verbose=verbose,
        )

        model = run["model"]
        device = run["device"]

        # ----------------------------------------------------
        # Evaluate held-out repetition 3
        # ----------------------------------------------------

        test_metrics = evaluate_model(
            model,
            bundle["test_loader"],
            device,
        )

        summary = run["summary"]

        results.append(
            {
                "internal_ticks": tick_budget,
                "best_epoch": summary["best_epoch"],
                "train_accuracy_most_certain": summary["train_accuracy_most_certain"],
                "val_accuracy_most_certain": summary["val_accuracy_most_certain"],
                "test_accuracy_most_certain": test_metrics["accuracy_most_certain"],
                "test_accuracy_final_tick": test_metrics["accuracy_final_tick"],
                "test_loss": test_metrics["loss"],
                "test_mean_most_certain_tick": test_metrics["mean_most_certain_tick"],
                "test_mean_selected_certainty": test_metrics["mean_selected_certainty"],
                "checkpoint_dir": str(run["output_dir"]),
            }
        )

        print("\nHeld-out test result:")

        print(
            "MC accuracy:",
            f"{test_metrics['accuracy_most_certain']:.3f}",
        )

        print(
            "final-tick accuracy:",
            f"{test_metrics['accuracy_final_tick']:.3f}",
        )

        print(
            "mean most-certain tick:",
            f"{test_metrics['mean_most_certain_tick']:.2f}",
        )

        # ----------------------------------------------------
        # Model is already saved by run_rtg_baseline()
        # ----------------------------------------------------

        run["model"] = None

        del model
        del run

        if device.type == "mps" and hasattr(
            torch,
            "mps",
        ):
            torch.mps.empty_cache()

    return pd.DataFrame(results)


# ============================================================
# Reproducible training loader
# ============================================================


def rebuild_training_loader(
    loader,
    seed,
):
    """
    Rebuild a shuffled DataLoader with a freshly seeded
    generator.

    This ensures that repeated experiment runs using the same
    seed begin with the same sample-ordering process instead
    of inheriting the advanced state of an existing DataLoader
    generator.
    """

    generator = torch.Generator().manual_seed(seed)

    return DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        shuffle=True,
        generator=generator,
    )


# ============================================================
# Limited-evidence internal-tick training
# ============================================================


def run_limited_evidence_tick_sweep(
    bundle,
    config,
    prefix_length=10,
    tick_budgets=(5, 10, 15, 25),
    max_epochs=60,
    verbose=False,
):
    """
    Train separate CTMs with different internal-tick budgets
    while keeping external movement evidence fixed to a
    partial trajectory prefix.

    Example:
        prefix_length=10

    means that every model receives only the first 10 of the
    100 resampled movement positions.

    The same prefix length is used for train, validation,
    and test partitions.

    Model selection uses validation performance.
    The held-out test set is evaluated only after restoring
    the best validation checkpoint.
    """

    from dataclasses import replace

    prefix_length = int(prefix_length)

    if prefix_length <= 0:
        raise ValueError("prefix_length must be positive.")

    full_timesteps = bundle["train_loader"].dataset.tensors[0].shape[-1]

    if prefix_length > full_timesteps:
        raise ValueError(
            f"prefix_length={prefix_length} exceeds "
            f"full trajectory length={full_timesteps}."
        )

    # --------------------------------------------------------
    # Build one fixed limited-evidence bundle
    # --------------------------------------------------------

    prefix_bundle = dict(bundle)

    prefix_bundle["train_loader"] = build_prefix_loader(
        loader=bundle["train_loader"],
        prefix_length=prefix_length,
        shuffle=False,
    )

    prefix_bundle["val_loader"] = build_prefix_loader(
        loader=bundle["val_loader"],
        prefix_length=prefix_length,
        shuffle=False,
    )

    prefix_bundle["test_loader"] = build_prefix_loader(
        loader=bundle["test_loader"],
        prefix_length=prefix_length,
        shuffle=False,
    )

    results = []

    # --------------------------------------------------------
    # Sweep internal-tick budgets
    # --------------------------------------------------------

    for tick_budget in tick_budgets:
        tick_budget = int(tick_budget)

        if tick_budget <= 0:
            raise ValueError("tick_budget must be positive.")

        print()
        print("=" * 72)

        print(
            f"External evidence: "
            f"{prefix_length}/{full_timesteps} "
            f"({100 * prefix_length / full_timesteps:.0f}%)"
        )

        print(f"Internal-tick budget: {tick_budget}")

        print("=" * 72)

        # ----------------------------------------------------
        # Condition-specific CTM configuration
        # ----------------------------------------------------

        tick_config = replace(
            config,
            experiment_name=(f"b1_b4_prefix_{prefix_length}_ticks_{tick_budget}"),
            iterations=tick_budget,
        )

        # ----------------------------------------------------
        # Train and select checkpoint using validation
        # ----------------------------------------------------

        run = run_rtg_baseline(
            bundle=prefix_bundle,
            config=tick_config,
            max_epochs=max_epochs,
            verbose=verbose,
        )

        model = run["model"]
        device = run["device"]

        # ----------------------------------------------------
        # Held-out repetition-3 evaluation
        # ----------------------------------------------------

        test_metrics = evaluate_model(
            model,
            prefix_bundle["test_loader"],
            device,
        )

        summary = run["summary"]

        results.append(
            {
                "prefix_length": prefix_length,
                "percent_observed": 100.0 * prefix_length / full_timesteps,
                "internal_ticks": tick_budget,
                "best_epoch": summary["best_epoch"],
                "train_accuracy_most_certain": summary["train_accuracy_most_certain"],
                "val_accuracy_most_certain": summary["val_accuracy_most_certain"],
                "test_accuracy_most_certain": test_metrics["accuracy_most_certain"],
                "test_accuracy_final_tick": test_metrics["accuracy_final_tick"],
                "test_loss": test_metrics["loss"],
                "test_mean_most_certain_tick": test_metrics["mean_most_certain_tick"],
                "test_mean_selected_certainty": test_metrics["mean_selected_certainty"],
                "checkpoint_dir": str(run["output_dir"]),
            }
        )

        print("\nHeld-out test result:")

        print(
            "MC accuracy:",
            f"{test_metrics['accuracy_most_certain']:.3f}",
        )

        print(
            "final-tick accuracy:",
            f"{test_metrics['accuracy_final_tick']:.3f}",
        )

        print(
            "mean most-certain tick:",
            f"{test_metrics['mean_most_certain_tick']:.2f}",
        )

        # ----------------------------------------------------
        # Release device memory
        # ----------------------------------------------------

        run["model"] = None

        del model
        del run

        if device.type == "mps" and hasattr(
            torch,
            "mps",
        ):
            torch.mps.empty_cache()

    return pd.DataFrame(results)


# ============================================================
# Temporal-position training
# ============================================================


def build_window_loader(
    loader,
    window_start,
    window_length,
    shuffle=False,
    seed=0,
):
    """
    Build a DataLoader containing one fixed temporal window
    from each already-standardized trajectory.

    Example:
        window_start=37
        window_length=25

    extracts:
        X[:, :, 37:62]

    from every recording.
    """

    if not isinstance(
        loader.dataset,
        TensorDataset,
    ):
        raise TypeError("Expected loader.dataset to be a TensorDataset.")

    X, y = loader.dataset.tensors

    if X.ndim != 3:
        raise ValueError(f"Expected X with shape [N, C, T], got {X.shape}")

    window_start = int(window_start)

    window_length = int(window_length)

    window_end = window_start + window_length

    if window_start < 0:
        raise ValueError("window_start must be non-negative.")

    if window_length <= 0:
        raise ValueError("window_length must be positive.")

    if window_end > X.shape[-1]:
        raise ValueError(
            f"Window [{window_start}:{window_end}] "
            f"exceeds trajectory length {X.shape[-1]}."
        )

    X_window = X[
        :,
        :,
        window_start:window_end,
    ]

    dataset = TensorDataset(
        X_window,
        y,
    )

    if shuffle:
        generator = torch.Generator().manual_seed(seed)

        return DataLoader(
            dataset,
            batch_size=loader.batch_size,
            shuffle=True,
            generator=generator,
        )

    return DataLoader(
        dataset,
        batch_size=loader.batch_size,
        shuffle=False,
    )


def run_temporal_position_training(
    bundle,
    config,
    window_length=25,
    window_specs=(
        ("beginning", 0),
        ("middle", 37),
        ("end", 75),
    ),
    max_epochs=60,
    verbose=False,
):
    """
    Train separate CTMs on fixed temporal windows taken from
    different positions of the reach-to-grasp movement.

    All windows have identical length.

    Default conditions reproduce the temporal-position logic
    used in the synthetic experiments:

        beginning: [0:25]
        middle:    [37:62]
        end:       [75:100]

    Model selection is performed on repetition 2.
    Repetition 3 is evaluated only after restoring the best
    validation checkpoint.
    """

    from dataclasses import replace

    results = []

    full_timesteps = bundle["train_loader"].dataset.tensors[0].shape[-1]

    for position, window_start in window_specs:
        window_end = window_start + window_length

        print()
        print("=" * 72)

        print(f"Temporal position: {position}")

        print(f"Window: [{window_start}:{window_end}] length={window_length}")

        print("=" * 72)

        # ----------------------------------------------------
        # Build matched train / validation / test windows
        # ----------------------------------------------------

        window_bundle = dict(bundle)

        window_bundle["train_loader"] = build_window_loader(
            loader=bundle["train_loader"],
            window_start=window_start,
            window_length=window_length,
            shuffle=False,
        )

        window_bundle["val_loader"] = build_window_loader(
            loader=bundle["val_loader"],
            window_start=window_start,
            window_length=window_length,
            shuffle=False,
        )

        window_bundle["test_loader"] = build_window_loader(
            loader=bundle["test_loader"],
            window_start=window_start,
            window_length=window_length,
            shuffle=False,
        )

        # ----------------------------------------------------
        # Position-specific configuration
        # ----------------------------------------------------

        position_config = replace(
            config,
            experiment_name=(f"b1_b4_window_{position}_L{window_length}"),
        )

        # ----------------------------------------------------
        # Train and select using validation
        # ----------------------------------------------------

        run = run_rtg_baseline(
            bundle=window_bundle,
            config=position_config,
            max_epochs=max_epochs,
            verbose=verbose,
        )

        model = run["model"]
        device = run["device"]

        test_metrics = evaluate_model(
            model,
            window_bundle["test_loader"],
            device,
        )

        summary = run["summary"]

        results.append(
            {
                "position": position,
                "window_start": window_start,
                "window_end": window_end,
                "window_length": window_length,
                "window_center": window_start + window_length / 2,
                "fraction_observed": window_length / full_timesteps,
                "best_epoch": summary["best_epoch"],
                "train_accuracy_most_certain": summary["train_accuracy_most_certain"],
                "val_accuracy_most_certain": summary["val_accuracy_most_certain"],
                "test_accuracy_most_certain": test_metrics["accuracy_most_certain"],
                "test_accuracy_final_tick": test_metrics["accuracy_final_tick"],
                "test_loss": test_metrics["loss"],
                "test_mean_most_certain_tick": test_metrics["mean_most_certain_tick"],
                "test_mean_selected_certainty": test_metrics["mean_selected_certainty"],
                "checkpoint_dir": str(run["output_dir"]),
            }
        )

        print("\nHeld-out test result:")

        print(
            "MC accuracy:",
            f"{test_metrics['accuracy_most_certain']:.3f}",
        )

        print(
            "final-tick accuracy:",
            f"{test_metrics['accuracy_final_tick']:.3f}",
        )

        run["model"] = None

        del model
        del run

        if device.type == "mps" and hasattr(
            torch,
            "mps",
        ):
            torch.mps.empty_cache()

    return pd.DataFrame(results)


# ============================================================
# Wiener-noise robustness evaluation
# ============================================================


def build_wiener_noisy_loader(
    loader,
    sigma,
    seed=0,
):
    """
    Apply temporally correlated Wiener noise to an existing
    standardized dataset.

    The labels and recording order are preserved.

    A fresh RNG with the supplied seed is created for every
    call. Therefore, when the same seed is used across sigma
    values, the underlying Wiener realization is the same and
    only its magnitude changes.

    sigma=0 returns an unperturbed copy of the data.
    """

    if not isinstance(
        loader.dataset,
        TensorDataset,
    ):
        raise TypeError("Expected loader.dataset to be a TensorDataset.")

    X, y = loader.dataset.tensors

    if X.ndim != 3:
        raise ValueError(f"Expected X with shape [N, C, T], got {X.shape}")

    sigma = float(sigma)

    if sigma < 0:
        raise ValueError("sigma must be non-negative.")

    if sigma == 0.0:
        X_noisy = X.clone()

    else:
        rng = np.random.default_rng(seed)

        X_np = X.detach().cpu().numpy().copy()

        X_noisy_np = add_wiener_noise(
            X_np,
            sigma=sigma,
            rng=rng,
            time_axis=-1,
        )

        X_noisy = torch.tensor(
            X_noisy_np,
            dtype=X.dtype,
        )

    noisy_dataset = TensorDataset(
        X_noisy,
        y.clone(),
    )

    return DataLoader(
        noisy_dataset,
        batch_size=loader.batch_size,
        shuffle=False,
    )


def evaluate_wiener_robustness(
    model,
    loader,
    device,
    sigmas=(0.0, 0.1, 0.5, 1.0, 2.0),
    noise_seed=0,
):
    """
    Evaluate one frozen CTM checkpoint under increasing levels
    of temporally correlated Wiener noise.

    No retraining or parameter updates are performed.

    The same underlying Wiener realization is reused across
    sigma values by resetting the RNG to the same seed for
    every condition.
    """

    results = []

    model.eval()

    for sigma in sigmas:
        sigma = float(sigma)

        noisy_loader = build_wiener_noisy_loader(
            loader=loader,
            sigma=sigma,
            seed=noise_seed,
        )

        metrics = evaluate_model(
            model,
            noisy_loader,
            device,
        )

        results.append(
            {
                "wiener_sigma": sigma,
                "test_loss": metrics["loss"],
                "test_accuracy_most_certain": metrics["accuracy_most_certain"],
                "test_accuracy_final_tick": metrics["accuracy_final_tick"],
                "test_mean_most_certain_tick": metrics["mean_most_certain_tick"],
                "test_mean_selected_certainty": metrics["mean_selected_certainty"],
            }
        )

        print(
            f"sigma={sigma:>4.1f} | "
            f"MC acc="
            f"{metrics['accuracy_most_certain']:.3f} | "
            f"final acc="
            f"{metrics['accuracy_final_tick']:.3f} | "
            f"mean tick="
            f"{metrics['mean_most_certain_tick']:.2f} | "
            f"certainty="
            f"{metrics['mean_selected_certainty']:.3f}"
        )

    return pd.DataFrame(results)


# ============================================================
# B0 -> B1-B4 transfer experiment
# ============================================================


def build_transfer_metadata(
    data_root=DATA_ROOT,
):
    """
    Reconstruct the validated recording metadata without
    running the source-specific trajectory extraction.

    This allows the transfer experiment to derive a separate
    harmonized movement representation while preserving the
    original data_loader.py unchanged.
    """

    data_root = Path(data_root)

    inventory_df = build_recording_inventory(data_root)

    raw_metadata_df = pd.DataFrame(
        [extract_raw_metadata(path) for path in inventory_df["path"]]
    )

    selected_df, excluded_df = select_successful_recordings(raw_metadata_df)

    corrected_df = pd.DataFrame(
        [
            apply_rep0_target_correction(
                row.to_dict(),
                selected_df,
            )
            for _, row in selected_df.iterrows()
        ]
    )

    mapped_df = pd.DataFrame(
        [
            derive_experimental_mapping(row.to_dict())
            for _, row in corrected_df.iterrows()
        ]
    )

    return (
        mapped_df,
        excluded_df,
    )


def extract_harmonized_transfer_trajectory(
    df,
    board_sensor_index,
):
    """
    Extract one trajectory using the same reach-to-lift rule
    for both B0 and B1-B4.

    Common transfer definition:

        start:
            first hand_active == 1

        end:
            first crossing of 10% of the maximum relative
            target-board displacement

        export:
            no additional post-lift samples

    The target board sensor itself remains block-specific and
    is supplied through validated metadata.
    """

    board_sensor_index = int(board_sensor_index)

    if not 0 <= board_sensor_index <= 7:
        raise ValueError(f"Unexpected board sensor index: {board_sensor_index}")

    # --------------------------------------------------------
    # Common movement onset
    # --------------------------------------------------------

    hand_active = df["hand_active"].to_numpy()

    active_indices = np.where(hand_active == 1)[0]

    if len(active_indices) == 0:
        raise ValueError("No hand_active == 1 sample.")

    start = int(active_indices[0])

    # --------------------------------------------------------
    # Common target-board endpoint
    # --------------------------------------------------------

    board_columns = [f"board_data_{i}" for i in range(8)]

    board_data = df[board_columns].to_numpy(dtype=float)

    # Use one robust invalid-value treatment for both domains.
    board_data = replace_with_highest(
        board_data,
        target=8191.0,
    )

    board_data = replace_with_highest(
        board_data,
        target=8190.0,
    )

    board_signal = board_data[
        start:-1,
        board_sensor_index,
    ]

    if len(board_signal) < 2:
        raise ValueError("Board signal too short after movement onset.")

    # Zero baseline at movement onset.
    board_signal = board_signal - board_signal[0]

    max_displacement = float(board_signal.max())

    if max_displacement <= 0:
        raise ValueError("Target board signal has no positive displacement.")

    end_threshold = 0.1 * max_displacement

    t_lift = first_change_index(
        board_signal,
        end_threshold,
    )

    if t_lift is None:
        raise ValueError("Could not detect object lift.")

    end = int(start + t_lift)

    if start >= end:
        raise ValueError(f"Invalid bounds: start={start}, end={end}")

    # --------------------------------------------------------
    # Fingertip trajectory
    # --------------------------------------------------------

    fingertip_data = df[FINGERTIP_COLUMNS].to_numpy(dtype=np.float32)

    trajectory = fingertip_data[start:end]

    if trajectory.ndim != 2:
        raise ValueError(f"Expected 2D trajectory, got {trajectory.shape}")

    if trajectory.shape[1] != 15:
        raise ValueError(f"Expected 15 channels, got {trajectory.shape[1]}")

    if len(trajectory) < 2:
        raise ValueError(f"Trajectory too short: {len(trajectory)} samples.")

    if not np.isfinite(trajectory).all():
        raise ValueError("Trajectory contains non-finite values.")

    bounds = {
        "start_idx": start,
        "end_idx": end,
        "trajectory_length": len(trajectory),
        "board_sensor_index": board_sensor_index,
        "end_threshold": float(end_threshold),
        "board_max_displacement": max_displacement,
    }

    return (
        trajectory,
        bounds,
    )


def build_harmonized_transfer_dataset(
    data_root=DATA_ROOT,
    n_timepoints=100,
):
    """
    Build a harmonized B0/B1-B4 dataset specifically for the
    transfer experiment.

    Every retained recording uses the same temporal extraction
    rule before fixed-length resampling.
    """

    metadata_df, excluded_df = build_transfer_metadata(
        data_root=data_root,
    )

    X = []
    y = []
    records = []
    extraction_failures = []

    for _, row in metadata_df.iterrows():
        try:
            df = pd.read_csv(row["path"])

            (
                trajectory,
                bounds,
            ) = extract_harmonized_transfer_trajectory(
                df=df,
                board_sensor_index=(row["board_sensor_index"]),
            )

            resampled = resample_trajectory(
                trajectory,
                n_timepoints=n_timepoints,
            )

            if resampled.shape != (
                15,
                n_timepoints,
            ):
                raise ValueError(f"Unexpected resampled shape: {resampled.shape}")

            if not np.isfinite(resampled).all():
                raise ValueError("Non-finite resampled trajectory.")

            X.append(resampled)

            y.append(int(row["class_label"]))

            record = row.to_dict()

            record["transfer_start_idx"] = bounds["start_idx"]

            record["transfer_end_idx"] = bounds["end_idx"]

            record["transfer_trajectory_length"] = bounds["trajectory_length"]

            record["transfer_extraction_method"] = "harmonized_hand_active_board10"

            records.append(record)

        except Exception as exc:
            extraction_failures.append(
                {
                    "file": row["file"],
                    "participant": row["participant"],
                    "block": row["block"],
                    "trial_number": row["trial_number"],
                    "rep_number": row["rep_number"],
                    "error": str(exc),
                }
            )

    if len(X) == 0:
        raise RuntimeError("No transfer trajectories were extracted.")

    X = np.stack(
        X,
        axis=0,
    ).astype(np.float32)

    y = np.asarray(
        y,
        dtype=np.int64,
    )

    transfer_metadata = pd.DataFrame(records)

    failure_df = pd.DataFrame(extraction_failures)

    return {
        "X": X,
        "y": y,
        "metadata": transfer_metadata,
        "logical_exclusions": excluded_df,
        "extraction_failures": failure_df,
    }


def build_b0_to_b1_b4_transfer_bundle(
    config,
    b0_train_reps=(0, 1, 2, 3, 4, 5),
    b0_val_reps=(6, 7),
):
    """
    Construct the B0 -> B1-B4 transfer experiment.

    Training:
        B0 repetitions 0-5

    Validation:
        B0 repetitions 6-7

    Transfer test:
        all B1-B4 recordings

    Scaling is fit ONLY on the B0 training partition.
    """

    transfer_data = build_harmonized_transfer_dataset(
        data_root=config.data_root,
        n_timepoints=config.full_timesteps,
    )

    X = transfer_data["X"]
    y = transfer_data["y"]

    metadata = transfer_data["metadata"].copy().reset_index(drop=True)

    block = metadata["block"].to_numpy(dtype=int)

    rep = metadata["rep_number"].to_numpy(dtype=int)

    train_mask = (block == 0) & np.isin(
        rep,
        b0_train_reps,
    )

    val_mask = (block == 0) & np.isin(
        rep,
        b0_val_reps,
    )

    test_mask = np.isin(
        block,
        [1, 2, 3, 4],
    )

    if np.any(train_mask & val_mask):
        raise RuntimeError("Train and validation overlap.")

    if np.any(train_mask & test_mask):
        raise RuntimeError("Train and test overlap.")

    if np.any(val_mask & test_mask):
        raise RuntimeError("Validation and test overlap.")

    X_train = X[train_mask]

    y_train = y[train_mask]

    X_val = X[val_mask]

    y_val = y[val_mask]

    X_test = X[test_mask]

    y_test = y[test_mask]

    # --------------------------------------------------------
    # Fit scaling only on B0 training data
    # --------------------------------------------------------

    scaler = fit_channel_scaler(
        X_train,
        epsilon=config.scale_epsilon,
    )

    X_train = apply_channel_scaler(
        X_train,
        scaler,
    )

    X_val = apply_channel_scaler(
        X_val,
        scaler,
    )

    X_test = apply_channel_scaler(
        X_test,
        scaler,
    )

    # --------------------------------------------------------
    # Record split assignment
    # --------------------------------------------------------

    metadata["split"] = "unused"

    metadata.loc[
        train_mask,
        "split",
    ] = "train"

    metadata.loc[
        val_mask,
        "split",
    ] = "val"

    metadata.loc[
        test_mask,
        "split",
    ] = "test"

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(
                X_train,
                dtype=torch.float32,
            ),
            torch.tensor(
                y_train,
                dtype=torch.long,
            ),
        ),
        batch_size=config.batch_size,
        shuffle=False,
    )

    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(
                X_val,
                dtype=torch.float32,
            ),
            torch.tensor(
                y_val,
                dtype=torch.long,
            ),
        ),
        batch_size=config.batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(
                X_test,
                dtype=torch.float32,
            ),
            torch.tensor(
                y_test,
                dtype=torch.long,
            ),
        ),
        batch_size=config.batch_size,
        shuffle=False,
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "metadata": metadata,
        "scaler": scaler,
        "logical_exclusions": transfer_data["logical_exclusions"],
        "extraction_failures": transfer_data["extraction_failures"],
    }


# ============================================================
# Unseen-participant generalization
# ============================================================


def build_b1_b4_participant_base(
    config,
):
    """
    Build the unstandardized B1-B4 dataset used for
    leave-one-participant-out generalization.

    Standardization is deliberately deferred until each fold
    is constructed so that the held-out participant cannot
    influence scaling statistics.
    """

    data_root = Path(config.data_root)

    (
        X,
        y,
        metadata_df,
        excluded_df,
    ) = build_reach_to_grasp_dataset(
        data_root=data_root,
        n_timepoints=config.full_timesteps,
    )

    block_mask = metadata_df["block"].isin([1, 2, 3, 4]).to_numpy()

    X_b1_b4 = X[block_mask]

    y_b1_b4 = y[block_mask]

    metadata_b1_b4 = metadata_df.loc[block_mask].copy().reset_index(drop=True)

    if len(X_b1_b4) != 640:
        raise RuntimeError(f"Expected 640 B1-B4 recordings, got {len(X_b1_b4)}.")

    return {
        "X": X_b1_b4,
        "y": y_b1_b4,
        "metadata": metadata_b1_b4,
        "excluded": excluded_df,
    }


def build_unseen_participant_bundle(
    base_data,
    config,
    held_out_participant,
    train_reps=(0, 1, 2),
    val_reps=(3,),
):
    """
    Construct one unseen-participant fold.

    Training:
        participants other than held-out participant,
        repetitions 0, 1, and 2

    Validation:
        participants other than held-out participant,
        repetition 3

    Test:
        all B1-B4 recordings from the held-out participant

    Scaling is fitted only on the training partition.
    """

    held_out_participant = int(held_out_participant)

    X = base_data["X"]
    y = base_data["y"]

    metadata = base_data["metadata"].copy().reset_index(drop=True)

    participant = metadata["participant"].to_numpy(dtype=int)

    rep = metadata["rep_number"].to_numpy(dtype=int)

    source_mask = participant != held_out_participant

    test_mask = participant == held_out_participant

    train_mask = source_mask & np.isin(
        rep,
        train_reps,
    )

    val_mask = source_mask & np.isin(
        rep,
        val_reps,
    )

    if np.any(train_mask & val_mask):
        raise RuntimeError("Train and validation overlap.")

    if np.any(train_mask & test_mask):
        raise RuntimeError("Train and test overlap.")

    if np.any(val_mask & test_mask):
        raise RuntimeError("Validation and test overlap.")

    X_train = X[train_mask]

    y_train = y[train_mask]

    X_val = X[val_mask]

    y_val = y[val_mask]

    X_test = X[test_mask]

    y_test = y[test_mask]

    # --------------------------------------------------------
    # Training-partition-only standardization
    # --------------------------------------------------------

    scaler = fit_channel_scaler(
        X_train,
        epsilon=config.scale_epsilon,
    )

    X_train = apply_channel_scaler(
        X_train,
        scaler,
    )

    X_val = apply_channel_scaler(
        X_val,
        scaler,
    )

    X_test = apply_channel_scaler(
        X_test,
        scaler,
    )

    # --------------------------------------------------------
    # Split metadata
    # --------------------------------------------------------

    metadata["split"] = "unused"

    metadata.loc[
        train_mask,
        "split",
    ] = "train"

    metadata.loc[
        val_mask,
        "split",
    ] = "val"

    metadata.loc[
        test_mask,
        "split",
    ] = "test"

    # --------------------------------------------------------
    # DataLoaders
    #
    # run_rtg_baseline() rebuilds the training loader using a
    # fresh seeded shuffle generator.
    # --------------------------------------------------------

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(
                X_train,
                dtype=torch.float32,
            ),
            torch.tensor(
                y_train,
                dtype=torch.long,
            ),
        ),
        batch_size=config.batch_size,
        shuffle=False,
    )

    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(
                X_val,
                dtype=torch.float32,
            ),
            torch.tensor(
                y_val,
                dtype=torch.long,
            ),
        ),
        batch_size=config.batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(
                X_test,
                dtype=torch.float32,
            ),
            torch.tensor(
                y_test,
                dtype=torch.long,
            ),
        ),
        batch_size=config.batch_size,
        shuffle=False,
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "metadata": metadata,
        "scaler": scaler,
        "held_out_participant": held_out_participant,
    }


def run_unseen_participant_sweep(
    config,
    participants=(1, 2, 3, 4, 5),
    train_reps=(0, 1, 2),
    val_reps=(3,),
    max_epochs=60,
    verbose=False,
):
    """
    Run a leave-one-participant-out B1-B4 experiment.

    Each participant is held out completely from training,
    validation, and feature standardization.

    A fresh CTM is trained for every held-out participant.
    """

    from dataclasses import replace

    base_data = build_b1_b4_participant_base(config)

    results = []

    for held_out_participant in participants:
        print()
        print("=" * 72)

        print(
            "Held-out participant:",
            held_out_participant,
        )

        print("=" * 72)

        fold_bundle = build_unseen_participant_bundle(
            base_data=base_data,
            config=config,
            held_out_participant=(held_out_participant),
            train_reps=train_reps,
            val_reps=val_reps,
        )

        train_size = len(fold_bundle["train_loader"].dataset)

        val_size = len(fold_bundle["val_loader"].dataset)

        test_size = len(fold_bundle["test_loader"].dataset)

        print(
            "train:",
            train_size,
            "| val:",
            val_size,
            "| test:",
            test_size,
        )

        fold_config = replace(
            config,
            experiment_name=(f"b1_b4_unseen_participant_{held_out_participant}"),
        )

        run = run_rtg_baseline(
            bundle=fold_bundle,
            config=fold_config,
            max_epochs=max_epochs,
            verbose=verbose,
        )

        model = run["model"]
        device = run["device"]

        test_metrics = evaluate_model(
            model,
            fold_bundle["test_loader"],
            device,
        )

        summary = run["summary"]

        results.append(
            {
                "held_out_participant": held_out_participant,
                "n_train": train_size,
                "n_val": val_size,
                "n_test": test_size,
                "best_epoch": summary["best_epoch"],
                "train_accuracy_most_certain": summary["train_accuracy_most_certain"],
                "val_accuracy_most_certain": summary["val_accuracy_most_certain"],
                "test_accuracy_most_certain": test_metrics["accuracy_most_certain"],
                "test_accuracy_final_tick": test_metrics["accuracy_final_tick"],
                "test_loss": test_metrics["loss"],
                "test_mean_most_certain_tick": test_metrics["mean_most_certain_tick"],
                "test_mean_selected_certainty": test_metrics["mean_selected_certainty"],
                "checkpoint_dir": str(run["output_dir"]),
            }
        )

        print("\nHeld-out test result:")

        print(
            "MC accuracy:",
            f"{test_metrics['accuracy_most_certain']:.3f}",
        )

        print(
            "final-tick accuracy:",
            f"{test_metrics['accuracy_final_tick']:.3f}",
        )

        run["model"] = None

        del model
        del run

        if device.type == "mps" and hasattr(
            torch,
            "mps",
        ):
            torch.mps.empty_cache()

    return pd.DataFrame(results)


# ============================================================
# Final all-B0 -> B1-B4 transfer protocol
# ============================================================


def build_final_b0_transfer_bundle(
    config,
):
    """
    Build the final supervisor-aligned RTG transfer split.

    Training:
        all successful B0 recordings

    Testing:
        all B1-B4 recordings

    No B0 recordings are reserved for validation.

    The harmonized reach-to-lift representation is used for
    both source and transfer domains.

    Feature standardization is fitted exclusively on B0.
    """

    transfer_data = build_harmonized_transfer_dataset(
        data_root=Path(config.data_root),
        n_timepoints=(config.full_timesteps),
    )

    X = transfer_data["X"]
    y = transfer_data["y"]

    metadata = transfer_data["metadata"].copy().reset_index(drop=True)

    block = metadata["block"].to_numpy(dtype=int)

    train_mask = block == 0

    test_mask = np.isin(
        block,
        [1, 2, 3, 4],
    )

    if np.any(train_mask & test_mask):
        raise RuntimeError("B0 training and B1-B4 test sets overlap.")

    X_train = X[train_mask]

    y_train = y[train_mask]

    X_test = X[test_mask]

    y_test = y[test_mask]

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    if len(X_train) != 318:
        raise RuntimeError(
            f"Expected 318 successful B0 recordings, got {len(X_train)}."
        )

    if len(X_test) != 640:
        raise RuntimeError(f"Expected 640 B1-B4 recordings, got {len(X_test)}.")

    # --------------------------------------------------------
    # Fit standardization ONLY on all B0 training data
    # --------------------------------------------------------

    scaler = fit_channel_scaler(
        X_train,
        epsilon=config.scale_epsilon,
    )

    X_train = apply_channel_scaler(
        X_train,
        scaler,
    )

    X_test = apply_channel_scaler(
        X_test,
        scaler,
    )

    # --------------------------------------------------------
    # Record final split
    # --------------------------------------------------------

    metadata["split"] = "unused"

    metadata.loc[
        train_mask,
        "split",
    ] = "train"

    metadata.loc[
        test_mask,
        "split",
    ] = "test"

    # --------------------------------------------------------
    # DataLoaders
    #
    # Training is initially unshuffled here.
    # The fixed-epoch runner rebuilds it with a fresh seeded
    # shuffle generator for reproducibility.
    # --------------------------------------------------------

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(
                X_train,
                dtype=torch.float32,
            ),
            torch.tensor(
                y_train,
                dtype=torch.long,
            ),
        ),
        batch_size=config.batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(
                X_test,
                dtype=torch.float32,
            ),
            torch.tensor(
                y_test,
                dtype=torch.long,
            ),
        ),
        batch_size=config.batch_size,
        shuffle=False,
    )

    return {
        "train_loader": train_loader,
        "test_loader": test_loader,
        "metadata": metadata,
        "scaler": scaler,
        "logical_exclusions": transfer_data["logical_exclusions"],
        "extraction_failures": transfer_data["extraction_failures"],
    }


# ============================================================
# Participant-specific B0 -> B1-B4 transfer bundle
# ============================================================


def build_participant_specific_transfer_bundle(
    config,
    participant,
):
    """
    Build one participant-specific RTG transfer experiment.

    Training:
        all successful B0 recordings from one participant

    Testing:
        all B1-B4 recordings from the same participant

    Scaling:
        fit only on that participant's B0 recordings

    No B1-B4 recording is used for fitting or model
    selection.
    """

    participant = int(participant)

    if participant not in [
        1,
        2,
        3,
        4,
        5,
    ]:
        raise ValueError(f"Unexpected participant: {participant}")

    transfer_data = build_harmonized_transfer_dataset(
        data_root=Path(config.data_root),
        n_timepoints=(config.full_timesteps),
    )

    X = transfer_data["X"]

    y = transfer_data["y"]

    metadata = transfer_data["metadata"].copy().reset_index(drop=True)

    participant_values = metadata["participant"].to_numpy(dtype=int)

    block_values = metadata["block"].to_numpy(dtype=int)

    train_mask = (participant_values == participant) & (block_values == 0)

    test_mask = (participant_values == participant) & np.isin(
        block_values,
        [
            1,
            2,
            3,
            4,
        ],
    )

    if np.any(train_mask & test_mask):
        raise RuntimeError("Participant-specific train/test overlap.")

    X_train = X[train_mask]

    y_train = y[train_mask]

    X_test = X[test_mask]

    y_test = y[test_mask]

    if len(X_train) == 0:
        raise RuntimeError(f"No B0 training data for P{participant}.")

    if len(X_test) != 128:
        raise RuntimeError(
            f"Expected 128 B1-B4 recordings for P{participant}, got {len(X_test)}."
        )

    # --------------------------------------------------------
    # Participant-specific B0 scaler
    # --------------------------------------------------------

    scaler = fit_channel_scaler(
        X_train,
        epsilon=(config.scale_epsilon),
    )

    X_train = apply_channel_scaler(
        X_train,
        scaler,
    )

    X_test = apply_channel_scaler(
        X_test,
        scaler,
    )

    # --------------------------------------------------------
    # Record participant-specific split
    # --------------------------------------------------------

    metadata["split"] = "unused"

    metadata.loc[
        train_mask,
        "split",
    ] = "train"

    metadata.loc[
        test_mask,
        "split",
    ] = "test"

    participant_metadata = (
        metadata[train_mask | test_mask].copy().reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Loaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(
                X_train,
                dtype=torch.float32,
            ),
            torch.tensor(
                y_train,
                dtype=torch.long,
            ),
        ),
        batch_size=(config.batch_size),
        shuffle=False,
    )

    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(
                X_test,
                dtype=torch.float32,
            ),
            torch.tensor(
                y_test,
                dtype=torch.long,
            ),
        ),
        batch_size=(config.batch_size),
        shuffle=False,
    )

    return {
        "participant": participant,
        "train_loader": train_loader,
        "test_loader": test_loader,
        "metadata": participant_metadata,
        "scaler": scaler,
        "logical_exclusions": transfer_data["logical_exclusions"],
        "extraction_failures": transfer_data["extraction_failures"],
    }


def run_fixed_epoch_rtg_training(
    bundle,
    config,
    epochs,
    verbose=True,
):
    """
    Train one CTM for a pre-specified number of epochs.

    No validation set or validation-based checkpoint
    selection is used.

    The test partition is evaluated exactly once after
    training has finished.

    This function is intended for the final all-B0 ->
    B1-B4 transfer protocol.
    """

    import time
    from datetime import datetime, timezone

    epochs = int(epochs)

    if epochs <= 0:
        raise ValueError("epochs must be positive.")

    started_at = time.perf_counter()

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    set_run_seed(config.run_seed)

    device = resolve_device(config.device)

    print(
        "Device:",
        device,
    )

    train_loader = rebuild_training_loader(
        bundle["train_loader"],
        seed=config.run_seed,
    )

    test_loader = bundle["test_loader"]

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = build_model(
        config,
        device,
    )

    dry_X, _ = next(iter(train_loader))

    dry_X = dry_X.to(device)

    with torch.no_grad():
        dry_predictions, (dry_certainties), _ = model(dry_X)

    print(
        "Prediction shape:",
        tuple(dry_predictions.shape),
    )

    print(
        "Certainty shape:",
        tuple(dry_certainties.shape),
    )

    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    print(
        "Trainable parameters:",
        f"{parameter_count:,}",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=(config.weight_decay),
    )

    history = []

    # --------------------------------------------------------
    # Fixed training budget
    # --------------------------------------------------------

    for epoch in range(
        1,
        epochs + 1,
    ):
        model.train()

        epoch_loss = 0.0
        epoch_correct = 0
        epoch_seen = 0

        for (
            X_batch,
            y_batch,
        ) in train_loader:
            X_batch = X_batch.to(device)

            y_batch = y_batch.to(device)

            optimizer.zero_grad(set_to_none=True)

            (
                predictions,
                certainties,
                _,
            ) = model(X_batch)

            (
                loss,
                idx_max_certainty,
            ) = ctm_loss(
                predictions,
                certainties,
                y_batch,
            )

            loss.backward()

            optimizer.step()

            batch_size = y_batch.size(0)

            batch_indices = torch.arange(
                batch_size,
                device=device,
            )

            selected_logits = predictions[
                batch_indices,
                :,
                idx_max_certainty,
            ]

            predicted_classes = selected_logits.argmax(dim=1)

            epoch_loss += loss.item() * batch_size

            epoch_correct += (predicted_classes == y_batch).sum().item()

            epoch_seen += batch_size

        train_loss = epoch_loss / epoch_seen

        train_accuracy = epoch_correct / epoch_seen

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
            }
        )

        if verbose:
            print(
                f"epoch {epoch:03d} | "
                f"train loss "
                f"{train_loss:.4f} | "
                f"train MC acc "
                f"{train_accuracy:.3f}"
            )

    # --------------------------------------------------------
    # Final frozen-model evaluation
    # --------------------------------------------------------

    train_metrics = evaluate_model(
        model,
        train_loader,
        device,
    )

    # IMPORTANT:
    # B1-B4 is touched only here, after training is complete.
    test_metrics = evaluate_model(
        model,
        test_loader,
        device,
    )

    finished_at = time.perf_counter()

    # --------------------------------------------------------
    # Save final artifacts
    # --------------------------------------------------------

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    output_dir = (
        Path("results")
        / "rtg"
        / (f"{timestamp}_{config.experiment_name}_seed{config.run_seed}")
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "model_state_dict": {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        },
        "epochs": epochs,
        "run_seed": config.run_seed,
        "parameter_count": parameter_count,
    }

    torch.save(
        checkpoint,
        output_dir / "checkpoint.pt",
    )

    np.savez(
        output_dir / "scaler.npz",
        mean=bundle["scaler"]["mean"],
        std=bundle["scaler"]["std"],
    )

    bundle["metadata"].to_csv(
        output_dir / "split_metadata.csv",
        index=False,
    )

    pd.DataFrame(history).to_csv(
        output_dir / "training_history.csv",
        index=False,
    )

    summary = {
        "epochs": epochs,
        "parameter_count": parameter_count,
        "train_accuracy_most_certain": train_metrics["accuracy_most_certain"],
        "train_accuracy_final_tick": train_metrics["accuracy_final_tick"],
        "test_accuracy_most_certain": test_metrics["accuracy_most_certain"],
        "test_accuracy_final_tick": test_metrics["accuracy_final_tick"],
        "test_loss": test_metrics["loss"],
        "test_mean_most_certain_tick": test_metrics["mean_most_certain_tick"],
        "test_mean_selected_certainty": test_metrics["mean_selected_certainty"],
        "runtime_seconds": (finished_at - started_at),
    }

    print()
    print("Final fixed-budget training complete")

    print(
        "epochs:",
        epochs,
    )

    print(
        "B0 train MC accuracy:",
        f"{train_metrics['accuracy_most_certain']:.3f}",
    )

    print(
        "B1-B4 test MC accuracy:",
        f"{test_metrics['accuracy_most_certain']:.3f}",
    )

    print(
        "B1-B4 final-tick accuracy:",
        f"{test_metrics['accuracy_final_tick']:.3f}",
    )

    print(
        "saved to:",
        output_dir,
    )

    return {
        "model": model,
        "device": device,
        "history": pd.DataFrame(history),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "summary": summary,
        "output_dir": output_dir,
        "config": config,
    }


# ============================================================
# Physical-time anticipation helpers
# ============================================================


def infer_timestamp_scale_to_ms(
    timestamps,
):
    """
    Infer the scale needed to express glove timestamps in ms.

    The RTG glove was sampled at approximately 66.7 Hz, so
    the expected inter-sample interval is about 15 ms.

    Returns
    -------
    scale_to_ms : float
        Multiply raw timestamps by this value to obtain ms.

    median_step_ms : float
        Median positive timestamp difference after conversion.
    """

    timestamps = np.asarray(
        timestamps,
        dtype=float,
    )

    differences = np.diff(timestamps)

    positive = differences[np.isfinite(differences) & (differences > 0)]

    if len(positive) == 0:
        raise ValueError("No positive timestamp differences found.")

    median_raw_step = float(np.median(positive))

    # --------------------------------------------------------
    # Common timestamp units
    # --------------------------------------------------------

    # Already milliseconds:
    # expected step ~15
    if 5.0 <= median_raw_step <= 50.0:
        scale_to_ms = 1.0

    # Seconds:
    # expected step ~0.015
    elif 0.005 <= median_raw_step <= 0.05:
        scale_to_ms = 1000.0

    # Microseconds:
    # expected step ~15000
    elif 5000.0 <= median_raw_step <= 50000.0:
        scale_to_ms = 0.001

    else:
        raise ValueError(
            f"Could not infer timestamp unit. Median raw step = {median_raw_step}"
        )

    median_step_ms = median_raw_step * scale_to_ms

    return (
        scale_to_ms,
        median_step_ms,
    )


def audit_b1_b4_physical_durations(
    metadata,
):
    """
    Measure harmonized onset-to-lift duration for every
    B1-B4 test recording using the original glove timestamps.

    Uses exactly the same harmonized movement boundaries as
    the final B0 -> B1-B4 transfer experiment.
    """

    test_metadata = metadata.query("split == 'test'").reset_index(drop=True)

    records = []

    for _, row in test_metadata.iterrows():
        df = pd.read_csv(row["path"])

        (
            trajectory,
            bounds,
        ) = extract_harmonized_transfer_trajectory(
            df=df,
            board_sensor_index=(row["board_sensor_index"]),
        )

        timestamps = df["glove_timestamps"].to_numpy(dtype=float)

        (
            scale_to_ms,
            median_step_ms,
        ) = infer_timestamp_scale_to_ms(timestamps)

        timestamps_ms = timestamps * scale_to_ms

        start = int(bounds["start_idx"])

        end = int(bounds["end_idx"])

        # The extraction is [start:end].
        #
        # first_change_index() returns one index after the
        # threshold crossing, so end - 1 is the last retained
        # sample and represents the detected lift crossing.
        lift_index = end - 1

        if start < 0 or lift_index >= len(timestamps_ms) or lift_index <= start:
            raise ValueError(
                f"Invalid physical-time bounds "
                f"for {row['file']}: "
                f"start={start}, end={end}"
            )

        onset_time_ms = float(timestamps_ms[start])

        lift_time_ms = float(timestamps_ms[lift_index])

        duration_ms = lift_time_ms - onset_time_ms

        if not np.isfinite(duration_ms) or duration_ms <= 0:
            raise ValueError(
                f"Invalid movement duration for {row['file']}: {duration_ms}"
            )

        records.append(
            {
                "participant": int(row["participant"]),
                "block": int(row["block"]),
                "class_label": int(row["class_label"]),
                "trial_number": int(row["trial_number"]),
                "rep_number": int(row["rep_number"]),
                "file": row["file"],
                "path": row["path"],
                "start_idx": start,
                "lift_idx": lift_index,
                "raw_samples": len(trajectory),
                "onset_time_ms": onset_time_ms,
                "lift_time_ms": lift_time_ms,
                "duration_ms": duration_ms,
                "median_sample_step_ms": median_step_ms,
            }
        )

    return pd.DataFrame(records)


def build_physical_time_cutoff_loader(
    metadata,
    scaler,
    config,
    lead_time_ms,
):
    """
    Build B1-B4 inputs containing only movement observations
    available up to a specified time before detected object
    lift.

    Example
    -------
    lead_time_ms = 500

    means that the input stops at approximately 500 ms before
    the detected object-lift event.

    The available raw prefix is selected using original glove
    timestamps and is then resampled to the same 100-position
    CTM representation used during final training.

    Trials whose movement has not yet started at the requested
    cutoff are excluded from that cutoff condition.
    """

    lead_time_ms = float(lead_time_ms)

    if lead_time_ms < 0:
        raise ValueError("lead_time_ms must be non-negative.")

    test_metadata = metadata.query("split == 'test'").reset_index(drop=True)

    X = []
    y = []
    retained_records = []
    excluded_records = []

    for _, row in test_metadata.iterrows():
        df = pd.read_csv(row["path"])

        (
            full_trajectory,
            bounds,
        ) = extract_harmonized_transfer_trajectory(
            df=df,
            board_sensor_index=(row["board_sensor_index"]),
        )

        timestamps = df["glove_timestamps"].to_numpy(dtype=float)

        (
            scale_to_ms,
            _,
        ) = infer_timestamp_scale_to_ms(timestamps)

        timestamps_ms = timestamps * scale_to_ms

        start = int(bounds["start_idx"])

        end = int(bounds["end_idx"])

        lift_index = end - 1

        segment_times_ms = timestamps_ms[start:end]

        if len(segment_times_ms) != len(full_trajectory):
            raise RuntimeError(
                f"Timestamp and trajectory lengths do not match for {row['file']}."
            )

        lift_time_ms = float(timestamps_ms[lift_index])

        cutoff_time_ms = lift_time_ms - lead_time_ms

        # Number of observations actually available at or
        # before the requested physical cutoff.
        n_available = int(
            np.searchsorted(
                segment_times_ms,
                cutoff_time_ms,
                side="right",
            )
        )

        if n_available < 2:
            excluded_records.append(
                {
                    "participant": int(row["participant"]),
                    "block": int(row["block"]),
                    "class_label": int(row["class_label"]),
                    "file": row["file"],
                    "lead_time_ms": lead_time_ms,
                    "reason": "fewer_than_2_samples_available",
                }
            )

            continue

        available_trajectory = full_trajectory[:n_available]

        resampled = resample_trajectory(
            available_trajectory,
            n_timepoints=(config.full_timesteps),
        )

        X.append(resampled)

        y.append(int(row["class_label"]))

        actual_last_time_ms = float(segment_times_ms[n_available - 1])

        actual_lead_ms = lift_time_ms - actual_last_time_ms

        retained_records.append(
            {
                "participant": int(row["participant"]),
                "block": int(row["block"]),
                "class_label": int(row["class_label"]),
                "trial_number": int(row["trial_number"]),
                "rep_number": int(row["rep_number"]),
                "attempt_number": int(row["attempt_number"]),
                "file": row["file"],
                "path": row["path"],
                "lead_time_ms": lead_time_ms,
                "actual_lead_ms": actual_lead_ms,
                "n_raw_available": n_available,
                "n_raw_full": len(full_trajectory),
                "raw_fraction_available": (n_available / len(full_trajectory)),
            }
        )

    if len(X) == 0:
        raise ValueError(f"No valid trials at {lead_time_ms} ms before lift.")

    X = np.stack(
        X,
        axis=0,
    ).astype(np.float32)

    y = np.asarray(
        y,
        dtype=np.int64,
    )

    # --------------------------------------------------------
    # Use EXACTLY the scaler fitted on final B0 training data
    # --------------------------------------------------------

    X = apply_channel_scaler(
        X,
        scaler,
    )

    loader = DataLoader(
        TensorDataset(
            torch.tensor(
                X,
                dtype=torch.float32,
            ),
            torch.tensor(
                y,
                dtype=torch.long,
            ),
        ),
        batch_size=(config.batch_size),
        shuffle=False,
    )

    return {
        "loader": loader,
        "metadata": pd.DataFrame(retained_records),
        "excluded": pd.DataFrame(excluded_records),
        "lead_time_ms": lead_time_ms,
    }


def evaluate_physical_time_cutoffs(
    model,
    device,
    bundle,
    config,
    lead_times_ms,
):
    """
    Evaluate one frozen CTM at multiple physical lead times
    before detected object lift.
    """

    summaries = []
    outputs_by_lead = {}
    metadata_by_lead = {}

    for lead_time_ms in lead_times_ms:
        cutoff_bundle = build_physical_time_cutoff_loader(
            metadata=(bundle["metadata"]),
            scaler=(bundle["scaler"]),
            config=config,
            lead_time_ms=(lead_time_ms),
        )

        outputs = collect_split_predictions(
            model=model,
            loader=(cutoff_bundle["loader"]),
            device=device,
        )

        targets = outputs["targets"]

        predictions = outputs["most_certain_prediction"]

        accuracy = (predictions == targets).mean()

        metadata_df = cutoff_bundle["metadata"].copy().reset_index(drop=True)

        metadata_df["target"] = targets

        metadata_df["prediction"] = predictions

        metadata_df["correct"] = predictions == targets

        summaries.append(
            {
                "lead_time_ms": float(lead_time_ms),
                "n_valid": len(targets),
                "n_excluded": len(cutoff_bundle["excluded"]),
                "accuracy": float(accuracy),
                "mean_actual_lead_ms": float(metadata_df["actual_lead_ms"].mean()),
                "mean_raw_fraction_available": float(
                    metadata_df["raw_fraction_available"].mean()
                ),
                "mean_most_certain_tick": float(outputs["most_certain_tick"].mean()),
            }
        )

        outputs_by_lead[float(lead_time_ms)] = outputs

        metadata_by_lead[float(lead_time_ms)] = metadata_df

    return {
        "summary": pd.DataFrame(summaries)
        .sort_values(
            "lead_time_ms",
            ascending=False,
        )
        .reset_index(drop=True),
        "outputs_by_lead": outputs_by_lead,
        "metadata_by_lead": metadata_by_lead,
    }


# ============================================================
# Raw variable-length physical-time evaluation
# ============================================================


def evaluate_raw_variable_length_cutoff(
    model,
    device,
    metadata,
    scaler,
    lead_time_ms,
):
    """
    Evaluate a frozen CTM using the raw fingertip observations
    physically available at one requested lead time before lift.

    Unlike build_physical_time_cutoff_loader(), the available
    trajectory is NOT resampled to 100 positions.

    Each recording is evaluated individually so no padding,
    masking, or artificial observations are required.

    Returns
    -------
    dict containing:
        summary
        metadata
        excluded
        outputs
    """

    lead_time_ms = float(lead_time_ms)

    if lead_time_ms < 0:
        raise ValueError("lead_time_ms must be non-negative.")

    test_metadata = metadata.query("split == 'test'").reset_index(drop=True)

    retained_records = []
    excluded_records = []

    all_targets = []
    all_logits = []
    all_certainties = []
    all_predictions_per_tick = []
    all_most_certain_tick = []
    all_most_certain_prediction = []
    all_final_prediction = []
    all_most_certain_value = []
    all_losses = []

    model.eval()

    with torch.inference_mode():
        for _, row in test_metadata.iterrows():
            df = pd.read_csv(row["path"])

            (
                full_trajectory,
                bounds,
            ) = extract_harmonized_transfer_trajectory(
                df=df,
                board_sensor_index=(row["board_sensor_index"]),
            )

            timestamps = df["glove_timestamps"].to_numpy(dtype=float)

            (
                scale_to_ms,
                _,
            ) = infer_timestamp_scale_to_ms(timestamps)

            timestamps_ms = timestamps * scale_to_ms

            start = int(bounds["start_idx"])

            end = int(bounds["end_idx"])

            lift_index = end - 1

            segment_times_ms = timestamps_ms[start:end]

            if len(segment_times_ms) != len(full_trajectory):
                raise RuntimeError(
                    f"Timestamp and trajectory lengths do not match for {row['path']}."
                )

            lift_time_ms = float(timestamps_ms[lift_index])

            cutoff_time_ms = lift_time_ms - lead_time_ms

            n_available = int(
                np.searchsorted(
                    segment_times_ms,
                    cutoff_time_ms,
                    side="right",
                )
            )

            # ------------------------------------------------
            # Explicit exclusion
            # ------------------------------------------------

            if n_available < 2:
                excluded_records.append(
                    {
                        "participant": int(row["participant"]),
                        "block": int(row["block"]),
                        "trial_number": int(row["trial_number"]),
                        "rep_number": int(row["rep_number"]),
                        "attempt_number": int(row["attempt_number"]),
                        "class_label": int(row["class_label"]),
                        "file": row["file"],
                        "path": row["path"],
                        "lead_time_ms": lead_time_ms,
                        "n_raw_available": n_available,
                        "reason": "fewer_than_2_samples_available",
                    }
                )

                continue

            # ------------------------------------------------
            # Raw available trajectory
            #
            # full_trajectory:
            #     [time, channels]
            #
            # CTM input:
            #     [batch, channels, time]
            # ------------------------------------------------

            available_trajectory = full_trajectory[:n_available]

            X = available_trajectory.T[
                None,
                :,
                :,
            ].astype(np.float32)

            # Apply exactly the B0-fitted scaler.
            X = apply_channel_scaler(
                X,
                scaler,
            )

            X_tensor = torch.tensor(
                X,
                dtype=torch.float32,
                device=device,
            )

            target = int(row["class_label"])

            y_tensor = torch.tensor(
                [target],
                dtype=torch.long,
                device=device,
            )

            (
                predictions,
                certainties,
                _,
            ) = model(X_tensor)

            if predictions.shape != (
                1,
                8,
                25,
            ):
                raise RuntimeError(
                    "Unexpected prediction shape "
                    f"for {row['path']}: "
                    f"{tuple(predictions.shape)}"
                )

            if certainties.shape != (
                1,
                2,
                25,
            ):
                raise RuntimeError(
                    "Unexpected certainty shape "
                    f"for {row['path']}: "
                    f"{tuple(certainties.shape)}"
                )

            if not (
                torch.isfinite(predictions).all() and torch.isfinite(certainties).all()
            ):
                raise RuntimeError(f"Non-finite CTM output for {row['path']}.")

            # ------------------------------------------------
            # Loss and prediction readouts
            # ------------------------------------------------

            (
                loss,
                idx_max_certainty,
            ) = ctm_loss(
                predictions,
                certainties,
                y_tensor,
            )

            most_certain_idx = int(idx_max_certainty[0].item())

            logits = predictions[0]

            certainty = certainties[0]

            predictions_per_tick = logits.argmax(dim=0)

            most_certain_prediction = int(logits[:, most_certain_idx].argmax().item())

            final_prediction = int(logits[:, -1].argmax().item())

            most_certain_value = float(certainty[1, most_certain_idx].item())

            actual_last_time_ms = float(segment_times_ms[n_available - 1])

            actual_lead_ms = lift_time_ms - actual_last_time_ms

            recording_key = (
                f"P{int(row['participant'])}"
                f"|B{int(row['block'])}"
                f"|T{int(row['trial_number'])}"
                f"|R{int(row['rep_number'])}"
                f"|A{int(row['attempt_number'])}"
            )

            retained_records.append(
                {
                    "recording_key": recording_key,
                    "participant": int(row["participant"]),
                    "block": int(row["block"]),
                    "trial_number": int(row["trial_number"]),
                    "rep_number": int(row["rep_number"]),
                    "attempt_number": int(row["attempt_number"]),
                    "class_label": target,
                    "file": row["file"],
                    "path": row["path"],
                    "lead_time_ms": lead_time_ms,
                    "actual_lead_ms": actual_lead_ms,
                    "input_length": n_available,
                    "n_raw_full": len(full_trajectory),
                    "raw_fraction_available": (n_available / len(full_trajectory)),
                    "prediction_most_certain": most_certain_prediction,
                    "prediction_final_tick": final_prediction,
                    "correct_most_certain": (most_certain_prediction == target),
                    "correct_final_tick": (final_prediction == target),
                    "most_certain_tick": (most_certain_idx + 1),
                    "selected_certainty": most_certain_value,
                    "loss": float(loss.item()),
                }
            )

            all_targets.append(target)

            all_logits.append(logits.detach().cpu().numpy())

            all_certainties.append(certainty.detach().cpu().numpy())

            all_predictions_per_tick.append(predictions_per_tick.detach().cpu().numpy())

            all_most_certain_tick.append(most_certain_idx + 1)

            all_most_certain_prediction.append(most_certain_prediction)

            all_final_prediction.append(final_prediction)

            all_most_certain_value.append(most_certain_value)

            all_losses.append(float(loss.item()))

    if len(retained_records) == 0:
        raise RuntimeError(f"No valid recordings remained at {lead_time_ms} ms.")

    retained_df = pd.DataFrame(retained_records)

    excluded_df = pd.DataFrame(excluded_records)

    input_lengths = retained_df["input_length"].to_numpy()

    summary = {
        "lead_time_ms": lead_time_ms,
        "n_valid": len(retained_df),
        "n_excluded": len(excluded_df),
        "accuracy_most_certain": float(retained_df["correct_most_certain"].mean()),
        "accuracy_final_tick": float(retained_df["correct_final_tick"].mean()),
        "mean_selected_certainty": float(retained_df["selected_certainty"].mean()),
        "mean_most_certain_tick": float(retained_df["most_certain_tick"].mean()),
        "mean_actual_lead_ms": float(retained_df["actual_lead_ms"].mean()),
        "input_length_min": int(np.min(input_lengths)),
        "input_length_q25": float(
            np.quantile(
                input_lengths,
                0.25,
            )
        ),
        "input_length_median": float(np.median(input_lengths)),
        "input_length_mean": float(np.mean(input_lengths)),
        "input_length_q75": float(
            np.quantile(
                input_lengths,
                0.75,
            )
        ),
        "input_length_max": int(np.max(input_lengths)),
        "mean_raw_fraction_available": float(
            retained_df["raw_fraction_available"].mean()
        ),
        "mean_loss": float(np.mean(all_losses)),
    }

    outputs = {
        "targets": np.asarray(
            all_targets,
            dtype=np.int64,
        ),
        "logits": np.stack(
            all_logits,
            axis=0,
        ),
        "certainties": np.stack(
            all_certainties,
            axis=0,
        ),
        "predictions_per_tick": np.stack(
            all_predictions_per_tick,
            axis=0,
        ),
        "most_certain_tick": np.asarray(
            all_most_certain_tick,
            dtype=np.int64,
        ),
        "most_certain_prediction": np.asarray(
            all_most_certain_prediction,
            dtype=np.int64,
        ),
        "final_prediction": np.asarray(
            all_final_prediction,
            dtype=np.int64,
        ),
        "most_certain_value": np.asarray(
            all_most_certain_value,
            dtype=np.float32,
        ),
    }

    return {
        "summary": summary,
        "metadata": retained_df,
        "excluded": excluded_df,
        "outputs": outputs,
    }


def evaluate_raw_variable_length_physical_cutoffs(
    model,
    device,
    bundle,
    lead_times_ms=(
        500,
        400,
        300,
        200,
        100,
        0,
    ),
):
    """
    Evaluate one frozen CTM on raw variable-length movement
    prefixes at multiple physical lead times.
    """

    summaries = []
    metadata_by_lead = {}
    excluded_by_lead = {}
    outputs_by_lead = {}

    for lead_time_ms in lead_times_ms:
        result = evaluate_raw_variable_length_cutoff(
            model=model,
            device=device,
            metadata=(bundle["metadata"]),
            scaler=(bundle["scaler"]),
            lead_time_ms=(lead_time_ms),
        )

        summaries.append(result["summary"])

        metadata_by_lead[float(lead_time_ms)] = result["metadata"]

        excluded_by_lead[float(lead_time_ms)] = result["excluded"]

        outputs_by_lead[float(lead_time_ms)] = result["outputs"]

    return {
        "summary": pd.DataFrame(summaries)
        .sort_values(
            "lead_time_ms",
            ascending=False,
        )
        .reset_index(drop=True),
        "metadata_by_lead": metadata_by_lead,
        "excluded_by_lead": excluded_by_lead,
        "outputs_by_lead": outputs_by_lead,
    }
