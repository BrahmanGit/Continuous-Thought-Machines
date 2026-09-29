from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from synthetic_data import add_wiener_noise, make_synthetic_grasps

sys.path.insert(
    0,
    "/Users/arthu/Desktop/CTM_Res-Pro/models/ctm_official",
)

from models.ctm import ContinuousThoughtMachine
from synthetic_data import make_synthetic_grasps


@dataclass(frozen=True)
class ExperimentConfig:
    # Identification
    experiment_name: str = "baseline"

    # Seeds
    data_seed: int = 0
    split_seed: int = 0
    run_seed: int = 0

    # Synthetic data
    n_per_class: int = 200
    n_classes: int = 8
    n_channels: int = 15
    full_timesteps: int = 100
    noise_std: float = 0.05
    wiener_sigma: float = 0.0

    # Observation window
    window_start: int = 0
    window_length: int | None = None

    # Dataset split
    train_fraction: float = 0.70
    val_fraction: float = 0.15

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
    epochs: int = 15

    # Execution
    device: str = "auto"
    evaluate_test: bool = False


def set_run_seed(seed: int) -> None:
    """Control model initialization, CTM pairing, and batch-order randomness."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    return torch.device(requested_device)


def apply_observation_window(
    X: torch.Tensor,
    config: ExperimentConfig,
) -> torch.Tensor:
    """
    Crop an observation window from a full trajectory.

    X has shape [N, channels, full_timesteps].
    """
    start = config.window_start

    if config.window_length is None:
        end = X.size(-1)
    else:
        end = start + config.window_length

    if start < 0:
        raise ValueError("window_start must be non-negative.")

    if end > X.size(-1):
        raise ValueError(
            f"Requested window ends at frame {end}, "
            f"but trajectories contain only {X.size(-1)} frames."
        )

    if end <= start:
        raise ValueError("Observation window must contain at least one frame.")

    return X[:, :, start:end]


def stratified_split_indices(
    y: torch.Tensor,
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    """
    Split each class separately so train, validation, and test remain balanced.
    """
    if train_fraction <= 0 or val_fraction <= 0:
        raise ValueError("Train and validation fractions must be positive.")

    if train_fraction + val_fraction >= 1:
        raise ValueError("train_fraction + val_fraction must be less than 1.")

    rng = np.random.default_rng(seed)

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    labels = y.cpu().numpy()

    for class_id in np.unique(labels):
        class_indices = np.flatnonzero(labels == class_id)
        rng.shuffle(class_indices)

        n_class = len(class_indices)
        n_train = int(n_class * train_fraction)
        n_val = int(n_class * val_fraction)

        train_indices.extend(class_indices[:n_train].tolist())
        val_indices.extend(class_indices[n_train : n_train + n_val].tolist())
        test_indices.extend(class_indices[n_train + n_val :].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    return train_indices, val_indices, test_indices


def build_loaders(
    config: ExperimentConfig,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, int]]:
    """
    Generate data, crop the observation window, and create stratified splits.
    """
    X, y = make_synthetic_grasps(
        n_per_class=config.n_per_class,
        n_classes=config.n_classes,
        n_channels=config.n_channels,
        n_timesteps=config.full_timesteps,
        noise_std=config.noise_std,
        seed=config.data_seed,
    )

    # adding wiener process here

    if config.wiener_sigma > 0:
        rng = np.random.default_rng(config.data_seed)

        X_np = X.numpy()

        X_np = add_wiener_noise(
            X_np,
            sigma=config.wiener_sigma,
            rng=rng,
            time_axis=-1,
        )

        X = torch.tensor(X_np, dtype=torch.float32)

    X = apply_observation_window(X, config)

    dataset = TensorDataset(X, y)

    train_indices, val_indices, test_indices = stratified_split_indices(
        y=y,
        train_fraction=config.train_fraction,
        val_fraction=config.val_fraction,
        seed=config.split_seed,
    )

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # The run seed controls shuffled training-batch order.
    loader_generator = torch.Generator().manual_seed(config.run_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=loader_generator,
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

    sizes = {
        "n_train": len(train_dataset),
        "n_val": len(val_dataset),
        "n_test": len(test_dataset),
        "effective_timesteps": X.size(-1),
    }

    return train_loader, val_loader, test_loader, sizes


def build_model(
    config: ExperimentConfig,
    device: torch.device,
) -> ContinuousThoughtMachine:
    if config.d_input % config.heads != 0:
        raise ValueError("d_input must be divisible by the number of attention heads.")

    model = ContinuousThoughtMachine(
        iterations=config.iterations,
        d_model=config.d_model,
        d_input=config.d_input,
        heads=config.heads,
        n_synch_out=config.n_synch_out,
        n_synch_action=config.n_synch_action,
        synapse_depth=config.synapse_depth,
        memory_length=config.memory_length,
        deep_nlms=config.deep_nlms,
        memory_hidden_dims=config.memory_hidden_dims,
        do_layernorm_nlm=config.do_layernorm_nlm,
        backbone_type="none",
        positional_embedding_type="none",
        out_dims=config.n_classes,
        prediction_reshaper=[-1],
        dropout=config.dropout,
        neuron_select_type=config.neuron_select_type,
        n_random_pairing_self=config.n_random_pairing_self,
    )

    return model.to(device)


def ctm_loss(
    predictions: torch.Tensor,
    certainties: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    predictions: [B, classes, internal_ticks]
    certainties: [B, 2, internal_ticks]
    targets:     [B]
    """
    n_ticks = predictions.size(-1)

    targets_expanded = targets.unsqueeze(-1).expand(-1, n_ticks)

    losses = nn.functional.cross_entropy(
        predictions,
        targets_expanded,
        reduction="none",
    )

    idx_min_loss = losses.argmin(dim=-1)
    idx_max_certainty = certainties[:, 1].argmax(dim=-1)

    batch_indices = torch.arange(
        predictions.size(0),
        device=predictions.device,
    )

    minimum_loss = losses[
        batch_indices,
        idx_min_loss,
    ].mean()

    most_certain_loss = losses[
        batch_indices,
        idx_max_certainty,
    ].mean()

    loss = 0.5 * (minimum_loss + most_certain_loss)

    return loss, idx_max_certainty


def evaluate_model(
    model: ContinuousThoughtMachine,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_correct_most_certain = 0
    total_correct_final_tick = 0
    total_selected_tick = 0.0
    total_selected_certainty = 0.0
    total_seen = 0

    with torch.inference_mode():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            predictions, certainties, _ = model(X_batch)

            loss, idx_max_certainty = ctm_loss(
                predictions,
                certainties,
                y_batch,
            )

            batch_size = y_batch.size(0)
            batch_indices = torch.arange(
                batch_size,
                device=device,
            )

            most_certain_logits = predictions[
                batch_indices,
                :,
                idx_max_certainty,
            ]

            most_certain_classes = most_certain_logits.argmax(dim=1)
            final_tick_classes = predictions[:, :, -1].argmax(dim=1)

            selected_certainties = certainties[
                batch_indices,
                1,
                idx_max_certainty,
            ]

            total_loss += loss.item() * batch_size

            total_correct_most_certain += (most_certain_classes == y_batch).sum().item()

            total_correct_final_tick += (final_tick_classes == y_batch).sum().item()

            # Add one so reported ticks run from 1 to T rather than 0 to T-1.
            total_selected_tick += (idx_max_certainty.float() + 1).sum().item()

            total_selected_certainty += selected_certainties.sum().item()

            total_seen += batch_size

    return {
        "loss": total_loss / total_seen,
        "accuracy_most_certain": (total_correct_most_certain / total_seen),
        "accuracy_final_tick": (total_correct_final_tick / total_seen),
        "mean_most_certain_tick": (total_selected_tick / total_seen),
        "mean_selected_certainty": (total_selected_certainty / total_seen),
    }


def make_config_hash(config: ExperimentConfig) -> str:
    payload = json.dumps(
        asdict(config),
        sort_keys=True,
        default=str,
    )

    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def copy_state_dict_to_cpu(
    model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def run_experiment(
    config: ExperimentConfig,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Build, train, validate, and return one traceable result dictionary.
    """
    started_at = time.perf_counter()

    set_run_seed(config.run_seed)
    device = resolve_device(config.device)

    train_loader, val_loader, test_loader, sizes = build_loaders(config)

    model = build_model(config, device)

    # Dry forward pass for LazyLinear initialization.
    dry_X, _ = next(iter(train_loader))
    dry_X = dry_X.to(device)

    # Lazy modules materialize their parameters during this first forward pass.
    # Use no_grad rather than inference_mode so those parameters remain usable
    # by autograd during training.
    with torch.no_grad():
        dry_predictions, dry_certainties, _ = model(dry_X)

    if model.kv_proj[0].in_features != config.n_channels:
        raise RuntimeError(
            "The CTM inferred the wrong feature dimension. "
            f"Expected {config.n_channels}, "
            f"received {model.kv_proj[0].in_features}."
        )

    if dry_predictions.shape[1] != config.n_classes:
        raise RuntimeError("Unexpected number of output classes.")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_val_accuracy = float("-inf")
    best_val_loss = float("inf")

    for epoch in range(1, config.epochs + 1):
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

        validation = evaluate_model(
            model,
            val_loader,
            device,
        )

        train_loss = epoch_loss / epoch_seen
        train_accuracy = epoch_correct / epoch_seen

        is_better = validation["accuracy_most_certain"] > best_val_accuracy or (
            validation["accuracy_most_certain"] == best_val_accuracy
            and validation["loss"] < best_val_loss
        )

        if is_better:
            best_val_accuracy = validation["accuracy_most_certain"]
            best_val_loss = validation["loss"]
            best_epoch = epoch
            best_state = copy_state_dict_to_cpu(model)

        if verbose:
            print(
                f"epoch {epoch:02d} | "
                f"train loss {train_loss:.4f} | "
                f"train acc {train_accuracy:.3f} | "
                f"val loss {validation['loss']:.4f} | "
                f"val acc {validation['accuracy_most_certain']:.3f}"
            )

    if best_state is None:
        raise RuntimeError("No best model state was recorded.")

    # Evaluate the model selected using validation performance.
    model.load_state_dict(best_state)

    train_metrics = evaluate_model(model, train_loader, device)
    val_metrics = evaluate_model(model, val_loader, device)

    if config.evaluate_test:
        test_metrics = evaluate_model(model, test_loader, device)
    else:
        test_metrics = {
            "loss": float("nan"),
            "accuracy_most_certain": float("nan"),
            "accuracy_final_tick": float("nan"),
            "mean_most_certain_tick": float("nan"),
            "mean_selected_certainty": float("nan"),
        }

    finished_at = time.perf_counter()

    config_hash = make_config_hash(config)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    result: dict[str, Any] = {
        "run_id": f"{timestamp}_{config_hash}",
        "config_hash": config_hash,
        **asdict(config),
        **sizes,
        "device_used": str(device),
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "runtime_seconds": finished_at - started_at,
        "train_loss": train_metrics["loss"],
        "train_accuracy_most_certain": (train_metrics["accuracy_most_certain"]),
        "train_accuracy_final_tick": (train_metrics["accuracy_final_tick"]),
        "val_loss": val_metrics["loss"],
        "val_accuracy_most_certain": (val_metrics["accuracy_most_certain"]),
        "val_accuracy_final_tick": (val_metrics["accuracy_final_tick"]),
        "val_mean_most_certain_tick": (val_metrics["mean_most_certain_tick"]),
        "val_mean_selected_certainty": (val_metrics["mean_selected_certainty"]),
        "test_loss": test_metrics["loss"],
        "test_accuracy_most_certain": (test_metrics["accuracy_most_certain"]),
        "test_accuracy_final_tick": (test_metrics["accuracy_final_tick"]),
        "status": "completed",
    }

    del model

    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()

    return result


def append_result_to_csv(
    result: dict[str, Any],
    csv_path: str | Path,
) -> None:
    """
    Append one result row while preserving a stable CSV schema.
    """
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(result.keys())
    file_exists = path.exists()

    if file_exists:
        with path.open("r", newline="", encoding="utf-8") as existing_file:
            reader = csv.reader(existing_file)
            existing_header = next(reader, None)

        if existing_header != fieldnames:
            raise ValueError(
                "CSV columns do not match the current result schema. "
                "Use a new CSV file or deliberately migrate the old one."
            )

    with path.open("a", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(result)
