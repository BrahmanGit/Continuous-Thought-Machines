import sys
import random
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(
    0,
    "/Users/arthu/Desktop/CTM_Res-Pro/models/ctm_official",
)

from models.ctm import ContinuousThoughtMachine
from synthetic_data import make_synthetic_grasps


# Reproducibility
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


# Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)


# Data
X, y = make_synthetic_grasps(seed=seed)

dataset = TensorDataset(X, y)
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
)


# Model
model = ContinuousThoughtMachine(
    iterations=25,
    d_model=256,
    d_input=64,
    heads=4,
    n_synch_out=32,
    n_synch_action=32,
    synapse_depth=1,
    memory_length=15,
    deep_nlms=True,
    memory_hidden_dims=16,
    do_layernorm_nlm=False,
    backbone_type="none",
    positional_embedding_type="none",
    out_dims=8,
    prediction_reshaper=[-1],
    dropout=0.0,
    neuron_select_type="random-pairing",
    n_random_pairing_self=0,
).to(device)


# One dry forward pass to initialize lazy layers
X_batch, y_batch = next(iter(loader))

# Keep [B, channels, timesteps] = [B, 15, 100]
X_batch = X_batch.to(device)
y_batch = y_batch.to(device)

with torch.no_grad():
    preds, certs, synch = model(X_batch)

print("Input batch:", X_batch.shape)
print("Predictions:", preds.shape)
print("Certainties:", certs.shape)
print("KV input features:", model.kv_proj[0].in_features)

assert X_batch.shape[1:] == (15, 100)
assert preds.shape == (X_batch.size(0), 8, 25)
assert model.kv_proj[0].in_features == 15


def ctm_loss(predictions, certainties, targets):
    """
    predictions: [B, C, T]
    certainties: [B, 2, T]
    targets:     [B]
    """
    n_ticks = predictions.size(-1)

    targets_expanded = targets.unsqueeze(-1).expand(-1, n_ticks)

    losses = nn.functional.cross_entropy(
        predictions,
        targets_expanded,
        reduction="none",
    )  # [B, T]

    idx_min_loss = losses.argmin(dim=-1)
    idx_max_cert = certainties[:, 1].argmax(dim=-1)

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
        idx_max_cert,
    ].mean()

    loss = 0.5 * (minimum_loss + most_certain_loss)

    return loss, idx_max_cert


# Optimizer only after lazy layers are initialized
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=0.0,
)


# Training
n_epochs = 15

for epoch in range(n_epochs):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for X_batch, y_batch in loader:
        # No transpose
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad(set_to_none=True)

        preds, certs, _ = model(X_batch)

        loss, idx_cert = ctm_loss(
            preds,
            certs,
            y_batch,
        )

        loss.backward()
        optimizer.step()

        batch_indices = torch.arange(
            preds.size(0),
            device=device,
        )

        selected_logits = preds[
            batch_indices,
            :,
            idx_cert,
        ]

        predicted_classes = selected_logits.argmax(dim=1)

        total_correct += (predicted_classes == y_batch).sum().item()

        batch_size = y_batch.size(0)
        total_seen += batch_size
        total_loss += loss.item() * batch_size

    epoch_loss = total_loss / total_seen
    epoch_accuracy = total_correct / total_seen

    print(f"epoch {epoch + 1:02d} | loss {epoch_loss:.4f} | acc {epoch_accuracy:.3f}")
