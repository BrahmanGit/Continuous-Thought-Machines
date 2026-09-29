# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 08:45:35 2025

@author: jonas
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
from tqdm import tqdm


def create_windows(sequences, labels, window_size, stride=1):
    X_new = []
    y_new = []

    for seq, label in zip(sequences, labels):
        L = len(seq)
        if L < window_size:
            continue  # skip too-short sequences

        for start in range(0, L - window_size + 1, stride):
            window = seq[start : start + window_size]
            X_new.append(window)
            y_new.append(label)  # or label[start+window_size-1] if timestep label

    return X_new, y_new

class SeqDataset(Dataset):
    def __init__(self, sequences, labels, task="classification"):
        self.X = sequences
        self.y = labels
        self.task = task

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        seq = torch.tensor(self.X[idx], dtype=torch.float32)

        if self.task == "classification":
            label = torch.tensor(self.y[idx], dtype=torch.long)   # <-- IMPORTANT
        else:
            label = torch.tensor(self.y[idx], dtype=torch.float32)

        return seq, label


def collate_fn(batch):
    """
    Pads sequences in a batch for LSTM input.
    Returns:
      padded_seq: [batch, max_len, features]
      lengths: actual lengths per sequence
      labels: targets
    """
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in sequences])
    padded_seq = pad_sequence(sequences, batch_first=True)
    labels = torch.stack(labels)
    return padded_seq, lengths, labels


class BalancedClassSampler(Sampler):
    """
    Samples an equal number of examples from each class per epoch.
    The number of samples per class is limited by the smallest class size.
    """
    def __init__(self, labels, shuffle=True):
        """
        Args:
            labels: List or array of labels (class indices)
            shuffle: Whether to shuffle samples within each class
        """
        self.labels = np.array(labels)
        self.shuffle = shuffle
        
        # Get indices for each class
        self.class_indices = {}
        unique_classes = np.unique(self.labels)
        
        for c in unique_classes:
            self.class_indices[c] = np.where(self.labels == c)[0]
        
        # Find the minimum class size
        self.min_class_size = min(len(indices) for indices in self.class_indices.values())
        self.num_classes = len(unique_classes)
  
    def __iter__(self):
        # For each class, randomly sample min_class_size examples
        sampled_indices = []
        
        for class_idx, indices in self.class_indices.items():
            if self.shuffle:
                # Randomly sample without replacement
                selected = np.random.choice(indices, size=self.min_class_size, replace=False)
            else:
                # Take first min_class_size samples
                selected = indices[:self.min_class_size]
            
            sampled_indices.extend(selected)
        
        # Shuffle all selected indices together
        if self.shuffle:
            np.random.shuffle(sampled_indices)
        
        return iter(sampled_indices)
    
    def __len__(self):
        return self.min_class_size * self.num_classes


class LSTMNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, fc_dim, output_dim, task="classification"):
        super().__init__()
        self.task = task

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_dim, fc_dim)
        self.output_layer = nn.Linear(fc_dim, output_dim)

        if task == "classification":
            self.softmax = nn.Softmax(dim=1)

    def forward(self, padded_seq, lengths):
        # pack padded sequence for LSTM efficiency
        packed = pack_padded_sequence(padded_seq, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h, c) = self.lstm(packed)

        # h contains the final hidden states for each sequence: [1, batch, hidden_dim]
        last_output = h[-1]

        x = self.dropout(last_output)
        x = torch.relu(self.fc1(x))
        x = self.output_layer(x)

        if self.task == "classification":
            x = self.softmax(x)
        return x
    

class CNNNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, fc_dim, output_dim, task="classification"):
        super().__init__()
        self.task = task
        
        # 1D Convolutional layers
        # Conv1: input_dim -> hidden_dim
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            padding=1
        )
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        # Conv2: hidden_dim -> hidden_dim (deeper feature extraction)
        self.conv2 = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            padding=1
        )
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Global pooling to get fixed-size representation
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_dim, fc_dim)
        self.output_layer = nn.Linear(fc_dim, output_dim)
        
        if task == "classification":
            self.softmax = nn.Softmax(dim=1)
    
    def forward(self, padded_seq, lengths):
        # padded_seq shape: [batch, seq_len, input_dim]
        # Conv1d expects: [batch, channels, seq_len]
        x = padded_seq.transpose(1, 2)  # [batch, input_dim, seq_len]
        
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        
        # Global pooling: [batch, hidden_dim, seq_len] -> [batch, hidden_dim, 1]
        x = self.global_pool(x)
        x = x.squeeze(-1)  # [batch, hidden_dim]
        
        # Fully connected layers
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.output_layer(x)
        
        if self.task == "classification":
            x = self.softmax(x)
        
        return x
    
class smallCNNNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, fc_dim, output_dim, task="classification"):
        super().__init__()
        self.task = task
        
        # 1D Convolutional layers
        # Conv1: input_dim -> hidden_dim
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=2,
            padding=1
        )
        self.bn1 = nn.BatchNorm1d(hidden_dim)
   
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        
        self.dropout = nn.Dropout(0.2)
        # self.fc1 = nn.Linear(hidden_dim, fc_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        if task == "classification":
            self.softmax = nn.Softmax(dim=1)
    
    def forward(self, padded_seq, lengths):
        # padded_seq shape: [batch, seq_len, input_dim]
        # Conv1d expects: [batch, channels, seq_len]
        x = padded_seq.transpose(1, 2)  # [batch, input_dim, seq_len]
        
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        
        # Global pooling: [batch, hidden_dim, seq_len] -> [batch, hidden_dim, 1]
        x = self.global_pool(x)
        x = x.squeeze(-1)  # [batch, hidden_dim]
        
        # Fully connected layers
        # x = self.dropout(x)
        # x = torch.relu(self.fc1(x))
        x = self.output_layer(x)
        x = self.softmax(x)
        return x
    
class tinyCNNNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, fc_dim, output_dim, task="classification"):
        super().__init__()

        # Minimal model: ONLY ONE CONV layer
        # kernel_size=1 ensures minimal parameters
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=output_dim,  # one kernel per class
            kernel_size=1
        )

        if task == "classification":
            self.softmax = nn.Softmax(dim=1)

        self.task = task

    def forward(self, padded_seq, lengths=None):
        # padded_seq: [batch, seq_len, input_dim]
        x = padded_seq.transpose(1, 2)   # -> [batch, input_dim, seq_len]

        x = self.conv1(x)                # -> [batch, output_dim, seq_len]

        # Average across time dimension (NO parameters)
        x = x.mean(dim=2)                # -> [batch, output_dim]

        if self.task == "classification":
            x = self.softmax(x)

        return x
    
    
def train_model(model, train_loader, val_loader, task="classification", epochs=20, plot=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Loss function
    if task == "regression":
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters())

    # Track metrics
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    # ---- Best model tracking ----
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for padded_seq, lengths, labels in pbar:
            padded_seq = padded_seq.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(padded_seq, lengths)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Accuracy
            if task == "classification":
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            pbar.set_postfix({"train_loss": loss.item()})

        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total if task == "classification" else 0

        # ---- Validation ----
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for padded_seq, lengths, labels in val_loader:
                padded_seq = padded_seq.to(device)
                lengths = lengths.to(device)
                labels = labels.to(device)

                outputs = model(padded_seq, lengths)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                if task == "classification":
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total if task == "classification" else 0

        # Store metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # ---- Save best model ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Print metrics
        if task == "classification":
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        else:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # ---- Restore the best model ----
    model.load_state_dict(best_model_state)
    model.to(device)

    # ---- Optional Plot ----
    if plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
        axes[0].plot(history['val_loss'], label='Val Loss', marker='o')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        if task == "classification":
            axes[1].plot(history['train_acc'], label='Train Accuracy', marker='o')
            axes[1].plot(history['val_acc'], label='Val Accuracy', marker='o')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy (%)')
            axes[1].set_title('Training and Validation Accuracy')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].axis('off')
            axes[1].text(0.5, 0.5, 'Accuracy only for classification tasks',
                         ha='center', va='center', fontsize=12)

        plt.tight_layout()
        plt.show()

    return model, history




    