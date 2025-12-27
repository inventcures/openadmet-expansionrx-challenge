"""
Phase 2B: Multi-Task Neural Network for ADMET Prediction

Leverages endpoint relationships:
- Group A (Metabolic): HLM_CLint, MLM_CLint
- Group B (Binding): MPPB, MBPB, MGMB
- Group C (Permeability): Caco2_Papp, Caco2_Efflux
- Group D (Physicochemical): LogD, KSOL
"""
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import gc

# Suppress RDKit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TARGETS = ['LogD', 'KSOL', 'HLM CLint', 'MLM CLint',
           'Caco-2 Permeability Papp A>B', 'Caco-2 Permeability Efflux',
           'MPPB', 'MBPB', 'MGMB']

# Endpoint groupings for shared representations
ENDPOINT_GROUPS = {
    'metabolic': ['HLM CLint', 'MLM CLint'],
    'binding': ['MPPB', 'MBPB', 'MGMB'],
    'permeability': ['Caco-2 Permeability Papp A>B', 'Caco-2 Permeability Efflux'],
    'physicochemical': ['LogD', 'KSOL'],
}

VALID_RANGES = {
    'LogD': (-3.0, 6.0), 'KSOL': (0.001, 350.0),
    'HLM CLint': (0.0, 3000.0), 'MLM CLint': (0.0, 12000.0),
    'Caco-2 Permeability Papp A>B': (0.0, 60.0),
    'Caco-2 Permeability Efflux': (0.2, 120.0),
    'MPPB': (0.0, 100.0), 'MBPB': (0.0, 100.0), 'MGMB': (0.0, 100.0)
}


class MultiTaskDataset(Dataset):
    """Dataset for multi-task learning"""

    def __init__(self, X, y_dict, targets):
        self.X = torch.FloatTensor(X)
        self.targets = targets

        # Create target tensor with NaN mask
        self.y = torch.zeros(len(X), len(targets))
        self.mask = torch.zeros(len(X), len(targets))

        for i, target in enumerate(targets):
            values = y_dict[target]
            valid = ~np.isnan(values)
            self.y[valid, i] = torch.FloatTensor(values[valid])
            self.mask[valid, i] = 1.0

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.mask[idx]


class SharedTrunkNetwork(nn.Module):
    """
    Multi-task network with shared trunk and task-specific heads

    Architecture:
    - Shared trunk: 2 layers for common representations
    - Group-specific layers: 1 layer per endpoint group
    - Task heads: 1 layer per endpoint
    """

    def __init__(self, input_dim, hidden_dim=512, group_dim=256, head_dim=128, dropout=0.3):
        super().__init__()

        # Shared trunk (learns common molecular representations)
        self.shared_trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),
        )

        # Group-specific layers
        self.group_layers = nn.ModuleDict()
        for group_name in ENDPOINT_GROUPS.keys():
            self.group_layers[group_name] = nn.Sequential(
                nn.Linear(hidden_dim, group_dim),
                nn.BatchNorm1d(group_dim),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
            )

        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        for target in TARGETS:
            self.task_heads[target] = nn.Sequential(
                nn.Linear(group_dim, head_dim),
                nn.ReLU(),
                nn.Linear(head_dim, 1),
            )

        # Map targets to groups
        self.target_to_group = {}
        for group_name, targets in ENDPOINT_GROUPS.items():
            for target in targets:
                self.target_to_group[target] = group_name

    def forward(self, x):
        # Shared representation
        shared = self.shared_trunk(x)

        # Group representations
        group_reps = {name: layer(shared) for name, layer in self.group_layers.items()}

        # Task predictions
        outputs = {}
        for target in TARGETS:
            group_name = self.target_to_group[target]
            group_rep = group_reps[group_name]
            outputs[target] = self.task_heads[target](group_rep).squeeze(-1)

        return outputs


class MultiTaskTrainer:
    """Trainer for multi-task ADMET model"""

    def __init__(self, input_dim, lr=1e-3, weight_decay=1e-4,
                 hidden_dim=512, dropout=0.3, device=None):
        self.device = device or DEVICE
        self.input_dim = input_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.model = None
        self.scaler = StandardScaler()
        self.target_scalers = {t: StandardScaler() for t in TARGETS}

    def _create_model(self):
        return SharedTrunkNetwork(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout
        ).to(self.device)

    def _masked_mse_loss(self, predictions, targets, mask):
        """MSE loss with mask for missing values"""
        total_loss = 0.0
        count = 0

        for i, target in enumerate(TARGETS):
            pred = predictions[target]
            tgt = targets[:, i]
            m = mask[:, i]

            if m.sum() > 0:
                loss = ((pred - tgt) ** 2 * m).sum() / m.sum()
                total_loss += loss
                count += 1

        return total_loss / count if count > 0 else total_loss

    def fit(self, X, y_dict, epochs=100, batch_size=256, verbose=True):
        """Train the multi-task model"""

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Scale targets
        y_scaled = {}
        for target in TARGETS:
            values = y_dict[target].copy()
            valid = ~np.isnan(values)
            if valid.sum() > 0:
                values[valid] = self.target_scalers[target].fit_transform(
                    values[valid].reshape(-1, 1)
                ).flatten()
            y_scaled[target] = values

        # Create dataset
        dataset = MultiTaskDataset(X_scaled, y_scaled, TARGETS)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Create model
        self.model = self._create_model()
        optimizer = optim.AdamW(self.model.parameters(),
                                lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0

            for X_batch, y_batch, mask_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                mask_batch = mask_batch.to(self.device)

                optimizer.zero_grad()
                predictions = self.model(X_batch)
                loss = self._masked_mse_loss(predictions, y_batch, mask_batch)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()

            if verbose and (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(loader):.4f}")

        return self

    def predict(self, X):
        """Predict all endpoints"""
        self.model.eval()
        X_scaled = self.scaler.transform(X)

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            predictions = self.model(X_tensor)

        # Inverse transform predictions
        results = {}
        for target in TARGETS:
            pred = predictions[target].cpu().numpy()
            pred = self.target_scalers[target].inverse_transform(
                pred.reshape(-1, 1)
            ).flatten()
            pred = np.clip(pred, *VALID_RANGES[target])
            results[target] = pred

        return results


def train_multitask_cv(X, y_dict, n_folds=5, epochs=100, verbose=True):
    """
    Train multi-task model with cross-validation

    Returns OOF predictions and metrics
    """
    n_samples = len(X)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    oof_preds = {t: np.zeros(n_samples) for t in TARGETS}
    oof_counts = {t: np.zeros(n_samples) for t in TARGETS}

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        if verbose:
            print(f"\n  Fold {fold+1}/{n_folds}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train = {t: y_dict[t][train_idx] for t in TARGETS}

        # Train model
        trainer = MultiTaskTrainer(input_dim=X.shape[1], epochs=epochs)
        trainer.fit(X_train, y_train, epochs=epochs, verbose=verbose)

        # Predict
        preds = trainer.predict(X_val)

        for target in TARGETS:
            oof_preds[target][val_idx] = preds[target]
            mask = ~np.isnan(y_dict[target][val_idx])
            oof_counts[target][val_idx] = mask.astype(float)

        # Cleanup
        del trainer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

    # Calculate metrics
    results = {}
    for target in TARGETS:
        mask = ~np.isnan(y_dict[target])
        if mask.sum() == 0:
            continue

        y_true = y_dict[target][mask]
        y_pred = oof_preds[target][mask]

        mae = mean_absolute_error(y_true, y_pred)
        rae = mae / np.mean(np.abs(y_true - np.mean(y_true)))
        spear = spearmanr(y_true, y_pred)[0]

        results[target] = {'MAE': mae, 'RAE': rae, 'Spearman': spear}

        if verbose:
            print(f"  {target}: RAE={rae:.4f}, Spearman={spear:.4f}")

    return oof_preds, results


class MultiTaskEnsembleMember:
    """
    Multi-task model as an ensemble member

    Can be combined with GBDT models in stacking ensemble
    """

    def __init__(self, input_dim, n_models=3, epochs=80):
        self.input_dim = input_dim
        self.n_models = n_models
        self.epochs = epochs
        self.trainers = []

    def fit(self, X, y_dict, verbose=True):
        """Train ensemble of multi-task models"""
        for i in range(self.n_models):
            if verbose:
                print(f"  Training multi-task model {i+1}/{self.n_models}")

            trainer = MultiTaskTrainer(
                input_dim=self.input_dim,
                lr=1e-3 * (0.8 ** i),  # Slightly different LR
                hidden_dim=512 - i * 64,  # Slightly different architecture
            )
            trainer.fit(X, y_dict, epochs=self.epochs, verbose=False)
            self.trainers.append(trainer)

        return self

    def predict(self, X):
        """Average predictions from all models"""
        all_preds = {t: [] for t in TARGETS}

        for trainer in self.trainers:
            preds = trainer.predict(X)
            for target in TARGETS:
                all_preds[target].append(preds[target])

        # Average
        return {t: np.mean(all_preds[t], axis=0) for t in TARGETS}


if __name__ == "__main__":
    # Test with synthetic data
    print("Testing Multi-Task Neural Network")
    print("=" * 50)
    print(f"Device: {DEVICE}")

    # Create synthetic data
    n_samples = 1000
    n_features = 100

    np.random.seed(42)
    X = np.random.randn(n_samples, n_features).astype(np.float32)

    y_dict = {}
    for target in TARGETS:
        y = np.random.randn(n_samples) * 10 + 50
        # Add some NaN values
        y[np.random.choice(n_samples, size=100, replace=False)] = np.nan
        y_dict[target] = y

    print(f"\nX shape: {X.shape}")
    print(f"Targets: {len(TARGETS)}")

    # Test training
    print("\nTraining with 3-fold CV...")
    oof_preds, results = train_multitask_cv(X, y_dict, n_folds=3, epochs=30, verbose=True)

    print("\n" + "=" * 50)
    print("Results:")
    raes = [r['RAE'] for r in results.values()]
    print(f"MA-RAE: {np.mean(raes):.4f}")
