"""
TRUE Multi-Task Chemprop D-MPNN

Key difference from single-task:
- Shared message passing layers learn universal molecular representations
- Multiple output heads for each endpoint
- Auxiliary task learning improves generalization
- Based on paper: Multi-task Chemprop outperforms single-task

Version: 1.0
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import gc
warnings.filterwarnings('ignore')

# Check chemprop availability
try:
    import chemprop
    from chemprop import data, featurizers, models, nn
    from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
    CHEMPROP_AVAILABLE = True
    _chemprop_version = getattr(chemprop, '__version__', 'unknown')
except ImportError:
    CHEMPROP_AVAILABLE = False
    print("Chemprop not available - install with: pip install chemprop")

import torch
import torch.nn as tnn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TARGETS = ['LogD', 'KSOL', 'HLM CLint', 'MLM CLint',
           'Caco-2 Permeability Papp A>B', 'Caco-2 Permeability Efflux',
           'MPPB', 'MBPB', 'MGMB']

VALID_RANGES = {
    'LogD': (-3.0, 6.0), 'KSOL': (0.001, 350.0),
    'HLM CLint': (0.0, 3000.0), 'MLM CLint': (0.0, 12000.0),
    'Caco-2 Permeability Papp A>B': (0.0, 60.0),
    'Caco-2 Permeability Efflux': (0.2, 120.0),
    'MPPB': (0.0, 100.0), 'MBPB': (0.0, 100.0), 'MGMB': (0.0, 100.0)
}


class MultiTaskFFN(tnn.Module):
    """
    Multi-task Feed Forward Network with shared backbone and task-specific heads

    Architecture:
    - Shared layers: Learn common molecular property patterns
    - Task-specific heads: Specialize for each endpoint
    """

    def __init__(self, input_dim, hidden_dim=512, n_tasks=9, dropout=0.15):
        super().__init__()
        self.n_tasks = n_tasks

        # Shared backbone
        self.shared = tnn.Sequential(
            tnn.Linear(input_dim, hidden_dim),
            tnn.ReLU(),
            tnn.Dropout(dropout),
            tnn.Linear(hidden_dim, hidden_dim // 2),
            tnn.ReLU(),
            tnn.Dropout(dropout),
        )

        # Task-specific heads
        self.heads = tnn.ModuleList([
            tnn.Sequential(
                tnn.Linear(hidden_dim // 2, hidden_dim // 4),
                tnn.ReLU(),
                tnn.Dropout(dropout * 0.5),
                tnn.Linear(hidden_dim // 4, 1)
            ) for _ in range(n_tasks)
        ])

    def forward(self, x):
        """
        Forward pass

        Args:
            x: (batch_size, input_dim)

        Returns:
            (batch_size, n_tasks)
        """
        shared_repr = self.shared(x)
        outputs = [head(shared_repr) for head in self.heads]
        return torch.cat(outputs, dim=1)


class MultiTaskChemprop:
    """
    Multi-Task Chemprop D-MPNN

    Architecture:
    - Shared Bond Message Passing: Learns molecular graph representations
    - Shared Mean Aggregation: Converts node features to molecule features
    - Multi-Task FFN: Task-specific predictions with shared backbone

    Training:
    - Masked loss: Only compute loss for non-NaN targets
    - Task weighting: Balance endpoints with different sample sizes
    """

    def __init__(self,
                 hidden_size=500,
                 depth=4,
                 ffn_hidden_size=400,
                 dropout=0.15,
                 n_tasks=9,
                 device=None):
        self.hidden_size = hidden_size
        self.depth = depth
        self.ffn_hidden_size = ffn_hidden_size
        self.dropout = dropout
        self.n_tasks = n_tasks
        self.device = device or DEVICE

        self.message_passing = None
        self.aggregation = None
        self.ffn = None

        # Per-task normalization
        self.scaler_means = None
        self.scaler_stds = None

    def _create_model(self):
        """Create shared MPNN + multi-task FFN"""
        # Shared message passing
        self.message_passing = nn.BondMessagePassing(
            d_h=self.hidden_size,
            d_vd=0,
            depth=self.depth,
            dropout=self.dropout,
            activation='relu',
        ).to(self.device)

        # Shared aggregation
        self.aggregation = nn.MeanAggregation()

        # Multi-task FFN
        self.ffn = MultiTaskFFN(
            input_dim=self.hidden_size,
            hidden_dim=self.ffn_hidden_size,
            n_tasks=self.n_tasks,
            dropout=self.dropout
        ).to(self.device)

    def _prepare_data(self, smiles, targets=None):
        """
        Prepare multi-task dataset

        Args:
            smiles: List of SMILES
            targets: np.array of shape (n_samples, n_tasks), can have NaN

        Returns:
            MoleculeDataset
        """
        from rdkit import Chem

        datapoints = []
        valid_indices = []

        for i, smi in enumerate(smiles):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue

            try:
                if targets is not None:
                    # Convert to list, keeping NaN as-is (will mask in loss)
                    y_vals = [float(v) if not np.isnan(v) else 0.0 for v in targets[i]]
                    dp = MoleculeDatapoint.from_smi(smi, y=y_vals)
                else:
                    dp = MoleculeDatapoint.from_smi(smi)
                datapoints.append(dp)
                valid_indices.append(i)
            except Exception:
                continue

        if len(datapoints) == 0:
            raise ValueError("No valid molecules")

        return MoleculeDataset(datapoints), valid_indices

    def _masked_mse_loss(self, preds, targets, masks):
        """
        Compute MSE loss only for non-NaN targets

        Args:
            preds: (batch, n_tasks)
            targets: (batch, n_tasks)
            masks: (batch, n_tasks) - 1 for valid, 0 for NaN
        """
        # Apply mask
        masked_preds = preds * masks
        masked_targets = targets * masks

        # Compute per-task loss
        diff = (masked_preds - masked_targets) ** 2

        # Average over valid entries
        n_valid = masks.sum(dim=0).clamp(min=1)
        task_losses = (diff.sum(dim=0) / n_valid)

        # Return mean across tasks
        return task_losses.mean()

    def fit(self, smiles, targets_dict, val_smiles=None, val_targets_dict=None,
            epochs=60, batch_size=64, lr=1e-3, verbose=True):
        """
        Train multi-task model

        Args:
            smiles: List of SMILES
            targets_dict: Dict of {target_name: np.array}
            val_smiles: Validation SMILES (optional)
            val_targets_dict: Validation targets (optional)
            epochs: Training epochs
            batch_size: Batch size
            lr: Learning rate
            verbose: Print progress
        """
        # Build targets matrix (n_samples, n_tasks)
        n_samples = len(smiles)
        targets = np.full((n_samples, self.n_tasks), np.nan)

        for i, target_name in enumerate(TARGETS):
            if target_name in targets_dict:
                targets[:, i] = targets_dict[target_name]

        # Normalize per task
        self.scaler_means = np.nanmean(targets, axis=0)
        self.scaler_stds = np.nanstd(targets, axis=0) + 1e-8

        targets_scaled = (targets - self.scaler_means) / self.scaler_stds

        # Create mask for non-NaN entries
        train_masks = ~np.isnan(targets)
        targets_scaled = np.nan_to_num(targets_scaled, nan=0.0)

        # Prepare datasets
        train_dataset, train_indices = self._prepare_data(smiles, targets_scaled)
        train_loader = build_dataloader(train_dataset, batch_size=batch_size, shuffle=True)

        # Filter masks to valid molecules
        train_masks = train_masks[train_indices]

        # Validation if provided
        if val_smiles is not None and val_targets_dict is not None:
            val_targets = np.full((len(val_smiles), self.n_tasks), np.nan)
            for i, target_name in enumerate(TARGETS):
                if target_name in val_targets_dict:
                    val_targets[:, i] = val_targets_dict[target_name]

            val_targets_scaled = (val_targets - self.scaler_means) / self.scaler_stds
            val_masks = ~np.isnan(val_targets)
            val_targets_scaled = np.nan_to_num(val_targets_scaled, nan=0.0)

            val_dataset, val_indices = self._prepare_data(val_smiles, val_targets_scaled)
            val_loader = build_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
            val_masks = val_masks[val_indices]
        else:
            val_loader = None
            val_masks = None

        # Create model
        self._create_model()

        # Optimizer with weight decay
        all_params = list(self.message_passing.parameters()) + list(self.ffn.parameters())
        optimizer = AdamW(all_params, lr=lr, weight_decay=1e-4)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            self.message_passing.train()
            self.ffn.train()
            train_loss = 0.0
            n_batches = 0
            mask_idx = 0

            for batch in train_loader:
                try:
                    bmg = batch.bmg
                    bmg.to(self.device)

                    # Get targets
                    y = batch.Y.to(self.device)
                    batch_size_actual = y.shape[0]

                    # Get mask for this batch
                    batch_mask = torch.tensor(
                        train_masks[mask_idx:mask_idx + batch_size_actual],
                        device=self.device, dtype=torch.float32
                    )
                    mask_idx += batch_size_actual

                    # Forward pass
                    optimizer.zero_grad()
                    node_repr = self.message_passing(bmg)
                    mol_repr = self.aggregation(node_repr, bmg.batch)
                    preds = self.ffn(mol_repr)

                    # Masked loss
                    loss = self._masked_mse_loss(preds, y, batch_mask)
                    loss.backward()

                    tnn.utils.clip_grad_norm_(all_params, 1.0)
                    optimizer.step()

                    train_loss += loss.item()
                    n_batches += 1

                except Exception as e:
                    if n_batches == 0:
                        print(f"  Batch error: {e}")
                    continue

            scheduler.step()

            if n_batches == 0:
                raise RuntimeError("No valid batches")

            train_loss /= n_batches

            # Validation
            if val_loader is not None:
                self.message_passing.eval()
                self.ffn.eval()
                val_loss = 0.0
                n_val = 0
                mask_idx = 0

                with torch.no_grad():
                    for batch in val_loader:
                        try:
                            bmg = batch.bmg
                            bmg.to(self.device)
                            y = batch.Y.to(self.device)
                            batch_size_actual = y.shape[0]

                            batch_mask = torch.tensor(
                                val_masks[mask_idx:mask_idx + batch_size_actual],
                                device=self.device, dtype=torch.float32
                            )
                            mask_idx += batch_size_actual

                            node_repr = self.message_passing(bmg)
                            mol_repr = self.aggregation(node_repr, bmg.batch)
                            preds = self.ffn(mol_repr)

                            loss = self._masked_mse_loss(preds, y, batch_mask)
                            val_loss += loss.item()
                            n_val += 1
                        except Exception:
                            continue

                val_loss /= max(n_val, 1)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= 15:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch+1}")
                    break

            if verbose and (epoch + 1) % 10 == 0:
                msg = f"  Epoch {epoch+1}/{epochs}, Train: {train_loss:.4f}"
                if val_loader:
                    msg += f", Val: {val_loss:.4f}"
                print(msg)

        return self

    def predict(self, smiles):
        """
        Make predictions for all tasks

        Args:
            smiles: List of SMILES

        Returns:
            Dict of {target_name: predictions}
        """
        self.message_passing.eval()
        self.ffn.eval()

        dataset, valid_indices = self._prepare_data(smiles)
        loader = build_dataloader(dataset, batch_size=64, shuffle=False)

        all_preds = []

        with torch.no_grad():
            for batch in loader:
                try:
                    bmg = batch.bmg
                    bmg.to(self.device)

                    node_repr = self.message_passing(bmg)
                    mol_repr = self.aggregation(node_repr, bmg.batch)
                    preds = self.ffn(mol_repr)

                    all_preds.append(preds.cpu().numpy())
                except Exception:
                    continue

        if len(all_preds) == 0:
            return {t: np.full(len(smiles), np.nan) for t in TARGETS}

        preds_matrix = np.vstack(all_preds)

        # Inverse transform
        preds_matrix = preds_matrix * self.scaler_stds + self.scaler_means

        # Build result dict
        result = {}
        for i, target_name in enumerate(TARGETS):
            full_preds = np.full(len(smiles), self.scaler_means[i])
            full_preds[valid_indices] = preds_matrix[:, i]
            full_preds = np.clip(full_preds, *VALID_RANGES[target_name])
            result[target_name] = full_preds

        return result


class MultiTaskChempropEnsemble:
    """Ensemble of multi-task Chemprop models with different seeds"""

    def __init__(self, n_models=3, **kwargs):
        self.n_models = n_models
        self.kwargs = kwargs
        self.models = []

    def fit(self, smiles, targets_dict, val_smiles=None, val_targets_dict=None, verbose=True):
        """Train ensemble of models"""
        for i in range(self.n_models):
            if verbose:
                print(f"\n  Training multi-task model {i+1}/{self.n_models}")

            torch.manual_seed(42 + i)
            np.random.seed(42 + i)

            model = MultiTaskChemprop(**self.kwargs)
            model.fit(smiles, targets_dict, val_smiles, val_targets_dict, verbose=verbose)
            self.models.append(model)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        return self

    def predict(self, smiles):
        """Average predictions from all models"""
        all_preds = {t: [] for t in TARGETS}

        for model in self.models:
            preds = model.predict(smiles)
            for t in TARGETS:
                all_preds[t].append(preds[t])

        return {t: np.mean(all_preds[t], axis=0) for t in TARGETS}


def train_multitask_cv(smiles, y_dict, n_folds=5, n_models=3, verbose=True):
    """
    Train multi-task Chemprop with cross-validation

    Returns OOF predictions and metrics
    """
    n_samples = len(smiles)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    oof_preds = {t: np.zeros(n_samples) for t in TARGETS}
    results = {}

    for fold, (train_idx, val_idx) in enumerate(kf.split(smiles)):
        if verbose:
            print(f"\n{'='*50}")
            print(f"Multi-task Fold {fold+1}/{n_folds}")
            print(f"{'='*50}")

        train_smiles = [smiles[i] for i in train_idx]
        val_smiles = [smiles[i] for i in val_idx]

        train_targets = {t: y_dict[t][train_idx] for t in TARGETS if t in y_dict}
        val_targets = {t: y_dict[t][val_idx] for t in TARGETS if t in y_dict}

        # Train ensemble
        ensemble = MultiTaskChempropEnsemble(n_models=n_models)
        ensemble.fit(train_smiles, train_targets, val_smiles, val_targets, verbose=verbose)

        # Predict
        preds = ensemble.predict(val_smiles)
        for t in TARGETS:
            oof_preds[t][val_idx] = preds[t]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # Calculate metrics per endpoint
    for target in TARGETS:
        if target not in y_dict:
            continue

        y = y_dict[target]
        mask = ~np.isnan(y)

        if mask.sum() < 10:
            continue

        mae = mean_absolute_error(y[mask], oof_preds[target][mask])
        rae = mae / np.mean(np.abs(y[mask] - np.mean(y[mask])))
        spear = spearmanr(y[mask], oof_preds[target][mask])[0]

        results[target] = {'MAE': mae, 'RAE': rae, 'Spearman': spear}

        if verbose:
            print(f"\n{target}: RAE={rae:.4f}, Spearman={spear:.4f}")

    return oof_preds, results


if __name__ == "__main__":
    if not CHEMPROP_AVAILABLE:
        print("Chemprop not installed")
        sys.exit(1)

    print("Testing Multi-Task Chemprop")
    print("=" * 50)
    print(f"Device: {DEVICE}")
    print(f"Chemprop version: {_chemprop_version}")

    # Test with synthetic data
    test_smiles = ['CCO', 'CCCO', 'CCCCO', 'CC(C)O', 'CC(=O)O'] * 20
    n = len(test_smiles)

    test_targets = {
        'LogD': np.random.randn(n) * 2 + 2,
        'KSOL': np.random.rand(n) * 100,
        'HLM CLint': np.random.rand(n) * 500,
    }
    # Add some NaN to test masking
    test_targets['LogD'][5:10] = np.nan
    test_targets['KSOL'][15:25] = np.nan

    print(f"\nTest samples: {n}")
    print(f"Targets: {list(test_targets.keys())}")

    model = MultiTaskChemprop(hidden_size=256, depth=3)
    model.fit(test_smiles[:80], {k: v[:80] for k, v in test_targets.items()},
              test_smiles[80:], {k: v[80:] for k, v in test_targets.items()},
              epochs=30, verbose=True)

    preds = model.predict(test_smiles[80:])
    print(f"\nPrediction keys: {list(preds.keys())}")
    for t, p in preds.items():
        if t in test_targets:
            print(f"{t}: mean={p.mean():.2f}, std={p.std():.2f}")
