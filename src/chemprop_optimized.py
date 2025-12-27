"""
Phase 2B: Optimized Chemprop D-MPNN

Hyperparameter-tuned message passing neural network for molecular property prediction.
Trains ensemble of models with different seeds for robust predictions.

Version: 2.2 (fixed data preparation - numpy to Python float)
"""
_CHEMPROP_OPT_VERSION = "2.2"
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import tempfile
import shutil
warnings.filterwarnings('ignore')

# Check chemprop availability
try:
    import chemprop
    from chemprop import data, featurizers, models, nn
    from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
    CHEMPROP_AVAILABLE = True

    # Log version and API info for debugging
    _chemprop_version = getattr(chemprop, '__version__', 'unknown')
    _has_from_smi = hasattr(MoleculeDatapoint, 'from_smi')
    print(f"chemprop_optimized v{_CHEMPROP_OPT_VERSION} | Chemprop v{_chemprop_version} | from_smi: {_has_from_smi}")
except ImportError:
    CHEMPROP_AVAILABLE = False
    print("Chemprop not available - install with: pip install chemprop")

import torch
import torch.nn as tnn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
import gc

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

# Optimized hyperparameters per endpoint (from grid search)
CHEMPROP_HYPERPARAMS = {
    'LogD': {'hidden_size': 500, 'depth': 4, 'dropout': 0.1, 'ffn_hidden_size': 400, 'epochs': 50},
    'KSOL': {'hidden_size': 500, 'depth': 4, 'dropout': 0.15, 'ffn_hidden_size': 400, 'epochs': 50},
    'HLM CLint': {'hidden_size': 400, 'depth': 3, 'dropout': 0.2, 'ffn_hidden_size': 300, 'epochs': 40},
    'MLM CLint': {'hidden_size': 400, 'depth': 3, 'dropout': 0.2, 'ffn_hidden_size': 300, 'epochs': 40},
    'Caco-2 Permeability Papp A>B': {'hidden_size': 400, 'depth': 4, 'dropout': 0.15, 'ffn_hidden_size': 300, 'epochs': 45},
    'Caco-2 Permeability Efflux': {'hidden_size': 300, 'depth': 3, 'dropout': 0.2, 'ffn_hidden_size': 200, 'epochs': 40},
    'MPPB': {'hidden_size': 500, 'depth': 4, 'dropout': 0.1, 'ffn_hidden_size': 400, 'epochs': 50},
    'MBPB': {'hidden_size': 500, 'depth': 4, 'dropout': 0.1, 'ffn_hidden_size': 400, 'epochs': 50},
    'MGMB': {'hidden_size': 300, 'depth': 3, 'dropout': 0.25, 'ffn_hidden_size': 200, 'epochs': 35},
}


class ChempropModel:
    """Wrapper for Chemprop D-MPNN model"""

    def __init__(self, target, hp=None, device=None):
        self.target = target
        self.hp = hp or CHEMPROP_HYPERPARAMS.get(target, {})
        self.device = device or DEVICE
        self.model = None
        self.scaler_mean = None
        self.scaler_std = None

    def _create_model(self):
        """Create Chemprop MPNN model"""
        mp = nn.BondMessagePassing(
            d_h=self.hp.get('hidden_size', 400),
            d_vd=0,
            depth=self.hp.get('depth', 3),
            dropout=self.hp.get('dropout', 0.1),
            activation='relu',
        )

        agg = nn.MeanAggregation()

        ffn = nn.RegressionFFN(
            input_dim=self.hp.get('hidden_size', 400),
            hidden_dim=self.hp.get('ffn_hidden_size', 300),
            n_layers=2,
            dropout=self.hp.get('dropout', 0.1),
            activation='relu',
        )

        model = models.MPNN(
            message_passing=mp,
            agg=agg,
            predictor=ffn,
        )

        return model.to(self.device)

    def _prepare_data(self, smiles, targets=None):
        """Prepare data for Chemprop v2"""
        from rdkit import Chem

        datapoints = []
        for i, smi in enumerate(smiles):
            # Validate SMILES first
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue

            try:
                if targets is not None:
                    # IMPORTANT: Convert numpy float to Python float for Chemprop
                    y_val = [float(targets[i])]
                    dp = MoleculeDatapoint.from_smi(smi, y=y_val)
                else:
                    dp = MoleculeDatapoint.from_smi(smi)
                datapoints.append(dp)
            except Exception as e:
                # Skip problematic molecules
                continue

        if len(datapoints) == 0:
            raise ValueError("No valid molecules found in dataset")

        return MoleculeDataset(datapoints)

    def fit(self, smiles, targets, val_smiles=None, val_targets=None, verbose=True):
        """Train the model"""
        # Normalize targets
        self.scaler_mean = np.mean(targets)
        self.scaler_std = np.std(targets) + 1e-8
        targets_scaled = (targets - self.scaler_mean) / self.scaler_std

        # Prepare datasets
        train_dataset = self._prepare_data(smiles, targets_scaled)
        train_loader = build_dataloader(train_dataset, batch_size=64, shuffle=True)

        if val_smiles is not None and val_targets is not None:
            val_targets_scaled = (val_targets - self.scaler_mean) / self.scaler_std
            val_dataset = self._prepare_data(val_smiles, val_targets_scaled)
            val_loader = build_dataloader(val_dataset, batch_size=64, shuffle=False)
        else:
            val_loader = None

        # Create model
        self.model = self._create_model()
        optimizer = Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        criterion = tnn.MSELoss()

        epochs = self.hp.get('epochs', 50)
        best_val_loss = float('inf')
        patience_counter = 0

        def get_targets(batch):
            """Extract targets from chemprop v2 batch (handles API variations)"""
            # Try different attribute names used in chemprop versions
            for attr in ['Y', 'y', 'targets']:
                if hasattr(batch, attr):
                    t = getattr(batch, attr)
                    if t is not None:
                        if hasattr(t, 'to'):
                            return t.to(self.device)
                        return torch.tensor(t, device=self.device, dtype=torch.float32)
            raise AttributeError(f"Cannot find targets in batch. Attributes: {dir(batch)}")

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0

            n_batches = 0
            for batch in train_loader:
                try:
                    bmg = batch.bmg.to(self.device)
                    y = get_targets(batch)

                    optimizer.zero_grad()
                    preds = self.model(bmg)
                    loss = criterion(preds.squeeze(), y.squeeze())
                    loss.backward()
                    tnn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    train_loss += loss.item()
                    n_batches += 1
                except Exception as e:
                    # Log first few errors for debugging
                    if n_batches == 0:
                        print(f"    Batch error: {e}")
                    continue

            if n_batches == 0:
                raise RuntimeError("No valid batches found - check SMILES validity")

            train_loss /= max(n_batches, 1)

            # Validation
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                n_val_batches = 0
                with torch.no_grad():
                    for batch in val_loader:
                        try:
                            bmg = batch.bmg.to(self.device)
                            y = get_targets(batch)
                            preds = self.model(bmg)
                            loss = criterion(preds.squeeze(), y.squeeze())
                            val_loss += loss.item()
                            n_val_batches += 1
                        except Exception:
                            continue
                val_loss /= max(n_val_batches, 1)
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= 10:
                    if verbose:
                        print(f"    Early stopping at epoch {epoch+1}")
                    break

            if verbose and (epoch + 1) % 10 == 0:
                msg = f"    Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}"
                if val_loader:
                    msg += f", Val Loss: {val_loss:.4f}"
                print(msg)

        return self

    def predict(self, smiles):
        """Predict targets for SMILES"""
        self.model.eval()
        dataset = self._prepare_data(smiles)
        loader = build_dataloader(dataset, batch_size=64, shuffle=False)

        predictions = []
        with torch.no_grad():
            for batch in loader:
                try:
                    bmg = batch.bmg.to(self.device)
                    preds = self.model(bmg)
                    predictions.extend(preds.cpu().numpy().flatten())
                except Exception:
                    # For failed batches, use NaN (will be replaced with mean later)
                    predictions.extend([np.nan])

        # Inverse transform
        predictions = np.array(predictions) * self.scaler_std + self.scaler_mean

        # Replace NaN with mean (from training data)
        nan_mask = np.isnan(predictions)
        if nan_mask.any():
            predictions[nan_mask] = self.scaler_mean

        predictions = np.clip(predictions, *VALID_RANGES[self.target])

        return predictions


class ChempropEnsemble:
    """Ensemble of Chemprop models with different seeds"""

    def __init__(self, target, n_models=3, hp=None):
        self.target = target
        self.n_models = n_models
        self.hp = hp or CHEMPROP_HYPERPARAMS.get(target, {})
        self.models = []

    def fit(self, smiles, targets, val_smiles=None, val_targets=None, verbose=True):
        """Train ensemble of models"""
        for i in range(self.n_models):
            if verbose:
                print(f"  Training Chemprop model {i+1}/{self.n_models}")

            torch.manual_seed(42 + i)
            np.random.seed(42 + i)

            model = ChempropModel(self.target, self.hp)
            model.fit(smiles, targets, val_smiles, val_targets, verbose=verbose)
            self.models.append(model)

            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()

        return self

    def predict(self, smiles):
        """Average predictions from all models"""
        all_preds = [model.predict(smiles) for model in self.models]
        return np.mean(all_preds, axis=0)


def train_chemprop_cv(smiles, y_dict, n_folds=5, n_models=3, verbose=True):
    """
    Train Chemprop models with cross-validation

    Returns OOF predictions and metrics
    """
    n_samples = len(smiles)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    oof_preds = {t: np.zeros(n_samples) for t in TARGETS}
    results = {}

    for target in TARGETS:
        if verbose:
            print(f"\n{'='*50}")
            print(f"Chemprop: {target}")
            print(f"{'='*50}")

        y = y_dict[target]
        mask = ~np.isnan(y)
        valid_idx = np.where(mask)[0]
        valid_smiles = [smiles[i] for i in valid_idx]
        valid_y = y[mask]

        if verbose:
            print(f"Samples: {len(valid_y)}")

        fold_preds = np.zeros(len(valid_y))

        for fold, (train_idx, val_idx) in enumerate(kf.split(valid_smiles)):
            if verbose:
                print(f"\n  Fold {fold+1}/{n_folds}")

            train_smiles = [valid_smiles[i] for i in train_idx]
            val_smiles_fold = [valid_smiles[i] for i in val_idx]
            train_y = valid_y[train_idx]
            val_y = valid_y[val_idx]

            # Train ensemble
            ensemble = ChempropEnsemble(target, n_models=n_models)
            ensemble.fit(train_smiles, train_y, val_smiles_fold, val_y, verbose=verbose)

            # Predict
            preds = ensemble.predict(val_smiles_fold)
            fold_preds[val_idx] = preds

            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()

        # Store OOF predictions
        oof_preds[target][valid_idx] = fold_preds

        # Calculate metrics
        mae = mean_absolute_error(valid_y, fold_preds)
        rae = mae / np.mean(np.abs(valid_y - np.mean(valid_y)))
        spear = spearmanr(valid_y, fold_preds)[0]

        results[target] = {'MAE': mae, 'RAE': rae, 'Spearman': spear}

        if verbose:
            print(f"\n  {target}: RAE={rae:.4f}, Spearman={spear:.4f}")

    return oof_preds, results


def train_chemprop_final(smiles_train, y_dict, smiles_test, n_models=5, verbose=True):
    """
    Train final Chemprop models on all data and predict test set

    Returns predictions for test set
    """
    predictions = {}

    for target in TARGETS:
        if verbose:
            print(f"\n{'='*50}")
            print(f"Final Chemprop: {target}")
            print(f"{'='*50}")

        y = y_dict[target]
        mask = ~np.isnan(y)
        valid_smiles = [smiles_train[i] for i, m in enumerate(mask) if m]
        valid_y = y[mask]

        if verbose:
            print(f"Training on {len(valid_y)} samples")

        # Train ensemble on all data
        ensemble = ChempropEnsemble(target, n_models=n_models)
        ensemble.fit(valid_smiles, valid_y, verbose=verbose)

        # Predict test set
        preds = ensemble.predict(smiles_test)
        predictions[target] = preds

        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

    return predictions


if __name__ == "__main__":
    if not CHEMPROP_AVAILABLE:
        print("Chemprop not installed. Run: pip install chemprop")
        sys.exit(1)

    print("Testing Chemprop Optimized")
    print("=" * 50)
    print(f"Device: {DEVICE}")

    # Test with small synthetic data
    test_smiles = ['CCO', 'CCCO', 'CCCCO', 'CC(C)O', 'CC(=O)O'] * 20
    test_y = np.random.randn(len(test_smiles)) * 2 + 5

    print(f"\nTest SMILES: {len(test_smiles)}")

    model = ChempropModel('LogD')
    model.fit(test_smiles[:80], test_y[:80], test_smiles[80:], test_y[80:], verbose=True)

    preds = model.predict(test_smiles[80:])
    print(f"\nPredictions shape: {preds.shape}")
    print(f"Sample predictions: {preds[:5]}")
