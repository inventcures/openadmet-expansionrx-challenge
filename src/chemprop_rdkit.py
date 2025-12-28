"""
Chemprop-RDKit Wrapper for V4 Pipeline.

Chemprop-RDKit is the #1 model on TDC ADMET leaderboard:
- D-MPNN (Directed Message Passing Neural Network)
- 200 RDKit descriptors as extra features
- Hybrid approach outperforms pure GNNs and pure descriptors

Reference: https://chemprop.readthedocs.io/en/latest/
"""

import os
import warnings
from typing import List, Optional, Tuple, Dict, Any, Union
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# Suppress warnings during import
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        TORCH_AVAILABLE = True
    except ImportError:
        TORCH_AVAILABLE = False
        print("Warning: PyTorch not available")

    try:
        import chemprop
        from chemprop import data, featurizers, models, nn as chemprop_nn
        from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
        from chemprop.utils import make_mol
        CHEMPROP_AVAILABLE = True
    except ImportError:
        CHEMPROP_AVAILABLE = False
        print("Warning: Chemprop not available. Install with: pip install chemprop>=2.0")

    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Descriptors3D
        from rdkit.ML.Descriptors import MoleculeDescriptors
        RDKIT_AVAILABLE = True
    except ImportError:
        RDKIT_AVAILABLE = False
        print("Warning: RDKit not available")


def compute_rdkit_descriptors(smiles_list: List[str], verbose: bool = True) -> np.ndarray:
    """
    Compute 200 RDKit descriptors for molecules.

    Args:
        smiles_list: List of SMILES strings
        verbose: Show progress

    Returns:
        Array of shape (n_molecules, n_descriptors)
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit not available")

    # Get all 2D descriptor names
    descriptor_names = [x[0] for x in Descriptors._descList]

    # Create calculator
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

    descriptors = []
    from tqdm import tqdm

    iterator = tqdm(smiles_list, desc="Computing RDKit descriptors", disable=not verbose)

    for smi in iterator:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            # Return NaN for invalid molecules
            descriptors.append([np.nan] * len(descriptor_names))
        else:
            try:
                desc = calculator.CalcDescriptors(mol)
                descriptors.append(desc)
            except Exception:
                descriptors.append([np.nan] * len(descriptor_names))

    X = np.array(descriptors, dtype=np.float32)

    # Replace inf with nan
    X[~np.isfinite(X)] = np.nan

    # Fill NaN with column median
    for i in range(X.shape[1]):
        col = X[:, i]
        median = np.nanmedian(col)
        if np.isnan(median):
            median = 0.0
        X[np.isnan(col), i] = median

    return X


class ChempropRDKitModel:
    """
    Chemprop D-MPNN with RDKit descriptors.

    This is the TDC ADMET leaderboard #1 architecture.
    Combines:
    - Graph neural network (D-MPNN) for molecular structure
    - 200 RDKit descriptors for molecular properties
    - Feed-forward network for final prediction
    """

    def __init__(
        self,
        hidden_size: int = 300,
        depth: int = 3,
        dropout: float = 0.0,
        ffn_hidden_size: int = 300,
        ffn_num_layers: int = 2,
        epochs: int = 30,
        batch_size: int = 50,
        learning_rate: float = 1e-4,
        warmup_epochs: int = 2,
        use_rdkit_features: bool = True,
        aggregation: str = 'mean',
        device: str = 'auto',
        seed: int = 42,
        verbose: bool = True
    ):
        self.hidden_size = hidden_size
        self.depth = depth
        self.dropout = dropout
        self.ffn_hidden_size = ffn_hidden_size
        self.ffn_num_layers = ffn_num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.use_rdkit_features = use_rdkit_features
        self.aggregation = aggregation
        self.seed = seed
        self.verbose = verbose

        # Device setup
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = None
        self.rdkit_scaler = None
        self.y_scaler = None

    def _prepare_data(
        self,
        smiles: List[str],
        y: Optional[np.ndarray] = None,
        fit_scalers: bool = False
    ) -> Tuple[MoleculeDataset, Optional[np.ndarray]]:
        """Prepare Chemprop dataset with RDKit features."""

        # Compute RDKit features
        if self.use_rdkit_features:
            rdkit_features = compute_rdkit_descriptors(smiles, verbose=self.verbose)

            if fit_scalers:
                self.rdkit_scaler = StandardScaler()
                rdkit_features = self.rdkit_scaler.fit_transform(rdkit_features)
            elif self.rdkit_scaler is not None:
                rdkit_features = self.rdkit_scaler.transform(rdkit_features)
        else:
            rdkit_features = None

        # Scale targets
        scaled_y = None
        if y is not None:
            if fit_scalers:
                self.y_scaler = StandardScaler()
                scaled_y = self.y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
            elif self.y_scaler is not None:
                scaled_y = self.y_scaler.transform(y.reshape(-1, 1)).flatten()
            else:
                scaled_y = y

        # Create datapoints
        datapoints = []
        for i, smi in enumerate(smiles):
            mol = make_mol(smi, keep_h=False, add_h=False)

            if y is not None:
                target = [[float(scaled_y[i])]]
            else:
                target = None

            if rdkit_features is not None:
                x_d = rdkit_features[i]
            else:
                x_d = None

            dp = MoleculeDatapoint(mol=mol, y=target, x_d=x_d)
            datapoints.append(dp)

        dataset = MoleculeDataset(datapoints)

        return dataset, scaled_y

    def _build_model(self, x_d_dim: int = 0) -> models.MPNN:
        """Build Chemprop MPNN model."""

        # Message passing
        mp = chemprop_nn.BondMessagePassing(
            d_h=self.hidden_size,
            depth=self.depth,
            dropout=self.dropout
        )

        # Aggregation
        if self.aggregation == 'mean':
            agg = chemprop_nn.MeanAggregation()
        elif self.aggregation == 'sum':
            agg = chemprop_nn.SumAggregation()
        else:
            agg = chemprop_nn.NormAggregation()

        # FFN input size = hidden_size + x_d_dim
        ffn_input_dim = self.hidden_size + x_d_dim

        # Output FFN
        ffn = chemprop_nn.RegressionFFN(
            input_dim=ffn_input_dim,
            hidden_dim=self.ffn_hidden_size,
            n_layers=self.ffn_num_layers,
            dropout=self.dropout,
            n_tasks=1
        )

        # Full model
        model = models.MPNN(
            message_passing=mp,
            agg=agg,
            ffn=ffn
        )

        return model

    def fit(self, smiles: List[str], y: np.ndarray) -> 'ChempropRDKitModel':
        """
        Train the model.

        Args:
            smiles: List of SMILES strings
            y: Target values

        Returns:
            self
        """
        if not CHEMPROP_AVAILABLE:
            raise ImportError("Chemprop not available")

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Prepare data
        dataset, _ = self._prepare_data(smiles, y, fit_scalers=True)

        # Get x_d dimension
        x_d_dim = 0
        if self.use_rdkit_features and dataset[0].x_d is not None:
            x_d_dim = len(dataset[0].x_d)

        # Build model
        self.model = self._build_model(x_d_dim=x_d_dim)
        self.model = self.model.to(self.device)

        # Create dataloader
        train_loader = build_dataloader(dataset, batch_size=self.batch_size, shuffle=True)

        # Training setup
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Warmup scheduler
        def warmup_lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return (epoch + 1) / self.warmup_epochs
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lr_lambda)

        # Training loop
        self.model.train()
        from tqdm import trange

        epoch_iterator = trange(self.epochs, desc="Training", disable=not self.verbose)

        for epoch in epoch_iterator:
            total_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                # Move batch to device
                bmg, V_d, X_d, targets, *_ = batch

                # Handle device transfer for BatchMolGraph
                bmg = bmg.to(self.device)
                if X_d is not None:
                    X_d = X_d.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                preds = self.model(bmg, V_d, X_d)

                # MSE loss
                loss = torch.nn.functional.mse_loss(preds, targets)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            scheduler.step()

            avg_loss = total_loss / max(n_batches, 1)
            epoch_iterator.set_postfix(loss=f"{avg_loss:.4f}")

        return self

    def predict(self, smiles: List[str]) -> np.ndarray:
        """
        Make predictions.

        Args:
            smiles: List of SMILES strings

        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        # Prepare data
        dataset, _ = self._prepare_data(smiles, y=None, fit_scalers=False)

        # Create dataloader
        test_loader = build_dataloader(dataset, batch_size=self.batch_size, shuffle=False)

        # Prediction
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for batch in test_loader:
                bmg, V_d, X_d, *_ = batch

                bmg = bmg.to(self.device)
                if X_d is not None:
                    X_d = X_d.to(self.device)

                preds = self.model(bmg, V_d, X_d)
                all_preds.append(preds.cpu().numpy())

        predictions = np.concatenate(all_preds, axis=0).flatten()

        # Inverse scale
        if self.y_scaler is not None:
            predictions = self.y_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

        return predictions


class ChempropRDKitEnsemble:
    """
    Ensemble of Chemprop-RDKit models with different seeds.

    Provides:
    - Multi-seed training for robustness
    - K-fold cross-validation with OOF predictions
    - Averaged test predictions
    """

    def __init__(
        self,
        n_seeds: int = 5,
        n_folds: int = 5,
        base_seed: int = 42,
        **model_kwargs
    ):
        self.n_seeds = n_seeds
        self.n_folds = n_folds
        self.base_seed = base_seed
        self.model_kwargs = model_kwargs

        self.models: List[List[ChempropRDKitModel]] = []  # [seed][fold]
        self.oof_predictions: Optional[np.ndarray] = None

    def fit(
        self,
        smiles: List[str],
        y: np.ndarray,
        return_oof: bool = True
    ) -> Tuple['ChempropRDKitEnsemble', Optional[np.ndarray]]:
        """
        Train ensemble with K-fold CV.

        Args:
            smiles: List of SMILES
            y: Target values
            return_oof: Return OOF predictions

        Returns:
            Tuple of (self, oof_predictions)
        """
        smiles = list(smiles)
        y = np.array(y)
        n_samples = len(smiles)

        oof_predictions = np.zeros((n_samples, self.n_seeds))

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.base_seed)

        for seed_idx in range(self.n_seeds):
            seed = self.base_seed + seed_idx * 100
            print(f"\n{'='*50}")
            print(f"Seed {seed_idx + 1}/{self.n_seeds} (seed={seed})")
            print(f"{'='*50}")

            seed_models = []

            for fold, (train_idx, val_idx) in enumerate(kf.split(smiles)):
                print(f"\nFold {fold + 1}/{self.n_folds}")

                train_smiles = [smiles[i] for i in train_idx]
                train_y = y[train_idx]

                val_smiles = [smiles[i] for i in val_idx]

                # Train model
                model = ChempropRDKitModel(seed=seed + fold, **self.model_kwargs)
                model.fit(train_smiles, train_y)

                # OOF predictions
                val_preds = model.predict(val_smiles)
                oof_predictions[val_idx, seed_idx] = val_preds

                seed_models.append(model)

            self.models.append(seed_models)

        # Average OOF across seeds
        self.oof_predictions = np.mean(oof_predictions, axis=1)

        if return_oof:
            return self, self.oof_predictions
        return self, None

    def predict(self, smiles: List[str]) -> np.ndarray:
        """
        Make predictions (average of all models).

        Args:
            smiles: List of SMILES

        Returns:
            Averaged predictions
        """
        all_preds = []

        for seed_models in self.models:
            for model in seed_models:
                preds = model.predict(smiles)
                all_preds.append(preds)

        # Average all predictions
        predictions = np.mean(all_preds, axis=0)

        return predictions

    def fit_predict(
        self,
        smiles_train: List[str],
        y_train: np.ndarray,
        smiles_test: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit and return both OOF and test predictions.

        Args:
            smiles_train: Training SMILES
            y_train: Training targets
            smiles_test: Test SMILES

        Returns:
            Tuple of (oof_predictions, test_predictions)
        """
        self.fit(smiles_train, y_train, return_oof=True)
        test_preds = self.predict(smiles_test)

        return self.oof_predictions, test_preds


def check_dependencies():
    """Check if all dependencies are available."""
    status = {
        'torch': TORCH_AVAILABLE,
        'chemprop': CHEMPROP_AVAILABLE,
        'rdkit': RDKIT_AVAILABLE
    }

    missing = [k for k, v in status.items() if not v]

    if missing:
        print(f"Missing dependencies: {missing}")
        print("\nInstall with:")
        print("  pip install torch chemprop rdkit")
        return False

    print("All dependencies available!")
    return True


if __name__ == "__main__":
    if not check_dependencies():
        exit(1)

    # Quick test
    print("\nTesting Chemprop-RDKit model...")

    test_smiles = [
        "CCO",
        "CC(=O)O",
        "c1ccccc1",
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    ] * 10  # Repeat to have enough data

    test_y = np.random.randn(len(test_smiles))

    model = ChempropRDKitModel(
        epochs=5,
        batch_size=10,
        hidden_size=100,
        verbose=True
    )

    model.fit(test_smiles[:40], test_y[:40])
    preds = model.predict(test_smiles[40:])

    print(f"\nPredictions shape: {preds.shape}")
    print(f"Sample predictions: {preds[:5]}")
    print("\nTest passed!")
