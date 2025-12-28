"""
AttentiveFP Wrapper for V4 Pipeline.

AttentiveFP is a graph attention network for molecular property prediction:
- Extends GAT with gated recurrent units
- Atom-level and molecule-level attention
- State-of-the-art on MoleculeNet benchmarks

Reference: https://github.com/awslabs/dgl-lifesci
"""

import warnings
from typing import List, Optional, Tuple
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

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
        import dgl
        from dgllife.model import AttentiveFPPredictor
        from dgllife.utils import (
            AttentiveFPAtomFeaturizer,
            AttentiveFPBondFeaturizer,
            mol_to_bigraph,
            smiles_to_bigraph
        )
        DGL_AVAILABLE = True
    except ImportError:
        DGL_AVAILABLE = False
        print("Warning: DGL-LifeSci not available. Install with: pip install dgllife")

    try:
        from rdkit import Chem
        RDKIT_AVAILABLE = True
    except ImportError:
        RDKIT_AVAILABLE = False


def smiles_to_graph(smiles: str, atom_featurizer, bond_featurizer):
    """Convert SMILES to DGL graph with features."""
    try:
        g = smiles_to_bigraph(
            smiles,
            node_featurizer=atom_featurizer,
            edge_featurizer=bond_featurizer,
            add_self_loop=False
        )
        return g
    except Exception:
        return None


def collate_graphs(data_list):
    """Collate function for DataLoader."""
    graphs, labels = zip(*data_list)

    # Filter out None graphs
    valid_pairs = [(g, l) for g, l in zip(graphs, labels) if g is not None]

    if len(valid_pairs) == 0:
        return None, None

    graphs, labels = zip(*valid_pairs)
    batched_graph = dgl.batch(graphs)
    labels = torch.tensor(labels, dtype=torch.float32)

    return batched_graph, labels


def collate_graphs_inference(graphs):
    """Collate function for inference (no labels)."""
    # Filter out None graphs
    valid_graphs = [g for g in graphs if g is not None]

    if len(valid_graphs) == 0:
        return None

    return dgl.batch(valid_graphs)


class GraphDataset(torch.utils.data.Dataset):
    """Dataset for graph neural networks."""

    def __init__(self, graphs, labels=None):
        self.graphs = graphs
        self.labels = labels

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.graphs[idx], self.labels[idx]
        return self.graphs[idx]


class AttentiveFPModel:
    """
    AttentiveFP model for molecular property prediction.

    Uses graph attention mechanism with:
    - Atom-level attention for local interactions
    - Molecule-level attention for global aggregation
    - Gated recurrent units for message passing
    """

    def __init__(
        self,
        node_feat_size: int = 39,
        edge_feat_size: int = 10,
        num_layers: int = 2,
        num_timesteps: int = 2,
        graph_feat_size: int = 200,
        dropout: float = 0.2,
        epochs: int = 100,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        patience: int = 10,
        device: str = 'auto',
        seed: int = 42,
        verbose: bool = True
    ):
        self.node_feat_size = node_feat_size
        self.edge_feat_size = edge_feat_size
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.graph_feat_size = graph_feat_size
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.seed = seed
        self.verbose = verbose

        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = None
        self.y_scaler = None
        self.atom_featurizer = None
        self.bond_featurizer = None

    def _init_featurizers(self):
        """Initialize atom and bond featurizers."""
        self.atom_featurizer = AttentiveFPAtomFeaturizer(atom_data_field='h')
        self.bond_featurizer = AttentiveFPBondFeaturizer(bond_data_field='e')

    def _smiles_to_graphs(self, smiles_list: List[str], verbose: bool = True) -> List:
        """Convert SMILES to graphs."""
        if self.atom_featurizer is None:
            self._init_featurizers()

        graphs = []
        iterator = tqdm(smiles_list, desc="Converting SMILES to graphs", disable=not verbose)

        for smi in iterator:
            g = smiles_to_graph(smi, self.atom_featurizer, self.bond_featurizer)
            graphs.append(g)

        return graphs

    def _build_model(self):
        """Build AttentiveFP model."""
        model = AttentiveFPPredictor(
            node_feat_size=self.node_feat_size,
            edge_feat_size=self.edge_feat_size,
            num_layers=self.num_layers,
            num_timesteps=self.num_timesteps,
            graph_feat_size=self.graph_feat_size,
            n_tasks=1,
            dropout=self.dropout
        )
        return model

    def fit(
        self,
        smiles: List[str],
        y: np.ndarray,
        val_smiles: Optional[List[str]] = None,
        val_y: Optional[np.ndarray] = None
    ) -> 'AttentiveFPModel':
        """
        Train the model.

        Args:
            smiles: Training SMILES
            y: Target values
            val_smiles: Validation SMILES (optional, for early stopping)
            val_y: Validation targets

        Returns:
            self
        """
        if not DGL_AVAILABLE:
            raise ImportError("DGL-LifeSci not available")

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Scale targets
        self.y_scaler = StandardScaler()
        y_scaled = self.y_scaler.fit_transform(y.reshape(-1, 1)).flatten()

        # Convert to graphs
        train_graphs = self._smiles_to_graphs(smiles, verbose=self.verbose)

        # Filter valid graphs
        valid_indices = [i for i, g in enumerate(train_graphs) if g is not None]
        train_graphs = [train_graphs[i] for i in valid_indices]
        y_scaled = y_scaled[valid_indices]

        if len(train_graphs) == 0:
            raise ValueError("No valid molecules in training data")

        # Update feature sizes from actual graphs
        sample_graph = train_graphs[0]
        self.node_feat_size = sample_graph.ndata['h'].shape[1]
        self.edge_feat_size = sample_graph.edata['e'].shape[1]

        # Build model
        self.model = self._build_model()
        self.model = self.model.to(self.device)

        # Create dataloaders
        train_dataset = GraphDataset(train_graphs, y_scaled)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_graphs,
            drop_last=True
        )

        # Validation loader if provided
        val_loader = None
        if val_smiles is not None and val_y is not None:
            val_graphs = self._smiles_to_graphs(val_smiles, verbose=False)
            val_y_scaled = self.y_scaler.transform(val_y.reshape(-1, 1)).flatten()

            valid_val_indices = [i for i, g in enumerate(val_graphs) if g is not None]
            val_graphs = [val_graphs[i] for i in valid_val_indices]
            val_y_scaled = val_y_scaled[valid_val_indices]

            val_dataset = GraphDataset(val_graphs, val_y_scaled)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=collate_graphs
            )

        # Optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        epoch_iterator = tqdm(range(self.epochs), desc="Training", disable=not self.verbose)

        for epoch in epoch_iterator:
            # Training
            self.model.train()
            total_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                bg, labels = batch

                if bg is None:
                    continue

                bg = bg.to(self.device)
                labels = labels.to(self.device)

                node_feats = bg.ndata['h']
                edge_feats = bg.edata['e']

                preds = self.model(bg, node_feats, edge_feats)
                loss = nn.functional.mse_loss(preds.squeeze(), labels)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_train_loss = total_loss / max(n_batches, 1)

            # Validation
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                val_batches = 0

                with torch.no_grad():
                    for batch in val_loader:
                        bg, labels = batch
                        if bg is None:
                            continue

                        bg = bg.to(self.device)
                        labels = labels.to(self.device)

                        node_feats = bg.ndata['h']
                        edge_feats = bg.edata['e']

                        preds = self.model(bg, node_feats, edge_feats)
                        loss = nn.functional.mse_loss(preds.squeeze(), labels)

                        val_loss += loss.item()
                        val_batches += 1

                avg_val_loss = val_loss / max(val_batches, 1)
                scheduler.step(avg_val_loss)

                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    if self.verbose:
                        print(f"\nEarly stopping at epoch {epoch + 1}")
                    break

                epoch_iterator.set_postfix(
                    train_loss=f"{avg_train_loss:.4f}",
                    val_loss=f"{avg_val_loss:.4f}"
                )
            else:
                epoch_iterator.set_postfix(loss=f"{avg_train_loss:.4f}")

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self

    def predict(self, smiles: List[str]) -> np.ndarray:
        """
        Make predictions.

        Args:
            smiles: List of SMILES

        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        # Convert to graphs
        graphs = self._smiles_to_graphs(smiles, verbose=self.verbose)

        # Track valid/invalid
        valid_mask = [g is not None for g in graphs]
        valid_graphs = [g for g in graphs if g is not None]

        if len(valid_graphs) == 0:
            return np.zeros(len(smiles))

        # Create dataloader
        test_loader = DataLoader(
            valid_graphs,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_graphs_inference
        )

        # Predict
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for bg in test_loader:
                if bg is None:
                    continue

                bg = bg.to(self.device)
                node_feats = bg.ndata['h']
                edge_feats = bg.edata['e']

                preds = self.model(bg, node_feats, edge_feats)
                all_preds.append(preds.cpu().numpy())

        valid_preds = np.concatenate(all_preds, axis=0).flatten()

        # Inverse scale
        valid_preds = self.y_scaler.inverse_transform(valid_preds.reshape(-1, 1)).flatten()

        # Fill in predictions for all molecules (0 for invalid)
        predictions = np.zeros(len(smiles))
        valid_idx = 0
        for i, is_valid in enumerate(valid_mask):
            if is_valid:
                predictions[i] = valid_preds[valid_idx]
                valid_idx += 1

        return predictions


class AttentiveFPEnsemble:
    """
    Ensemble of AttentiveFP models with K-fold CV.

    Provides:
    - Multi-seed training for robustness
    - K-fold cross-validation with OOF predictions
    - Averaged test predictions
    """

    def __init__(
        self,
        n_seeds: int = 3,
        n_folds: int = 5,
        base_seed: int = 42,
        **model_kwargs
    ):
        self.n_seeds = n_seeds
        self.n_folds = n_folds
        self.base_seed = base_seed
        self.model_kwargs = model_kwargs

        self.models: List[List[AttentiveFPModel]] = []
        self.oof_predictions: Optional[np.ndarray] = None

    def fit(
        self,
        smiles: List[str],
        y: np.ndarray,
        return_oof: bool = True
    ) -> Tuple['AttentiveFPEnsemble', Optional[np.ndarray]]:
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
            print(f"AttentiveFP Seed {seed_idx + 1}/{self.n_seeds} (seed={seed})")
            print(f"{'='*50}")

            seed_models = []

            for fold, (train_idx, val_idx) in enumerate(kf.split(smiles)):
                print(f"\nFold {fold + 1}/{self.n_folds}")

                train_smiles = [smiles[i] for i in train_idx]
                train_y = y[train_idx]
                val_smiles = [smiles[i] for i in val_idx]
                val_y = y[val_idx]

                # Train with validation for early stopping
                model = AttentiveFPModel(seed=seed + fold, **self.model_kwargs)
                model.fit(train_smiles, train_y, val_smiles, val_y)

                # OOF predictions
                val_preds = model.predict(val_smiles)
                oof_predictions[val_idx, seed_idx] = val_preds

                seed_models.append(model)

            self.models.append(seed_models)

        self.oof_predictions = np.mean(oof_predictions, axis=1)

        if return_oof:
            return self, self.oof_predictions
        return self, None

    def predict(self, smiles: List[str]) -> np.ndarray:
        """Make predictions (average of all models)."""
        all_preds = []

        for seed_models in self.models:
            for model in seed_models:
                preds = model.predict(smiles)
                all_preds.append(preds)

        return np.mean(all_preds, axis=0)

    def fit_predict(
        self,
        smiles_train: List[str],
        y_train: np.ndarray,
        smiles_test: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit and return both OOF and test predictions."""
        self.fit(smiles_train, y_train, return_oof=True)
        test_preds = self.predict(smiles_test)
        return self.oof_predictions, test_preds


def check_dependencies():
    """Check if all dependencies are available."""
    status = {
        'torch': TORCH_AVAILABLE,
        'dgl-lifesci': DGL_AVAILABLE,
        'rdkit': RDKIT_AVAILABLE
    }

    missing = [k for k, v in status.items() if not v]

    if missing:
        print(f"Missing dependencies: {missing}")
        print("\nInstall with:")
        print("  pip install torch dgl dgllife rdkit")
        return False

    print("All dependencies available!")
    return True


if __name__ == "__main__":
    if not check_dependencies():
        exit(1)

    print("\nTesting AttentiveFP model...")

    test_smiles = [
        "CCO",
        "CC(=O)O",
        "c1ccccc1",
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    ] * 20

    test_y = np.random.randn(len(test_smiles))

    model = AttentiveFPModel(
        epochs=10,
        batch_size=16,
        num_layers=2,
        graph_feat_size=100,
        verbose=True
    )

    model.fit(test_smiles[:80], test_y[:80])
    preds = model.predict(test_smiles[80:])

    print(f"\nPredictions shape: {preds.shape}")
    print(f"Sample predictions: {preds[:5]}")
    print("\nTest passed!")
