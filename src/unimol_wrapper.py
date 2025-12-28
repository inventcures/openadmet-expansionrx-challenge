"""
Uni-Mol Wrapper for V4 Pipeline.

Uni-Mol is a 3D molecular representation learning framework:
- Pre-trained on 209M 3D conformations
- SE(3)-equivariant transformer architecture
- Outperforms SOTA in 14/15 molecular property tasks
- Uni-Mol2 (Nov 2024): 1.1B parameters, 800M conformations

Reference: https://github.com/deepmodeling/Uni-Mol
Documentation: https://unimol.readthedocs.io/
"""

import os
import warnings
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import tempfile
import shutil

# Try to import unimol_tools
try:
    from unimol_tools import MolTrain, MolPredict
    UNIMOL_AVAILABLE = True
except ImportError:
    UNIMOL_AVAILABLE = False
    print("Warning: unimol-tools not available. Install with: pip install unimol-tools")

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


def generate_3d_conformer(smiles: str, n_conformers: int = 1) -> Optional[Dict]:
    """
    Generate 3D conformer for a molecule.

    Args:
        smiles: SMILES string
        n_conformers: Number of conformers to generate

    Returns:
        Dictionary with atoms and coordinates, or None if failed
    """
    if not RDKIT_AVAILABLE:
        return None

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        mol = Chem.AddHs(mol)

        # Generate conformer
        result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        if result == -1:
            # Try with random coordinates
            AllChem.EmbedMolecule(mol, useRandomCoords=True)

        # Optimize
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            pass

        # Extract atoms and coordinates
        conf = mol.GetConformer()
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        coords = []
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            coords.append([pos.x, pos.y, pos.z])

        return {
            'atoms': atoms,
            'coordinates': [np.array(coords)]
        }

    except Exception:
        return None


class UniMolModel:
    """
    Uni-Mol model wrapper for molecular property prediction.

    Uses the unimol-tools package for easy fine-tuning of the
    pre-trained Uni-Mol model on custom datasets.
    """

    def __init__(
        self,
        task: str = 'regression',
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        metrics: str = 'mse',
        early_stopping: int = 10,
        use_3d: bool = True,
        model_version: str = 'v1',  # 'v1' or 'v2'
        save_dir: Optional[str] = None,
        seed: int = 42,
        verbose: bool = True
    ):
        self.task = task
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.metrics = metrics
        self.early_stopping = early_stopping
        self.use_3d = use_3d
        self.model_version = model_version
        self.seed = seed
        self.verbose = verbose

        # Create temp directory if no save_dir specified
        if save_dir is None:
            self.save_dir = tempfile.mkdtemp(prefix='unimol_')
            self._temp_dir = True
        else:
            self.save_dir = save_dir
            self._temp_dir = False
            os.makedirs(save_dir, exist_ok=True)

        self.trainer = None
        self.predictor = None

    def _prepare_data_dict(
        self,
        smiles: List[str],
        y: Optional[np.ndarray] = None
    ) -> Dict:
        """Prepare data dictionary for unimol-tools."""

        data = {'smiles': smiles}

        if y is not None:
            data['target'] = y.tolist()

        return data

    def _prepare_data_csv(
        self,
        smiles: List[str],
        y: Optional[np.ndarray] = None,
        filename: str = 'data.csv'
    ) -> str:
        """Prepare CSV file for unimol-tools."""

        filepath = os.path.join(self.save_dir, filename)

        df = pd.DataFrame({'SMILES': smiles})
        if y is not None:
            df['target'] = y

        df.to_csv(filepath, index=False)

        return filepath

    def fit(self, smiles: List[str], y: np.ndarray) -> 'UniMolModel':
        """
        Train the model.

        Args:
            smiles: List of SMILES strings
            y: Target values

        Returns:
            self
        """
        if not UNIMOL_AVAILABLE:
            raise ImportError("unimol-tools not available")

        # Prepare training data
        train_csv = self._prepare_data_csv(smiles, y, 'train.csv')

        # Initialize trainer
        self.trainer = MolTrain(
            task=self.task,
            data_type='molecule',
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            metrics=self.metrics,
            early_stopping=self.early_stopping,
            save_path=self.save_dir,
            seed=self.seed,
            remove_hs=True
        )

        # Fit
        self.trainer.fit(data=train_csv)

        # Initialize predictor with trained model
        self.predictor = MolPredict(load_model=self.save_dir)

        return self

    def predict(self, smiles: List[str]) -> np.ndarray:
        """
        Make predictions.

        Args:
            smiles: List of SMILES strings

        Returns:
            Array of predictions
        """
        if self.predictor is None:
            raise ValueError("Model not trained. Call fit() first.")

        # Prepare test data
        test_csv = self._prepare_data_csv(smiles, filename='test.csv')

        # Predict
        predictions = self.predictor.predict(data=test_csv)

        return np.array(predictions).flatten()

    def __del__(self):
        """Clean up temp directory."""
        if hasattr(self, '_temp_dir') and self._temp_dir:
            try:
                shutil.rmtree(self.save_dir)
            except Exception:
                pass


class UniMolEnsemble:
    """
    Ensemble of Uni-Mol models with K-fold CV.

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
        save_dir: Optional[str] = None,
        **model_kwargs
    ):
        self.n_seeds = n_seeds
        self.n_folds = n_folds
        self.base_seed = base_seed
        self.model_kwargs = model_kwargs

        if save_dir is None:
            self.save_dir = tempfile.mkdtemp(prefix='unimol_ensemble_')
            self._temp_dir = True
        else:
            self.save_dir = save_dir
            self._temp_dir = False
            os.makedirs(save_dir, exist_ok=True)

        self.models: List[List[UniMolModel]] = []
        self.oof_predictions: Optional[np.ndarray] = None

    def fit(
        self,
        smiles: List[str],
        y: np.ndarray,
        return_oof: bool = True
    ) -> Tuple['UniMolEnsemble', Optional[np.ndarray]]:
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
            print(f"Uni-Mol Seed {seed_idx + 1}/{self.n_seeds} (seed={seed})")
            print(f"{'='*50}")

            seed_models = []

            for fold, (train_idx, val_idx) in enumerate(kf.split(smiles)):
                print(f"\nFold {fold + 1}/{self.n_folds}")

                train_smiles = [smiles[i] for i in train_idx]
                train_y = y[train_idx]
                val_smiles = [smiles[i] for i in val_idx]

                # Create model save directory
                model_dir = os.path.join(
                    self.save_dir,
                    f'seed_{seed}_fold_{fold}'
                )

                # Train model
                model = UniMolModel(
                    seed=seed + fold,
                    save_dir=model_dir,
                    **self.model_kwargs
                )
                model.fit(train_smiles, train_y)

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

    def __del__(self):
        """Clean up temp directory."""
        if hasattr(self, '_temp_dir') and self._temp_dir:
            try:
                shutil.rmtree(self.save_dir)
            except Exception:
                pass


class UniMolFallback:
    """
    Fallback implementation when unimol-tools is not available.

    Uses RDKit 3D conformers + simple descriptor-based model as a
    stand-in for Uni-Mol's 3D representation learning.
    """

    def __init__(
        self,
        epochs: int = 100,
        batch_size: int = 32,
        seed: int = 42,
        verbose: bool = True
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.verbose = verbose

        self.model = None

    def _compute_3d_descriptors(self, smiles: str) -> np.ndarray:
        """Compute 3D descriptors from conformer."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.zeros(10)

            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
            AllChem.MMFFOptimizeMolecule(mol)

            # Compute 3D descriptors
            from rdkit.Chem import Descriptors3D

            descriptors = [
                Descriptors3D.Asphericity(mol),
                Descriptors3D.Eccentricity(mol),
                Descriptors3D.InertialShapeFactor(mol),
                Descriptors3D.NPR1(mol),
                Descriptors3D.NPR2(mol),
                Descriptors3D.PMI1(mol),
                Descriptors3D.PMI2(mol),
                Descriptors3D.PMI3(mol),
                Descriptors3D.RadiusOfGyration(mol),
                Descriptors3D.SpherocityIndex(mol),
            ]

            return np.array(descriptors)

        except Exception:
            return np.zeros(10)

    def fit(self, smiles: List[str], y: np.ndarray) -> 'UniMolFallback':
        """Train fallback model."""
        from sklearn.ensemble import GradientBoostingRegressor
        from tqdm import tqdm

        # Compute 3D descriptors
        X = []
        iterator = tqdm(smiles, desc="Computing 3D descriptors", disable=not self.verbose)

        for smi in iterator:
            X.append(self._compute_3d_descriptors(smi))

        X = np.array(X)

        # Handle NaN
        X = np.nan_to_num(X)

        # Train model
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=self.seed
        )
        self.model.fit(X, y)

        return self

    def predict(self, smiles: List[str]) -> np.ndarray:
        """Make predictions."""
        from tqdm import tqdm

        X = []
        iterator = tqdm(smiles, desc="Computing 3D descriptors", disable=not self.verbose)

        for smi in iterator:
            X.append(self._compute_3d_descriptors(smi))

        X = np.array(X)
        X = np.nan_to_num(X)

        return self.model.predict(X)


def get_unimol_model(**kwargs):
    """
    Factory function to get Uni-Mol model.

    Returns UniMolModel if unimol-tools is available, otherwise UniMolFallback.
    """
    if UNIMOL_AVAILABLE:
        return UniMolModel(**kwargs)
    else:
        print("Warning: Using fallback 3D descriptor model instead of Uni-Mol")
        return UniMolFallback(**kwargs)


def check_dependencies():
    """Check if all dependencies are available."""
    status = {
        'unimol-tools': UNIMOL_AVAILABLE,
        'rdkit': RDKIT_AVAILABLE
    }

    print("Dependency status:")
    for name, available in status.items():
        status_str = "✓" if available else "✗"
        print(f"  {status_str} {name}")

    if not UNIMOL_AVAILABLE:
        print("\nInstall unimol-tools with:")
        print("  pip install unimol-tools")
        print("\nNote: Fallback 3D descriptor model will be used if unavailable.")

    return UNIMOL_AVAILABLE


if __name__ == "__main__":
    check_dependencies()

    print("\nTesting Uni-Mol wrapper...")

    test_smiles = [
        "CCO",
        "CC(=O)O",
        "c1ccccc1",
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    ] * 20

    test_y = np.random.randn(len(test_smiles))

    # Test 3D conformer generation
    print("\nTesting 3D conformer generation:")
    for smi in test_smiles[:3]:
        result = generate_3d_conformer(smi)
        if result:
            print(f"  {smi}: {len(result['atoms'])} atoms")
        else:
            print(f"  {smi}: Failed")

    # Test model
    model = get_unimol_model(epochs=5, verbose=True)
    model.fit(test_smiles[:80], test_y[:80])
    preds = model.predict(test_smiles[80:])

    print(f"\nPredictions shape: {preds.shape}")
    print(f"Sample predictions: {preds[:5]}")
    print("\nTest passed!")
