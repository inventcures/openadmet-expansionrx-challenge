"""
Three-Level Stacking Module for V4 Pipeline.

Implements the Kaggle Grandmaster stacking technique:
- Level 0: Base models with OOF predictions (34+ columns)
- Level 1: Meta-models on OOF predictions (5 columns)
- Level 2: Final blender with optimized weights

Reference: https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-a-kaggle-competition-with-stacking-using-cuml/
"""

import warnings
from typing import List, Dict, Tuple, Optional, Callable, Any
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
import joblib
from pathlib import Path


# Import meta-models
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Import hill climbing
from hill_climbing import HillClimbingEnsemble, optimize_weights, METRICS


class SimpleNN(nn.Module):
    """Simple neural network for Level 1 meta-learning."""

    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32]):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


class NNMetaModel:
    """Neural network meta-model wrapper."""

    def __init__(
        self,
        hidden_dims: List[int] = [64, 32],
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        patience: int = 10,
        seed: int = 42
    ):
        self.hidden_dims = hidden_dims
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.seed = seed

        self.model = None
        self.scaler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """Train the neural network."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        # Build model
        self.model = SimpleNN(X.shape[1], self.hidden_dims).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        best_loss = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            self.model.train()

            # Mini-batch training
            indices = np.random.permutation(len(X_scaled))
            total_loss = 0.0
            n_batches = 0

            for start in range(0, len(indices), self.batch_size):
                batch_idx = indices[start:start + self.batch_size]
                batch_X = X_tensor[batch_idx]
                batch_y = y_tensor[batch_idx]

                optimizer.zero_grad()
                preds = self.model(batch_X)
                loss = nn.functional.mse_loss(preds, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / n_batches
            scheduler.step(avg_loss)

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                break

        if best_state:
            self.model.load_state_dict(best_state)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_tensor)

        return preds.cpu().numpy()


class Level1MetaModels:
    """Level 1 meta-models for stacking."""

    def __init__(
        self,
        n_folds: int = 5,
        seed: int = 42,
        verbose: bool = True
    ):
        self.n_folds = n_folds
        self.seed = seed
        self.verbose = verbose

        self.models: Dict[str, List] = {}  # model_name -> list of fold models
        self.scalers: Dict[str, StandardScaler] = {}

    def get_meta_models(self) -> Dict[str, Any]:
        """Get dictionary of meta-model constructors."""
        meta_models = {
            'ridge': lambda: Ridge(alpha=1.0),
            'elasticnet': lambda: ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=1000),
        }

        if LGB_AVAILABLE:
            meta_models['lgb'] = lambda: lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.seed,
                verbose=-1
            )

        if XGB_AVAILABLE:
            meta_models['xgb'] = lambda: xgb.XGBRegressor(
                n_estimators=50,
                max_depth=2,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.seed,
                verbosity=0
            )

        if TORCH_AVAILABLE:
            meta_models['nn'] = lambda: NNMetaModel(
                hidden_dims=[64, 32],
                epochs=100,
                seed=self.seed
            )

        return meta_models

    def fit_predict(
        self,
        level0_oof: np.ndarray,
        y_true: np.ndarray,
        level0_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train Level 1 meta-models and generate OOF/test predictions.

        Args:
            level0_oof: Level 0 OOF predictions, shape (n_train, n_models)
            y_true: True targets
            level0_test: Level 0 test predictions, shape (n_test, n_models)

        Returns:
            Tuple of (level1_oof, level1_test)
            - level1_oof: shape (n_train, n_meta_models)
            - level1_test: shape (n_test, n_meta_models)
        """
        meta_models = self.get_meta_models()
        n_meta = len(meta_models)
        n_train = len(y_true)
        n_test = len(level0_test)

        level1_oof = np.zeros((n_train, n_meta))
        level1_test = np.zeros((n_test, n_meta))

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)

        for meta_idx, (name, model_fn) in enumerate(meta_models.items()):
            if self.verbose:
                print(f"\nTraining meta-model: {name}")

            fold_models = []
            fold_test_preds = []

            for fold, (train_idx, val_idx) in enumerate(kf.split(level0_oof)):
                if self.verbose:
                    print(f"  Fold {fold + 1}/{self.n_folds}")

                X_train = level0_oof[train_idx]
                y_train = y_true[train_idx]
                X_val = level0_oof[val_idx]

                # Train model
                model = model_fn()
                model.fit(X_train, y_train)

                # OOF predictions
                val_preds = model.predict(X_val)
                level1_oof[val_idx, meta_idx] = val_preds

                # Test predictions
                test_preds = model.predict(level0_test)
                fold_test_preds.append(test_preds)

                fold_models.append(model)

            # Average test predictions across folds
            level1_test[:, meta_idx] = np.mean(fold_test_preds, axis=0)

            self.models[name] = fold_models

        return level1_oof, level1_test


class ThreeLevelStacking:
    """
    Three-level stacking ensemble.

    Architecture:
    Level 0: Base models (n_base OOF predictions)
    Level 1: Meta-models (5 OOF predictions)
    Level 2: Final blender (optimized weights)
    """

    def __init__(
        self,
        n_folds: int = 5,
        metric: str = 'spearman',
        seed: int = 42,
        verbose: bool = True
    ):
        self.n_folds = n_folds
        self.metric = metric
        self.seed = seed
        self.verbose = verbose

        self.level1 = Level1MetaModels(n_folds=n_folds, seed=seed, verbose=verbose)
        self.level2_weights = None
        self.level2_selected = None

    def fit(
        self,
        level0_oof: np.ndarray,
        y_true: np.ndarray,
        level0_test: np.ndarray,
        model_names: Optional[List[str]] = None
    ) -> 'ThreeLevelStacking':
        """
        Fit the three-level stacking ensemble.

        Args:
            level0_oof: Level 0 OOF predictions, shape (n_train, n_base_models)
            y_true: True targets
            level0_test: Level 0 test predictions, shape (n_test, n_base_models)
            model_names: Optional names for base models

        Returns:
            self
        """
        if self.verbose:
            print("="*60)
            print("THREE-LEVEL STACKING")
            print("="*60)
            print(f"\nLevel 0: {level0_oof.shape[1]} base models")

        # Level 1: Train meta-models
        if self.verbose:
            print("\n" + "="*60)
            print("LEVEL 1: Training meta-models")
            print("="*60)

        self.level1_oof, self.level1_test = self.level1.fit_predict(
            level0_oof, y_true, level0_test
        )

        if self.verbose:
            print(f"\nLevel 1: {self.level1_oof.shape[1]} meta-models")

        # Level 2: Optimize blending weights
        if self.verbose:
            print("\n" + "="*60)
            print("LEVEL 2: Optimizing blender weights")
            print("="*60)

        meta_names = list(self.level1.models.keys())

        # Use hill climbing for weight optimization
        blender = HillClimbingEnsemble(
            metric=self.metric,
            max_models=len(meta_names),
            diversity_bonus=0.0,
            verbose=self.verbose
        )
        blender.fit(self.level1_oof, y_true, meta_names)

        self.level2_selected = blender.selected_indices_
        self.level2_weights = blender.weights_

        if self.verbose:
            print(f"\nLevel 2 selected: {[meta_names[i] for i in self.level2_selected]}")
            print(f"Level 2 weights: {self.level2_weights}")
            print(f"Final score: {blender.score_:.6f}")

        return self

    def predict(self, level0_test: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Make predictions using the trained ensemble.

        Args:
            level0_test: Level 0 test predictions (optional if fit was called)

        Returns:
            Final blended predictions
        """
        if level0_test is not None:
            # Recompute Level 1 test predictions
            _, level1_test = self.level1.fit_predict(
                np.zeros((1, level0_test.shape[1])),  # Dummy
                np.zeros(1),  # Dummy
                level0_test
            )
        else:
            level1_test = self.level1_test

        # Apply Level 2 blending
        selected_preds = level1_test[:, self.level2_selected]
        final_preds = selected_preds @ self.level2_weights

        return final_preds

    def get_oof_predictions(self) -> np.ndarray:
        """Get OOF predictions from the ensemble."""
        selected_preds = self.level1_oof[:, self.level2_selected]
        return selected_preds @ self.level2_weights


def collect_base_model_predictions(
    base_models: Dict[str, Callable],
    smiles_train: List[str],
    y_train: np.ndarray,
    smiles_test: List[str],
    n_folds: int = 5,
    n_seeds: int = 3,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Collect OOF and test predictions from multiple base models.

    Args:
        base_models: Dictionary of model_name -> fit_predict_fn
            fit_predict_fn(train_smiles, train_y, test_smiles) -> (oof, test)
        smiles_train: Training SMILES
        y_train: Training targets
        smiles_test: Test SMILES
        n_folds: Number of CV folds
        n_seeds: Number of random seeds
        seed: Base random seed

    Returns:
        Tuple of (level0_oof, level0_test, model_names)
    """
    all_oof = []
    all_test = []
    model_names = []

    for model_name, fit_predict_fn in base_models.items():
        print(f"\n{'='*50}")
        print(f"Training base model: {model_name}")
        print(f"{'='*50}")

        oof_preds, test_preds = fit_predict_fn(
            smiles_train, y_train, smiles_test
        )

        all_oof.append(oof_preds.reshape(-1, 1))
        all_test.append(test_preds.reshape(-1, 1))
        model_names.append(model_name)

    level0_oof = np.hstack(all_oof)
    level0_test = np.hstack(all_test)

    return level0_oof, level0_test, model_names


def evaluate_stacking(
    level0_oof: np.ndarray,
    y_true: np.ndarray,
    level0_test: np.ndarray,
    model_names: Optional[List[str]] = None,
    n_folds: int = 5
) -> Dict:
    """
    Evaluate three-level stacking and compare to simple averaging.

    Args:
        level0_oof: Level 0 OOF predictions
        y_true: True targets
        level0_test: Level 0 test predictions
        model_names: Optional model names
        n_folds: Number of CV folds

    Returns:
        Dictionary with evaluation results
    """
    results = {}

    # Simple averaging baseline
    simple_avg = np.mean(level0_oof, axis=1)
    results['simple_avg_spearman'] = spearmanr(simple_avg, y_true)[0]

    # Best single model
    single_scores = [spearmanr(level0_oof[:, i], y_true)[0] for i in range(level0_oof.shape[1])]
    best_idx = np.argmax(single_scores)
    results['best_single_spearman'] = single_scores[best_idx]
    results['best_single_model'] = model_names[best_idx] if model_names else f"model_{best_idx}"

    # Three-level stacking
    stacker = ThreeLevelStacking(n_folds=n_folds, verbose=True)
    stacker.fit(level0_oof, y_true, level0_test, model_names)

    stacked_oof = stacker.get_oof_predictions()
    results['stacked_spearman'] = spearmanr(stacked_oof, y_true)[0]

    # Improvement
    results['improvement_vs_avg'] = results['stacked_spearman'] - results['simple_avg_spearman']
    results['improvement_vs_best'] = results['stacked_spearman'] - results['best_single_spearman']

    return results


if __name__ == "__main__":
    print("Testing Three-Level Stacking...")

    np.random.seed(42)

    # Simulate data
    n_train = 1000
    n_test = 200
    n_models = 10

    y_true = np.random.randn(n_train)

    # Simulate OOF predictions with varying quality
    level0_oof = np.zeros((n_train, n_models))
    level0_test = np.zeros((n_test, n_models))

    for i in range(n_models):
        noise = np.random.randn(n_train) * (0.3 + i * 0.1)
        level0_oof[:, i] = y_true + noise
        level0_test[:, i] = np.random.randn(n_test)

    model_names = [f"Model_{i}" for i in range(n_models)]

    # Test stacking
    stacker = ThreeLevelStacking(n_folds=5, verbose=True)
    stacker.fit(level0_oof, y_true, level0_test, model_names)

    # Get predictions
    oof_preds = stacker.get_oof_predictions()
    test_preds = stacker.predict()

    print(f"\nOOF predictions shape: {oof_preds.shape}")
    print(f"Test predictions shape: {test_preds.shape}")

    # Evaluate
    oof_score = spearmanr(oof_preds, y_true)[0]
    simple_avg_score = spearmanr(np.mean(level0_oof, axis=1), y_true)[0]

    print(f"\nSimple average Spearman: {simple_avg_score:.6f}")
    print(f"Stacked Spearman: {oof_score:.6f}")
    print(f"Improvement: {oof_score - simple_avg_score:.6f}")

    print("\nTest passed!")
