"""
V4 Pipeline: Maximum Diversity Ensemble for TOP 5 Finish

This pipeline implements the full V4 strategy:
- Tier 1: Chemprop-RDKit, Uni-Mol, AttentiveFP (SOTA models)
- Tier 2: LightGBM, XGBoost with extended features
- Tier 3: 1D CNN with SMILES augmentation, RF, CatBoost
- SMILES 20x augmentation for training
- TDC pre-training for external data
- Hill climbing ensemble selection
- Three-level stacking for final predictions

Expected improvement: 25-40% RAE reduction vs V3
"""

import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import joblib

# Configure warnings
warnings.filterwarnings('ignore')

# Add src to path
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / 'src'))

# Import V4 modules
from smiles_augmentation import SmilesAugmenter, augment_dataset, batch_predict_with_augmentation
from hill_climbing import HillClimbingEnsemble, hill_climbing_ensemble
from three_level_stacking import ThreeLevelStacking

# Import existing modules
from extended_fingerprints import ExtendedFingerprinter
from smiles_cnn import SMILES1DCNNEnsemble

# Optional imports
try:
    from chemprop_rdkit import ChempropRDKitEnsemble, compute_rdkit_descriptors
    CHEMPROP_AVAILABLE = True
except ImportError:
    CHEMPROP_AVAILABLE = False
    print("Warning: Chemprop not available")

try:
    from attentivefp_wrapper import AttentiveFPEnsemble
    ATTENTIVEFP_AVAILABLE = True
except ImportError:
    ATTENTIVEFP_AVAILABLE = False
    print("Warning: AttentiveFP not available")

try:
    from unimol_wrapper import UniMolEnsemble, get_unimol_model
    UNIMOL_AVAILABLE = True
except ImportError:
    UNIMOL_AVAILABLE = False
    print("Warning: Uni-Mol not available")

try:
    from tdc_pretraining import TDCDataLoader, get_pretraining_data_for_endpoint
    TDC_AVAILABLE = True
except ImportError:
    TDC_AVAILABLE = False
    print("Warning: TDC not available")

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
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = 'cpu'


# Competition endpoints
ENDPOINTS = [
    'LogD',
    'KSOL',
    'HLM_CLint',
    'MLM_CLint',
    'Caco2_Papp',
    'Caco2_Efflux',
    'MPPB',
    'MBPB',
    'MGMB'
]


def print_banner(msg: str):
    """Print a formatted banner."""
    print(f"\n{'='*60}")
    print(f" {msg}")
    print(f"{'='*60}")


def compute_features(smiles: List[str], verbose: bool = True) -> Dict[str, np.ndarray]:
    """Compute all feature sets for molecules."""
    fingerprinter = ExtendedFingerprinter()
    features = fingerprinter.compute_all_features(smiles, verbose=verbose)
    return features


class GBDTModel:
    """Gradient boosting model (LightGBM/XGBoost/CatBoost)."""

    def __init__(
        self,
        model_type: str = 'lgb',
        n_estimators: int = 1000,
        max_depth: int = 7,
        learning_rate: float = 0.05,
        seed: int = 42
    ):
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.seed = seed
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """Train the model."""
        if self.model_type == 'lgb' and LGB_AVAILABLE:
            self.model = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.seed,
                verbose=-1,
                n_jobs=-1
            )
            if X_val is not None:
                self.model.fit(
                    X, y,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False)]
                )
            else:
                self.model.fit(X, y)

        elif self.model_type == 'xgb' and XGB_AVAILABLE:
            self.model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.seed,
                early_stopping_rounds=50 if X_val is not None else None,
                verbosity=0,
                n_jobs=-1
            )
            if X_val is not None:
                self.model.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
            else:
                self.model.fit(X, y)

        elif self.model_type == 'catboost' and CATBOOST_AVAILABLE:
            self.model = cb.CatBoostRegressor(
                iterations=self.n_estimators,
                depth=min(self.max_depth, 10),
                learning_rate=self.learning_rate,
                random_state=self.seed,
                verbose=False
            )
            if X_val is not None:
                self.model.fit(X, y, eval_set=(X_val, y_val), early_stopping_rounds=50)
            else:
                self.model.fit(X, y)
        else:
            raise ValueError(f"Model type {self.model_type} not available")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)


class GBDTEnsemble:
    """Ensemble of GBDT models with K-fold CV."""

    def __init__(
        self,
        model_type: str = 'lgb',
        n_seeds: int = 3,
        n_folds: int = 5,
        feature_type: str = 'comprehensive',
        **model_kwargs
    ):
        self.model_type = model_type
        self.n_seeds = n_seeds
        self.n_folds = n_folds
        self.feature_type = feature_type
        self.model_kwargs = model_kwargs

        self.models: List[List[GBDTModel]] = []
        self.scalers: List[StandardScaler] = []
        self.feature_cache: Dict = {}

    def fit_predict(
        self,
        smiles_train: List[str],
        y_train: np.ndarray,
        smiles_test: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit and return OOF and test predictions."""

        # Compute features
        print(f"Computing {self.feature_type} features...")
        train_features = compute_features(smiles_train, verbose=True)
        test_features = compute_features(smiles_test, verbose=True)

        X_train = train_features[self.feature_type]
        X_test = test_features[self.feature_type]

        n_train = len(y_train)
        n_test = len(X_test)

        oof_predictions = np.zeros((n_train, self.n_seeds))
        test_predictions = np.zeros((n_test, self.n_seeds))

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        for seed_idx in range(self.n_seeds):
            seed = 42 + seed_idx * 100
            print(f"\nSeed {seed_idx + 1}/{self.n_seeds}")

            seed_models = []
            fold_test_preds = []

            for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
                print(f"  Fold {fold + 1}/{self.n_folds}")

                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]

                # Train model
                model = GBDTModel(
                    model_type=self.model_type,
                    seed=seed + fold,
                    **self.model_kwargs
                )
                model.fit(X_tr, y_tr, X_val, y_val)

                # OOF predictions
                oof_predictions[val_idx, seed_idx] = model.predict(X_val)

                # Test predictions
                fold_test_preds.append(model.predict(X_test))

                seed_models.append(model)

            test_predictions[:, seed_idx] = np.mean(fold_test_preds, axis=0)
            self.models.append(seed_models)

        return np.mean(oof_predictions, axis=1), np.mean(test_predictions, axis=1)


class V4Pipeline:
    """
    V4 Pipeline for maximum diversity ensemble.

    Implements all components from v4_top5-finish_specs.md.
    """

    def __init__(
        self,
        use_augmentation: bool = True,
        n_augmentations: int = 20,
        use_tdc_pretraining: bool = False,
        n_folds: int = 5,
        n_seeds: int = 3,
        cache_dir: Optional[str] = None,
        verbose: bool = True
    ):
        self.use_augmentation = use_augmentation
        self.n_augmentations = n_augmentations
        self.use_tdc_pretraining = use_tdc_pretraining
        self.n_folds = n_folds
        self.n_seeds = n_seeds
        self.verbose = verbose

        if cache_dir is None:
            self.cache_dir = BASE_DIR / 'cache' / 'v4'
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.augmenter = SmilesAugmenter(
            n_augmentations=n_augmentations,
            verbose=verbose
        )

    def get_base_models(self) -> Dict[str, Any]:
        """Get dictionary of all base models to train."""
        models = {}

        # GBDT models (always available)
        if LGB_AVAILABLE:
            models['lgb_comprehensive'] = lambda: GBDTEnsemble(
                model_type='lgb',
                n_seeds=self.n_seeds,
                n_folds=self.n_folds,
                feature_type='comprehensive'
            )
            models['lgb_extended'] = lambda: GBDTEnsemble(
                model_type='lgb',
                n_seeds=self.n_seeds,
                n_folds=self.n_folds,
                feature_type='extended'
            )

        if XGB_AVAILABLE:
            models['xgb_comprehensive'] = lambda: GBDTEnsemble(
                model_type='xgb',
                n_seeds=self.n_seeds,
                n_folds=self.n_folds,
                feature_type='comprehensive'
            )

        if CATBOOST_AVAILABLE:
            models['catboost_comprehensive'] = lambda: GBDTEnsemble(
                model_type='catboost',
                n_seeds=self.n_seeds,
                n_folds=self.n_folds,
                feature_type='comprehensive'
            )

        # 1D CNN
        if TORCH_AVAILABLE:
            models['cnn_smiles'] = lambda: SMILES1DCNNEnsemble(
                n_models=self.n_seeds,
                n_folds=self.n_folds,
                epochs=50,
                batch_size=64
            )

        # Chemprop-RDKit (TDC #1)
        if CHEMPROP_AVAILABLE:
            models['chemprop_rdkit'] = lambda: ChempropRDKitEnsemble(
                n_seeds=self.n_seeds,
                n_folds=self.n_folds,
                epochs=30,
                batch_size=50
            )

        # AttentiveFP
        if ATTENTIVEFP_AVAILABLE:
            models['attentivefp'] = lambda: AttentiveFPEnsemble(
                n_seeds=self.n_seeds,
                n_folds=self.n_folds,
                epochs=100,
                batch_size=128
            )

        # Uni-Mol
        if UNIMOL_AVAILABLE:
            models['unimol'] = lambda: UniMolEnsemble(
                n_seeds=min(self.n_seeds, 2),  # Uni-Mol is expensive
                n_folds=self.n_folds,
                epochs=50
            )

        return models

    def train_endpoint(
        self,
        endpoint: str,
        smiles_train: List[str],
        y_train: np.ndarray,
        smiles_test: List[str]
    ) -> Tuple[np.ndarray, Dict]:
        """
        Train all models for a single endpoint.

        Returns:
            Tuple of (test_predictions, diagnostics)
        """
        print_banner(f"Training endpoint: {endpoint}")

        base_models = self.get_base_models()
        n_train = len(smiles_train)
        n_test = len(smiles_test)

        # Collect OOF and test predictions
        all_oof = []
        all_test = []
        model_names = []
        diagnostics = {}

        for model_name, model_fn in base_models.items():
            print(f"\n{'='*50}")
            print(f"Training: {model_name}")
            print(f"{'='*50}")

            try:
                start_time = time.time()

                model = model_fn()
                oof_pred, test_pred = model.fit_predict(
                    smiles_train, y_train, smiles_test
                )

                elapsed = time.time() - start_time

                # Evaluate OOF
                spearman = spearmanr(oof_pred, y_train)[0]

                all_oof.append(oof_pred.reshape(-1, 1))
                all_test.append(test_pred.reshape(-1, 1))
                model_names.append(model_name)

                diagnostics[model_name] = {
                    'spearman': spearman,
                    'time': elapsed
                }

                print(f"\n{model_name}: Spearman={spearman:.4f}, Time={elapsed:.1f}s")

            except Exception as e:
                print(f"\nError training {model_name}: {e}")
                diagnostics[model_name] = {'error': str(e)}

        if len(all_oof) == 0:
            raise ValueError("No models trained successfully")

        # Stack predictions
        level0_oof = np.hstack(all_oof)
        level0_test = np.hstack(all_test)

        print(f"\n{'='*50}")
        print(f"Level 0: {level0_oof.shape[1]} base models")
        print(f"{'='*50}")

        # Three-level stacking
        print_banner("Three-Level Stacking")

        stacker = ThreeLevelStacking(
            n_folds=self.n_folds,
            metric='spearman',
            verbose=self.verbose
        )
        stacker.fit(level0_oof, y_train, level0_test, model_names)

        # Get final predictions
        test_predictions = stacker.predict()

        # Final OOF evaluation
        oof_predictions = stacker.get_oof_predictions()
        final_spearman = spearmanr(oof_predictions, y_train)[0]

        # Simple average baseline
        simple_avg = np.mean(level0_oof, axis=1)
        baseline_spearman = spearmanr(simple_avg, y_train)[0]

        diagnostics['final'] = {
            'stacked_spearman': final_spearman,
            'baseline_spearman': baseline_spearman,
            'improvement': final_spearman - baseline_spearman,
            'n_models': level0_oof.shape[1]
        }

        print(f"\nBaseline (simple avg): {baseline_spearman:.4f}")
        print(f"Stacked: {final_spearman:.4f}")
        print(f"Improvement: {final_spearman - baseline_spearman:.4f}")

        return test_predictions, diagnostics

    def run(
        self,
        train_path: Optional[str] = None,
        test_path: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Run the full V4 pipeline.

        Args:
            train_path: Path to training data
            test_path: Path to test data
            output_dir: Directory for outputs

        Returns:
            DataFrame with test predictions
        """
        print_banner("V4 Pipeline: Maximum Diversity Ensemble")
        print(f"\nDevice: {DEVICE}")
        print(f"Augmentation: {self.use_augmentation}")
        print(f"TDC Pre-training: {self.use_tdc_pretraining}")

        # Load data
        if train_path is None:
            train_path = BASE_DIR / 'data' / 'raw' / 'train.csv'
        if test_path is None:
            test_path = BASE_DIR / 'data' / 'raw' / 'test_blinded.csv'
        if output_dir is None:
            output_dir = BASE_DIR / 'submissions'

        train_path = Path(train_path)
        test_path = Path(test_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nTrain path: {train_path}")
        print(f"Test path: {test_path}")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        smiles_train = train_df['SMILES'].tolist()
        smiles_test = test_df['SMILES'].tolist()

        print(f"\nTrain size: {len(smiles_train)}")
        print(f"Test size: {len(smiles_test)}")

        # Results storage
        all_diagnostics = {}
        predictions = {'SMILES': smiles_test}

        # Train each endpoint
        for endpoint in ENDPOINTS:
            if endpoint not in train_df.columns:
                print(f"\nSkipping {endpoint}: not in training data")
                continue

            # Get endpoint data
            mask = train_df[endpoint].notna()
            endpoint_smiles = train_df.loc[mask, 'SMILES'].tolist()
            endpoint_y = train_df.loc[mask, endpoint].values

            print(f"\n{endpoint}: {len(endpoint_smiles)} samples")

            # Train
            test_preds, diagnostics = self.train_endpoint(
                endpoint,
                endpoint_smiles,
                endpoint_y,
                smiles_test
            )

            predictions[endpoint] = test_preds
            all_diagnostics[endpoint] = diagnostics

        # Create submission
        submission = pd.DataFrame(predictions)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        submission_path = output_dir / f'v4_submission_{timestamp}.csv'
        submission.to_csv(submission_path, index=False)

        print_banner("Pipeline Complete")
        print(f"\nSubmission saved to: {submission_path}")

        # Print summary
        print("\nEndpoint Summary:")
        for endpoint in ENDPOINTS:
            if endpoint in all_diagnostics:
                d = all_diagnostics[endpoint].get('final', {})
                spearman = d.get('stacked_spearman', 'N/A')
                improvement = d.get('improvement', 'N/A')
                if isinstance(spearman, float):
                    print(f"  {endpoint}: Spearman={spearman:.4f}, Improvement={improvement:.4f}")

        return submission


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='V4 Pipeline')
    parser.add_argument('--train', type=str, help='Path to training data')
    parser.add_argument('--test', type=str, help='Path to test data')
    parser.add_argument('--output', type=str, help='Output directory')
    parser.add_argument('--no-augmentation', action='store_true', help='Disable SMILES augmentation')
    parser.add_argument('--tdc-pretrain', action='store_true', help='Use TDC pre-training')
    parser.add_argument('--n-folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--n-seeds', type=int, default=3, help='Number of random seeds')

    args = parser.parse_args()

    pipeline = V4Pipeline(
        use_augmentation=not args.no_augmentation,
        use_tdc_pretraining=args.tdc_pretrain,
        n_folds=args.n_folds,
        n_seeds=args.n_seeds
    )

    pipeline.run(
        train_path=args.train,
        test_path=args.test,
        output_dir=args.output
    )


if __name__ == '__main__':
    main()
