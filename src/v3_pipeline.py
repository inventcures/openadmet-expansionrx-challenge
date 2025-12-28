"""
V3 Pipeline - TOP 3 Strategy Implementation

Multi-architecture ensemble with diversity weighting:
1. 1D CNN on SMILES (BELKA winner technique)
2. Extended fingerprints (ECFP4/6, FCFP4, MACCS)
3. ChemBERTa embeddings
4. Multi-task Chemprop
5. Diversity-weighted ensemble

Based on: docs/v3_deep-research_specs.md
"""
import os
import sys
import signal
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import gc
import argparse
import time
import json
import pickle
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
warnings.filterwarnings('ignore')

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import torch
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
from tqdm import tqdm

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
    from catboost import CatBoostRegressor
    CB_AVAILABLE = True
except ImportError:
    CB_AVAILABLE = False

BASE_DIR = Path(__file__).parent.parent
CHECKPOINT_DIR = BASE_DIR / ".checkpoints" / "v3"

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


def setup_logging():
    """Setup logging"""
    log_dir = BASE_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"v3_pipeline_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


class V3CheckpointManager:
    """Checkpoint management for V3 pipeline"""

    def __init__(self, checkpoint_dir: Path = CHECKPOINT_DIR):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.checkpoint_dir / "v3_state.json"
        self.state = self._load_state()

        signal.signal(signal.SIGINT, self._signal_handler)
        self._interrupted = False

    def _signal_handler(self, signum, frame):
        print("\n\nSIGINT - saving checkpoint...")
        self._interrupted = True
        self.save_state()
        print("Checkpoint saved. Run again to resume.")
        sys.exit(0)

    def _load_state(self) -> Dict[str, Any]:
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            'stage': 'start',
            'completed_endpoints': [],
            'completed_models': [],
            'features_computed': {},
        }

    def save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def save_array(self, name: str, arr: np.ndarray):
        np.save(self.checkpoint_dir / f"{name}.npy", arr)
        self.state['features_computed'][name] = True
        self.save_state()

    def load_array(self, name: str) -> Optional[np.ndarray]:
        path = self.checkpoint_dir / f"{name}.npy"
        if path.exists():
            return np.load(path)
        return None

    def is_computed(self, name: str) -> bool:
        return self.state.get('features_computed', {}).get(name, False)

    def mark_model_complete(self, model_name: str):
        if model_name not in self.state['completed_models']:
            self.state['completed_models'].append(model_name)
            self.save_state()

    def is_model_complete(self, model_name: str) -> bool:
        return model_name in self.state.get('completed_models', [])

    def clear(self):
        """Clear all checkpoints"""
        import shutil
        if self.checkpoint_dir.exists():
            shutil.rmtree(self.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()


class V3Pipeline:
    """
    V3 Multi-Architecture Ensemble Pipeline

    Combines:
    - 1D CNN on SMILES
    - Extended fingerprints + GBDT
    - ChemBERTa embeddings + GBDT
    - Diversity-weighted ensembling
    """

    def __init__(
        self,
        n_folds: int = 5,
        n_seeds: int = 3,
        use_gpu: bool = True,
        checkpoint: bool = True
    ):
        self.n_folds = n_folds
        self.n_seeds = n_seeds
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')

        self.checkpoint_mgr = V3CheckpointManager() if checkpoint else None
        self.logger = setup_logging()

        # Model storage
        self.oof_predictions = {}  # {endpoint: {model_name: predictions}}
        self.test_predictions = {}  # {endpoint: {model_name: predictions}}
        self.ensemble_weights = {}  # {endpoint: weights}

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load training and test data"""
        train_path = BASE_DIR / "data" / "train.csv"
        test_path = BASE_DIR / "data" / "test.csv"

        self.logger.info(f"Loading data from {BASE_DIR / 'data'}")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        self.logger.info(f"Train: {len(train_df)} samples, Test: {len(test_df)} samples")
        return train_df, test_df

    def compute_extended_fingerprints(
        self,
        smiles_list: List[str],
        name_prefix: str = "train"
    ) -> Dict[str, np.ndarray]:
        """Compute extended fingerprints"""
        from extended_fingerprints import ExtendedFingerprinter, compute_all_features

        cache_name = f"{name_prefix}_extended_fp"
        if self.checkpoint_mgr and self.checkpoint_mgr.is_computed(cache_name):
            self.logger.info(f"Loading cached {cache_name}")
            return {
                'standard': self.checkpoint_mgr.load_array(f"{cache_name}_standard"),
                'extended': self.checkpoint_mgr.load_array(f"{cache_name}_extended"),
            }

        self.logger.info(f"Computing extended fingerprints for {len(smiles_list)} molecules")
        features = compute_all_features(smiles_list, verbose=True)

        if self.checkpoint_mgr:
            self.checkpoint_mgr.save_array(f"{cache_name}_standard", features['standard'])
            self.checkpoint_mgr.save_array(f"{cache_name}_extended", features['extended'])
            self.checkpoint_mgr.state['features_computed'][cache_name] = True
            self.checkpoint_mgr.save_state()

        return features

    def compute_chemberta_embeddings(
        self,
        smiles_list: List[str],
        name_prefix: str = "train"
    ) -> np.ndarray:
        """Compute ChemBERTa embeddings"""
        cache_name = f"{name_prefix}_chemberta"
        if self.checkpoint_mgr and self.checkpoint_mgr.is_computed(cache_name):
            self.logger.info(f"Loading cached {cache_name}")
            return self.checkpoint_mgr.load_array(cache_name)

        try:
            from chemberta_embeddings import ChemBERTaEmbedder
            self.logger.info(f"Computing ChemBERTa embeddings for {len(smiles_list)} molecules")
            embedder = ChemBERTaEmbedder()
            embedder.load_model()
            embeddings = embedder.embed_batch(smiles_list, batch_size=128, show_progress=True)

            if self.checkpoint_mgr:
                self.checkpoint_mgr.save_array(cache_name, embeddings)

            return embeddings
        except Exception as e:
            self.logger.warning(f"ChemBERTa failed: {e}, returning zeros")
            return np.zeros((len(smiles_list), 768), dtype=np.float32)

    def train_smiles_cnn(
        self,
        train_smiles: List[str],
        train_y: np.ndarray,
        test_smiles: List[str],
        endpoint: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Train 1D CNN on SMILES"""
        model_name = f"smiles_cnn_{endpoint}"

        if self.checkpoint_mgr and self.checkpoint_mgr.is_model_complete(model_name):
            self.logger.info(f"Loading cached {model_name}")
            oof = self.checkpoint_mgr.load_array(f"{model_name}_oof")
            test = self.checkpoint_mgr.load_array(f"{model_name}_test")
            return oof, test

        try:
            from smiles_cnn import SMILES1DCNNEnsemble
            self.logger.info(f"Training SMILES 1D CNN for {endpoint}")

            ensemble = SMILES1DCNNEnsemble(
                n_models=self.n_seeds,
                n_folds=self.n_folds,
                epochs=30,
                batch_size=64,
                device=self.device,
                hidden_dim=128,  # Smaller for faster training
            )

            oof_pred, test_pred = ensemble.fit_predict(
                train_smiles, train_y, test_smiles, verbose=True
            )

            if self.checkpoint_mgr:
                self.checkpoint_mgr.save_array(f"{model_name}_oof", oof_pred)
                self.checkpoint_mgr.save_array(f"{model_name}_test", test_pred)
                self.checkpoint_mgr.mark_model_complete(model_name)

            return oof_pred, test_pred
        except Exception as e:
            self.logger.warning(f"SMILES CNN failed: {e}")
            return np.zeros(len(train_smiles)), np.zeros(len(test_smiles))

    def train_gbdt_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        endpoint: str,
        model_type: str = 'lightgbm',
        feature_name: str = 'ecfp'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Train GBDT model with CV"""
        model_name = f"{model_type}_{feature_name}_{endpoint}"

        if self.checkpoint_mgr and self.checkpoint_mgr.is_model_complete(model_name):
            self.logger.info(f"Loading cached {model_name}")
            oof = self.checkpoint_mgr.load_array(f"{model_name}_oof")
            test = self.checkpoint_mgr.load_array(f"{model_name}_test")
            return oof, test

        self.logger.info(f"Training {model_type} ({feature_name}) for {endpoint}")

        n_samples = len(X_train)
        oof_pred = np.zeros(n_samples)
        test_preds = []

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            if model_type == 'lightgbm' and LGB_AVAILABLE:
                model = lgb.LGBMRegressor(
                    n_estimators=500,
                    max_depth=7,
                    learning_rate=0.05,
                    num_leaves=63,
                    min_child_samples=10,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    random_state=42 + fold,
                    verbose=-1,
                    n_jobs=-1
                )
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False)]
                )
            elif model_type == 'xgboost' and XGB_AVAILABLE:
                model = xgb.XGBRegressor(
                    n_estimators=500,
                    max_depth=7,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    random_state=42 + fold,
                    verbosity=0,
                    n_jobs=-1
                )
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=50,
                    verbose=False
                )
            elif model_type == 'catboost' and CB_AVAILABLE:
                model = CatBoostRegressor(
                    iterations=500,
                    depth=7,
                    learning_rate=0.05,
                    random_state=42 + fold,
                    verbose=False
                )
                model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50)
            else:
                # Fallback to Random Forest
                model = RandomForestRegressor(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_leaf=5,
                    random_state=42 + fold,
                    n_jobs=-1
                )
                model.fit(X_tr, y_tr)

            oof_pred[val_idx] = model.predict(X_val)
            test_preds.append(model.predict(X_test))

        test_pred = np.mean(test_preds, axis=0)

        # Evaluate
        corr, _ = spearmanr(oof_pred, y_train)
        self.logger.info(f"  {model_name}: Spearman={corr:.4f}")

        if self.checkpoint_mgr:
            self.checkpoint_mgr.save_array(f"{model_name}_oof", oof_pred)
            self.checkpoint_mgr.save_array(f"{model_name}_test", test_pred)
            self.checkpoint_mgr.mark_model_complete(model_name)

        return oof_pred, test_pred

    def train_all_models_for_endpoint(
        self,
        endpoint: str,
        train_smiles: List[str],
        train_y: np.ndarray,
        test_smiles: List[str],
        train_features: Dict[str, np.ndarray],
        test_features: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Train all model types for one endpoint"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Training models for: {endpoint}")
        self.logger.info(f"{'='*60}")

        oof_preds = {}
        test_preds = {}

        # 1. SMILES 1D CNN
        if self.use_gpu:
            oof, test = self.train_smiles_cnn(train_smiles, train_y, test_smiles, endpoint)
            oof_preds['smiles_cnn'] = oof
            test_preds['smiles_cnn'] = test

        # 2. LightGBM with standard fingerprints (ECFP4 + descriptors)
        if 'standard' in train_features:
            oof, test = self.train_gbdt_model(
                train_features['standard'], train_y, test_features['standard'],
                endpoint, 'lightgbm', 'standard'
            )
            oof_preds['lgb_standard'] = oof
            test_preds['lgb_standard'] = test

        # 3. XGBoost with extended fingerprints (ECFP6 + FCFP4 + MACCS)
        if 'extended' in train_features and XGB_AVAILABLE:
            oof, test = self.train_gbdt_model(
                train_features['extended'], train_y, test_features['extended'],
                endpoint, 'xgboost', 'extended'
            )
            oof_preds['xgb_extended'] = oof
            test_preds['xgb_extended'] = test

        # 4. LightGBM with ChemBERTa embeddings
        if 'chemberta' in train_features:
            oof, test = self.train_gbdt_model(
                train_features['chemberta'], train_y, test_features['chemberta'],
                endpoint, 'lightgbm', 'chemberta'
            )
            oof_preds['lgb_chemberta'] = oof
            test_preds['lgb_chemberta'] = test

        # 5. CatBoost with standard features (for diversity)
        if 'standard' in train_features and CB_AVAILABLE:
            oof, test = self.train_gbdt_model(
                train_features['standard'], train_y, test_features['standard'],
                endpoint, 'catboost', 'standard'
            )
            oof_preds['cb_standard'] = oof
            test_preds['cb_standard'] = test

        # 6. Combined features (extended + chemberta)
        if 'extended' in train_features and 'chemberta' in train_features:
            combined_train = np.hstack([train_features['extended'], train_features['chemberta']])
            combined_test = np.hstack([test_features['extended'], test_features['chemberta']])
            oof, test = self.train_gbdt_model(
                combined_train, train_y, combined_test,
                endpoint, 'lightgbm', 'combined'
            )
            oof_preds['lgb_combined'] = oof
            test_preds['lgb_combined'] = test

        return oof_preds, test_preds

    def ensemble_predictions(
        self,
        oof_preds: Dict[str, np.ndarray],
        test_preds: Dict[str, np.ndarray],
        y_true: np.ndarray,
        endpoint: str
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Create diversity-weighted ensemble"""
        from diversity_ensemble import DiversityWeightedEnsemble, analyze_ensemble_diversity

        model_names = list(oof_preds.keys())
        n_models = len(model_names)

        if n_models == 0:
            self.logger.warning(f"No models for {endpoint}")
            return np.zeros_like(y_true), np.zeros(len(list(test_preds.values())[0])), {}

        # Stack predictions
        oof_matrix = np.column_stack([oof_preds[name] for name in model_names])
        test_matrix = np.column_stack([test_preds[name] for name in model_names])

        # Analyze diversity
        analysis = analyze_ensemble_diversity(oof_matrix, y_true, model_names)
        self.logger.info(f"  Ensemble diversity: {analysis['mean_diversity']:.3f}")
        self.logger.info(f"  {analysis['recommendation']}")

        # Fit diversity-weighted ensemble
        ensemble = DiversityWeightedEnsemble(method='diversity', diversity_weight=0.4)
        ensemble.fit(oof_matrix, y_true, model_names)

        oof_ensemble = ensemble.predict(oof_matrix)
        test_ensemble = ensemble.predict(test_matrix)

        # Evaluate
        corr, _ = spearmanr(oof_ensemble, y_true)
        self.logger.info(f"  Ensemble Spearman: {corr:.4f}")

        # Get weights for analysis
        weights = ensemble.get_model_importance(model_names)
        self.logger.info(f"  Model weights: {weights}")

        return oof_ensemble, test_ensemble, weights

    def run(self, clear_cache: bool = False):
        """Run full V3 pipeline"""
        if clear_cache and self.checkpoint_mgr:
            self.logger.info("Clearing cache...")
            self.checkpoint_mgr.clear()

        start_time = time.time()
        self.logger.info("V3 Pipeline - TOP 3 Strategy")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Folds: {self.n_folds}, Seeds: {self.n_seeds}")

        # Load data
        train_df, test_df = self.load_data()
        train_smiles = train_df['SMILES'].tolist()
        test_smiles = test_df['SMILES'].tolist()

        # Compute features
        self.logger.info("\n" + "="*60)
        self.logger.info("Phase 1: Feature Extraction")
        self.logger.info("="*60)

        train_fp = self.compute_extended_fingerprints(train_smiles, "train")
        test_fp = self.compute_extended_fingerprints(test_smiles, "test")

        train_chemberta = self.compute_chemberta_embeddings(train_smiles, "train")
        test_chemberta = self.compute_chemberta_embeddings(test_smiles, "test")

        train_features = {
            'standard': train_fp['standard'],
            'extended': train_fp['extended'],
            'chemberta': train_chemberta
        }
        test_features = {
            'standard': test_fp['standard'],
            'extended': test_fp['extended'],
            'chemberta': test_chemberta
        }

        # Train models for each endpoint
        self.logger.info("\n" + "="*60)
        self.logger.info("Phase 2: Model Training")
        self.logger.info("="*60)

        final_predictions = {}
        all_weights = {}

        for endpoint in tqdm(TARGETS, desc="Endpoints"):
            # Get target values (handle missing)
            mask = ~train_df[endpoint].isna()
            train_y = train_df.loc[mask, endpoint].values

            # Filter features for non-missing samples
            train_feat_filtered = {
                k: v[mask.values] for k, v in train_features.items()
            }
            train_smiles_filtered = [s for s, m in zip(train_smiles, mask) if m]

            # Train all models
            oof_preds, test_preds = self.train_all_models_for_endpoint(
                endpoint,
                train_smiles_filtered,
                train_y,
                test_smiles,
                train_feat_filtered,
                test_features
            )

            # Ensemble
            oof_final, test_final, weights = self.ensemble_predictions(
                oof_preds, test_preds, train_y, endpoint
            )

            # Clip to valid range
            vmin, vmax = VALID_RANGES[endpoint]
            test_final = np.clip(test_final, vmin, vmax)

            final_predictions[endpoint] = test_final
            all_weights[endpoint] = weights

            # Store for analysis
            self.oof_predictions[endpoint] = oof_preds
            self.test_predictions[endpoint] = test_preds
            self.ensemble_weights[endpoint] = weights

        # Create submission
        self.logger.info("\n" + "="*60)
        self.logger.info("Phase 3: Create Submission")
        self.logger.info("="*60)

        submission = test_df[['SMILES']].copy()
        for endpoint in TARGETS:
            submission[endpoint] = final_predictions[endpoint]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        submission_path = BASE_DIR / "submissions" / f"v3_submission_{timestamp}.csv"
        submission_path.parent.mkdir(exist_ok=True)
        submission.to_csv(submission_path, index=False)

        elapsed = time.time() - start_time
        self.logger.info(f"\nPipeline completed in {elapsed/60:.1f} minutes")
        self.logger.info(f"Submission saved: {submission_path}")

        # Save weights analysis
        weights_path = BASE_DIR / "submissions" / f"v3_weights_{timestamp}.json"
        with open(weights_path, 'w') as f:
            json.dump(all_weights, f, indent=2)
        self.logger.info(f"Weights saved: {weights_path}")

        return submission


def main():
    parser = argparse.ArgumentParser(description='V3 Pipeline - TOP 3 Strategy')
    parser.add_argument('--folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--seeds', type=int, default=3, help='Number of seeds for ensembling')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU')
    parser.add_argument('--no-checkpoint', action='store_true', help='Disable checkpointing')
    parser.add_argument('--clear-cache', action='store_true', help='Clear cached checkpoints')

    args = parser.parse_args()

    pipeline = V3Pipeline(
        n_folds=args.folds,
        n_seeds=args.seeds,
        use_gpu=not args.no_gpu,
        checkpoint=not args.no_checkpoint
    )

    submission = pipeline.run(clear_cache=args.clear_cache)
    print(f"\nSubmission shape: {submission.shape}")
    print(submission.head())


if __name__ == "__main__":
    main()
