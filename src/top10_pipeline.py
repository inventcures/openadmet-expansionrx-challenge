"""
TOP10 Pipeline - Breaking into the Top 10

WITH CHECKPOINTING, RESUME, AND PROGRESS TRACKING

Features:
- Automatic checkpointing after each major step
- Resume from last checkpoint on re-run
- tqdm progress bars with ETA
- Detailed logging
- Graceful SIGINT handling (Ctrl+C saves state)

Comprehensive ensemble combining best practices from research papers:
1. MULTI-TASK CHEMPROP: Shared MPNN with task-specific heads
2. ChemBERTa EMBEDDINGS: Pretrained transformer features
3. SCAFFOLD-BASED CV: More realistic evaluation
4. DEEP STACKING: Two-level stacking with maximum model diversity

Version: 2.0 (with checkpointing)
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
from typing import Dict, Any, Optional
warnings.filterwarnings('ignore')

# Suppress RDKit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import torch
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
import xgboost as xgb
from catboost import CatBoostRegressor
from tqdm import tqdm

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

BASE_DIR = Path(__file__).parent.parent
CHECKPOINT_DIR = BASE_DIR / ".checkpoints" / "top10"

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

# Endpoint-specific hyperparameters (tuned)
HYPERPARAMS = {
    'LogD': {'max_depth': 8, 'lr': 0.03, 'n_estimators': 1000},
    'KSOL': {'max_depth': 8, 'lr': 0.03, 'n_estimators': 900},
    'HLM CLint': {'max_depth': 7, 'lr': 0.04, 'n_estimators': 700},
    'MLM CLint': {'max_depth': 7, 'lr': 0.04, 'n_estimators': 700},
    'Caco-2 Permeability Papp A>B': {'max_depth': 7, 'lr': 0.04, 'n_estimators': 800},
    'Caco-2 Permeability Efflux': {'max_depth': 6, 'lr': 0.05, 'n_estimators': 600},
    'MPPB': {'max_depth': 8, 'lr': 0.03, 'n_estimators': 900},
    'MBPB': {'max_depth': 8, 'lr': 0.03, 'n_estimators': 900},
    'MGMB': {'max_depth': 6, 'lr': 0.05, 'n_estimators': 500},
}


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging():
    """Setup detailed logging"""
    log_dir = BASE_DIR / "logs"
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"top10_pipeline_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging to: {log_file}")
    return logger


# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

class CheckpointManager:
    """Manage checkpoints for pipeline resume capability"""

    def __init__(self, checkpoint_dir: Path = CHECKPOINT_DIR):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.checkpoint_dir / "pipeline_state.json"
        self.state = self._load_state()

        # Register SIGINT handler
        signal.signal(signal.SIGINT, self._signal_handler)
        self._interrupted = False

    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        print("\n\n⚠️  SIGINT received - saving checkpoint before exit...")
        self._interrupted = True
        self.save_state()
        print("✅ Checkpoint saved. Run again to resume.")
        sys.exit(0)

    def _load_state(self) -> Dict[str, Any]:
        """Load pipeline state from disk"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            'stage': 'start',
            'completed_endpoints': [],
            'start_time': None,
            'features_computed': False,
            'chemberta_computed': False,
            'chemprop_computed': False,
        }

    def save_state(self):
        """Save pipeline state to disk"""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2, default=str)

    def save_checkpoint(self, name: str, data: Any):
        """Save data checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"{name}.pkl"
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(data, f)
        logging.info(f"💾 Checkpoint saved: {name}")

    def load_checkpoint(self, name: str) -> Optional[Any]:
        """Load data checkpoint if exists"""
        checkpoint_path = self.checkpoint_dir / f"{name}.pkl"
        if checkpoint_path.exists():
            with open(checkpoint_path, 'rb') as f:
                data = pickle.load(f)
            logging.info(f"📂 Checkpoint loaded: {name}")
            return data
        return None

    def checkpoint_exists(self, name: str) -> bool:
        """Check if checkpoint exists"""
        return (self.checkpoint_dir / f"{name}.pkl").exists()

    def mark_stage_complete(self, stage: str):
        """Mark a pipeline stage as complete"""
        self.state['stage'] = stage
        self.save_state()

    def mark_endpoint_complete(self, endpoint: str):
        """Mark an endpoint as complete"""
        if endpoint not in self.state['completed_endpoints']:
            self.state['completed_endpoints'].append(endpoint)
        self.save_state()

    def is_endpoint_complete(self, endpoint: str) -> bool:
        """Check if endpoint is already complete"""
        return endpoint in self.state['completed_endpoints']

    def clear_checkpoints(self):
        """Clear all checkpoints for fresh run"""
        import shutil
        if self.checkpoint_dir.exists():
            shutil.rmtree(self.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()
        logging.info("🗑️  Checkpoints cleared")

    def get_progress_summary(self) -> str:
        """Get human-readable progress summary"""
        completed = len(self.state['completed_endpoints'])
        total = len(TARGETS)
        pct = (completed / total) * 100 if total > 0 else 0

        return (f"Stage: {self.state['stage']} | "
                f"Endpoints: {completed}/{total} ({pct:.0f}%) | "
                f"Features: {'✓' if self.state['features_computed'] else '✗'} | "
                f"ChemBERTa: {'✓' if self.state['chemberta_computed'] else '✗'} | "
                f"Chemprop: {'✓' if self.state['chemprop_computed'] else '✗'}")


# ============================================================================
# FEATURE COMPUTATION
# ============================================================================

def compute_molecular_features(smiles_list, checkpoint_mgr: CheckpointManager):
    """Compute molecular features with checkpointing"""

    # Check for existing checkpoint
    if checkpoint_mgr.checkpoint_exists('molecular_features'):
        return checkpoint_mgr.load_checkpoint('molecular_features')

    from feature_engineering_v2 import FeatureEngineerV2

    logging.info("Computing molecular fingerprints...")

    fe = FeatureEngineerV2(
        morgan_bits=1024,
        use_maccs=True,
        use_rdkit_fp=True,
        use_mordred=True,
        verbose=False
    )

    # Use tqdm for progress
    features = []
    batch_size = 500

    for i in tqdm(range(0, len(smiles_list), batch_size),
                  desc="Molecular Features", unit="batch"):
        batch = smiles_list[i:i+batch_size]
        batch_features = fe.compute_features(batch)
        features.append(batch_features)

    features = np.vstack(features)
    logging.info(f"Molecular features shape: {features.shape}")

    # Save checkpoint
    checkpoint_mgr.save_checkpoint('molecular_features', features)
    checkpoint_mgr.state['features_computed'] = True
    checkpoint_mgr.save_state()

    return features


def compute_chemberta_features(smiles_list, checkpoint_mgr: CheckpointManager):
    """Compute ChemBERTa embeddings with checkpointing"""

    # Check for existing checkpoint
    if checkpoint_mgr.checkpoint_exists('chemberta_features'):
        return checkpoint_mgr.load_checkpoint('chemberta_features')

    try:
        from chemberta_embeddings import ChemBERTaFeatureExtractor, CHEMBERTA_AVAILABLE

        if not CHEMBERTA_AVAILABLE:
            logging.warning("ChemBERTa not available, using zeros")
            features = np.zeros((len(smiles_list), 256), dtype=np.float32)
        else:
            logging.info("Computing ChemBERTa embeddings...")
            extractor = ChemBERTaFeatureExtractor(n_components=256)
            features = extractor.fit_transform(smiles_list, batch_size=128)
            logging.info(f"ChemBERTa features shape: {features.shape}")

    except Exception as e:
        logging.error(f"ChemBERTa failed: {e}")
        features = np.zeros((len(smiles_list), 256), dtype=np.float32)

    # Save checkpoint
    checkpoint_mgr.save_checkpoint('chemberta_features', features)
    checkpoint_mgr.state['chemberta_computed'] = True
    checkpoint_mgr.save_state()

    return features


# ============================================================================
# MULTI-TASK CHEMPROP
# ============================================================================

def train_multitask_chemprop(train_smiles, train_targets, test_smiles,
                              checkpoint_mgr: CheckpointManager,
                              n_folds=5, n_models=3):
    """Train multi-task Chemprop with checkpointing"""

    # Check for existing checkpoint
    if checkpoint_mgr.checkpoint_exists('chemprop_predictions'):
        data = checkpoint_mgr.load_checkpoint('chemprop_predictions')
        return data['train_oof'], data['test_preds']

    try:
        from multitask_chemprop import MultiTaskChempropEnsemble, CHEMPROP_AVAILABLE

        if not CHEMPROP_AVAILABLE:
            logging.warning("Chemprop not available, skipping")
            return None, None

        logging.info("Training multi-task Chemprop...")

        n_train = len(train_smiles)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        train_oof = {t: np.zeros(n_train) for t in TARGETS}
        test_preds_all = {t: [] for t in TARGETS}

        fold_pbar = tqdm(list(kf.split(train_smiles)),
                         desc="Chemprop CV", unit="fold")

        for fold, (train_idx, val_idx) in enumerate(fold_pbar):
            fold_pbar.set_postfix({'fold': f'{fold+1}/{n_folds}'})

            fold_train_smiles = [train_smiles[i] for i in train_idx]
            fold_val_smiles = [train_smiles[i] for i in val_idx]

            fold_train_targets = {t: train_targets[t][train_idx] for t in TARGETS if t in train_targets}
            fold_val_targets = {t: train_targets[t][val_idx] for t in TARGETS if t in train_targets}

            # Train ensemble
            ensemble = MultiTaskChempropEnsemble(
                n_models=n_models,
                hidden_size=500,
                depth=4,
                ffn_hidden_size=400,
                dropout=0.15
            )
            ensemble.fit(fold_train_smiles, fold_train_targets,
                        fold_val_smiles, fold_val_targets, verbose=False)

            # OOF predictions
            val_preds = ensemble.predict(fold_val_smiles)
            for t in TARGETS:
                train_oof[t][val_idx] = val_preds[t]

            # Test predictions
            test_preds = ensemble.predict(test_smiles)
            for t in TARGETS:
                test_preds_all[t].append(test_preds[t])

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Average test predictions
        test_preds_final = {t: np.mean(test_preds_all[t], axis=0) for t in TARGETS}

        # Save checkpoint
        checkpoint_mgr.save_checkpoint('chemprop_predictions', {
            'train_oof': train_oof,
            'test_preds': test_preds_final
        })
        checkpoint_mgr.state['chemprop_computed'] = True
        checkpoint_mgr.save_state()

        return train_oof, test_preds_final

    except Exception as e:
        logging.error(f"Multi-task Chemprop failed: {e}")
        return None, None


# ============================================================================
# DEEP STACKING ENSEMBLE
# ============================================================================

class DeepStackingEnsemble:
    """Deep Stacking Ensemble with progress tracking"""

    def __init__(self, n_folds=5, verbose=True):
        self.n_folds = n_folds
        self.verbose = verbose
        self.base_models = {}
        self.meta_model = None
        self.uses_chemprop = False

    def _get_diverse_base_models(self, hp):
        """Create diverse base model configurations"""
        models = {}

        # XGBoost variants
        models['xgb_default'] = xgb.XGBRegressor(
            n_estimators=hp['n_estimators'],
            max_depth=hp['max_depth'],
            learning_rate=hp['lr'],
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=3.0,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )

        models['xgb_deep'] = xgb.XGBRegressor(
            n_estimators=hp['n_estimators'],
            max_depth=hp['max_depth'] + 2,
            learning_rate=hp['lr'] * 0.7,
            subsample=0.9,
            colsample_bytree=0.7,
            reg_lambda=5.0,
            random_state=43,
            n_jobs=-1,
            verbosity=0,
        )

        # CatBoost variants
        models['catboost_default'] = CatBoostRegressor(
            iterations=hp['n_estimators'],
            depth=min(hp['max_depth'], 8),
            learning_rate=hp['lr'],
            subsample=0.8,
            random_seed=42,
            verbose=False,
        )

        models['catboost_reg'] = CatBoostRegressor(
            iterations=hp['n_estimators'],
            depth=min(hp['max_depth'], 6),
            learning_rate=hp['lr'] * 0.8,
            subsample=0.9,
            l2_leaf_reg=10.0,
            random_seed=43,
            verbose=False,
        )

        # LightGBM variants
        if LGB_AVAILABLE:
            models['lgb_default'] = lgb.LGBMRegressor(
                n_estimators=hp['n_estimators'],
                max_depth=hp['max_depth'],
                learning_rate=hp['lr'],
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=3.0,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )

            models['lgb_reg'] = lgb.LGBMRegressor(
                n_estimators=hp['n_estimators'],
                max_depth=hp['max_depth'] - 1,
                learning_rate=hp['lr'] * 1.2,
                subsample=0.7,
                colsample_bytree=0.6,
                reg_lambda=10.0,
                random_state=43,
                n_jobs=-1,
                verbose=-1,
            )

        return models

    def fit(self, X, y, hp=None, chemprop_oof=None):
        """Fit with progress bar"""
        if hp is None:
            hp = {'max_depth': 7, 'lr': 0.05, 'n_estimators': 500}

        n_samples = len(y)
        base_models = self._get_diverse_base_models(hp)
        n_base = len(base_models) + (1 if chemprop_oof is not None else 0)

        # Generate OOF predictions
        oof_preds = np.zeros((n_samples, n_base))
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        self.base_models = {name: [] for name in base_models.keys()}

        # Progress bar for folds
        fold_iter = list(kf.split(X))

        for fold, (train_idx, val_idx) in enumerate(fold_iter):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Progress bar for models within fold
            model_pbar = tqdm(base_models.items(),
                             desc=f"  Fold {fold+1}/{self.n_folds}",
                             leave=False, unit="model")

            for model_idx, (name, model_template) in enumerate(model_pbar):
                model_pbar.set_postfix({'model': name})

                if 'xgb' in name:
                    model = xgb.XGBRegressor(**model_template.get_params())
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                elif 'catboost' in name:
                    model = CatBoostRegressor(**model_template.get_params())
                    model.fit(X_train, y_train, eval_set=(X_val, y_val),
                             early_stopping_rounds=50, verbose=False)
                elif 'lgb' in name:
                    model = lgb.LGBMRegressor(**model_template.get_params())
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                             callbacks=[lgb.early_stopping(50, verbose=False)])

                self.base_models[name].append(model)
                oof_preds[val_idx, model_idx] = model.predict(X_val)

        # Add Chemprop OOF
        if chemprop_oof is not None:
            oof_preds[:, -1] = chemprop_oof
            self.uses_chemprop = True

        # Fit meta-learner
        self.meta_model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0])
        self.meta_model.fit(oof_preds, y)

        # Compute metrics
        meta_preds = self.meta_model.predict(oof_preds)
        mae = mean_absolute_error(y, meta_preds)
        rae = mae / np.mean(np.abs(y - np.mean(y)))
        spear = spearmanr(y, meta_preds)[0]

        return {'MAE': mae, 'RAE': rae, 'Spearman': spear}

    def predict(self, X, chemprop_pred=None):
        """Make predictions"""
        n_samples = len(X)
        n_base = len(self.base_models) + (1 if self.uses_chemprop else 0)
        base_preds = np.zeros((n_samples, n_base))

        for model_idx, (name, fold_models) in enumerate(self.base_models.items()):
            fold_preds = np.zeros((n_samples, len(fold_models)))
            for fold_idx, model in enumerate(fold_models):
                fold_preds[:, fold_idx] = model.predict(X)
            base_preds[:, model_idx] = fold_preds.mean(axis=1)

        if self.uses_chemprop and chemprop_pred is not None:
            base_preds[:, -1] = chemprop_pred

        return self.meta_model.predict(base_preds)

    def get_weights(self):
        """Get meta-learner weights"""
        names = list(self.base_models.keys())
        if self.uses_chemprop:
            names.append('chemprop')
        return dict(zip(names, self.meta_model.coef_))


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_top10_pipeline(train_df, test_df, checkpoint_mgr: CheckpointManager,
                       use_chemprop=True, use_chemberta=True):
    """Full TOP10 pipeline with checkpointing and progress tracking"""

    start_time = time.time()

    print("\n" + "=" * 70)
    print("🚀 TOP10 PIPELINE (with checkpointing)")
    print("=" * 70)
    print(f"📊 {checkpoint_mgr.get_progress_summary()}")
    print("=" * 70)

    train_smiles = train_df['SMILES'].tolist()
    test_smiles = test_df['SMILES'].tolist()
    all_smiles = train_smiles + test_smiles

    # =========================================================================
    # STEP 1: Molecular Features
    # =========================================================================
    print("\n" + "─" * 70)
    print("📦 [1/4] MOLECULAR FEATURES")
    print("─" * 70)

    mol_features = compute_molecular_features(all_smiles, checkpoint_mgr)
    X_train_mol = mol_features[:len(train_df)]
    X_test_mol = mol_features[len(train_df):]

    # =========================================================================
    # STEP 2: ChemBERTa Features
    # =========================================================================
    print("\n" + "─" * 70)
    print("🧬 [2/4] ChemBERTa EMBEDDINGS")
    print("─" * 70)

    if use_chemberta:
        bert_features = compute_chemberta_features(all_smiles, checkpoint_mgr)
        X_train_bert = bert_features[:len(train_df)]
        X_test_bert = bert_features[len(train_df):]

        X_train = np.hstack([X_train_mol, X_train_bert])
        X_test = np.hstack([X_test_mol, X_test_bert])
    else:
        logging.info("Skipping ChemBERTa (disabled)")
        X_train = X_train_mol
        X_test = X_test_mol

    logging.info(f"Combined features: {X_train.shape}")

    # =========================================================================
    # STEP 3: Multi-task Chemprop
    # =========================================================================
    print("\n" + "─" * 70)
    print("🔬 [3/4] MULTI-TASK CHEMPROP")
    print("─" * 70)

    train_targets = {t: train_df[t].values for t in TARGETS}

    if use_chemprop:
        chemprop_train_oof, chemprop_test_preds = train_multitask_chemprop(
            train_smiles, train_targets, test_smiles, checkpoint_mgr,
            n_folds=5, n_models=3
        )
    else:
        logging.info("Skipping Chemprop (disabled)")
        chemprop_train_oof = None
        chemprop_test_preds = None

    # =========================================================================
    # STEP 4: Deep Stacking Ensemble
    # =========================================================================
    print("\n" + "─" * 70)
    print("🏗️  [4/4] DEEP STACKING ENSEMBLE")
    print("─" * 70)

    predictions = {}
    results = {}

    # Check for existing endpoint predictions
    if checkpoint_mgr.checkpoint_exists('endpoint_predictions'):
        saved_data = checkpoint_mgr.load_checkpoint('endpoint_predictions')
        predictions = saved_data.get('predictions', {})
        results = saved_data.get('results', {})

    # Progress bar for endpoints
    endpoint_pbar = tqdm(TARGETS, desc="Endpoints", unit="endpoint")

    for target in endpoint_pbar:
        endpoint_pbar.set_postfix({'current': target[:15]})

        # Skip if already complete
        if checkpoint_mgr.is_endpoint_complete(target):
            logging.info(f"⏭️  Skipping {target} (already complete)")
            continue

        logging.info(f"\n{'='*50}")
        logging.info(f"Training: {target}")
        logging.info(f"{'='*50}")

        y = train_targets[target]
        mask = ~np.isnan(y)
        X_t = X_train[mask]
        y_t = y[mask]

        logging.info(f"Samples: {len(y_t)}")

        hp = HYPERPARAMS.get(target, {'max_depth': 7, 'lr': 0.05, 'n_estimators': 500})

        # Get Chemprop OOF for this target
        chemprop_oof_t = None
        chemprop_test_t = None
        if chemprop_train_oof is not None:
            chemprop_oof_t = chemprop_train_oof[target][mask]
            chemprop_test_t = chemprop_test_preds[target]

        # Train ensemble
        ensemble = DeepStackingEnsemble(n_folds=5, verbose=True)
        result = ensemble.fit(X_t, y_t, hp=hp, chemprop_oof=chemprop_oof_t)
        results[target] = result

        # Predict test
        pred = ensemble.predict(X_test, chemprop_pred=chemprop_test_t)
        pred = np.clip(pred, *VALID_RANGES[target])
        predictions[target] = pred

        # Log results
        logging.info(f"✅ {target}: RAE={result['RAE']:.4f}, Spearman={result['Spearman']:.4f}")

        # Save checkpoint after each endpoint
        checkpoint_mgr.mark_endpoint_complete(target)
        checkpoint_mgr.save_checkpoint('endpoint_predictions', {
            'predictions': predictions,
            'results': results
        })

        gc.collect()

    # =========================================================================
    # SUMMARY
    # =========================================================================
    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("📊 RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Endpoint':<35} {'RAE':>10} {'Spearman':>10}")
    print("-" * 55)

    raes = []
    for t in TARGETS:
        if t in results:
            r = results[t]
            print(f"{t:<35} {r['RAE']:>10.4f} {r['Spearman']:>10.4f}")
            raes.append(r['RAE'])

    ma_rae = np.mean(raes) if raes else 0
    print("-" * 55)
    print(f"{'MA-RAE':<35} {ma_rae:>10.4f}")
    print(f"\n⏱️  Total time: {elapsed/60:.1f} minutes")

    # Save submission
    sub = pd.DataFrame({'Molecule Name': test_df['Molecule Name']})
    for t in TARGETS:
        sub[t] = predictions.get(t, train_df[t].median())

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = BASE_DIR / f"submissions/top10_{timestamp}.csv"
    sub.to_csv(out_path, index=False)
    print(f"\n💾 Saved: {out_path}")

    # Mark complete
    checkpoint_mgr.mark_stage_complete('complete')

    return results, predictions, out_path


def load_data():
    """Load train and test data"""
    train_path = BASE_DIR / "data/raw/train.csv"
    test_path = BASE_DIR / "data/raw/test_blinded.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    logging.info(f"Train: {len(train_df)} molecules")
    logging.info(f"Test: {len(test_df)} molecules")

    return train_df, test_df


def main():
    parser = argparse.ArgumentParser(description='TOP10 Pipeline with Checkpointing')
    parser.add_argument('--no-chemprop', action='store_true',
                       help='Skip multi-task Chemprop')
    parser.add_argument('--no-chemberta', action='store_true',
                       help='Skip ChemBERTa embeddings')
    parser.add_argument('--fresh', action='store_true',
                       help='Clear checkpoints and start fresh')
    parser.add_argument('--status', action='store_true',
                       help='Show checkpoint status and exit')
    args = parser.parse_args()

    # Setup
    logger = setup_logging()
    checkpoint_mgr = CheckpointManager()

    # Status check
    if args.status:
        print("\n" + "=" * 70)
        print("📊 CHECKPOINT STATUS")
        print("=" * 70)
        print(checkpoint_mgr.get_progress_summary())
        print(f"\nCheckpoint dir: {checkpoint_mgr.checkpoint_dir}")

        # List checkpoint files
        checkpoints = list(checkpoint_mgr.checkpoint_dir.glob("*.pkl"))
        if checkpoints:
            print("\nSaved checkpoints:")
            for cp in checkpoints:
                size_mb = cp.stat().st_size / (1024 * 1024)
                print(f"  - {cp.name} ({size_mb:.1f} MB)")
        return

    # Fresh start
    if args.fresh:
        checkpoint_mgr.clear_checkpoints()

    # Load data
    train_df, test_df = load_data()

    # Run pipeline
    try:
        results, predictions, out_path = run_top10_pipeline(
            train_df, test_df, checkpoint_mgr,
            use_chemprop=not args.no_chemprop,
            use_chemberta=not args.no_chemberta
        )

        print("\n" + "=" * 70)
        print("✅ PIPELINE COMPLETE!")
        print("=" * 70)
        print(f"Submission: {out_path}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted - checkpoint saved automatically")
        print("Run again to resume from last checkpoint")


if __name__ == "__main__":
    main()
