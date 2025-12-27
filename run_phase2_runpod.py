#!/usr/bin/env python3
"""
Phase 2 Pipeline - RunPod RTX 4090 Optimized (Hardened)

Features:
- Detailed logging with timestamps
- Progress bars (tqdm) with percentage
- Idempotent checkpointing - resume from last known state
- Automatic recovery on failure/interruption
"""
import os
import sys
import json
import pickle
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Suppress warnings before any imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

import warnings
warnings.filterwarnings('ignore')

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import pandas as pd
import numpy as np
import gc
import argparse
import time
from tqdm import tqdm

import torch
import xgboost as xgb
from catboost import CatBoostRegressor
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    lgb = None

from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

# Optional imports
try:
    from src.multitask_nn import MultiTaskTrainer
    MULTITASK_AVAILABLE = True
except ImportError:
    MULTITASK_AVAILABLE = False

try:
    from src.chemprop_optimized import ChempropEnsemble
    CHEMPROP_AVAILABLE = True
except ImportError:
    CHEMPROP_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModel
    from src.chemberta_embeddings import ChemBERTaEmbedder
    CHEMBERTA_AVAILABLE = True
except ImportError:
    CHEMBERTA_AVAILABLE = False
    ChemBERTaEmbedder = None

try:
    from src.unimol_features import compute_3d_features, UNIMOL_AVAILABLE
except ImportError:
    UNIMOL_AVAILABLE = False

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, MACCSkeys

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent
CHECKPOINT_DIR = BASE_DIR / ".checkpoints_phase2"

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

HYPERPARAMS_GPU = {
    'LogD': {'max_depth': 10, 'lr': 0.02, 'n_estimators': 1500},
    'KSOL': {'max_depth': 10, 'lr': 0.02, 'n_estimators': 1200},
    'HLM CLint': {'max_depth': 9, 'lr': 0.03, 'n_estimators': 1000},
    'MLM CLint': {'max_depth': 9, 'lr': 0.03, 'n_estimators': 1000},
    'Caco-2 Permeability Papp A>B': {'max_depth': 9, 'lr': 0.03, 'n_estimators': 1000},
    'Caco-2 Permeability Efflux': {'max_depth': 8, 'lr': 0.04, 'n_estimators': 800},
    'MPPB': {'max_depth': 10, 'lr': 0.02, 'n_estimators': 1200},
    'MBPB': {'max_depth': 10, 'lr': 0.02, 'n_estimators': 1200},
    'MGMB': {'max_depth': 8, 'lr': 0.05, 'n_estimators': 600},
}

# Pipeline stages for checkpointing
STAGES = [
    'init',
    'features_base',
    'features_chemberta',
    'features_unimol',
    'gbdt_training',
    'multitask_training',
    'chemprop_training',
    'blending',
    'complete'
]

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(mode: str) -> logging.Logger:
    """Setup detailed logging with timestamps"""
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    log_file = CHECKPOINT_DIR / f"phase2_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-5s | %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger('Phase2')
    logger.info(f"Logging to: {log_file}")
    return logger

# ============================================================================
# CHECKPOINT MANAGER
# ============================================================================

class CheckpointManager:
    """Manage pipeline checkpoints for idempotent execution"""

    def __init__(self, mode: str, config: Dict[str, Any]):
        self.mode = mode
        self.config = config
        self.checkpoint_dir = CHECKPOINT_DIR
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Create unique run ID based on config
        config_str = json.dumps(config, sort_keys=True)
        self.run_id = hashlib.md5(config_str.encode()).hexdigest()[:8]
        self.checkpoint_file = self.checkpoint_dir / f"checkpoint_{mode}_{self.run_id}.json"
        self.data_dir = self.checkpoint_dir / f"data_{mode}_{self.run_id}"
        self.data_dir.mkdir(exist_ok=True)

        self.state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        """Load existing checkpoint state"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                state = json.load(f)
                logging.info(f"Loaded checkpoint: stage={state.get('current_stage')}")
                return state
        return {
            'current_stage': 'init',
            'completed_stages': [],
            'start_time': datetime.now().isoformat(),
            'config': self.config,
            'results': {},
            'predictions': {},
        }

    def save_state(self):
        """Save current state to checkpoint"""
        self.state['last_update'] = datetime.now().isoformat()
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.state, f, indent=2, default=str)

    def is_stage_complete(self, stage: str) -> bool:
        """Check if a stage is already complete"""
        return stage in self.state.get('completed_stages', [])

    def mark_stage_complete(self, stage: str):
        """Mark a stage as complete"""
        if stage not in self.state['completed_stages']:
            self.state['completed_stages'].append(stage)
        self.state['current_stage'] = stage
        self.save_state()
        logging.info(f"Stage complete: {stage}")

    def save_array(self, name: str, arr: np.ndarray):
        """Save numpy array to checkpoint"""
        path = self.data_dir / f"{name}.npy"
        np.save(path, arr)
        logging.debug(f"Saved array: {name} {arr.shape}")

    def load_array(self, name: str) -> Optional[np.ndarray]:
        """Load numpy array from checkpoint"""
        path = self.data_dir / f"{name}.npy"
        if path.exists():
            arr = np.load(path)
            logging.debug(f"Loaded array: {name} {arr.shape}")
            return arr
        return None

    def save_pickle(self, name: str, obj: Any):
        """Save object to pickle"""
        path = self.data_dir / f"{name}.pkl"
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

    def load_pickle(self, name: str) -> Optional[Any]:
        """Load object from pickle"""
        path = self.data_dir / f"{name}.pkl"
        if path.exists():
            with open(path, 'rb') as f:
                return pickle.load(f)
        return None

    def save_predictions(self, stage: str, predictions: Dict[str, np.ndarray]):
        """Save predictions for a stage"""
        for target, pred in predictions.items():
            self.save_array(f"{stage}_{target}", pred)
        self.state['predictions'][stage] = list(predictions.keys())
        self.save_state()

    def load_predictions(self, stage: str) -> Optional[Dict[str, np.ndarray]]:
        """Load predictions for a stage"""
        if stage not in self.state.get('predictions', {}):
            return None
        predictions = {}
        for target in self.state['predictions'][stage]:
            arr = self.load_array(f"{stage}_{target}")
            if arr is not None:
                predictions[target] = arr
        return predictions if predictions else None

    def cleanup(self):
        """Remove checkpoint files after successful completion"""
        import shutil
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
        if self.data_dir.exists():
            shutil.rmtree(self.data_dir)
        logging.info("Checkpoint files cleaned up")

# ============================================================================
# GPU UTILITIES
# ============================================================================

def detect_gpu() -> tuple:
    """Detect available GPU with detailed info"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_mem_free = (torch.cuda.get_device_properties(0).total_memory -
                       torch.cuda.memory_allocated(0)) / 1e9
        logging.info(f"GPU: {gpu_name}")
        logging.info(f"GPU Memory: {gpu_mem:.1f} GB total, {gpu_mem_free:.1f} GB free")
        return True, gpu_name
    logging.warning("No CUDA GPU detected, using CPU")
    return False, None

def clear_gpu_memory():
    """Clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def get_descriptors(mol) -> np.ndarray:
    """Get 50 RDKit descriptors"""
    try:
        return np.array([
            Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol), Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol), Descriptors.NumRotatableBonds(mol),
            Descriptors.NumAromaticRings(mol), Descriptors.FractionCSP3(mol),
            Descriptors.HeavyAtomCount(mol), Descriptors.RingCount(mol),
            rdMolDescriptors.CalcNumAliphaticRings(mol),
            rdMolDescriptors.CalcNumHeterocycles(mol),
            Descriptors.LabuteASA(mol), Descriptors.qed(mol),
            rdMolDescriptors.CalcNumAmideBonds(mol),
            Descriptors.BertzCT(mol), Descriptors.Chi0(mol),
            Descriptors.Chi1(mol), Descriptors.Kappa1(mol),
            Descriptors.Kappa2(mol), Descriptors.HallKierAlpha(mol),
            rdMolDescriptors.CalcNumSpiroAtoms(mol),
            rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
            Descriptors.NumValenceElectrons(mol),
            Descriptors.NumRadicalElectrons(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol),
            rdMolDescriptors.CalcNumSaturatedRings(mol),
            rdMolDescriptors.CalcNumAromaticHeterocycles(mol),
            rdMolDescriptors.CalcNumAromaticCarbocycles(mol),
            rdMolDescriptors.CalcNumSaturatedHeterocycles(mol),
            rdMolDescriptors.CalcNumSaturatedCarbocycles(mol),
            rdMolDescriptors.CalcNumHeteroatoms(mol),
            rdMolDescriptors.CalcNumLipinskiHBA(mol),
            rdMolDescriptors.CalcNumLipinskiHBD(mol),
            Descriptors.PEOE_VSA1(mol), Descriptors.PEOE_VSA2(mol),
            Descriptors.SMR_VSA1(mol), Descriptors.SMR_VSA2(mol),
            Descriptors.SlogP_VSA1(mol), Descriptors.SlogP_VSA2(mol),
            Descriptors.EState_VSA1(mol), Descriptors.EState_VSA2(mol),
            Descriptors.MaxPartialCharge(mol), Descriptors.MinPartialCharge(mol),
            Descriptors.MaxAbsPartialCharge(mol), Descriptors.MinAbsPartialCharge(mol),
            Descriptors.NHOHCount(mol), Descriptors.NOCount(mol),
            Descriptors.NumAliphaticCarbocycles(mol),
            Descriptors.NumAliphaticHeterocycles(mol),
        ], dtype=np.float32)
    except Exception:
        return np.zeros(50, dtype=np.float32)


def compute_features_gpu(smiles_list, use_maccs=True, use_rdkit_fp=True,
                         checkpoint_mgr=None) -> np.ndarray:
    """Compute features with progress bar and checkpointing"""

    # Check for cached features
    cache_name = f"features_{'maccs' if use_maccs else ''}{'_rdkit' if use_rdkit_fp else ''}"
    if checkpoint_mgr:
        cached = checkpoint_mgr.load_array(cache_name)
        if cached is not None:
            logging.info(f"Loaded cached features: {cached.shape}")
            return cached

    n_mols = len(smiles_list)
    morgan_bits = 1024
    maccs_bits = 167 if use_maccs else 0
    rdkit_bits = 2048 if use_rdkit_fp else 0
    desc_count = 50
    total_dim = morgan_bits + maccs_bits + rdkit_bits + desc_count

    logging.info(f"Computing {total_dim} features for {n_mols} molecules...")

    features = np.zeros((n_mols, total_dim), dtype=np.float32)

    for i, smi in enumerate(tqdm(smiles_list, desc="Fingerprints", unit="mol",
                                  ncols=80, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        try:
            idx = 0
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=morgan_bits)
            features[i, idx:idx+morgan_bits] = np.array(fp)
            idx += morgan_bits

            if use_maccs:
                maccs = MACCSkeys.GenMACCSKeys(mol)
                features[i, idx:idx+maccs_bits] = np.array(maccs)
                idx += maccs_bits

            if use_rdkit_fp:
                rdkit_fp = Chem.RDKFingerprint(mol, fpSize=rdkit_bits)
                features[i, idx:idx+rdkit_bits] = np.array(rdkit_fp)
                idx += rdkit_bits

            desc = get_descriptors(mol)
            features[i, idx:idx+desc_count] = desc
        except Exception:
            continue

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # Cache features
    if checkpoint_mgr:
        checkpoint_mgr.save_array(cache_name, features)

    logging.info(f"Features computed: {features.shape}")
    return features

# ============================================================================
# GPU STACKING ENSEMBLE
# ============================================================================

class GPUStackingEnsemble:
    """Stacking ensemble with progress tracking"""

    def __init__(self, use_gpu=True, n_folds=5):
        self.use_gpu = use_gpu
        self.n_folds = n_folds
        self.base_models = {}
        self.meta_model = None

    def _create_xgb(self, hp):
        return xgb.XGBRegressor(
            n_estimators=hp['n_estimators'],
            max_depth=hp['max_depth'],
            learning_rate=hp['lr'],
            tree_method='hist',
            device='cuda' if self.use_gpu else 'cpu',
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=3.0,
            reg_alpha=0.1,
            max_bin=256,
            n_jobs=-1,
            verbosity=0,
        )

    def _create_catboost(self, hp):
        params = {
            'iterations': hp['n_estimators'],
            'depth': min(hp['max_depth'], 10),
            'learning_rate': hp['lr'],
            'l2_leaf_reg': 3.0,
            'random_seed': 42,
            'verbose': False,
        }
        if self.use_gpu:
            params['task_type'] = 'GPU'
            params['devices'] = '0'
        return CatBoostRegressor(**params)

    def _create_lgb(self, hp):
        if not LGB_AVAILABLE:
            return None
        return lgb.LGBMRegressor(
            n_estimators=hp['n_estimators'],
            max_depth=hp['max_depth'],
            learning_rate=hp['lr'],
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=3.0,
            n_jobs=-1,
            verbose=-1,
        )

    def _create_rf(self, hp):
        return RandomForestRegressor(
            n_estimators=min(hp['n_estimators'], 500),
            max_depth=hp['max_depth'] + 2,
            min_samples_leaf=3,
            n_jobs=-1,
        )

    def fit(self, X, y, hp, target_name=""):
        """Fit with detailed progress"""
        n_samples = len(y)
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        model_names = ['xgb', 'catboost', 'lgb', 'rf']
        oof_preds = {name: np.zeros(n_samples) for name in model_names}
        self.base_models = {name: [] for name in model_names}

        fold_pbar = tqdm(enumerate(kf.split(X)), total=self.n_folds,
                         desc=f"  {target_name[:15]:15s}", unit="fold",
                         ncols=80, leave=False)

        for fold, (tr_idx, va_idx) in fold_pbar:
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            # XGBoost
            xgb_model = self._create_xgb(hp)
            xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            oof_preds['xgb'][va_idx] = xgb_model.predict(X_va)
            self.base_models['xgb'].append(xgb_model)

            # CatBoost
            cb_model = self._create_catboost(hp)
            cb_model.fit(X_tr, y_tr, eval_set=(X_va, y_va),
                        early_stopping_rounds=100, verbose=False)
            oof_preds['catboost'][va_idx] = cb_model.predict(X_va)
            self.base_models['catboost'].append(cb_model)

            # LightGBM
            if LGB_AVAILABLE and lgb is not None:
                lgb_model = self._create_lgb(hp)
                try:
                    # LightGBM >= 4.0 API
                    lgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                                 callbacks=[lgb.early_stopping(100, verbose=False)])
                except TypeError:
                    # Fallback for older LightGBM
                    lgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                                 early_stopping_rounds=100, verbose=False)
                oof_preds['lgb'][va_idx] = lgb_model.predict(X_va)
                self.base_models['lgb'].append(lgb_model)

            # RandomForest
            rf_model = self._create_rf(hp)
            rf_model.fit(X_tr, y_tr)
            oof_preds['rf'][va_idx] = rf_model.predict(X_va)
            self.base_models['rf'].append(rf_model)

            clear_gpu_memory()

        # Meta-learner
        stack_features = np.column_stack([oof_preds[name] for name in model_names
                                          if len(self.base_models[name]) > 0])
        self.meta_model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        self.meta_model.fit(stack_features, y)

        oof_final = self.meta_model.predict(stack_features)
        mae = mean_absolute_error(y, oof_final)
        y_var = np.mean(np.abs(y - np.mean(y)))
        rae = mae / y_var if y_var > 1e-10 else mae  # Avoid division by zero
        spear_result = spearmanr(y, oof_final)
        spear = spear_result[0] if not np.isnan(spear_result[0]) else 0.0

        return {'MAE': mae, 'RAE': rae, 'Spearman': spear}

    def predict(self, X):
        model_names = ['xgb', 'catboost', 'lgb', 'rf']
        base_preds = []
        for name in model_names:
            if len(self.base_models[name]) > 0:
                fold_preds = np.column_stack([m.predict(X) for m in self.base_models[name]])
                base_preds.append(fold_preds.mean(axis=1))
        stack_features = np.column_stack(base_preds)
        return self.meta_model.predict(stack_features)

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline(mode='full', use_multitask=True, use_chemprop=True,
                 use_chemberta=True, use_unimol=True, resume=True):
    """Run Phase 2 pipeline with checkpointing and progress tracking"""

    # Setup
    logger = setup_logging(mode)
    config = {
        'mode': mode,
        'use_multitask': use_multitask,
        'use_chemprop': use_chemprop,
        'use_chemberta': use_chemberta,
        'use_unimol': use_unimol,
    }

    ckpt = CheckpointManager(mode, config)

    logger.info("=" * 60)
    logger.info(f"PHASE 2 PIPELINE - RTX 4090 ({mode.upper()} MODE)")
    logger.info("=" * 60)
    logger.info(f"Config: {config}")
    logger.info(f"Run ID: {ckpt.run_id}")

    if resume and ckpt.state['completed_stages']:
        logger.info(f"Resuming from stage: {ckpt.state['current_stage']}")
        logger.info(f"Completed stages: {ckpt.state['completed_stages']}")

    use_gpu, gpu_name = detect_gpu()

    # ========================================================================
    # STAGE 1: Load Data
    # ========================================================================
    logger.info("\n[1/7] Loading data...")
    train_path = BASE_DIR / "data/raw/train.csv"
    test_path = BASE_DIR / "data/raw/test_blinded.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Validate required columns
    required_train_cols = ['SMILES'] + TARGETS
    required_test_cols = ['SMILES', 'Molecule Name']
    missing_train = [c for c in required_train_cols if c not in train_df.columns]
    missing_test = [c for c in required_test_cols if c not in test_df.columns]

    if missing_train:
        raise ValueError(f"Missing columns in train.csv: {missing_train}")
    if missing_test:
        raise ValueError(f"Missing columns in test_blinded.csv: {missing_test}")

    logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")

    all_smiles = pd.concat([train_df['SMILES'], test_df['SMILES']]).values
    n_train = len(train_df)

    # ========================================================================
    # STAGE 2: Base Features
    # ========================================================================
    if not ckpt.is_stage_complete('features_base'):
        logger.info("\n[2/7] Computing base features...")
        use_rdkit_fp = (mode == 'full')
        X_all = compute_features_gpu(all_smiles, use_maccs=True,
                                     use_rdkit_fp=use_rdkit_fp, checkpoint_mgr=ckpt)
        ckpt.save_array('X_all', X_all)
        ckpt.mark_stage_complete('features_base')
    else:
        logger.info("\n[2/7] Loading cached base features...")
        X_all = ckpt.load_array('X_all')
        use_rdkit_fp = (mode == 'full')
        expected_dim = 1024 + 167 + 50 + (2048 if use_rdkit_fp else 0)

        if X_all is None:
            logger.warning("Cached features not found, recomputing...")
            X_all = compute_features_gpu(all_smiles, use_maccs=True,
                                         use_rdkit_fp=use_rdkit_fp, checkpoint_mgr=ckpt)
            ckpt.save_array('X_all', X_all)
        elif X_all.shape[0] != len(all_smiles):
            logger.warning(f"Cached features have wrong sample count ({X_all.shape[0]} vs {len(all_smiles)}), recomputing...")
            X_all = compute_features_gpu(all_smiles, use_maccs=True,
                                         use_rdkit_fp=use_rdkit_fp, checkpoint_mgr=ckpt)
            ckpt.save_array('X_all', X_all)
        elif X_all.shape[1] != expected_dim:
            logger.warning(f"Cached features dimension mismatch ({X_all.shape[1]} vs {expected_dim}), recomputing...")
            X_all = compute_features_gpu(all_smiles, use_maccs=True,
                                         use_rdkit_fp=use_rdkit_fp, checkpoint_mgr=ckpt)
            ckpt.save_array('X_all', X_all)

    X_train = X_all[:n_train]
    X_test = X_all[n_train:]
    logger.info(f"Base features: {X_train.shape}")

    # ========================================================================
    # STAGE 3: ChemBERTa Embeddings
    # ========================================================================
    if use_chemberta and CHEMBERTA_AVAILABLE and ChemBERTaEmbedder is not None:
        if not ckpt.is_stage_complete('features_chemberta'):
            logger.info("\n[3/7] Computing ChemBERTa embeddings...")
            try:
                embedder = ChemBERTaEmbedder()
                embedder.load_model()

                chemberta_all = embedder.embed_batch(
                    all_smiles.tolist(), batch_size=32, show_progress=True
                )
                ckpt.save_array('chemberta_all', chemberta_all)
                ckpt.mark_stage_complete('features_chemberta')
                clear_gpu_memory()

                X_train = np.hstack([X_train, chemberta_all[:n_train]])
                X_test = np.hstack([X_test, chemberta_all[n_train:]])
                logger.info(f"With ChemBERTa: {X_train.shape}")
            except Exception as e:
                logger.warning(f"ChemBERTa failed: {e}. Skipping...")
                ckpt.mark_stage_complete('features_chemberta')
        else:
            logger.info("\n[3/7] Loading cached ChemBERTa embeddings...")
            chemberta_all = ckpt.load_array('chemberta_all')
            if chemberta_all is not None:
                X_train = np.hstack([X_train, chemberta_all[:n_train]])
                X_test = np.hstack([X_test, chemberta_all[n_train:]])
                logger.info(f"With ChemBERTa: {X_train.shape}")
    else:
        if use_chemberta:
            logger.info("\n[3/7] ChemBERTa not available (install: pip install transformers). Skipping...")
        ckpt.mark_stage_complete('features_chemberta')

    # ========================================================================
    # STAGE 4: Uni-Mol 3D Features
    # ========================================================================
    if use_unimol and UNIMOL_AVAILABLE and mode == 'full':
        if not ckpt.is_stage_complete('features_unimol'):
            logger.info("\n[4/7] Computing Uni-Mol 3D features...")
            try:
                unimol_all = compute_3d_features(all_smiles.tolist(), use_unimol=True)
                ckpt.save_array('unimol_all', unimol_all)
                ckpt.mark_stage_complete('features_unimol')
                clear_gpu_memory()

                X_train = np.hstack([X_train, unimol_all[:n_train]])
                X_test = np.hstack([X_test, unimol_all[n_train:]])
                logger.info(f"With Uni-Mol: {X_train.shape}")
            except Exception as e:
                logger.warning(f"Uni-Mol failed: {e}. Skipping...")
                ckpt.mark_stage_complete('features_unimol')
        else:
            logger.info("\n[4/7] Loading cached Uni-Mol features...")
            unimol_all = ckpt.load_array('unimol_all')
            if unimol_all is not None:
                X_train = np.hstack([X_train, unimol_all[:n_train]])
                X_test = np.hstack([X_test, unimol_all[n_train:]])
                logger.info(f"With Uni-Mol: {X_train.shape}")
    else:
        if use_unimol and mode == 'full':
            logger.info("\n[4/7] Uni-Mol not available (install: pip install unimol_tools). Skipping...")
        ckpt.mark_stage_complete('features_unimol')

    # ========================================================================
    # STAGE 5: GBDT Training
    # ========================================================================
    gbdt_predictions = ckpt.load_predictions('gbdt')
    results = ckpt.state.get('results', {})

    if not ckpt.is_stage_complete('gbdt_training') or gbdt_predictions is None:
        logger.info("\n[5/7] Training GBDT stacking ensembles...")
        gbdt_predictions = {}

        target_pbar = tqdm(TARGETS, desc="GBDT Training", unit="target", ncols=80)
        for target in target_pbar:
            target_pbar.set_postfix_str(target[:20])

            y = train_df[target].values
            mask = ~np.isnan(y)
            X_t, y_t = X_train[mask], y[mask]

            if len(y_t) < 20:
                logger.warning(f"Skipping GBDT for {target}: only {len(y_t)} valid samples")
                # Use mean as fallback prediction
                gbdt_predictions[target] = np.full(len(X_test), np.mean(y_t) if len(y_t) > 0 else 0.0)
                continue

            hp = HYPERPARAMS_GPU[target].copy()
            if mode == 'quick':
                hp['n_estimators'] = min(hp['n_estimators'], 400)

            try:
                ensemble = GPUStackingEnsemble(use_gpu=use_gpu, n_folds=5)
                result = ensemble.fit(X_t, y_t, hp, target_name=target)
                results[target] = result

                pred = ensemble.predict(X_test)
                pred = np.clip(pred, *VALID_RANGES[target])
                gbdt_predictions[target] = pred

                logger.info(f"  {target}: RAE={result['RAE']:.4f}, Spearman={result['Spearman']:.4f}")
            except Exception as e:
                logger.error(f"GBDT training failed for {target}: {e}")
                gbdt_predictions[target] = np.full(len(X_test), np.mean(y_t))
            finally:
                clear_gpu_memory()

        ckpt.save_predictions('gbdt', gbdt_predictions)
        ckpt.state['results'] = results
        ckpt.mark_stage_complete('gbdt_training')
    else:
        logger.info("\n[5/7] Loaded cached GBDT predictions")

    # ========================================================================
    # STAGE 6: Multi-task Neural Network
    # ========================================================================
    mt_predictions = None
    if use_multitask and MULTITASK_AVAILABLE:
        mt_predictions = ckpt.load_predictions('multitask')

        if not ckpt.is_stage_complete('multitask_training') or mt_predictions is None:
            logger.info("\n[6/7] Training Multi-task Neural Network...")
            try:
                y_dict = {t: train_df[t].values for t in TARGETS}
                epochs = 50 if mode == 'quick' else 100

                mt_trainer = MultiTaskTrainer(
                    input_dim=X_train.shape[1],
                    hidden_dim=512,
                    lr=1e-3,
                )
                mt_trainer.fit(X_train, y_dict, epochs=epochs, verbose=True)
                mt_predictions = mt_trainer.predict(X_test)

                ckpt.save_predictions('multitask', mt_predictions)
                ckpt.mark_stage_complete('multitask_training')
            except Exception as e:
                logger.error(f"Multi-task training failed: {e}")
                logger.warning("Continuing without multi-task predictions")
                mt_predictions = None
                ckpt.mark_stage_complete('multitask_training')
            finally:
                clear_gpu_memory()
        else:
            logger.info("\n[6/7] Loaded cached Multi-task predictions")
    else:
        ckpt.mark_stage_complete('multitask_training')

    # ========================================================================
    # STAGE 7: Chemprop D-MPNN
    # ========================================================================
    cp_predictions = None
    if use_chemprop and CHEMPROP_AVAILABLE:
        cp_predictions = ckpt.load_predictions('chemprop')

        if not ckpt.is_stage_complete('chemprop_training') or cp_predictions is None:
            logger.info("\n[7/7] Training Chemprop D-MPNN...")
            try:
                train_smiles = train_df['SMILES'].tolist()
                test_smiles = test_df['SMILES'].tolist()
                n_models = 2 if mode == 'quick' else 3

                cp_predictions = {}
                for target in tqdm(TARGETS, desc="Chemprop", unit="target", ncols=80):
                    y = train_df[target].values
                    mask = ~np.isnan(y)
                    valid_smiles = [train_smiles[i] for i, m in enumerate(mask) if m]
                    valid_y = y[mask]

                    if len(valid_smiles) < 10:
                        logger.warning(f"Skipping Chemprop for {target}: only {len(valid_smiles)} valid samples")
                        continue

                    ensemble = ChempropEnsemble(target, n_models=n_models)
                    ensemble.fit(valid_smiles, valid_y, verbose=False)
                    cp_predictions[target] = ensemble.predict(test_smiles)
                    clear_gpu_memory()

                ckpt.save_predictions('chemprop', cp_predictions)
                ckpt.mark_stage_complete('chemprop_training')
            except Exception as e:
                logger.error(f"Chemprop failed: {e}")
                logger.error("Chemprop is required. Fix the error or use --no-chemprop to skip.")
                raise RuntimeError(f"Chemprop training failed: {e}") from e
        else:
            logger.info("\n[7/7] Loaded cached Chemprop predictions")
    else:
        if use_chemprop:
            logger.info("\n[7/7] Chemprop not available (install: pip install chemprop). Skipping...")
        ckpt.mark_stage_complete('chemprop_training')

    # ========================================================================
    # BLENDING
    # ========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("BLENDING PREDICTIONS")
    logger.info("=" * 60)

    predictions = {}
    n_model_types = 1 + (1 if mt_predictions else 0) + (1 if cp_predictions else 0)

    if gbdt_predictions is None:
        raise RuntimeError("GBDT predictions not available - cannot blend")

    for target in TARGETS:
        if target not in gbdt_predictions:
            logger.error(f"Missing GBDT prediction for {target}")
            raise RuntimeError(f"Missing GBDT prediction for {target}")

        gbdt_pred = gbdt_predictions[target]

        if n_model_types == 1:
            blended = gbdt_pred
        elif n_model_types == 2:
            if mt_predictions and target in mt_predictions:
                blended = 0.7 * gbdt_pred + 0.3 * mt_predictions[target]
            elif cp_predictions and target in cp_predictions:
                blended = 0.75 * gbdt_pred + 0.25 * cp_predictions[target]
            else:
                blended = gbdt_pred  # Fallback to GBDT only
        else:
            mt_pred = mt_predictions.get(target, gbdt_pred) if mt_predictions else gbdt_pred
            cp_pred = cp_predictions.get(target, gbdt_pred) if cp_predictions else gbdt_pred
            blended = (0.6 * gbdt_pred +
                      0.25 * mt_pred +
                      0.15 * cp_pred)

        predictions[target] = np.clip(blended, *VALID_RANGES[target])

    blend_desc = f"GBDT"
    if mt_predictions:
        blend_desc += " + Multi-task"
    if cp_predictions:
        blend_desc += " + Chemprop"
    logger.info(f"Blended {n_model_types} model types: {blend_desc}")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)

    raes = [results[t]['RAE'] for t in TARGETS if t in results]
    for t in TARGETS:
        if t in results:
            r = results[t]
            logger.info(f"{t:35s}: RAE={r['RAE']:.4f}, Spearman={r['Spearman']:.4f}")

    if raes:
        ma_rae = np.mean(raes)
        logger.info(f"\n{'MA-RAE':35s}: {ma_rae:.4f}")
    else:
        logger.warning("No RAE scores available (results may have been loaded from cache)")

    # Save submission
    sub = pd.DataFrame({'Molecule Name': test_df['Molecule Name']})
    for t in TARGETS:
        sub[t] = predictions[t]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = BASE_DIR / f"submissions/phase2_{mode}_{timestamp}.csv"
    out_path.parent.mkdir(exist_ok=True)
    sub.to_csv(out_path, index=False)
    logger.info(f"\nSubmission saved: {out_path}")

    # Mark complete and optionally cleanup
    ckpt.mark_stage_complete('complete')
    # ckpt.cleanup()  # Uncomment to auto-cleanup on success

    return results, out_path


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Phase 2 Pipeline - RunPod RTX 4090 (Hardened)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_phase2_runpod.py --mode full          # Full pipeline (~150 min)
  python run_phase2_runpod.py --mode quick         # Quick pipeline (~45 min)
  python run_phase2_runpod.py --gbdt-only          # GBDT only (~20 min)
  python run_phase2_runpod.py --no-resume          # Fresh start, ignore checkpoints
        """
    )

    parser.add_argument('--mode', choices=['quick', 'full'], default='full',
                       help='quick (~45 min) or full (~150 min)')
    parser.add_argument('--no-multitask', action='store_true',
                       help='Disable multi-task neural network')
    parser.add_argument('--no-chemprop', action='store_true',
                       help='Disable Chemprop D-MPNN')
    parser.add_argument('--no-chemberta', action='store_true',
                       help='Disable ChemBERTa embeddings')
    parser.add_argument('--no-unimol', action='store_true',
                       help='Disable Uni-Mol 3D features')
    parser.add_argument('--gbdt-only', action='store_true',
                       help='GBDT stacking only (fastest)')
    parser.add_argument('--no-resume', action='store_true',
                       help='Start fresh, ignore existing checkpoints')
    parser.add_argument('--clean', action='store_true',
                       help='Remove all checkpoints and exit')

    args = parser.parse_args()

    # Handle cleanup
    if args.clean:
        import shutil
        if CHECKPOINT_DIR.exists():
            shutil.rmtree(CHECKPOINT_DIR)
            print(f"Removed: {CHECKPOINT_DIR}")
        sys.exit(0)

    # Determine flags
    use_mt = not args.no_multitask and not args.gbdt_only
    use_cp = not args.no_chemprop and not args.gbdt_only
    use_cb = not args.no_chemberta and not args.gbdt_only
    use_um = not args.no_unimol and not args.gbdt_only

    start = time.time()

    try:
        results, out_path = run_pipeline(
            mode=args.mode,
            use_multitask=use_mt,
            use_chemprop=use_cp,
            use_chemberta=use_cb,
            use_unimol=use_um,
            resume=not args.no_resume
        )
        elapsed = time.time() - start
        print(f"\n{'='*60}")
        print(f"SUCCESS! Total time: {elapsed/60:.1f} minutes")
        print(f"Submit: {out_path}")
        print(f"{'='*60}")

    except KeyboardInterrupt:
        print("\n\nInterrupted! Progress saved to checkpoint.")
        print("Re-run the same command to resume.")
        sys.exit(1)

    except Exception as e:
        logging.exception(f"Error: {e}")
        print("\n\nError occurred! Progress saved to checkpoint.")
        print("Re-run the same command to resume from last checkpoint.")
        sys.exit(1)
