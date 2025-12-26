#!/usr/bin/env python3
"""
ADMET Challenge - Local M3 Pro Training Script
Idempotent with checkpointing, resume on failure, detailed logging.

Usage:
    python run_local_m3.py                    # Resume from checkpoint (medium mode)
    python run_local_m3.py --mode quick       # Quick test run
    python run_local_m3.py --mode full        # Full training
    python run_local_m3.py --force            # Force restart from scratch
    python run_local_m3.py --status           # Show checkpoint status
"""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['RDKit_DEPRECATION_WARNINGS'] = '0'

import sys
import argparse
import signal
import json
import pickle
import hashlib
import time
import gc
import logging
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum

import numpy as np
import pandas as pd
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "raw"
SUB_DIR = BASE_DIR / "submissions"
LOG_DIR = BASE_DIR / "logs"
CHECKPOINT_DIR = BASE_DIR / ".checkpoints"

TARGETS = [
    'LogD', 'KSOL', 'HLM CLint', 'MLM CLint',
    'Caco-2 Permeability Papp A>B', 'Caco-2 Permeability Efflux',
    'MPPB', 'MBPB', 'MGMB'
]

VALID_RANGES = {
    'LogD': (-3.0, 6.0), 'KSOL': (0.001, 350.0),
    'HLM CLint': (0.0, 3000.0), 'MLM CLint': (0.0, 12000.0),
    'Caco-2 Permeability Papp A>B': (0.0, 60.0),
    'Caco-2 Permeability Efflux': (0.2, 120.0),
    'MPPB': (0.0, 100.0), 'MBPB': (0.0, 100.0), 'MGMB': (0.0, 100.0)
}

# Mode configurations
MODE_CONFIG = {
    'quick': {
        'iterations': 200,
        'chemprop_epochs': 5,
        'n_bits': 1024,
        'description': 'Quick test (~15-20 min)'
    },
    'medium': {
        'iterations': 600,
        'chemprop_epochs': 15,
        'n_bits': 2048,
        'description': 'Balanced (~60-80 min)'
    },
    'full': {
        'iterations': 1200,
        'chemprop_epochs': 30,
        'n_bits': 2048,
        'description': 'Full accuracy (~2-3 hours)'
    }
}

# ============================================================================
# LOGGING SETUP
# ============================================================================

class ColorFormatter(logging.Formatter):
    """Custom formatter with colors for terminal output"""
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        record.levelname = f"{color}{record.levelname:8}{reset}"
        return super().format(record)


def setup_logging(log_file: Path) -> logging.Logger:
    """Setup dual logging to file and console"""
    logger = logging.getLogger('admet')
    logger.setLevel(logging.DEBUG)
    logger.handlers = []

    # File handler (detailed)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(fh)

    # Console handler (colored)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(ColorFormatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    ))
    logger.addHandler(ch)

    return logger


# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

class TrainingStage(Enum):
    INIT = "init"
    FEATURES = "features"
    CATBOOST = "catboost"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CHEMPROP = "chemprop"
    ENSEMBLE = "ensemble"
    COMPLETE = "complete"


@dataclass
class Checkpoint:
    """Checkpoint state for resumable training"""
    mode: str
    stage: str
    current_target_idx: int = 0
    started_at: str = ""
    updated_at: str = ""

    # Completed work
    completed_stages: List[str] = field(default_factory=list)
    completed_targets: Dict[str, List[str]] = field(default_factory=dict)

    # Results
    results: Dict[str, Dict] = field(default_factory=dict)
    predictions: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)

    # Feature hash (to detect data changes)
    data_hash: str = ""

    # Timing
    stage_times: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'Checkpoint':
        return cls(**d)


class CheckpointManager:
    """Manages checkpoint save/load with atomic writes"""

    def __init__(self, checkpoint_dir: Path, mode: str):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.checkpoint_file = checkpoint_dir / f"checkpoint_{mode}.json"
        self.features_file = checkpoint_dir / f"features_{mode}.pkl"
        self.checkpoint: Optional[Checkpoint] = None
        self.logger = logging.getLogger('admet')

    def exists(self) -> bool:
        return self.checkpoint_file.exists()

    def load(self) -> Optional[Checkpoint]:
        """Load checkpoint if exists"""
        if not self.exists():
            return None
        try:
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)
            self.checkpoint = Checkpoint.from_dict(data)
            self.logger.info(f"Loaded checkpoint: stage={self.checkpoint.stage}, "
                           f"targets={len(self.checkpoint.completed_targets)}")
            return self.checkpoint
        except Exception as e:
            self.logger.warning(f"Failed to load checkpoint: {e}")
            return None

    def save(self, checkpoint: Checkpoint):
        """Atomic checkpoint save"""
        checkpoint.updated_at = datetime.now().isoformat()
        self.checkpoint = checkpoint

        # Write to temp file first, then rename (atomic on POSIX)
        temp_file = self.checkpoint_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(checkpoint.to_dict(), f, indent=2)
        temp_file.rename(self.checkpoint_file)

    def save_features(self, X_train: np.ndarray, X_test: np.ndarray):
        """Save computed features"""
        with open(self.features_file, 'wb') as f:
            pickle.dump({'X_train': X_train, 'X_test': X_test}, f)
        self.logger.debug(f"Saved features to {self.features_file}")

    def load_features(self) -> Optional[tuple]:
        """Load cached features if exist"""
        if not self.features_file.exists():
            return None
        try:
            with open(self.features_file, 'rb') as f:
                data = pickle.load(f)
            self.logger.info("Loaded cached features")
            return data['X_train'], data['X_test']
        except Exception as e:
            self.logger.warning(f"Failed to load features: {e}")
            return None

    def clear(self):
        """Clear all checkpoints"""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
        if self.features_file.exists():
            self.features_file.unlink()
        self.checkpoint = None
        self.logger.info("Cleared checkpoints")

    def get_status(self) -> str:
        """Get human-readable status"""
        if not self.exists():
            return "No checkpoint found"

        cp = self.load()
        if not cp:
            return "Checkpoint corrupted"

        lines = [
            f"Mode: {cp.mode}",
            f"Stage: {cp.stage}",
            f"Started: {cp.started_at}",
            f"Updated: {cp.updated_at}",
            f"Completed stages: {', '.join(cp.completed_stages) or 'None'}",
        ]

        if cp.results:
            lines.append("\nResults so far:")
            for model, data in cp.results.items():
                if 'ma_rae' in data:
                    lines.append(f"  {model}: MA-RAE = {data['ma_rae']:.4f}")

        return "\n".join(lines)


# ============================================================================
# SIGNAL HANDLING
# ============================================================================

class GracefulInterrupt:
    """Context manager for graceful interrupt handling"""

    def __init__(self, checkpoint_manager: CheckpointManager, checkpoint: Checkpoint):
        self.cm = checkpoint_manager
        self.checkpoint = checkpoint
        self.interrupted = False
        self.logger = logging.getLogger('admet')
        self._original_sigint = None
        self._original_sigterm = None

    def __enter__(self):
        self._original_sigint = signal.signal(signal.SIGINT, self._handler)
        self._original_sigterm = signal.signal(signal.SIGTERM, self._handler)
        return self

    def __exit__(self, *args):
        signal.signal(signal.SIGINT, self._original_sigint)
        signal.signal(signal.SIGTERM, self._original_sigterm)

    def _handler(self, signum, frame):
        if self.interrupted:
            self.logger.warning("Force quit - saving checkpoint...")
            self.cm.save(self.checkpoint)
            sys.exit(1)

        self.interrupted = True
        self.logger.warning("\n" + "="*60)
        self.logger.warning("INTERRUPT RECEIVED - Saving checkpoint...")
        self.logger.warning("Press Ctrl+C again to force quit")
        self.logger.warning("="*60)
        self.cm.save(self.checkpoint)
        self.logger.info(f"Checkpoint saved at stage: {self.checkpoint.stage}")
        self.logger.info("Run again to resume from this point")
        sys.exit(0)

    def check(self):
        """Check if interrupted (for use in loops)"""
        return self.interrupted


# ============================================================================
# FEATURE COMPUTATION
# ============================================================================

def compute_data_hash(train_df: pd.DataFrame, test_df: pd.DataFrame) -> str:
    """Compute hash of input data for cache invalidation"""
    h = hashlib.md5()
    h.update(str(len(train_df)).encode())
    h.update(str(len(test_df)).encode())
    h.update(train_df['SMILES'].iloc[0].encode())
    h.update(test_df['SMILES'].iloc[0].encode())
    return h.hexdigest()[:12]


def compute_features(smiles_list: list, n_bits: int = 2048,
                    logger: logging.Logger = None) -> np.ndarray:
    """Compute molecular features with progress bar"""
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

    if logger:
        logger.info(f"Computing features: {len(smiles_list)} molecules, {n_bits}-bit FP")

    n_desc = 25
    features = []

    pbar = tqdm(smiles_list, desc="Computing features", unit="mol",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

    for smi in pbar:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            features.append(np.zeros(n_bits + n_desc))
            continue
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
            desc = [
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
                rdMolDescriptors.CalcNumHBA(mol)
            ]
            features.append(np.concatenate([np.array(fp), desc]))
        except:
            features.append(np.zeros(n_bits + n_desc))

    return np.array(features, dtype=np.float32)


def compute_rae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Relative Absolute Error"""
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(y_true, y_pred)
    baseline = np.mean(np.abs(y_true - np.mean(y_true)))
    return mae / baseline if baseline > 0 else 1.0


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_model(model_name: str, X_train_all: np.ndarray, train_df: pd.DataFrame,
                X_test: np.ndarray, config: dict, checkpoint: Checkpoint,
                checkpoint_manager: CheckpointManager,
                interrupt_handler: GracefulInterrupt,
                logger: logging.Logger) -> Dict:
    """Train a single model type across all targets"""
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_absolute_error

    logger.info(f"Training {model_name.upper()} (iterations={config['iterations']})")

    # Initialize storage
    if model_name not in checkpoint.predictions:
        checkpoint.predictions[model_name] = {}
    if model_name not in checkpoint.completed_targets:
        checkpoint.completed_targets[model_name] = []

    results = {}
    start_time = time.time()

    # Get model class
    if model_name == 'catboost':
        from catboost import CatBoostRegressor
        ModelClass = CatBoostRegressor
    elif model_name == 'xgboost':
        from xgboost import XGBRegressor
        ModelClass = XGBRegressor
    elif model_name == 'lightgbm':
        from lightgbm import LGBMRegressor
        ModelClass = LGBMRegressor

    # Progress bar for targets
    completed = checkpoint.completed_targets.get(model_name, [])
    remaining_targets = [t for t in TARGETS if t not in completed]

    if not remaining_targets:
        logger.info(f"  All targets already completed for {model_name}")
        return checkpoint.results.get(model_name, {})

    logger.info(f"  Remaining targets: {len(remaining_targets)}/{len(TARGETS)}")

    pbar = tqdm(remaining_targets, desc=f"{model_name.upper()}", unit="target",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

    for target in pbar:
        if interrupt_handler.check():
            break

        pbar.set_postfix_str(target[:20])

        y = train_df[target].values
        mask = ~np.isnan(y)
        X_t, y_t = X_train_all[mask], y[mask]

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        oof = np.zeros(len(y_t))
        test_preds = []

        for fold, (tr_idx, va_idx) in enumerate(kf.split(X_t)):
            if interrupt_handler.check():
                break

            if model_name == 'catboost':
                model = ModelClass(
                    iterations=config['iterations'], depth=7, learning_rate=0.03,
                    subsample=0.8, colsample_bylevel=0.8, l2_leaf_reg=3.0,
                    verbose=False, random_seed=42+fold, early_stopping_rounds=50
                )
                model.fit(X_t[tr_idx], y_t[tr_idx],
                         eval_set=(X_t[va_idx], y_t[va_idx]), verbose=False)
            elif model_name == 'xgboost':
                model = ModelClass(
                    n_estimators=config['iterations'], max_depth=7, learning_rate=0.03,
                    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                    random_state=42+fold, verbosity=0, n_jobs=-1
                )
                model.fit(X_t[tr_idx], y_t[tr_idx],
                         eval_set=[(X_t[va_idx], y_t[va_idx])], verbose=False)
            elif model_name == 'lightgbm':
                model = ModelClass(
                    n_estimators=config['iterations'], max_depth=7, learning_rate=0.03,
                    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                    random_state=42+fold, verbosity=-1, n_jobs=-1
                )
                model.fit(X_t[tr_idx], y_t[tr_idx],
                         eval_set=[(X_t[va_idx], y_t[va_idx])])

            oof[va_idx] = model.predict(X_t[va_idx])
            test_preds.append(model.predict(X_test))

        if interrupt_handler.check():
            break

        # Compute metrics
        rae = compute_rae(y_t, oof)
        results[target] = rae

        # Store predictions
        test_pred = np.clip(np.mean(test_preds, axis=0), *VALID_RANGES[target])
        checkpoint.predictions[model_name][target] = test_pred.tolist()

        # Mark target complete
        checkpoint.completed_targets[model_name].append(target)

        # Save checkpoint after each target
        checkpoint_manager.save(checkpoint)

        logger.debug(f"  {target}: RAE={rae:.4f} (n={len(y_t)})")

    # Compute MA-RAE
    if results:
        ma_rae = np.mean(list(results.values()))
        elapsed = time.time() - start_time

        checkpoint.results[model_name] = {
            'ma_rae': ma_rae,
            'per_target': results,
            'time': elapsed
        }
        checkpoint.stage_times[model_name] = elapsed

        logger.info(f"  {model_name.upper()} MA-RAE: {ma_rae:.4f} ({elapsed:.0f}s)")

    gc.collect()
    return checkpoint.results.get(model_name, {})


def train_chemprop(train_df: pd.DataFrame, test_df: pd.DataFrame,
                   config: dict, checkpoint: Checkpoint,
                   checkpoint_manager: CheckpointManager,
                   interrupt_handler: GracefulInterrupt,
                   logger: logging.Logger) -> Dict:
    """Train Chemprop D-MPNN model"""
    try:
        import torch
        import lightning as pl
        from chemprop import data, models, nn

        # Suppress lightning logs
        logging.getLogger('lightning.pytorch').setLevel(logging.ERROR)

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"Training CHEMPROP D-MPNN (epochs={config['chemprop_epochs']}, device={device})")

        if 'chemprop' not in checkpoint.predictions:
            checkpoint.predictions['chemprop'] = {}
        if 'chemprop' not in checkpoint.completed_targets:
            checkpoint.completed_targets['chemprop'] = []

        completed = checkpoint.completed_targets.get('chemprop', [])
        remaining_targets = [t for t in TARGETS if t not in completed]

        if not remaining_targets:
            logger.info("  All targets already completed for chemprop")
            return checkpoint.results.get('chemprop', {})

        start_time = time.time()

        pbar = tqdm(remaining_targets, desc="CHEMPROP", unit="target",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

        for target in pbar:
            if interrupt_handler.check():
                break

            pbar.set_postfix_str(target[:20])

            mask = train_df[target].notna()
            train_smiles = train_df.loc[mask, 'SMILES'].tolist()
            train_y = train_df.loc[mask, target].values.reshape(-1, 1)
            test_smiles = test_df['SMILES'].tolist()

            train_data = [data.MoleculeDatapoint.from_smi(smi, y)
                          for smi, y in zip(train_smiles, train_y)]
            test_data = [data.MoleculeDatapoint.from_smi(smi) for smi in test_smiles]

            train_dset = data.MoleculeDataset(train_data)
            test_dset = data.MoleculeDataset(test_data)

            n_train = int(len(train_dset) * 0.85)
            train_dset, val_dset = data.split_data_by_indices(
                train_dset, [list(range(n_train)), list(range(n_train, len(train_dset)))]
            )

            mp = nn.BondMessagePassing()
            agg = nn.MeanAggregation()
            ffn = nn.RegressionFFN()
            mpnn = models.MPNN(mp, agg, ffn)

            trainer = pl.Trainer(
                max_epochs=config['chemprop_epochs'],
                accelerator=device,
                enable_progress_bar=False,
                logger=False,
                enable_checkpointing=False
            )

            train_loader = data.build_dataloader(train_dset, batch_size=64, shuffle=True)
            val_loader = data.build_dataloader(val_dset, batch_size=64)
            test_loader = data.build_dataloader(test_dset, batch_size=64)

            trainer.fit(mpnn, train_loader, val_loader)
            preds = trainer.predict(mpnn, test_loader)
            test_pred = np.concatenate([p.numpy() for p in preds]).flatten()

            checkpoint.predictions['chemprop'][target] = np.clip(
                test_pred, *VALID_RANGES[target]
            ).tolist()
            checkpoint.completed_targets['chemprop'].append(target)
            checkpoint_manager.save(checkpoint)

            logger.debug(f"  {target}: completed (n={mask.sum()})")

            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()

        elapsed = time.time() - start_time
        checkpoint.results['chemprop'] = {'time': elapsed}
        checkpoint.stage_times['chemprop'] = elapsed
        logger.info(f"  CHEMPROP completed ({elapsed:.0f}s)")

        return checkpoint.results.get('chemprop', {})

    except Exception as e:
        logger.warning(f"Chemprop failed: {e}")
        logger.warning("Continuing without Chemprop...")
        return {}


def create_ensembles(checkpoint: Checkpoint, test_df: pd.DataFrame,
                     mode: str, logger: logging.Logger):
    """Create ensemble submissions"""
    logger.info("Creating ensemble submissions...")

    available_models = [m for m in ['catboost', 'xgboost', 'lightgbm']
                        if m in checkpoint.predictions and checkpoint.predictions[m]]

    if len(available_models) < 2:
        logger.warning("Not enough models for ensemble")
        return

    # Ensemble 1: Equal weights
    logger.info(f"  Equal weights: {', '.join(available_models)}")
    ens_equal = pd.DataFrame({'Molecule Name': test_df['Molecule Name']})
    for t in TARGETS:
        preds = [np.array(checkpoint.predictions[m][t]) for m in available_models]
        ens_equal[t] = np.mean(preds, axis=0)
    ens_equal.to_csv(SUB_DIR / f"ensemble_equal_{mode}.csv", index=False)

    # Ensemble 2: Weighted by inverse RAE
    if all(m in checkpoint.results and 'per_target' in checkpoint.results[m]
           for m in available_models):
        logger.info("  Weighted by inverse RAE")
        ens_weighted = pd.DataFrame({'Molecule Name': test_df['Molecule Name']})
        for t in TARGETS:
            raes = [checkpoint.results[m]['per_target'][t] for m in available_models]
            weights = np.array([1/r for r in raes])
            weights = weights / weights.sum()
            preds = [np.array(checkpoint.predictions[m][t]) for m in available_models]
            ens_weighted[t] = sum(w * p for w, p in zip(weights, preds))
        ens_weighted.to_csv(SUB_DIR / f"ensemble_weighted_{mode}.csv", index=False)

    # Ensemble 3: Include Chemprop
    if 'chemprop' in checkpoint.predictions and checkpoint.predictions['chemprop']:
        logger.info("  With Chemprop")
        all_models = available_models + ['chemprop']
        ens_all = pd.DataFrame({'Molecule Name': test_df['Molecule Name']})
        for t in TARGETS:
            preds = [np.array(checkpoint.predictions[m][t]) for m in all_models
                     if t in checkpoint.predictions.get(m, {})]
            if preds:
                ens_all[t] = np.mean(preds, axis=0)
        ens_all.to_csv(SUB_DIR / f"ensemble_with_chemprop_{mode}.csv", index=False)


def save_individual_submissions(checkpoint: Checkpoint, test_df: pd.DataFrame,
                                mode: str, logger: logging.Logger):
    """Save individual model submissions"""
    for model_name, preds in checkpoint.predictions.items():
        if not preds:
            continue
        sub = pd.DataFrame({'Molecule Name': test_df['Molecule Name']})
        for t in TARGETS:
            if t in preds:
                sub[t] = preds[t]
        out_path = SUB_DIR / f"{model_name}_{mode}.csv"
        sub.to_csv(out_path, index=False)
        logger.info(f"  Saved: {out_path.name}")


def print_summary(checkpoint: Checkpoint, logger: logging.Logger):
    """Print final summary"""
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    # Per-model results
    print("\n" + "-"*50)
    print("MODEL RESULTS (MA-RAE)")
    print("-"*50)

    best_model = None
    best_rae = float('inf')

    for model in ['catboost', 'xgboost', 'lightgbm']:
        if model in checkpoint.results and 'ma_rae' in checkpoint.results[model]:
            ma_rae = checkpoint.results[model]['ma_rae']
            t = checkpoint.results[model].get('time', 0)
            print(f"  {model.upper():<12} MA-RAE: {ma_rae:.4f}  (time: {t:.0f}s)")
            if ma_rae < best_rae:
                best_rae = ma_rae
                best_model = model

    print("\n" + "-"*50)
    print("PER-TARGET RAE COMPARISON")
    print("-"*50)
    print(f"{'Target':<35} {'CatBoost':>10} {'XGBoost':>10} {'LightGBM':>10}")
    print("-"*70)

    for t in TARGETS:
        values = []
        for model in ['catboost', 'xgboost', 'lightgbm']:
            if model in checkpoint.results and 'per_target' in checkpoint.results[model]:
                values.append(checkpoint.results[model]['per_target'].get(t, float('nan')))
            else:
                values.append(float('nan'))
        print(f"{t:<35} {values[0]:>10.4f} {values[1]:>10.4f} {values[2]:>10.4f}")

    print("\n" + "="*50)
    if best_model:
        print(f"BEST SINGLE MODEL: {best_model.upper()} = {best_rae:.4f}")
    print(f"TARGET:            0.5593 (leader 'pebble')")
    if best_model:
        print(f"GAP:               {best_rae - 0.5593:+.4f}")
    print("="*50)

    # Total time
    total_time = sum(checkpoint.stage_times.values())
    print(f"\nTotal training time: {timedelta(seconds=int(total_time))}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='ADMET Challenge Training - Idempotent with checkpointing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_local_m3.py                  # Resume from checkpoint (medium mode)
  python run_local_m3.py --mode quick     # Quick test run (~15-20 min)
  python run_local_m3.py --mode full      # Full training (~2-3 hours)
  python run_local_m3.py --force          # Force restart from scratch
  python run_local_m3.py --status         # Show checkpoint status
        """
    )
    parser.add_argument('--mode', choices=['quick', 'medium', 'full'], default='medium',
                        help='Training mode (default: medium)')
    parser.add_argument('--force', action='store_true',
                        help='Force restart, ignore existing checkpoint')
    parser.add_argument('--status', action='store_true',
                        help='Show checkpoint status and exit')
    parser.add_argument('--skip-chemprop', action='store_true',
                        help='Skip Chemprop training')
    args = parser.parse_args()

    # Create directories
    SUB_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOG_DIR / f"training_{args.mode}_{timestamp}.log"
    logger = setup_logging(log_file)

    # Checkpoint manager
    cm = CheckpointManager(CHECKPOINT_DIR, args.mode)

    # Status only
    if args.status:
        print(cm.get_status())
        return

    # Force restart
    if args.force:
        logger.info("Force flag set - clearing checkpoints")
        cm.clear()

    # Load or create checkpoint
    checkpoint = cm.load()
    config = MODE_CONFIG[args.mode]

    if checkpoint:
        logger.info(f"Resuming from checkpoint: stage={checkpoint.stage}")
    else:
        checkpoint = Checkpoint(
            mode=args.mode,
            stage=TrainingStage.INIT.value,
            started_at=datetime.now().isoformat()
        )
        logger.info(f"Starting fresh training run")

    # Header
    print("="*70)
    print("ADMET CHALLENGE - LOCAL M3 PRO TRAINING")
    print(f"Mode: {args.mode} ({config['description']})")
    print(f"Log: {log_file}")
    print("="*70)

    # Setup interrupt handler
    with GracefulInterrupt(cm, checkpoint) as interrupt:

        # Load data
        logger.info("Loading data...")
        train_df = pd.read_csv(DATA_DIR / "train.csv")
        test_df = pd.read_csv(DATA_DIR / "test_blinded.csv")
        logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")

        data_hash = compute_data_hash(train_df, test_df)

        # Compute or load features
        if checkpoint.stage == TrainingStage.INIT.value or checkpoint.data_hash != data_hash:
            logger.info("Computing features...")
            checkpoint.stage = TrainingStage.FEATURES.value
            cm.save(checkpoint)

            all_smiles = pd.concat([train_df['SMILES'], test_df['SMILES']]).values
            X_all = compute_features(all_smiles, n_bits=config['n_bits'], logger=logger)
            X_train_all, X_test = X_all[:len(train_df)], X_all[len(train_df):]

            checkpoint.data_hash = data_hash
            cm.save_features(X_train_all, X_test)
            checkpoint.completed_stages.append(TrainingStage.FEATURES.value)
            cm.save(checkpoint)
        else:
            cached = cm.load_features()
            if cached:
                X_train_all, X_test = cached
            else:
                logger.info("Feature cache missing, recomputing...")
                all_smiles = pd.concat([train_df['SMILES'], test_df['SMILES']]).values
                X_all = compute_features(all_smiles, n_bits=config['n_bits'], logger=logger)
                X_train_all, X_test = X_all[:len(train_df)], X_all[len(train_df):]
                cm.save_features(X_train_all, X_test)

        logger.info(f"Features: {X_train_all.shape}")

        # Train CatBoost
        if TrainingStage.CATBOOST.value not in checkpoint.completed_stages:
            checkpoint.stage = TrainingStage.CATBOOST.value
            cm.save(checkpoint)
            train_model('catboost', X_train_all, train_df, X_test, config,
                       checkpoint, cm, interrupt, logger)
            if not interrupt.check():
                checkpoint.completed_stages.append(TrainingStage.CATBOOST.value)
                cm.save(checkpoint)

        if interrupt.check():
            return

        # Train XGBoost
        if TrainingStage.XGBOOST.value not in checkpoint.completed_stages:
            checkpoint.stage = TrainingStage.XGBOOST.value
            cm.save(checkpoint)
            train_model('xgboost', X_train_all, train_df, X_test, config,
                       checkpoint, cm, interrupt, logger)
            if not interrupt.check():
                checkpoint.completed_stages.append(TrainingStage.XGBOOST.value)
                cm.save(checkpoint)

        if interrupt.check():
            return

        # Train LightGBM
        if TrainingStage.LIGHTGBM.value not in checkpoint.completed_stages:
            checkpoint.stage = TrainingStage.LIGHTGBM.value
            cm.save(checkpoint)
            train_model('lightgbm', X_train_all, train_df, X_test, config,
                       checkpoint, cm, interrupt, logger)
            if not interrupt.check():
                checkpoint.completed_stages.append(TrainingStage.LIGHTGBM.value)
                cm.save(checkpoint)

        if interrupt.check():
            return

        # Train Chemprop
        if not args.skip_chemprop and TrainingStage.CHEMPROP.value not in checkpoint.completed_stages:
            checkpoint.stage = TrainingStage.CHEMPROP.value
            cm.save(checkpoint)
            train_chemprop(train_df, test_df, config, checkpoint, cm, interrupt, logger)
            if not interrupt.check():
                checkpoint.completed_stages.append(TrainingStage.CHEMPROP.value)
                cm.save(checkpoint)

        if interrupt.check():
            return

        # Create ensembles
        checkpoint.stage = TrainingStage.ENSEMBLE.value
        cm.save(checkpoint)

        logger.info("Saving individual submissions...")
        save_individual_submissions(checkpoint, test_df, args.mode, logger)

        create_ensembles(checkpoint, test_df, args.mode, logger)

        # Mark complete
        checkpoint.stage = TrainingStage.COMPLETE.value
        checkpoint.completed_stages.append(TrainingStage.COMPLETE.value)
        cm.save(checkpoint)

        # Save results JSON
        results_file = LOG_DIR / f"results_{args.mode}_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'mode': args.mode,
                'config': config,
                'results': checkpoint.results,
                'stage_times': checkpoint.stage_times,
                'completed_at': datetime.now().isoformat()
            }, f, indent=2)
        logger.info(f"Results saved: {results_file}")

        # Print summary
        print_summary(checkpoint, logger)

        print(f"\nSubmissions in: {SUB_DIR}/")
        for f in sorted(SUB_DIR.glob(f"*_{args.mode}.csv")):
            print(f"  {f.name}")

        print(f"\nTraining complete! Log: {log_file}")


if __name__ == "__main__":
    main()
