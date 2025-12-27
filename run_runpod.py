#!/usr/bin/env python3
"""
ADMET Challenge - RunPod GPU-Optimized Training Script
MAXIMALLY OPTIMIZED for RTX 4090 / A100 / H100 with CUDA acceleration.

Usage:
    python run_runpod.py                    # Resume from checkpoint (full mode)
    python run_runpod.py --mode medium      # Medium training
    python run_runpod.py --force            # Force restart from scratch
    python run_runpod.py --status           # Show checkpoint status

RTX 4090 Optimizations:
    - TensorFloat-32 (TF32) for Ampere/Ada GPUs
    - cuDNN benchmark mode for fastest kernels
    - Mixed precision (FP16/BF16) where supported
    - Optimized batch sizes for 24GB VRAM
    - XGBoost 2.0+ with device='cuda' and tree_method='hist'
    - CatBoost with task_type='GPU'
    - LightGBM CPU (GPU requires special build)
"""

import warnings
warnings.filterwarnings('ignore')
import os

# Suppress RDKit warnings
os.environ['RDKit_DEPRECATION_WARNINGS'] = '0'

# CUDA optimizations - set before importing torch
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async CUDA ops
os.environ['PYTORCH_ALLOC_CONF'] = 'max_split_size_mb:512'  # Better memory management (new name)

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

# GPU-Optimized Mode configurations - MAXIMIZED for RTX 4090
MODE_CONFIG = {
    'quick': {
        'catboost_iters': 500,
        'xgboost_iters': 500,
        'lightgbm_iters': 500,
        'chemprop_epochs': 15,
        'chemprop_batch_size': 256,  # RTX 4090 can handle large batches
        'n_bits': 2048,
        'description': 'Quick GPU test (~8-12 min)'
    },
    'medium': {
        'catboost_iters': 1000,
        'xgboost_iters': 1000,
        'lightgbm_iters': 1000,
        'chemprop_epochs': 30,
        'chemprop_batch_size': 512,
        'n_bits': 2048,
        'description': 'Balanced GPU (~20-30 min)'
    },
    'full': {
        'catboost_iters': 2000,
        'xgboost_iters': 2000,
        'lightgbm_iters': 2000,
        'chemprop_epochs': 50,
        'chemprop_batch_size': 512,
        'n_bits': 2048,
        'description': 'Full GPU accuracy (~40-60 min)'
    }
}

# ============================================================================
# GPU DETECTION & OPTIMIZATION
# ============================================================================

def setup_gpu():
    """Detect and configure GPU with maximum optimizations for RTX 4090"""
    gpu_info = {
        'cuda_available': False,
        'device_count': 0,
        'device_name': 'CPU',
        'memory_gb': 0,
        'compute_capability': (0, 0),
        'is_ampere_or_newer': False
    }

    try:
        import torch
        if torch.cuda.is_available():
            gpu_info['cuda_available'] = True
            gpu_info['device_count'] = torch.cuda.device_count()
            gpu_info['device_name'] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            gpu_info['memory_gb'] = props.total_memory / 1e9
            gpu_info['compute_capability'] = (props.major, props.minor)

            # Check if Ampere (8.x) or Ada (8.9) or newer
            gpu_info['is_ampere_or_newer'] = props.major >= 8

            # === RTX 4090 / Ampere+ Optimizations ===

            # Enable TensorFloat-32 (TF32) for massive speedup on Ampere/Ada
            # TF32 uses 19 bits (10 mantissa) vs FP32's 23 bits - 3x faster matmul
            if gpu_info['is_ampere_or_newer']:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print(f"[GPU] Enabled TF32 for {gpu_info['device_name']}")

            # Enable cuDNN autotuning - finds fastest convolution algorithms
            torch.backends.cudnn.benchmark = True

            # Enable flash attention if available (PyTorch 2.0+)
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)

            # Pre-allocate GPU memory to avoid fragmentation
            torch.cuda.empty_cache()

            print(f"[GPU] {gpu_info['device_name']}: {gpu_info['memory_gb']:.1f}GB, "
                  f"Compute {props.major}.{props.minor}")
    except ImportError:
        pass

    return gpu_info


# ============================================================================
# LOGGING SETUP
# ============================================================================

class ColorFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[36m', 'INFO': '\033[32m', 'WARNING': '\033[33m',
        'ERROR': '\033[31m', 'CRITICAL': '\033[35m', 'RESET': '\033[0m'
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        record.levelname = f"{color}{record.levelname:8}{reset}"
        return super().format(record)


def setup_logging(log_file: Path) -> logging.Logger:
    logger = logging.getLogger('admet_gpu')
    logger.setLevel(logging.DEBUG)
    logger.handlers = []

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(fh)

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
    mode: str
    stage: str
    current_target_idx: int = 0
    started_at: str = ""
    updated_at: str = ""
    completed_stages: List[str] = field(default_factory=list)
    completed_targets: Dict[str, List[str]] = field(default_factory=dict)
    results: Dict[str, Dict] = field(default_factory=dict)
    predictions: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)
    data_hash: str = ""
    stage_times: Dict[str, float] = field(default_factory=dict)
    gpu_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'Checkpoint':
        return cls(**d)


class CheckpointManager:
    def __init__(self, checkpoint_dir: Path, mode: str):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.checkpoint_file = checkpoint_dir / f"checkpoint_gpu_{mode}.json"
        self.features_file = checkpoint_dir / f"features_gpu_{mode}.pkl"
        self.checkpoint: Optional[Checkpoint] = None
        self.logger = logging.getLogger('admet_gpu')

    def exists(self) -> bool:
        return self.checkpoint_file.exists()

    def load(self) -> Optional[Checkpoint]:
        if not self.exists():
            return None
        try:
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)
            self.checkpoint = Checkpoint.from_dict(data)
            self.logger.info(f"Loaded checkpoint: stage={self.checkpoint.stage}")
            return self.checkpoint
        except Exception as e:
            self.logger.warning(f"Failed to load checkpoint: {e}")
            return None

    def save(self, checkpoint: Checkpoint):
        checkpoint.updated_at = datetime.now().isoformat()
        self.checkpoint = checkpoint
        temp_file = self.checkpoint_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(checkpoint.to_dict(), f, indent=2)
        temp_file.rename(self.checkpoint_file)

    def save_features(self, X_train: np.ndarray, X_test: np.ndarray):
        with open(self.features_file, 'wb') as f:
            pickle.dump({'X_train': X_train, 'X_test': X_test}, f)

    def load_features(self) -> Optional[tuple]:
        if not self.features_file.exists():
            return None
        try:
            with open(self.features_file, 'rb') as f:
                data = pickle.load(f)
            return data['X_train'], data['X_test']
        except:
            return None

    def clear(self):
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
        if self.features_file.exists():
            self.features_file.unlink()
        self.checkpoint = None

    def get_status(self) -> str:
        if not self.exists():
            return "No checkpoint found"
        cp = self.load()
        if not cp:
            return "Checkpoint corrupted"
        lines = [
            f"Mode: {cp.mode} (GPU)",
            f"Stage: {cp.stage}",
            f"GPU: {cp.gpu_info.get('device_name', 'Unknown')}",
            f"Started: {cp.started_at}",
            f"Updated: {cp.updated_at}",
            f"Completed: {', '.join(cp.completed_stages) or 'None'}",
        ]
        if cp.results:
            lines.append("\nResults:")
            for model, data in cp.results.items():
                if 'ma_rae' in data:
                    lines.append(f"  {model}: MA-RAE = {data['ma_rae']:.4f}")
        return "\n".join(lines)


# ============================================================================
# SIGNAL HANDLING
# ============================================================================

class GracefulInterrupt:
    def __init__(self, checkpoint_manager: CheckpointManager, checkpoint: Checkpoint):
        self.cm = checkpoint_manager
        self.checkpoint = checkpoint
        self.interrupted = False
        self.logger = logging.getLogger('admet_gpu')
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
        self.logger.warning("INTERRUPT - Saving checkpoint...")
        self.logger.warning("="*60)
        self.cm.save(self.checkpoint)
        sys.exit(0)

    def check(self):
        return self.interrupted


# ============================================================================
# FEATURE COMPUTATION
# ============================================================================

def compute_data_hash(train_df: pd.DataFrame, test_df: pd.DataFrame) -> str:
    h = hashlib.md5()
    h.update(str(len(train_df)).encode())
    h.update(str(len(test_df)).encode())
    h.update(train_df['SMILES'].iloc[0].encode())
    return h.hexdigest()[:12]


def compute_features(smiles_list: list, n_bits: int = 2048,
                    logger: logging.Logger = None) -> np.ndarray:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

    if logger:
        logger.info(f"Computing features: {len(smiles_list)} molecules, {n_bits}-bit FP")

    n_desc = 25
    features = []

    pbar = tqdm(smiles_list, desc="Features", unit="mol",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

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
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(y_true, y_pred)
    baseline = np.mean(np.abs(y_true - np.mean(y_true)))
    return mae / baseline if baseline > 0 else 1.0


# ============================================================================
# GPU-OPTIMIZED MODEL TRAINING - RTX 4090 MAXED
# ============================================================================

def train_catboost_gpu(X_train_all: np.ndarray, train_df: pd.DataFrame,
                       X_test: np.ndarray, config: dict, checkpoint: Checkpoint,
                       checkpoint_manager: CheckpointManager,
                       interrupt_handler: GracefulInterrupt,
                       logger: logging.Logger, gpu_info: dict) -> Dict:
    """Train CatBoost with RTX 4090 GPU optimizations"""
    from sklearn.model_selection import KFold
    from catboost import CatBoostRegressor

    use_gpu = gpu_info['cuda_available']
    task_type = 'GPU' if use_gpu else 'CPU'
    iters = config['catboost_iters']
    logger.info(f"Training CATBOOST ({task_type}, iterations={iters})")

    if 'catboost' not in checkpoint.predictions:
        checkpoint.predictions['catboost'] = {}
    if 'catboost' not in checkpoint.completed_targets:
        checkpoint.completed_targets['catboost'] = []

    results = {}
    start_time = time.time()

    completed = checkpoint.completed_targets.get('catboost', [])
    remaining_targets = [t for t in TARGETS if t not in completed]

    if not remaining_targets:
        logger.info("  All targets completed")
        return checkpoint.results.get('catboost', {})

    pbar = tqdm(remaining_targets, desc="CATBOOST", unit="target")

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

            # RTX 4090 optimized CatBoost params
            if use_gpu:
                model = CatBoostRegressor(
                    iterations=iters,
                    depth=10,  # Deeper trees on GPU
                    learning_rate=0.03,
                    l2_leaf_reg=3.0,
                    task_type='GPU',
                    devices='0',
                    bootstrap_type='Bernoulli',
                    subsample=0.8,
                    grow_policy='Lossguide',  # Faster on GPU
                    max_leaves=64,  # For Lossguide
                    verbose=False,
                    random_seed=42+fold,
                    early_stopping_rounds=100
                )
            else:
                model = CatBoostRegressor(
                    iterations=iters,
                    depth=8,
                    learning_rate=0.03,
                    l2_leaf_reg=3.0,
                    subsample=0.8,
                    colsample_bylevel=0.8,
                    verbose=False,
                    random_seed=42+fold,
                    early_stopping_rounds=100
                )

            model.fit(X_t[tr_idx], y_t[tr_idx],
                     eval_set=(X_t[va_idx], y_t[va_idx]), verbose=False)
            oof[va_idx] = model.predict(X_t[va_idx])
            test_preds.append(model.predict(X_test))

        if interrupt_handler.check():
            break

        rae = compute_rae(y_t, oof)
        results[target] = rae
        test_pred = np.clip(np.mean(test_preds, axis=0), *VALID_RANGES[target])
        checkpoint.predictions['catboost'][target] = test_pred.tolist()
        checkpoint.completed_targets['catboost'].append(target)
        checkpoint_manager.save(checkpoint)
        logger.debug(f"  {target}: RAE={rae:.4f}")

    if results:
        ma_rae = np.mean(list(results.values()))
        elapsed = time.time() - start_time
        checkpoint.results['catboost'] = {
            'ma_rae': ma_rae, 'per_target': results, 'time': elapsed
        }
        checkpoint.stage_times['catboost'] = elapsed
        logger.info(f"  CATBOOST MA-RAE: {ma_rae:.4f} ({elapsed:.0f}s)")

    gc.collect()
    if use_gpu:
        import torch
        torch.cuda.empty_cache()
    return checkpoint.results.get('catboost', {})


def train_xgboost_gpu(X_train_all: np.ndarray, train_df: pd.DataFrame,
                      X_test: np.ndarray, config: dict, checkpoint: Checkpoint,
                      checkpoint_manager: CheckpointManager,
                      interrupt_handler: GracefulInterrupt,
                      logger: logging.Logger, gpu_info: dict) -> Dict:
    """Train XGBoost with RTX 4090 GPU optimizations (XGBoost 2.0+ API)"""
    from sklearn.model_selection import KFold
    from xgboost import XGBRegressor

    use_gpu = gpu_info['cuda_available']
    iters = config['xgboost_iters']

    # XGBoost 2.0+ API: use device='cuda' instead of tree_method='gpu_hist'
    device = 'cuda' if use_gpu else 'cpu'

    logger.info(f"Training XGBOOST (device={device}, iterations={iters})")

    if 'xgboost' not in checkpoint.predictions:
        checkpoint.predictions['xgboost'] = {}
    if 'xgboost' not in checkpoint.completed_targets:
        checkpoint.completed_targets['xgboost'] = []

    results = {}
    start_time = time.time()

    completed = checkpoint.completed_targets.get('xgboost', [])
    remaining_targets = [t for t in TARGETS if t not in completed]

    if not remaining_targets:
        logger.info("  All targets completed")
        return checkpoint.results.get('xgboost', {})

    pbar = tqdm(remaining_targets, desc="XGBOOST", unit="target")

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

            # XGBoost 2.0+ API: device='cuda' for GPU, tree_method='hist' is default
            model = XGBRegressor(
                n_estimators=iters,
                max_depth=10 if use_gpu else 8,  # Deeper on GPU
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                colsample_bylevel=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                tree_method='hist',  # 'hist' works for both CPU and GPU in 2.0+
                device=device,       # 'cuda' or 'cpu'
                max_bin=256,
                random_state=42+fold,
                verbosity=0,
                n_jobs=-1 if device == 'cpu' else 1  # n_jobs only for CPU
            )
            model.fit(X_t[tr_idx], y_t[tr_idx],
                     eval_set=[(X_t[va_idx], y_t[va_idx])], verbose=False)
            oof[va_idx] = model.predict(X_t[va_idx])
            test_preds.append(model.predict(X_test))

        if interrupt_handler.check():
            break

        rae = compute_rae(y_t, oof)
        results[target] = rae
        test_pred = np.clip(np.mean(test_preds, axis=0), *VALID_RANGES[target])
        checkpoint.predictions['xgboost'][target] = test_pred.tolist()
        checkpoint.completed_targets['xgboost'].append(target)
        checkpoint_manager.save(checkpoint)
        logger.debug(f"  {target}: RAE={rae:.4f}")

    if results:
        ma_rae = np.mean(list(results.values()))
        elapsed = time.time() - start_time
        checkpoint.results['xgboost'] = {
            'ma_rae': ma_rae, 'per_target': results, 'time': elapsed
        }
        checkpoint.stage_times['xgboost'] = elapsed
        logger.info(f"  XGBOOST MA-RAE: {ma_rae:.4f} ({elapsed:.0f}s)")

    gc.collect()
    if use_gpu:
        import torch
        torch.cuda.empty_cache()
    return checkpoint.results.get('xgboost', {})


def train_lightgbm_gpu(X_train_all: np.ndarray, train_df: pd.DataFrame,
                       X_test: np.ndarray, config: dict, checkpoint: Checkpoint,
                       checkpoint_manager: CheckpointManager,
                       interrupt_handler: GracefulInterrupt,
                       logger: logging.Logger, gpu_info: dict) -> Dict:
    """Train LightGBM (CPU - GPU requires special build)"""
    from sklearn.model_selection import KFold
    from lightgbm import LGBMRegressor

    # Note: LightGBM GPU requires special compilation, use CPU for reliability
    iters = config['lightgbm_iters']
    logger.info(f"Training LIGHTGBM (cpu, iterations={iters})")

    if 'lightgbm' not in checkpoint.predictions:
        checkpoint.predictions['lightgbm'] = {}
    if 'lightgbm' not in checkpoint.completed_targets:
        checkpoint.completed_targets['lightgbm'] = []

    results = {}
    start_time = time.time()

    completed = checkpoint.completed_targets.get('lightgbm', [])
    remaining_targets = [t for t in TARGETS if t not in completed]

    if not remaining_targets:
        logger.info("  All targets completed")
        return checkpoint.results.get('lightgbm', {})

    pbar = tqdm(remaining_targets, desc="LIGHTGBM", unit="target")

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

            # LightGBM CPU params (GPU requires special build)
            model = LGBMRegressor(
                n_estimators=iters,
                max_depth=8,
                num_leaves=63,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42+fold,
                verbosity=-1,
                n_jobs=-1,
                force_col_wise=True,  # Better for wide datasets
            )
            model.fit(X_t[tr_idx], y_t[tr_idx],
                     eval_set=[(X_t[va_idx], y_t[va_idx])])
            oof[va_idx] = model.predict(X_t[va_idx])
            test_preds.append(model.predict(X_test))

        if interrupt_handler.check():
            break

        rae = compute_rae(y_t, oof)
        results[target] = rae
        test_pred = np.clip(np.mean(test_preds, axis=0), *VALID_RANGES[target])
        checkpoint.predictions['lightgbm'][target] = test_pred.tolist()
        checkpoint.completed_targets['lightgbm'].append(target)
        checkpoint_manager.save(checkpoint)
        logger.debug(f"  {target}: RAE={rae:.4f}")

    if results:
        ma_rae = np.mean(list(results.values()))
        elapsed = time.time() - start_time
        checkpoint.results['lightgbm'] = {
            'ma_rae': ma_rae, 'per_target': results, 'time': elapsed
        }
        checkpoint.stage_times['lightgbm'] = elapsed
        logger.info(f"  LIGHTGBM MA-RAE: {ma_rae:.4f} ({elapsed:.0f}s)")

    gc.collect()
    return checkpoint.results.get('lightgbm', {})


def train_chemprop_gpu(train_df: pd.DataFrame, test_df: pd.DataFrame,
                       config: dict, checkpoint: Checkpoint,
                       checkpoint_manager: CheckpointManager,
                       interrupt_handler: GracefulInterrupt,
                       logger: logging.Logger, gpu_info: dict) -> Dict:
    """Train Chemprop D-MPNN with RTX 4090 CUDA optimizations"""
    try:
        import torch
        import lightning as pl
        from chemprop import data, models, nn

        logging.getLogger('lightning.pytorch').setLevel(logging.ERROR)
        logging.getLogger('lightning.fabric').setLevel(logging.ERROR)

        use_gpu = gpu_info['cuda_available']
        device = "cuda" if use_gpu else "cpu"
        batch_size = config['chemprop_batch_size']
        epochs = config['chemprop_epochs']

        logger.info(f"Training CHEMPROP ({device}, epochs={epochs}, batch={batch_size})")

        if 'chemprop' not in checkpoint.predictions:
            checkpoint.predictions['chemprop'] = {}
        if 'chemprop' not in checkpoint.completed_targets:
            checkpoint.completed_targets['chemprop'] = []

        completed = checkpoint.completed_targets.get('chemprop', [])
        remaining_targets = [t for t in TARGETS if t not in completed]

        if not remaining_targets:
            logger.info("  All targets completed")
            return checkpoint.results.get('chemprop', {})

        start_time = time.time()

        pbar = tqdm(remaining_targets, desc="CHEMPROP", unit="target")

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

            # RTX 4090 optimized trainer
            trainer_kwargs = {
                'max_epochs': epochs,
                'accelerator': 'gpu' if use_gpu else 'cpu',
                'devices': 1,
                'enable_progress_bar': False,
                'logger': False,
                'enable_checkpointing': False,
            }

            # Use mixed precision on GPU for speed
            if use_gpu:
                # BF16 is better on Ampere+, FP16 otherwise
                if gpu_info.get('is_ampere_or_newer', False):
                    trainer_kwargs['precision'] = 'bf16-mixed'
                else:
                    trainer_kwargs['precision'] = '16-mixed'

            trainer = pl.Trainer(**trainer_kwargs)

            # Use more workers for data loading on GPU
            num_workers = 4 if use_gpu else 0
            train_loader = data.build_dataloader(train_dset, batch_size=batch_size,
                                                  shuffle=True, num_workers=num_workers)
            val_loader = data.build_dataloader(val_dset, batch_size=batch_size,
                                                num_workers=num_workers)
            test_loader = data.build_dataloader(test_dset, batch_size=batch_size,
                                                 num_workers=num_workers)

            trainer.fit(mpnn, train_loader, val_loader)
            preds = trainer.predict(mpnn, test_loader)
            test_pred = np.concatenate([p.cpu().numpy() for p in preds]).flatten()

            checkpoint.predictions['chemprop'][target] = np.clip(
                test_pred, *VALID_RANGES[target]
            ).tolist()
            checkpoint.completed_targets['chemprop'].append(target)
            checkpoint_manager.save(checkpoint)
            logger.debug(f"  {target}: completed")

            gc.collect()
            if use_gpu:
                torch.cuda.empty_cache()

        elapsed = time.time() - start_time
        checkpoint.results['chemprop'] = {'time': elapsed}
        checkpoint.stage_times['chemprop'] = elapsed
        logger.info(f"  CHEMPROP completed ({elapsed:.0f}s)")

        return checkpoint.results.get('chemprop', {})

    except Exception as e:
        logger.warning(f"Chemprop failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return {}


# ============================================================================
# ENSEMBLE & SUBMISSION
# ============================================================================

def create_ensembles(checkpoint: Checkpoint, test_df: pd.DataFrame,
                     mode: str, logger: logging.Logger):
    logger.info("Creating ensembles...")

    available_models = [m for m in ['catboost', 'xgboost', 'lightgbm']
                        if m in checkpoint.predictions and checkpoint.predictions[m]]

    if len(available_models) < 2:
        logger.warning("Not enough models for ensemble")
        return

    # Equal weights
    ens_equal = pd.DataFrame({'Molecule Name': test_df['Molecule Name']})
    for t in TARGETS:
        preds = [np.array(checkpoint.predictions[m][t]) for m in available_models]
        ens_equal[t] = np.mean(preds, axis=0)
    ens_equal.to_csv(SUB_DIR / f"ensemble_equal_gpu_{mode}.csv", index=False)
    logger.info(f"  Saved: ensemble_equal_gpu_{mode}.csv")

    # Weighted by inverse RAE
    if all(m in checkpoint.results and 'per_target' in checkpoint.results[m]
           for m in available_models):
        ens_weighted = pd.DataFrame({'Molecule Name': test_df['Molecule Name']})
        for t in TARGETS:
            raes = [checkpoint.results[m]['per_target'][t] for m in available_models]
            weights = np.array([1/r for r in raes])
            weights = weights / weights.sum()
            preds = [np.array(checkpoint.predictions[m][t]) for m in available_models]
            ens_weighted[t] = sum(w * p for w, p in zip(weights, preds))
        ens_weighted.to_csv(SUB_DIR / f"ensemble_weighted_gpu_{mode}.csv", index=False)
        logger.info(f"  Saved: ensemble_weighted_gpu_{mode}.csv")

    # With Chemprop
    if 'chemprop' in checkpoint.predictions and checkpoint.predictions['chemprop']:
        all_models = available_models + ['chemprop']
        ens_all = pd.DataFrame({'Molecule Name': test_df['Molecule Name']})
        for t in TARGETS:
            preds = [np.array(checkpoint.predictions[m][t]) for m in all_models
                     if t in checkpoint.predictions.get(m, {})]
            if preds:
                ens_all[t] = np.mean(preds, axis=0)
        ens_all.to_csv(SUB_DIR / f"ensemble_all_gpu_{mode}.csv", index=False)
        logger.info(f"  Saved: ensemble_all_gpu_{mode}.csv")


def save_individual_submissions(checkpoint: Checkpoint, test_df: pd.DataFrame,
                                mode: str, logger: logging.Logger):
    for model_name, preds in checkpoint.predictions.items():
        if not preds:
            continue
        sub = pd.DataFrame({'Molecule Name': test_df['Molecule Name']})
        for t in TARGETS:
            if t in preds:
                sub[t] = preds[t]
        out_path = SUB_DIR / f"{model_name}_gpu_{mode}.csv"
        sub.to_csv(out_path, index=False)
        logger.info(f"  Saved: {out_path.name}")


def print_summary(checkpoint: Checkpoint, gpu_info: dict, logger: logging.Logger):
    print("\n" + "="*70)
    print("FINAL SUMMARY (GPU-OPTIMIZED)")
    print("="*70)
    print(f"\nGPU: {gpu_info.get('device_name', 'N/A')} ({gpu_info.get('memory_gb', 0):.1f} GB)")
    if gpu_info.get('is_ampere_or_newer'):
        print("     TF32 enabled, BF16 mixed precision")

    print("\n" + "-"*50)
    print("MODEL RESULTS (MA-RAE)")
    print("-"*50)

    best_model, best_rae = None, float('inf')
    for model in ['catboost', 'xgboost', 'lightgbm']:
        if model in checkpoint.results and 'ma_rae' in checkpoint.results[model]:
            ma_rae = checkpoint.results[model]['ma_rae']
            t = checkpoint.results[model].get('time', 0)
            print(f"  {model.upper():<12} MA-RAE: {ma_rae:.4f}  ({t:.0f}s)")
            if ma_rae < best_rae:
                best_rae = ma_rae
                best_model = model

    print("\n" + "="*50)
    if best_model:
        print(f"BEST MODEL: {best_model.upper()} = {best_rae:.4f}")
    print(f"TARGET:     0.5593 (leader 'pebble')")
    if best_model:
        print(f"GAP:        {best_rae - 0.5593:+.4f}")
    print("="*50)

    total_time = sum(checkpoint.stage_times.values())
    print(f"\nTotal time: {timedelta(seconds=int(total_time))}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='ADMET - RunPod GPU Training (RTX 4090 Optimized)')
    parser.add_argument('--mode', choices=['quick', 'medium', 'full'], default='full')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--skip-chemprop', action='store_true')
    args = parser.parse_args()

    SUB_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOG_DIR / f"training_gpu_{args.mode}_{timestamp}.log"
    logger = setup_logging(log_file)

    # GPU detection with optimizations
    gpu_info = setup_gpu()

    cm = CheckpointManager(CHECKPOINT_DIR, args.mode)

    if args.status:
        print(cm.get_status())
        return

    if args.force:
        logger.info("Force flag - clearing checkpoints")
        cm.clear()

    checkpoint = cm.load()
    config = MODE_CONFIG[args.mode]

    if checkpoint:
        logger.info(f"Resuming from checkpoint: stage={checkpoint.stage}")
    else:
        checkpoint = Checkpoint(
            mode=args.mode,
            stage=TrainingStage.INIT.value,
            started_at=datetime.now().isoformat(),
            gpu_info=gpu_info
        )

    print("="*70)
    print("ADMET CHALLENGE - RUNPOD GPU TRAINING (RTX 4090 OPTIMIZED)")
    print(f"Mode: {args.mode} ({config['description']})")
    if gpu_info['cuda_available']:
        print(f"GPU: {gpu_info['device_name']} ({gpu_info['memory_gb']:.1f} GB)")
        if gpu_info['is_ampere_or_newer']:
            print("     TF32 + BF16 mixed precision enabled")
    else:
        print("GPU: Not available (CPU mode)")
    print(f"Log: {log_file}")
    print("="*70)

    if not gpu_info['cuda_available']:
        logger.warning("CUDA not available - running in CPU mode")

    with GracefulInterrupt(cm, checkpoint) as interrupt:
        logger.info("Loading data...")
        train_df = pd.read_csv(DATA_DIR / "train.csv")
        test_df = pd.read_csv(DATA_DIR / "test_blinded.csv")
        logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")

        data_hash = compute_data_hash(train_df, test_df)

        # Features
        if checkpoint.stage == TrainingStage.INIT.value or checkpoint.data_hash != data_hash:
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
                all_smiles = pd.concat([train_df['SMILES'], test_df['SMILES']]).values
                X_all = compute_features(all_smiles, n_bits=config['n_bits'], logger=logger)
                X_train_all, X_test = X_all[:len(train_df)], X_all[len(train_df):]
                cm.save_features(X_train_all, X_test)

        logger.info(f"Features: {X_train_all.shape}")

        # CatBoost
        if TrainingStage.CATBOOST.value not in checkpoint.completed_stages:
            checkpoint.stage = TrainingStage.CATBOOST.value
            cm.save(checkpoint)
            train_catboost_gpu(X_train_all, train_df, X_test, config,
                              checkpoint, cm, interrupt, logger, gpu_info)
            if not interrupt.check():
                checkpoint.completed_stages.append(TrainingStage.CATBOOST.value)
                cm.save(checkpoint)

        if interrupt.check():
            return

        # XGBoost
        if TrainingStage.XGBOOST.value not in checkpoint.completed_stages:
            checkpoint.stage = TrainingStage.XGBOOST.value
            cm.save(checkpoint)
            train_xgboost_gpu(X_train_all, train_df, X_test, config,
                             checkpoint, cm, interrupt, logger, gpu_info)
            if not interrupt.check():
                checkpoint.completed_stages.append(TrainingStage.XGBOOST.value)
                cm.save(checkpoint)

        if interrupt.check():
            return

        # LightGBM
        if TrainingStage.LIGHTGBM.value not in checkpoint.completed_stages:
            checkpoint.stage = TrainingStage.LIGHTGBM.value
            cm.save(checkpoint)
            train_lightgbm_gpu(X_train_all, train_df, X_test, config,
                              checkpoint, cm, interrupt, logger, gpu_info)
            if not interrupt.check():
                checkpoint.completed_stages.append(TrainingStage.LIGHTGBM.value)
                cm.save(checkpoint)

        if interrupt.check():
            return

        # Chemprop
        if not args.skip_chemprop and TrainingStage.CHEMPROP.value not in checkpoint.completed_stages:
            checkpoint.stage = TrainingStage.CHEMPROP.value
            cm.save(checkpoint)
            train_chemprop_gpu(train_df, test_df, config, checkpoint, cm, interrupt, logger, gpu_info)
            if not interrupt.check():
                checkpoint.completed_stages.append(TrainingStage.CHEMPROP.value)
                cm.save(checkpoint)

        if interrupt.check():
            return

        # Ensembles
        checkpoint.stage = TrainingStage.ENSEMBLE.value
        cm.save(checkpoint)

        logger.info("Saving submissions...")
        save_individual_submissions(checkpoint, test_df, args.mode, logger)
        create_ensembles(checkpoint, test_df, args.mode, logger)

        checkpoint.stage = TrainingStage.COMPLETE.value
        checkpoint.completed_stages.append(TrainingStage.COMPLETE.value)
        cm.save(checkpoint)

        # Save results
        results_file = LOG_DIR / f"results_gpu_{args.mode}_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'mode': args.mode,
                'gpu': gpu_info,
                'config': config,
                'results': checkpoint.results,
                'stage_times': checkpoint.stage_times,
                'completed_at': datetime.now().isoformat()
            }, f, indent=2)

        print_summary(checkpoint, gpu_info, logger)

        print(f"\nSubmissions in: {SUB_DIR}/")
        for f in sorted(SUB_DIR.glob(f"*_gpu_{args.mode}.csv")):
            print(f"  {f.name}")

        print(f"\nComplete! Log: {log_file}")


if __name__ == "__main__":
    main()
