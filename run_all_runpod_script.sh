#!/bin/bash
#===============================================================================
# ADMET Challenge - Meta Orchestrator Script
# Runs all models sequentially with detailed logging
#
# Usage:
#   chmod +x run_all_runpod_script.sh
#   ./run_all_runpod_script.sh
#
# For MacBook M1/M3: ./run_all_runpod_script.sh --local
# For RunPod GPU:    ./run_all_runpod_script.sh --gpu
#===============================================================================

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="${LOG_DIR}/run_${TIMESTAMP}.log"

# Parse arguments
MODE="local"
if [[ "$1" == "--gpu" ]]; then
    MODE="gpu"
elif [[ "$1" == "--local" ]]; then
    MODE="local"
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

#===============================================================================
# Helper Functions
#===============================================================================

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo -e "$msg" | tee -a "$MAIN_LOG"
}

log_section() {
    echo "" | tee -a "$MAIN_LOG"
    echo "================================================================================" | tee -a "$MAIN_LOG"
    echo -e "${BLUE}$1${NC}" | tee -a "$MAIN_LOG"
    echo "================================================================================" | tee -a "$MAIN_LOG"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$MAIN_LOG"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$MAIN_LOG"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$MAIN_LOG"
}

#===============================================================================
# Setup
#===============================================================================

mkdir -p "$LOG_DIR"
mkdir -p "${SCRIPT_DIR}/submissions"
mkdir -p "${SCRIPT_DIR}/models"

log_section "ADMET CHALLENGE - TRAINING ORCHESTRATOR"
log "Started at: $(date)"
log "Mode: $MODE"
log "Script directory: $SCRIPT_DIR"
log "Log file: $MAIN_LOG"

# System info
log_section "SYSTEM INFORMATION"
log "OS: $(uname -s) $(uname -r)"
log "Architecture: $(uname -m)"
log "Python: $(python3 --version 2>&1)"

if [[ "$MODE" == "gpu" ]]; then
    if command -v nvidia-smi &> /dev/null; then
        log "GPU Info:"
        nvidia-smi --query-gpu=name,memory.total --format=csv | tee -a "$MAIN_LOG"
    fi
else
    log "Running on Apple Silicon (MPS acceleration available)"
fi

#===============================================================================
# Install Requirements
#===============================================================================

log_section "INSTALLING REQUIREMENTS"

# Check if uv is available
if command -v uv &> /dev/null; then
    log "Using uv for package installation..."

    # Create/activate virtual environment
    if [[ ! -d "${SCRIPT_DIR}/admet_challenge" ]]; then
        log "Creating virtual environment..."
        uv venv "${SCRIPT_DIR}/admet_challenge"
    fi

    source "${SCRIPT_DIR}/admet_challenge/bin/activate"

    log "Installing packages..."
    uv pip install -r "${SCRIPT_DIR}/requirements_runpod.txt" 2>&1 | tee -a "$MAIN_LOG"

    log_success "Requirements installed successfully"
else
    log_warning "uv not found, using pip..."
    pip install -r "${SCRIPT_DIR}/requirements_runpod.txt" 2>&1 | tee -a "$MAIN_LOG"
fi

# Verify installations
log "Verifying package installations..."
python3 -c "
import sys
packages = ['numpy', 'pandas', 'sklearn', 'catboost', 'xgboost', 'lightgbm', 'rdkit', 'torch']
for pkg in packages:
    try:
        __import__(pkg)
        print(f'  {pkg}: OK')
    except ImportError as e:
        print(f'  {pkg}: MISSING - {e}')
        sys.exit(1)
" 2>&1 | tee -a "$MAIN_LOG"

#===============================================================================
# Create Individual Model Scripts
#===============================================================================

log_section "CREATING MODEL TRAINING SCRIPTS"

# Create individual model training script
cat > "${SCRIPT_DIR}/src/train_single_model.py" << 'PYTHON_SCRIPT'
#!/usr/bin/env python3
"""
Single model trainer with detailed metrics logging
Usage: python train_single_model.py --model catboost --epochs medium
"""
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['RDKit_DEPRECATION_WARNINGS'] = '0'

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import json
import time
import gc
from datetime import datetime

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr, pearsonr

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
MODEL_DIR = BASE_DIR / "models"
SUB_DIR = BASE_DIR / "submissions"
LOG_DIR = BASE_DIR / "logs"

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

# Epoch configurations
EPOCH_CONFIG = {
    'quick': {'catboost': 200, 'xgboost': 200, 'lightgbm': 200, 'chemprop': 5},
    'medium': {'catboost': 600, 'xgboost': 500, 'lightgbm': 500, 'chemprop': 20},
    'full': {'catboost': 1200, 'xgboost': 1000, 'lightgbm': 1000, 'chemprop': 50},
}

def compute_features(smiles_list, n_bits=2048):
    """Compute molecular features"""
    print(f"Computing features for {len(smiles_list)} molecules...")
    features = []
    n_desc = 25
    for i, smi in enumerate(smiles_list):
        if (i+1) % 2000 == 0:
            print(f"  Progress: {i+1}/{len(smiles_list)}")
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

def compute_metrics(y_true, y_pred):
    """Compute comprehensive metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    spearman = spearmanr(y_true, y_pred)[0]
    pearson = pearsonr(y_true, y_pred)[0]

    # RAE (Relative Absolute Error) - competition metric
    baseline_mae = np.mean(np.abs(y_true - np.mean(y_true)))
    rae = mae / baseline_mae if baseline_mae > 0 else 1.0

    return {
        'MAE': mae, 'RMSE': rmse, 'R2': r2,
        'Spearman': spearman, 'Pearson': pearson, 'RAE': rae
    }

def train_catboost(X_train, y_train, X_test, n_iters, n_folds=5):
    from catboost import CatBoostRegressor

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof = np.zeros(len(y_train))
    test_preds = []
    fold_metrics = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train)):
        model = CatBoostRegressor(
            iterations=n_iters, depth=7, learning_rate=0.03,
            subsample=0.8, colsample_bylevel=0.8, l2_leaf_reg=3.0,
            verbose=False, random_seed=42+fold, early_stopping_rounds=50
        )
        model.fit(X_train[tr_idx], y_train[tr_idx],
                  eval_set=(X_train[va_idx], y_train[va_idx]), verbose=False)

        oof[va_idx] = model.predict(X_train[va_idx])
        test_preds.append(model.predict(X_test))

        fold_mae = mean_absolute_error(y_train[va_idx], oof[va_idx])
        fold_metrics.append({'fold': fold+1, 'val_mae': fold_mae})
        print(f"    Fold {fold+1}: MAE = {fold_mae:.4f}")

    return oof, np.mean(test_preds, axis=0), fold_metrics

def train_xgboost(X_train, y_train, X_test, n_iters, n_folds=5):
    from xgboost import XGBRegressor

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof = np.zeros(len(y_train))
    test_preds = []
    fold_metrics = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train)):
        model = XGBRegressor(
            n_estimators=n_iters, max_depth=7, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42+fold, verbosity=0, n_jobs=-1,
            early_stopping_rounds=50
        )
        model.fit(X_train[tr_idx], y_train[tr_idx],
                  eval_set=[(X_train[va_idx], y_train[va_idx])], verbose=False)

        oof[va_idx] = model.predict(X_train[va_idx])
        test_preds.append(model.predict(X_test))

        fold_mae = mean_absolute_error(y_train[va_idx], oof[va_idx])
        fold_metrics.append({'fold': fold+1, 'val_mae': fold_mae})
        print(f"    Fold {fold+1}: MAE = {fold_mae:.4f}")

    return oof, np.mean(test_preds, axis=0), fold_metrics

def train_lightgbm(X_train, y_train, X_test, n_iters, n_folds=5):
    from lightgbm import LGBMRegressor, early_stopping

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof = np.zeros(len(y_train))
    test_preds = []
    fold_metrics = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train)):
        model = LGBMRegressor(
            n_estimators=n_iters, max_depth=7, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42+fold, verbosity=-1, n_jobs=-1
        )
        model.fit(X_train[tr_idx], y_train[tr_idx],
                  eval_set=[(X_train[va_idx], y_train[va_idx])],
                  callbacks=[early_stopping(50, verbose=False)])

        oof[va_idx] = model.predict(X_train[va_idx])
        test_preds.append(model.predict(X_test))

        fold_mae = mean_absolute_error(y_train[va_idx], oof[va_idx])
        fold_metrics.append({'fold': fold+1, 'val_mae': fold_mae})
        print(f"    Fold {fold+1}: MAE = {fold_mae:.4f}")

    return oof, np.mean(test_preds, axis=0), fold_metrics

def train_chemprop(train_df, test_df, target, n_epochs):
    """Train Chemprop D-MPNN"""
    try:
        import chemprop
        from chemprop import data, models, nn
        import torch
        import lightning as pl

        # Detect device
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        print(f"    Using device: {device}")

        # Prepare data
        mask = train_df[target].notna()
        train_smiles = train_df.loc[mask, 'SMILES'].tolist()
        train_y = train_df.loc[mask, target].values.reshape(-1, 1)
        test_smiles = test_df['SMILES'].tolist()

        train_data = [data.MoleculeDatapoint.from_smi(smi, y)
                      for smi, y in zip(train_smiles, train_y)]
        test_data = [data.MoleculeDatapoint.from_smi(smi) for smi in test_smiles]

        train_dset = data.MoleculeDataset(train_data)
        test_dset = data.MoleculeDataset(test_data)

        # 85/15 split
        n_train = int(len(train_dset) * 0.85)
        train_dset, val_dset = data.split_data_by_indices(
            train_dset, [list(range(n_train)), list(range(n_train, len(train_dset)))]
        )

        # Model
        mp = nn.BondMessagePassing()
        agg = nn.MeanAggregation()
        ffn = nn.RegressionFFN()
        mpnn = models.MPNN(mp, agg, ffn)

        # Trainer
        trainer = pl.Trainer(
            max_epochs=n_epochs,
            accelerator="mps" if device == "mps" else ("gpu" if device == "cuda" else "cpu"),
            enable_progress_bar=True,
            logger=False,
            enable_checkpointing=False
        )

        train_loader = data.build_dataloader(train_dset, batch_size=64, shuffle=True)
        val_loader = data.build_dataloader(val_dset, batch_size=64)
        test_loader = data.build_dataloader(test_dset, batch_size=64)

        trainer.fit(mpnn, train_loader, val_loader)

        preds = trainer.predict(mpnn, test_loader)
        test_pred = np.concatenate([p.numpy() for p in preds]).flatten()

        return test_pred, None  # No OOF for chemprop simplified version

    except Exception as e:
        print(f"    Chemprop error: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        choices=['catboost', 'xgboost', 'lightgbm', 'chemprop', 'all'])
    parser.add_argument('--epochs', type=str, default='medium',
                        choices=['quick', 'medium', 'full'])
    args = parser.parse_args()

    print("="*70)
    print(f"ADMET CHALLENGE - {args.model.upper()} TRAINING")
    print(f"Epoch mode: {args.epochs}")
    print(f"Started: {datetime.now()}")
    print("="*70)

    # Load data
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test_blinded.csv")
    print(f"\nData: Train={len(train_df)}, Test={len(test_df)}")

    # Get iterations
    n_iters = EPOCH_CONFIG[args.epochs][args.model if args.model != 'all' else 'catboost']
    print(f"Iterations/Epochs: {n_iters}")

    # Compute features (skip for chemprop)
    if args.model != 'chemprop':
        all_smiles = pd.concat([train_df['SMILES'], test_df['SMILES']]).values
        X_all = compute_features(all_smiles)
        X_train_all, X_test = X_all[:len(train_df)], X_all[len(train_df):]
        print(f"Features: {X_train_all.shape}")

    # Results storage
    all_results = {}
    all_predictions = {}

    # Training loop
    for target in TARGETS:
        print(f"\n{'='*60}")
        print(f"TARGET: {target}")
        print(f"{'='*60}")

        start_time = time.time()

        if args.model == 'chemprop':
            n_epochs = EPOCH_CONFIG[args.epochs]['chemprop']
            test_pred, _ = train_chemprop(train_df, test_df, target, n_epochs)
            if test_pred is not None:
                all_predictions[target] = np.clip(test_pred, *VALID_RANGES[target])
                all_results[target] = {'time': time.time() - start_time}
        else:
            y = train_df[target].values
            mask = ~np.isnan(y)
            X_t, y_t = X_train_all[mask], y[mask]
            print(f"  Samples: {len(y_t)}")

            if args.model == 'catboost':
                oof, test_pred, fold_metrics = train_catboost(X_t, y_t, X_test, n_iters)
            elif args.model == 'xgboost':
                oof, test_pred, fold_metrics = train_xgboost(X_t, y_t, X_test, n_iters)
            elif args.model == 'lightgbm':
                oof, test_pred, fold_metrics = train_lightgbm(X_t, y_t, X_test, n_iters)

            # Compute metrics
            metrics = compute_metrics(y_t, oof)
            metrics['time'] = time.time() - start_time
            metrics['fold_metrics'] = fold_metrics
            all_results[target] = metrics
            all_predictions[target] = np.clip(test_pred, *VALID_RANGES[target])

            print(f"\n  METRICS for {target}:")
            print(f"    MAE:      {metrics['MAE']:.4f}")
            print(f"    RMSE:     {metrics['RMSE']:.4f}")
            print(f"    R2:       {metrics['R2']:.4f}")
            print(f"    Spearman: {metrics['Spearman']:.4f}")
            print(f"    Pearson:  {metrics['Pearson']:.4f}")
            print(f"    RAE:      {metrics['RAE']:.4f}  <-- Competition metric")
            print(f"    Time:     {metrics['time']:.1f}s")

        gc.collect()

    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    if args.model != 'chemprop':
        raes = [r['RAE'] for r in all_results.values() if 'RAE' in r]
        ma_rae = np.mean(raes)

        print(f"\nPer-endpoint RAE:")
        for target, metrics in all_results.items():
            if 'RAE' in metrics:
                print(f"  {target}: {metrics['RAE']:.4f}")

        print(f"\n{'='*40}")
        print(f"MA-RAE: {ma_rae:.4f}")
        print(f"Target: 0.5593 (leader 'pebble')")
        print(f"Gap:    {ma_rae - 0.5593:.4f}")
        print(f"{'='*40}")

    # Save submission
    if all_predictions:
        sub = pd.DataFrame({'Molecule Name': test_df['Molecule Name']})
        for t in TARGETS:
            if t in all_predictions:
                sub[t] = all_predictions[t]

        out_path = SUB_DIR / f"{args.model}_{args.epochs}.csv"
        sub.to_csv(out_path, index=False)
        print(f"\nSubmission saved: {out_path}")

    # Save detailed results
    results_path = LOG_DIR / f"{args.model}_{args.epochs}_results.json"

    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    json_results = {k: {kk: convert(vv) for kk, vv in v.items()}
                    for k, v in all_results.items()}

    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"Results saved: {results_path}")

    print(f"\nCompleted: {datetime.now()}")

if __name__ == "__main__":
    main()
PYTHON_SCRIPT

log_success "Created train_single_model.py"

#===============================================================================
# Run Models
#===============================================================================

EPOCH_MODE="${2:-medium}"  # Default to medium

log_section "TRAINING MODELS (Mode: $EPOCH_MODE)"

# Activate environment
source "${SCRIPT_DIR}/admet_challenge/bin/activate"

# Track overall results
declare -A MODEL_RESULTS

# Function to run a single model
run_model() {
    local model=$1
    local log_file="${LOG_DIR}/${model}_${TIMESTAMP}.log"

    log "Starting $model training..."
    log "Log file: $log_file"

    local start_time=$(date +%s)

    python3 "${SCRIPT_DIR}/src/train_single_model.py" \
        --model "$model" \
        --epochs "$EPOCH_MODE" \
        2>&1 | tee "$log_file"

    local exit_code=$?
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    if [ $exit_code -eq 0 ]; then
        log_success "$model completed in ${duration}s"
        MODEL_RESULTS[$model]="SUCCESS (${duration}s)"

        # Extract MA-RAE from log
        local ma_rae=$(grep "MA-RAE:" "$log_file" | tail -1 | awk '{print $2}')
        if [ -n "$ma_rae" ]; then
            log "$model MA-RAE: $ma_rae"
            MODEL_RESULTS[$model]="SUCCESS - MA-RAE: $ma_rae (${duration}s)"
        fi
    else
        log_error "$model failed with exit code $exit_code"
        MODEL_RESULTS[$model]="FAILED"
    fi

    echo ""
}

# Run each model
log_section "1/4 - CATBOOST"
run_model "catboost"

log_section "2/4 - XGBOOST"
run_model "xgboost"

log_section "3/4 - LIGHTGBM"
run_model "lightgbm"

log_section "4/4 - CHEMPROP (D-MPNN)"
run_model "chemprop"

#===============================================================================
# Create Ensemble
#===============================================================================

log_section "CREATING ENSEMBLE"

python3 << 'ENSEMBLE_SCRIPT'
import pandas as pd
import numpy as np
from pathlib import Path
import json

BASE_DIR = Path(".")
SUB_DIR = BASE_DIR / "submissions"
LOG_DIR = BASE_DIR / "logs"

TARGETS = ['LogD', 'KSOL', 'HLM CLint', 'MLM CLint',
           'Caco-2 Permeability Papp A>B', 'Caco-2 Permeability Efflux',
           'MPPB', 'MBPB', 'MGMB']

# Find available submissions
epoch_mode = "medium"  # Match the training mode
models = ['catboost', 'xgboost', 'lightgbm']
submissions = {}
weights = {}

for model in models:
    path = SUB_DIR / f"{model}_{epoch_mode}.csv"
    if path.exists():
        submissions[model] = pd.read_csv(path)
        print(f"Loaded: {path}")

        # Try to get RAE for weighting
        results_path = LOG_DIR / f"{model}_{epoch_mode}_results.json"
        if results_path.exists():
            with open(results_path) as f:
                results = json.load(f)
            avg_rae = np.mean([v.get('RAE', 1.0) for v in results.values()])
            weights[model] = 1.0 / avg_rae
        else:
            weights[model] = 1.0

if len(submissions) < 2:
    print("Not enough submissions for ensemble")
    exit(0)

# Normalize weights
total = sum(weights.values())
weights = {k: v/total for k, v in weights.items()}
print(f"\nEnsemble weights: {weights}")

# Create ensemble
base_df = list(submissions.values())[0]
ensemble = pd.DataFrame({'Molecule Name': base_df['Molecule Name']})

for target in TARGETS:
    weighted_sum = np.zeros(len(ensemble))
    for model, df in submissions.items():
        weighted_sum += weights[model] * df[target].values
    ensemble[target] = weighted_sum

out_path = SUB_DIR / f"ensemble_{epoch_mode}.csv"
ensemble.to_csv(out_path, index=False)
print(f"\nEnsemble saved: {out_path}")
ENSEMBLE_SCRIPT

log_success "Ensemble created"

#===============================================================================
# Final Summary
#===============================================================================

log_section "FINAL SUMMARY"

log "Model Results:"
for model in catboost xgboost lightgbm chemprop; do
    result="${MODEL_RESULTS[$model]:-NOT RUN}"
    log "  $model: $result"
done

log ""
log "Submissions created:"
ls -la "${SCRIPT_DIR}/submissions/"*.csv 2>/dev/null | while read line; do
    log "  $line"
done

log ""
log "Log files:"
ls -la "${LOG_DIR}/"*"${TIMESTAMP}"* 2>/dev/null | while read line; do
    log "  $line"
done

log ""
log "Total runtime: $(($(date +%s) - $(date -d "$TIMESTAMP" +%s 2>/dev/null || echo 0)))s"
log "Completed at: $(date)"

log_section "NEXT STEPS"
log "1. Review individual model logs in ${LOG_DIR}/"
log "2. Compare MA-RAE scores across models"
log "3. Submit best performing submission to leaderboard"
log "4. Target: MA-RAE < 0.5593"

echo ""
echo "Done! Check $MAIN_LOG for full log."
