#!/bin/bash
#===============================================================================
# ADMET Challenge - Local M3 Pro Training Script
#
# Usage:
#   chmod +x run_local_m3.sh
#   ./run_local_m3.sh [quick|medium|full]
#
# Default: medium (~60-80 min on M3 Pro)
#===============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="${1:-medium}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs"
mkdir -p "$LOG_DIR" submissions models

echo "============================================================"
echo "ADMET CHALLENGE - M3 PRO LOCAL TRAINING"
echo "Mode: $MODE | Started: $(date)"
echo "============================================================"

# Activate environment
source admet_challenge/bin/activate

# Install any missing packages
echo ""
echo "[1/6] Checking dependencies..."
uv pip install -q numpy pandas scikit-learn scipy rdkit catboost xgboost lightgbm torch lightning chemprop 2>/dev/null || true

python3 -c "import catboost, xgboost, lightgbm, torch; print('All packages OK')"

#===============================================================================
# TRAINING SCRIPT (embedded Python)
#===============================================================================

echo ""
echo "[2/6] Starting training pipeline..."
echo ""

python3 << PYTHON_EOF
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['RDKit_DEPRECATION_WARNINGS'] = '0'

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import time
import gc
import json
from datetime import datetime

# Suppress all warnings
import logging
logging.getLogger('lightning.pytorch').setLevel(logging.ERROR)

print("="*70)
print("ADMET CHALLENGE TRAINING - ALL MODELS")
print(f"Mode: ${MODE}")
print(f"Started: {datetime.now()}")
print("="*70)

#-------------------------------------------------------------------------------
# Configuration
#-------------------------------------------------------------------------------
BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data" / "raw"
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

ITERS = {'quick': 200, 'medium': 600, 'full': 1200}["${MODE}"]
CHEMPROP_EPOCHS = {'quick': 5, 'medium': 15, 'full': 30}["${MODE}"]

#-------------------------------------------------------------------------------
# Feature computation
#-------------------------------------------------------------------------------
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

def compute_features(smiles_list):
    """Morgan FP 2048 + 25 descriptors"""
    print(f"\nComputing features for {len(smiles_list)} molecules...")
    features = []
    for i, smi in enumerate(smiles_list):
        if (i+1) % 2000 == 0:
            print(f"  {i+1}/{len(smiles_list)}")
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            features.append(np.zeros(2073))
            continue
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
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
            features.append(np.zeros(2073))
    print("  Done!")
    return np.array(features, dtype=np.float32)

def compute_rae(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    baseline = np.mean(np.abs(y_true - np.mean(y_true)))
    return mae / baseline if baseline > 0 else 1.0

#-------------------------------------------------------------------------------
# Load Data
#-------------------------------------------------------------------------------
print("\n" + "="*70)
print("LOADING DATA")
print("="*70)

train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test_blinded.csv")
print(f"Train: {len(train_df)}, Test: {len(test_df)}")

all_smiles = pd.concat([train_df['SMILES'], test_df['SMILES']]).values
X_all = compute_features(all_smiles)
X_train_all, X_test = X_all[:len(train_df)], X_all[len(train_df):]
print(f"Feature shape: {X_train_all.shape}")

#-------------------------------------------------------------------------------
# Storage
#-------------------------------------------------------------------------------
ALL_RESULTS = {}
ALL_PREDICTIONS = {}

#===============================================================================
# MODEL 1: CATBOOST
#===============================================================================
print("\n" + "="*70)
print("MODEL 1/4: CATBOOST")
print(f"Iterations: {ITERS}")
print("="*70)

from catboost import CatBoostRegressor

cb_results = {}
cb_predictions = {}
cb_start = time.time()

for target in TARGETS:
    print(f"\n  {target}...")
    y = train_df[target].values
    mask = ~np.isnan(y)
    X_t, y_t = X_train_all[mask], y[mask]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y_t))
    test_preds = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_t)):
        model = CatBoostRegressor(
            iterations=ITERS, depth=7, learning_rate=0.03,
            subsample=0.8, colsample_bylevel=0.8, l2_leaf_reg=3.0,
            verbose=False, random_seed=42+fold, early_stopping_rounds=50
        )
        model.fit(X_t[tr_idx], y_t[tr_idx],
                  eval_set=(X_t[va_idx], y_t[va_idx]), verbose=False)
        oof[va_idx] = model.predict(X_t[va_idx])
        test_preds.append(model.predict(X_test))

    rae = compute_rae(y_t, oof)
    cb_results[target] = rae
    cb_predictions[target] = np.clip(np.mean(test_preds, axis=0), *VALID_RANGES[target])
    print(f"    RAE: {rae:.4f} (n={len(y_t)})")

cb_ma_rae = np.mean(list(cb_results.values()))
cb_time = time.time() - cb_start
print(f"\n  CATBOOST MA-RAE: {cb_ma_rae:.4f} ({cb_time:.0f}s)")

ALL_RESULTS['catboost'] = {'ma_rae': cb_ma_rae, 'per_target': cb_results, 'time': cb_time}
ALL_PREDICTIONS['catboost'] = cb_predictions
gc.collect()

# Save CatBoost submission
sub = pd.DataFrame({'Molecule Name': test_df['Molecule Name']})
for t in TARGETS:
    sub[t] = cb_predictions[t]
sub.to_csv(SUB_DIR / "catboost_${MODE}.csv", index=False)

#===============================================================================
# MODEL 2: XGBOOST
#===============================================================================
print("\n" + "="*70)
print("MODEL 2/4: XGBOOST")
print(f"Iterations: {ITERS}")
print("="*70)

from xgboost import XGBRegressor

xgb_results = {}
xgb_predictions = {}
xgb_start = time.time()

for target in TARGETS:
    print(f"\n  {target}...")
    y = train_df[target].values
    mask = ~np.isnan(y)
    X_t, y_t = X_train_all[mask], y[mask]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y_t))
    test_preds = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_t)):
        model = XGBRegressor(
            n_estimators=ITERS, max_depth=7, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42+fold, verbosity=0, n_jobs=-1
        )
        model.fit(X_t[tr_idx], y_t[tr_idx],
                  eval_set=[(X_t[va_idx], y_t[va_idx])], verbose=False)
        oof[va_idx] = model.predict(X_t[va_idx])
        test_preds.append(model.predict(X_test))

    rae = compute_rae(y_t, oof)
    xgb_results[target] = rae
    xgb_predictions[target] = np.clip(np.mean(test_preds, axis=0), *VALID_RANGES[target])
    print(f"    RAE: {rae:.4f} (n={len(y_t)})")

xgb_ma_rae = np.mean(list(xgb_results.values()))
xgb_time = time.time() - xgb_start
print(f"\n  XGBOOST MA-RAE: {xgb_ma_rae:.4f} ({xgb_time:.0f}s)")

ALL_RESULTS['xgboost'] = {'ma_rae': xgb_ma_rae, 'per_target': xgb_results, 'time': xgb_time}
ALL_PREDICTIONS['xgboost'] = xgb_predictions
gc.collect()

# Save XGBoost submission
sub = pd.DataFrame({'Molecule Name': test_df['Molecule Name']})
for t in TARGETS:
    sub[t] = xgb_predictions[t]
sub.to_csv(SUB_DIR / "xgboost_${MODE}.csv", index=False)

#===============================================================================
# MODEL 3: LIGHTGBM
#===============================================================================
print("\n" + "="*70)
print("MODEL 3/4: LIGHTGBM")
print(f"Iterations: {ITERS}")
print("="*70)

from lightgbm import LGBMRegressor

lgb_results = {}
lgb_predictions = {}
lgb_start = time.time()

for target in TARGETS:
    print(f"\n  {target}...")
    y = train_df[target].values
    mask = ~np.isnan(y)
    X_t, y_t = X_train_all[mask], y[mask]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y_t))
    test_preds = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_t)):
        model = LGBMRegressor(
            n_estimators=ITERS, max_depth=7, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42+fold, verbosity=-1, n_jobs=-1
        )
        model.fit(X_t[tr_idx], y_t[tr_idx],
                  eval_set=[(X_t[va_idx], y_t[va_idx])])
        oof[va_idx] = model.predict(X_t[va_idx])
        test_preds.append(model.predict(X_test))

    rae = compute_rae(y_t, oof)
    lgb_results[target] = rae
    lgb_predictions[target] = np.clip(np.mean(test_preds, axis=0), *VALID_RANGES[target])
    print(f"    RAE: {rae:.4f} (n={len(y_t)})")

lgb_ma_rae = np.mean(list(lgb_results.values()))
lgb_time = time.time() - lgb_start
print(f"\n  LIGHTGBM MA-RAE: {lgb_ma_rae:.4f} ({lgb_time:.0f}s)")

ALL_RESULTS['lightgbm'] = {'ma_rae': lgb_ma_rae, 'per_target': lgb_results, 'time': lgb_time}
ALL_PREDICTIONS['lightgbm'] = lgb_predictions
gc.collect()

# Save LightGBM submission
sub = pd.DataFrame({'Molecule Name': test_df['Molecule Name']})
for t in TARGETS:
    sub[t] = lgb_predictions[t]
sub.to_csv(SUB_DIR / "lightgbm_${MODE}.csv", index=False)

#===============================================================================
# MODEL 4: CHEMPROP (D-MPNN)
#===============================================================================
print("\n" + "="*70)
print("MODEL 4/4: CHEMPROP D-MPNN")
print(f"Epochs: {CHEMPROP_EPOCHS}")
print("="*70)

chp_predictions = {}
chp_start = time.time()

try:
    import torch
    import lightning as pl
    from chemprop import data, models, nn

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    for target in TARGETS:
        print(f"\n  {target}...")

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
            max_epochs=CHEMPROP_EPOCHS,
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

        chp_predictions[target] = np.clip(test_pred, *VALID_RANGES[target])
        print(f"    Done (n={mask.sum()})")

        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    chp_time = time.time() - chp_start
    print(f"\n  CHEMPROP completed ({chp_time:.0f}s)")

    ALL_PREDICTIONS['chemprop'] = chp_predictions

    # Save Chemprop submission
    sub = pd.DataFrame({'Molecule Name': test_df['Molecule Name']})
    for t in TARGETS:
        sub[t] = chp_predictions[t]
    sub.to_csv(SUB_DIR / "chemprop_${MODE}.csv", index=False)

except Exception as e:
    print(f"  Chemprop failed: {e}")
    print("  Continuing without Chemprop...")

#===============================================================================
# ENSEMBLES
#===============================================================================
print("\n" + "="*70)
print("CREATING ENSEMBLES")
print("="*70)

# Ensemble 1: Equal weights (3 GBDT models)
print("\n  Ensemble 1: Equal weights (CatBoost + XGBoost + LightGBM)")
ens_equal = pd.DataFrame({'Molecule Name': test_df['Molecule Name']})
for t in TARGETS:
    preds = [ALL_PREDICTIONS['catboost'][t],
             ALL_PREDICTIONS['xgboost'][t],
             ALL_PREDICTIONS['lightgbm'][t]]
    ens_equal[t] = np.mean(preds, axis=0)
ens_equal.to_csv(SUB_DIR / "ensemble_equal_${MODE}.csv", index=False)

# Ensemble 2: Weighted by inverse RAE
print("  Ensemble 2: Weighted by inverse RAE")
ens_weighted = pd.DataFrame({'Molecule Name': test_df['Molecule Name']})
for t in TARGETS:
    raes = [ALL_RESULTS['catboost']['per_target'][t],
            ALL_RESULTS['xgboost']['per_target'][t],
            ALL_RESULTS['lightgbm']['per_target'][t]]
    weights = np.array([1/r for r in raes])
    weights = weights / weights.sum()

    preds = [ALL_PREDICTIONS['catboost'][t],
             ALL_PREDICTIONS['xgboost'][t],
             ALL_PREDICTIONS['lightgbm'][t]]
    ens_weighted[t] = sum(w * p for w, p in zip(weights, preds))
ens_weighted.to_csv(SUB_DIR / "ensemble_weighted_${MODE}.csv", index=False)

# Ensemble 3: Include Chemprop if available
if 'chemprop' in ALL_PREDICTIONS:
    print("  Ensemble 3: All 4 models (including Chemprop)")
    ens_all = pd.DataFrame({'Molecule Name': test_df['Molecule Name']})
    for t in TARGETS:
        preds = [ALL_PREDICTIONS['catboost'][t],
                 ALL_PREDICTIONS['xgboost'][t],
                 ALL_PREDICTIONS['lightgbm'][t],
                 ALL_PREDICTIONS['chemprop'][t]]
        ens_all[t] = np.mean(preds, axis=0)
    ens_all.to_csv(SUB_DIR / "ensemble_with_chemprop_${MODE}.csv", index=False)

#===============================================================================
# FINAL SUMMARY
#===============================================================================
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)

print("\n" + "-"*50)
print("PER-TARGET RAE COMPARISON")
print("-"*50)
print(f"{'Target':<35} {'CatBoost':>10} {'XGBoost':>10} {'LightGBM':>10}")
print("-"*70)
for t in TARGETS:
    cb = ALL_RESULTS['catboost']['per_target'][t]
    xgb = ALL_RESULTS['xgboost']['per_target'][t]
    lgb = ALL_RESULTS['lightgbm']['per_target'][t]
    best = min(cb, xgb, lgb)
    print(f"{t:<35} {cb:>10.4f} {xgb:>10.4f} {lgb:>10.4f}  {'*' if best == cb else ''}")

print("\n" + "-"*50)
print("MODEL SUMMARY (MA-RAE)")
print("-"*50)
for model in ['catboost', 'xgboost', 'lightgbm']:
    ma_rae = ALL_RESULTS[model]['ma_rae']
    t = ALL_RESULTS[model]['time']
    print(f"  {model.upper():<12} MA-RAE: {ma_rae:.4f}  (time: {t:.0f}s)")

print("\n" + "="*50)
best_model = min(ALL_RESULTS.keys(), key=lambda x: ALL_RESULTS[x]['ma_rae'])
best_ma_rae = ALL_RESULTS[best_model]['ma_rae']
print(f"BEST SINGLE MODEL: {best_model.upper()} = {best_ma_rae:.4f}")
print(f"TARGET:            0.5593 (leader 'pebble')")
print(f"GAP:               {best_ma_rae - 0.5593:+.4f}")
print("="*50)

print("\n" + "-"*50)
print("SUBMISSIONS CREATED")
print("-"*50)
for f in sorted(SUB_DIR.glob("*_${MODE}.csv")):
    print(f"  {f.name}")

# Save results JSON
results_json = {
    'mode': '${MODE}',
    'timestamp': datetime.now().isoformat(),
    'models': {k: {'ma_rae': v['ma_rae'], 'time': v['time']} for k, v in ALL_RESULTS.items()},
    'best_model': best_model,
    'best_ma_rae': best_ma_rae,
    'target': 0.5593
}
with open(LOG_DIR / "results_${MODE}_${TIMESTAMP}.json", 'w') as f:
    json.dump(results_json, f, indent=2)

print(f"\nCompleted: {datetime.now()}")
print("="*70)
PYTHON_EOF

#===============================================================================
# Done
#===============================================================================

echo ""
echo "============================================================"
echo "TRAINING COMPLETE"
echo "============================================================"
echo ""
echo "Submissions in: submissions/"
ls -la submissions/*_${MODE}.csv 2>/dev/null || echo "  (none found)"
echo ""
echo "Logs in: logs/"
ls -la logs/results_${MODE}_*.json 2>/dev/null || echo "  (none found)"
echo ""
echo "Finished: $(date)"
echo "============================================================"
