#!/usr/bin/env python3
"""
Phase 2A Pipeline - RunPod RTX 4090 Optimized

GPU-optimized stacking ensemble with extended features.
Designed for RunPod with RTX 4090 (24GB VRAM).
"""
import os
import sys

# Suppress warnings before any imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

import warnings
warnings.filterwarnings('ignore')

# Suppress RDKit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import pandas as pd
import numpy as np
from pathlib import Path
import gc
import argparse
import time
from datetime import datetime

import torch
import xgboost as xgb
from catboost import CatBoostRegressor
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except:
    LGB_AVAILABLE = False

from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

# Multi-task neural network
try:
    from src.multitask_nn import MultiTaskTrainer, ENDPOINT_GROUPS
    MULTITASK_AVAILABLE = True
except ImportError:
    MULTITASK_AVAILABLE = False
    print("Multi-task NN not available")

# Chemprop D-MPNN
try:
    from src.chemprop_optimized import ChempropEnsemble, CHEMPROP_HYPERPARAMS
    CHEMPROP_AVAILABLE = True
except ImportError:
    CHEMPROP_AVAILABLE = False
    print("Chemprop not available")

# ChemBERTa embeddings
try:
    from src.chemberta_embeddings import ChemBERTaEmbedder, CHEMBERTA_AVAILABLE
except ImportError:
    CHEMBERTA_AVAILABLE = False
    print("ChemBERTa not available")

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, MACCSkeys

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent

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

# RTX 4090 optimized hyperparameters (more iterations, deeper trees)
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

# ============================================================================
# GPU DETECTION
# ============================================================================

def detect_gpu():
    """Detect available GPU"""
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        return True, gpu_name
    print("No CUDA GPU detected, using CPU")
    return False, None

# ============================================================================
# FEATURE ENGINEERING (GPU-friendly batch processing)
# ============================================================================

def compute_features_gpu(smiles_list, use_maccs=True, use_rdkit_fp=True):
    """
    Compute features with GPU-friendly batching
    Morgan (1024) + MACCS (167) + RDKit FP (2048) + Descriptors (50)
    """
    n_mols = len(smiles_list)

    # Calculate dimensions
    morgan_bits = 1024
    maccs_bits = 167 if use_maccs else 0
    rdkit_bits = 2048 if use_rdkit_fp else 0
    desc_count = 50
    total_dim = morgan_bits + maccs_bits + rdkit_bits + desc_count

    print(f"Computing {total_dim} features for {n_mols} molecules...")

    features = np.zeros((n_mols, total_dim), dtype=np.float32)

    for i, smi in enumerate(smiles_list):
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{n_mols}")

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        try:
            idx = 0

            # Morgan fingerprint (1024 bits)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=morgan_bits)
            features[i, idx:idx+morgan_bits] = np.array(fp)
            idx += morgan_bits

            # MACCS keys (167 bits)
            if use_maccs:
                maccs = MACCSkeys.GenMACCSKeys(mol)
                features[i, idx:idx+maccs_bits] = np.array(maccs)
                idx += maccs_bits

            # RDKit fingerprint (2048 bits)
            if use_rdkit_fp:
                rdkit_fp = Chem.RDKFingerprint(mol, fpSize=rdkit_bits)
                features[i, idx:idx+rdkit_bits] = np.array(rdkit_fp)
                idx += rdkit_bits

            # Descriptors (50)
            desc = get_descriptors(mol)
            features[i, idx:idx+desc_count] = desc

        except:
            continue

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"Features shape: {features.shape}")
    return features


def get_descriptors(mol):
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
    except:
        return np.zeros(50, dtype=np.float32)

# ============================================================================
# GPU-OPTIMIZED STACKING ENSEMBLE
# ============================================================================

class GPUStackingEnsemble:
    """Stacking ensemble optimized for RTX 4090"""

    def __init__(self, use_gpu=True, n_folds=5, use_multitask=True):
        self.use_gpu = use_gpu
        self.n_folds = n_folds
        self.use_multitask = use_multitask and MULTITASK_AVAILABLE
        self.base_models = {}
        self.meta_model = None
        self.feature_scaler = StandardScaler()

    def _create_xgb(self, hp):
        """XGBoost with CUDA"""
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
        """CatBoost with GPU"""
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
        """LightGBM (CPU only - GPU requires special build)"""
        return lgb.LGBMRegressor(
            n_estimators=hp['n_estimators'],
            max_depth=hp['max_depth'],
            learning_rate=hp['lr'],
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=3.0,
            n_jobs=-1,
            verbose=-1,
        ) if LGB_AVAILABLE else None

    def _create_rf(self, hp):
        """RandomForest (CPU)"""
        return RandomForestRegressor(
            n_estimators=min(hp['n_estimators'], 500),
            max_depth=hp['max_depth'] + 2,
            min_samples_leaf=3,
            n_jobs=-1,
        )

    def fit(self, X, y, hp):
        """Fit stacking ensemble with OOF predictions"""
        n_samples = len(y)
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        # Base model predictions
        model_names = ['xgb', 'catboost', 'lgb', 'rf']
        oof_preds = {name: np.zeros(n_samples) for name in model_names}
        self.base_models = {name: [] for name in model_names}

        for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
            print(f"  Fold {fold+1}/{self.n_folds}")
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
            if LGB_AVAILABLE:
                lgb_model = self._create_lgb(hp)
                lgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                             callbacks=[lgb.early_stopping(100, verbose=False)])
                oof_preds['lgb'][va_idx] = lgb_model.predict(X_va)
                self.base_models['lgb'].append(lgb_model)

            # RandomForest
            rf_model = self._create_rf(hp)
            rf_model.fit(X_tr, y_tr)
            oof_preds['rf'][va_idx] = rf_model.predict(X_va)
            self.base_models['rf'].append(rf_model)

            # Clear GPU memory
            if self.use_gpu:
                torch.cuda.empty_cache()
            gc.collect()

        # Stack predictions for meta-learner
        stack_features = np.column_stack([oof_preds[name] for name in model_names
                                          if len(self.base_models[name]) > 0])

        # Fit meta-learner
        self.meta_model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        self.meta_model.fit(stack_features, y)

        # Calculate OOF metrics
        oof_final = self.meta_model.predict(stack_features)
        mae = mean_absolute_error(y, oof_final)
        rae = mae / np.mean(np.abs(y - np.mean(y)))
        spear = spearmanr(y, oof_final)[0]

        print(f"  OOF: MAE={mae:.4f}, RAE={rae:.4f}, Spearman={spear:.4f}")
        return {'MAE': mae, 'RAE': rae, 'Spearman': spear}

    def predict(self, X):
        """Predict using ensemble"""
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

def run_pipeline(mode='full', use_multitask=True, use_chemprop=True, use_chemberta=True):
    """Run Phase 2B pipeline on RunPod with multi-task learning + Chemprop + ChemBERTa"""

    print("=" * 60)
    print(f"PHASE 2B PIPELINE - RUNPOD RTX 4090 ({mode.upper()} MODE)")
    print("=" * 60)

    # Detect GPU
    use_gpu, gpu_name = detect_gpu()

    # Load data
    print("\n[1/4] Loading data...")
    train_df = pd.read_csv(BASE_DIR / "data/raw/train.csv")
    test_df = pd.read_csv(BASE_DIR / "data/raw/test_blinded.csv")
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    # Compute features
    print("\n[2/4] Computing features...")
    all_smiles = pd.concat([train_df['SMILES'], test_df['SMILES']]).values

    if mode == 'quick':
        X_all = compute_features_gpu(all_smiles, use_maccs=True, use_rdkit_fp=False)
    else:
        X_all = compute_features_gpu(all_smiles, use_maccs=True, use_rdkit_fp=True)

    X_train = X_all[:len(train_df)]
    X_test = X_all[len(train_df):]

    # Add ChemBERTa embeddings if available
    if use_chemberta and CHEMBERTA_AVAILABLE:
        print("\n[2.5/4] Computing ChemBERTa embeddings...")
        embedder = ChemBERTaEmbedder()
        embedder.load_model()

        chemberta_all = embedder.embed_batch(all_smiles.tolist(), batch_size=32)
        chemberta_train = chemberta_all[:len(train_df)]
        chemberta_test = chemberta_all[len(train_df):]

        # Concatenate with fingerprint features
        X_train = np.hstack([X_train, chemberta_train])
        X_test = np.hstack([X_test, chemberta_test])
        print(f"Features with ChemBERTa: {X_train.shape}")

        if use_gpu:
            torch.cuda.empty_cache()
        gc.collect()

    # Train models
    print("\n[3/4] Training stacking ensembles...")
    predictions = {}
    results = {}

    for target in TARGETS:
        print(f"\n{'='*50}")
        print(f"{target}")
        print(f"{'='*50}")

        y = train_df[target].values
        mask = ~np.isnan(y)
        X_t, y_t = X_train[mask], y[mask]
        print(f"Samples: {len(y_t)}")

        hp = HYPERPARAMS_GPU[target]
        if mode == 'quick':
            hp = {**hp, 'n_estimators': min(hp['n_estimators'], 400)}

        ensemble = GPUStackingEnsemble(use_gpu=use_gpu, n_folds=5)
        result = ensemble.fit(X_t, y_t, hp)
        results[target] = result

        # Predict
        pred = ensemble.predict(X_test)
        pred = np.clip(pred, *VALID_RANGES[target])
        predictions[target] = pred

        # Clear memory
        if use_gpu:
            torch.cuda.empty_cache()
        gc.collect()

    # Multi-task neural network (Phase 2B)
    mt_predictions = None
    if use_multitask and MULTITASK_AVAILABLE:
        print("\n" + "=" * 60)
        print("MULTI-TASK NEURAL NETWORK")
        print("=" * 60)

        y_dict = {t: train_df[t].values for t in TARGETS}
        epochs = 50 if mode == 'quick' else 100

        mt_trainer = MultiTaskTrainer(
            input_dim=X_train.shape[1],
            hidden_dim=512,
            lr=1e-3,
        )
        mt_trainer.fit(X_train, y_dict, epochs=epochs, verbose=True)
        mt_predictions = mt_trainer.predict(X_test)

        if use_gpu:
            torch.cuda.empty_cache()
        gc.collect()

    # Chemprop D-MPNN (Phase 2B)
    cp_predictions = None
    if use_chemprop and CHEMPROP_AVAILABLE:
        print("\n" + "=" * 60)
        print("CHEMPROP D-MPNN")
        print("=" * 60)

        train_smiles = train_df['SMILES'].tolist()
        test_smiles = test_df['SMILES'].tolist()
        n_models = 2 if mode == 'quick' else 3

        cp_predictions = {}
        for target in TARGETS:
            print(f"\n{target}")
            y = train_df[target].values
            mask = ~np.isnan(y)
            valid_smiles = [train_smiles[i] for i, m in enumerate(mask) if m]
            valid_y = y[mask]
            print(f"  Samples: {len(valid_y)}")

            ensemble = ChempropEnsemble(target, n_models=n_models)
            ensemble.fit(valid_smiles, valid_y, verbose=True)
            cp_predictions[target] = ensemble.predict(test_smiles)

            if use_gpu:
                torch.cuda.empty_cache()
            gc.collect()

    # Blend all models
    print("\n" + "=" * 60)
    print("BLENDING PREDICTIONS")
    print("=" * 60)

    n_models = 1
    if mt_predictions:
        n_models += 1
    if cp_predictions:
        n_models += 1

    # Weights: GBDT 0.6, Multi-task 0.25, Chemprop 0.15
    for target in TARGETS:
        gbdt_pred = predictions[target]

        if n_models == 1:
            blended = gbdt_pred
        elif n_models == 2 and mt_predictions:
            blended = 0.7 * gbdt_pred + 0.3 * mt_predictions[target]
        elif n_models == 2 and cp_predictions:
            blended = 0.75 * gbdt_pred + 0.25 * cp_predictions[target]
        else:  # All 3 models
            blended = (0.6 * gbdt_pred +
                      0.25 * mt_predictions[target] +
                      0.15 * cp_predictions[target])

        predictions[target] = np.clip(blended, *VALID_RANGES[target])

    print(f"Blended {n_models} model types: GBDT" +
          (" + Multi-task" if mt_predictions else "") +
          (" + Chemprop" if cp_predictions else ""))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY" + (" (GBDT + Multi-task blend)" if mt_predictions else ""))
    print("=" * 60)
    raes = [r['RAE'] for r in results.values()]
    for t, r in results.items():
        print(f"{t}: RAE={r['RAE']:.4f}, Spearman={r['Spearman']:.4f}")
    print(f"\nMA-RAE: {np.mean(raes):.4f}")

    # Save submission
    print("\n[4/4] Saving submission...")
    sub = pd.DataFrame({'Molecule Name': test_df['Molecule Name']})
    for t in TARGETS:
        sub[t] = predictions[t]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = BASE_DIR / f"submissions/phase2_runpod_{mode}_{timestamp}.csv"
    out_path.parent.mkdir(exist_ok=True)
    sub.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    return results, out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Phase 2B RunPod Pipeline')
    parser.add_argument('--mode', choices=['quick', 'full'], default='full',
                       help='quick (~40 min) or full (~120 min)')
    parser.add_argument('--no-multitask', action='store_true',
                       help='Disable multi-task neural network')
    parser.add_argument('--no-chemprop', action='store_true',
                       help='Disable Chemprop D-MPNN')
    parser.add_argument('--no-chemberta', action='store_true',
                       help='Disable ChemBERTa embeddings')
    parser.add_argument('--gbdt-only', action='store_true',
                       help='GBDT stacking only (fastest)')
    args = parser.parse_args()

    use_mt = not args.no_multitask and not args.gbdt_only
    use_cp = not args.no_chemprop and not args.gbdt_only
    use_cb = not args.no_chemberta and not args.gbdt_only

    start = time.time()
    results, out_path = run_pipeline(
        args.mode,
        use_multitask=use_mt,
        use_chemprop=use_cp,
        use_chemberta=use_cb
    )
    elapsed = time.time() - start

    print(f"\nTotal time: {elapsed/60:.1f} minutes")
    print(f"Submit: {out_path}")
