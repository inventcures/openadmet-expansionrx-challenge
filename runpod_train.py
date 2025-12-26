#!/usr/bin/env python3
"""
RunPod Training Script for ADMET Challenge
Run on GPU instance: python runpod_train.py

Requirements:
pip install pandas numpy scikit-learn scipy rdkit catboost xgboost lightgbm chemprop torch lightning joblib tqdm

Target: MA-RAE < 0.5593 (current best: 0.5656)
"""
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['RDKit_DEPRECATION_WARNINGS'] = '0'

import pandas as pd
import numpy as np
from pathlib import Path
import gc
import joblib
from datetime import datetime

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr

# Models
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "raw"
MODEL_DIR = BASE_DIR / "models"
SUB_DIR = BASE_DIR / "submissions"

MODEL_DIR.mkdir(exist_ok=True)
SUB_DIR.mkdir(exist_ok=True)

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

def compute_features(smiles_list, verbose=True):
    """Morgan FP (2048 bits) + 25 descriptors = 2073 features"""
    if verbose:
        print(f"Computing features for {len(smiles_list)} molecules...")
    features = []
    for i, smi in enumerate(smiles_list):
        if verbose and (i+1) % 2000 == 0:
            print(f"  {i+1}/{len(smiles_list)}")
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            features.append(np.zeros(2073))
            continue
        try:
            # 2048-bit Morgan FP (larger than 1024)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            # Extended descriptors
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
                # Additional descriptors
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
    return np.array(features, dtype=np.float32)

def compute_rae(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    baseline = np.mean(np.abs(y_true - np.mean(y_true)))
    return mae / baseline if baseline > 0 else 1.0

def train_catboost(X_train, y_train, X_test, target_name, n_folds=5):
    """Train CatBoost with optimized hyperparameters"""
    print(f"\n  Training CatBoost for {target_name}...")

    # Endpoint-specific params
    params = {
        'LogD': {'depth': 8, 'lr': 0.025, 'iters': 1200},
        'KSOL': {'depth': 8, 'lr': 0.025, 'iters': 1000},
        'HLM CLint': {'depth': 7, 'lr': 0.04, 'iters': 800},
        'MLM CLint': {'depth': 7, 'lr': 0.04, 'iters': 800},
        'Caco-2 Permeability Papp A>B': {'depth': 7, 'lr': 0.03, 'iters': 900},
        'Caco-2 Permeability Efflux': {'depth': 6, 'lr': 0.04, 'iters': 600},
        'MPPB': {'depth': 8, 'lr': 0.025, 'iters': 1000},
        'MBPB': {'depth': 8, 'lr': 0.025, 'iters': 1000},
        'MGMB': {'depth': 6, 'lr': 0.05, 'iters': 500},
    }
    hp = params.get(target_name, {'depth': 7, 'lr': 0.03, 'iters': 800})

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof = np.zeros(len(y_train))
    test_preds = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train)):
        model = CatBoostRegressor(
            iterations=hp['iters'], depth=hp['depth'], learning_rate=hp['lr'],
            subsample=0.8, colsample_bylevel=0.8, l2_leaf_reg=3.0,
            random_strength=1.0, verbose=False, random_seed=42+fold,
            early_stopping_rounds=100
        )
        model.fit(X_train[tr_idx], y_train[tr_idx],
                  eval_set=(X_train[va_idx], y_train[va_idx]), verbose=False)
        oof[va_idx] = model.predict(X_train[va_idx])
        test_preds.append(model.predict(X_test))

    rae = compute_rae(y_train, oof)
    test_pred = np.mean(test_preds, axis=0)
    return oof, test_pred, rae

def train_xgboost(X_train, y_train, X_test, target_name, n_folds=5):
    """Train XGBoost with optimized hyperparameters"""
    print(f"\n  Training XGBoost for {target_name}...")

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof = np.zeros(len(y_train))
    test_preds = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train)):
        model = XGBRegressor(
            n_estimators=800, max_depth=7, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42+fold, verbosity=0, n_jobs=-1
        )
        model.fit(X_train[tr_idx], y_train[tr_idx],
                  eval_set=[(X_train[va_idx], y_train[va_idx])],
                  verbose=False)
        oof[va_idx] = model.predict(X_train[va_idx])
        test_preds.append(model.predict(X_test))

    rae = compute_rae(y_train, oof)
    test_pred = np.mean(test_preds, axis=0)
    return oof, test_pred, rae

def train_lightgbm(X_train, y_train, X_test, target_name, n_folds=5):
    """Train LightGBM with optimized hyperparameters"""
    print(f"\n  Training LightGBM for {target_name}...")

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof = np.zeros(len(y_train))
    test_preds = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train)):
        model = LGBMRegressor(
            n_estimators=800, max_depth=7, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42+fold, verbosity=-1, n_jobs=-1
        )
        model.fit(X_train[tr_idx], y_train[tr_idx],
                  eval_set=[(X_train[va_idx], y_train[va_idx])],
                  callbacks=[])
        oof[va_idx] = model.predict(X_train[va_idx])
        test_preds.append(model.predict(X_test))

    rae = compute_rae(y_train, oof)
    test_pred = np.mean(test_preds, axis=0)
    return oof, test_pred, rae

def train_chemprop(train_df, test_df, target_name):
    """Train Chemprop D-MPNN (GPU accelerated)"""
    try:
        import chemprop
        from chemprop import data, featurizers, models, nn
        import torch
        import lightning as pl

        print(f"\n  Training Chemprop D-MPNN for {target_name}...")

        # Check GPU
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"    Using device: {device}")

        # Prepare data
        mask = train_df[target_name].notna()
        train_smiles = train_df.loc[mask, 'SMILES'].tolist()
        train_y = train_df.loc[mask, target_name].values.reshape(-1, 1)
        test_smiles = test_df['SMILES'].tolist()

        # Create datasets
        train_data = [data.MoleculeDatapoint.from_smi(smi, y)
                      for smi, y in zip(train_smiles, train_y)]
        test_data = [data.MoleculeDatapoint.from_smi(smi) for smi in test_smiles]

        train_dset = data.MoleculeDataset(train_data)
        test_dset = data.MoleculeDataset(test_data)

        # Split for validation
        train_dset, val_dset = data.split_data_by_indices(
            train_dset, [list(range(int(len(train_dset)*0.85))),
                        list(range(int(len(train_dset)*0.85), len(train_dset)))]
        )

        # Create model
        mp = nn.BondMessagePassing()
        agg = nn.MeanAggregation()
        ffn = nn.RegressionFFN()
        mpnn = models.MPNN(mp, agg, ffn)

        # Train
        trainer = pl.Trainer(
            max_epochs=30,
            accelerator=device if device != "cpu" else "auto",
            enable_progress_bar=False,
            logger=False
        )

        train_loader = data.build_dataloader(train_dset, batch_size=64, shuffle=True)
        val_loader = data.build_dataloader(val_dset, batch_size=64)
        test_loader = data.build_dataloader(test_dset, batch_size=64)

        trainer.fit(mpnn, train_loader, val_loader)

        # Predict
        preds = trainer.predict(mpnn, test_loader)
        test_pred = np.concatenate([p.numpy() for p in preds]).flatten()

        return test_pred

    except Exception as e:
        print(f"    Chemprop failed: {e}")
        return None

def main():
    print("="*60)
    print("ADMET CHALLENGE - RUNPOD TRAINING")
    print(f"Started: {datetime.now()}")
    print("="*60)

    # Load data
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test_blinded.csv")
    print(f"\nTrain: {len(train_df)}, Test: {len(test_df)}")

    # Compute features
    all_smiles = pd.concat([train_df['SMILES'], test_df['SMILES']]).values
    X_all = compute_features(all_smiles)
    X_train_all, X_test = X_all[:len(train_df)], X_all[len(train_df):]
    print(f"Feature shape: {X_train_all.shape}")

    # Store results
    results = {model: {} for model in ['catboost', 'xgboost', 'lightgbm', 'chemprop']}
    predictions = {model: {} for model in ['catboost', 'xgboost', 'lightgbm', 'chemprop']}

    # Train all models for all targets
    for target in TARGETS:
        print(f"\n{'='*50}")
        print(f"TARGET: {target}")
        print(f"{'='*50}")

        y = train_df[target].values
        mask = ~np.isnan(y)
        X_t, y_t = X_train_all[mask], y[mask]
        print(f"Samples: {len(y_t)}")

        # CatBoost
        oof_cb, pred_cb, rae_cb = train_catboost(X_t, y_t, X_test, target)
        results['catboost'][target] = rae_cb
        predictions['catboost'][target] = np.clip(pred_cb, *VALID_RANGES[target])
        print(f"  CatBoost RAE: {rae_cb:.4f}")

        # XGBoost
        oof_xgb, pred_xgb, rae_xgb = train_xgboost(X_t, y_t, X_test, target)
        results['xgboost'][target] = rae_xgb
        predictions['xgboost'][target] = np.clip(pred_xgb, *VALID_RANGES[target])
        print(f"  XGBoost RAE: {rae_xgb:.4f}")

        # LightGBM
        oof_lgb, pred_lgb, rae_lgb = train_lightgbm(X_t, y_t, X_test, target)
        results['lightgbm'][target] = rae_lgb
        predictions['lightgbm'][target] = np.clip(pred_lgb, *VALID_RANGES[target])
        print(f"  LightGBM RAE: {rae_lgb:.4f}")

        # Chemprop (GPU)
        pred_chp = train_chemprop(train_df, test_df, target)
        if pred_chp is not None:
            predictions['chemprop'][target] = np.clip(pred_chp, *VALID_RANGES[target])

        gc.collect()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY - Individual Model MA-RAE")
    print("="*60)

    for model in ['catboost', 'xgboost', 'lightgbm']:
        raes = list(results[model].values())
        ma_rae = np.mean(raes)
        print(f"{model}: MA-RAE = {ma_rae:.4f}")

    # Save individual submissions
    for model in ['catboost', 'xgboost', 'lightgbm']:
        sub = pd.DataFrame({'Molecule Name': test_df['Molecule Name']})
        for t in TARGETS:
            sub[t] = predictions[model][t]
        out_path = SUB_DIR / f"runpod_{model}.csv"
        sub.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")

    # Create ensembles
    print("\n" + "="*60)
    print("CREATING ENSEMBLES")
    print("="*60)

    # Ensemble 1: Equal weights
    ensemble_equal = pd.DataFrame({'Molecule Name': test_df['Molecule Name']})
    for t in TARGETS:
        preds = [predictions['catboost'][t], predictions['xgboost'][t], predictions['lightgbm'][t]]
        ensemble_equal[t] = np.mean(preds, axis=0)
    ensemble_equal.to_csv(SUB_DIR / "runpod_ensemble_equal.csv", index=False)

    # Ensemble 2: Weighted by inverse RAE
    ensemble_weighted = pd.DataFrame({'Molecule Name': test_df['Molecule Name']})
    for t in TARGETS:
        raes = [results['catboost'][t], results['xgboost'][t], results['lightgbm'][t]]
        weights = [1/r for r in raes]
        weights = np.array(weights) / np.sum(weights)
        preds = [predictions['catboost'][t], predictions['xgboost'][t], predictions['lightgbm'][t]]
        ensemble_weighted[t] = sum(w * p for w, p in zip(weights, preds))
    ensemble_weighted.to_csv(SUB_DIR / "runpod_ensemble_weighted.csv", index=False)

    # Ensemble 3: Include Chemprop if available
    if predictions['chemprop']:
        ensemble_all = pd.DataFrame({'Molecule Name': test_df['Molecule Name']})
        for t in TARGETS:
            preds = [predictions['catboost'][t], predictions['xgboost'][t],
                     predictions['lightgbm'][t], predictions['chemprop'].get(t)]
            preds = [p for p in preds if p is not None]
            ensemble_all[t] = np.mean(preds, axis=0)
        ensemble_all.to_csv(SUB_DIR / "runpod_ensemble_with_chemprop.csv", index=False)

    print(f"\nCompleted: {datetime.now()}")
    print("="*60)

if __name__ == "__main__":
    main()
