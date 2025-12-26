"""
CatBoost + RandomForest Ensemble for ADMET Prediction
Memory-efficient implementation with enhanced fingerprints
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import gc
warnings.filterwarnings('ignore')

# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem import MACCSkeys
from rdkit.Avalon import pyAvalonTools

# ML
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
from catboost import CatBoostRegressor
import joblib

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
MODEL_DIR = BASE_DIR / "models"
SUBMISSION_DIR = BASE_DIR / "submissions"

# Targets
TARGETS = [
    'LogD', 'KSOL', 'HLM CLint', 'MLM CLint',
    'Caco-2 Permeability Papp A>B', 'Caco-2 Permeability Efflux',
    'MPPB', 'MBPB', 'MGMB'
]

# Valid ranges for clipping
VALID_RANGES = {
    'LogD': (-3.0, 6.0),
    'KSOL': (0.001, 350.0),
    'HLM CLint': (0.0, 3000.0),
    'MLM CLint': (0.0, 12000.0),
    'Caco-2 Permeability Papp A>B': (0.0, 60.0),
    'Caco-2 Permeability Efflux': (0.2, 120.0),
    'MPPB': (0.0, 100.0),
    'MBPB': (0.0, 100.0),
    'MGMB': (0.0, 100.0)
}


def compute_features(smiles_list):
    """Compute Morgan FP + Avalon FP + MACCS + RDKit descriptors"""
    print("Computing molecular features...")

    features_list = []
    failed = 0

    for i, smi in enumerate(smiles_list):
        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(smiles_list)} molecules...")

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            failed += 1
            # Return zeros for failed molecules
            features_list.append(np.zeros(2048 + 512 + 167 + 20))
            continue

        try:
            # Morgan fingerprint (1024 bits)
            morgan = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
            morgan_arr = np.array(morgan)

            # Morgan counts (1024)
            morgan_counts = AllChem.GetHashedMorganFingerprint(mol, radius=2, nBits=1024)
            morgan_counts_arr = np.zeros(1024)
            for idx, val in morgan_counts.GetNonzeroElements().items():
                morgan_counts_arr[idx % 1024] = val

            # Avalon fingerprint (512 bits)
            avalon = pyAvalonTools.GetAvalonFP(mol, nBits=512)
            avalon_arr = np.array(avalon)

            # MACCS keys (167 bits)
            maccs = MACCSkeys.GenMACCSKeys(mol)
            maccs_arr = np.array(maccs)

            # Key RDKit descriptors (20)
            desc = [
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.NumAromaticRings(mol),
                Descriptors.FractionCSP3(mol),
                Descriptors.HeavyAtomCount(mol),
                Descriptors.RingCount(mol),
                rdMolDescriptors.CalcNumAliphaticRings(mol),
                rdMolDescriptors.CalcNumHeterocycles(mol),
                Descriptors.LabuteASA(mol),
                Descriptors.NumRadicalElectrons(mol),
                Descriptors.NumValenceElectrons(mol),
                rdMolDescriptors.CalcNumAmideBonds(mol),
                rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
                rdMolDescriptors.CalcNumSpiroAtoms(mol),
                Descriptors.qed(mol),
                rdMolDescriptors.CalcTPSA(mol, includeSandP=True)
            ]
            desc_arr = np.array(desc, dtype=np.float32)

            # Concatenate all features
            combined = np.concatenate([morgan_arr, morgan_counts_arr, avalon_arr, maccs_arr, desc_arr])
            features_list.append(combined)

        except Exception as e:
            failed += 1
            features_list.append(np.zeros(2048 + 512 + 167 + 20))

    print(f"  Completed. Failed molecules: {failed}")
    return np.array(features_list, dtype=np.float32)


def train_model_for_target(X, y, target_name, n_folds=5):
    """Train CatBoost + RF ensemble for a single target"""
    print(f"\n{'='*60}")
    print(f"Training models for: {target_name}")
    print(f"{'='*60}")

    # Remove NaN values
    mask = ~np.isnan(y)
    X_clean = X[mask]
    y_clean = y[mask]
    print(f"  Samples: {len(y_clean)} (removed {(~mask).sum()} NaN)")

    if len(y_clean) < 50:
        print(f"  WARNING: Too few samples, skipping CV")
        return None

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    oof_cat = np.zeros(len(y_clean))
    oof_rf = np.zeros(len(y_clean))

    cat_models = []
    rf_models = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_clean)):
        X_train, X_val = X_clean[train_idx], X_clean[val_idx]
        y_train, y_val = y_clean[train_idx], y_clean[val_idx]

        print(f"  Fold {fold+1}/{n_folds}...")

        # CatBoost
        cat = CatBoostRegressor(
            iterations=800,
            depth=8,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bylevel=0.8,
            l2_leaf_reg=3.0,
            verbose=False,
            random_seed=42 + fold,
            early_stopping_rounds=100
        )
        cat.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        oof_cat[val_idx] = cat.predict(X_val)
        cat_models.append(cat)

        # RandomForest
        rf = RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42 + fold
        )
        rf.fit(X_train, y_train)
        oof_rf[val_idx] = rf.predict(X_val)
        rf_models.append(rf)

        # Clear memory
        gc.collect()

    # Find optimal weight
    best_weight = 0.5
    best_mae = float('inf')

    for w in np.arange(0.3, 0.8, 0.05):
        oof_ens = w * oof_cat + (1 - w) * oof_rf
        mae = mean_absolute_error(y_clean, oof_ens)
        if mae < best_mae:
            best_mae = mae
            best_weight = w

    oof_ensemble = best_weight * oof_cat + (1 - best_weight) * oof_rf

    # Metrics
    mae_cat = mean_absolute_error(y_clean, oof_cat)
    mae_rf = mean_absolute_error(y_clean, oof_rf)
    mae_ens = mean_absolute_error(y_clean, oof_ensemble)

    spear_cat = spearmanr(y_clean, oof_cat)[0]
    spear_rf = spearmanr(y_clean, oof_rf)[0]
    spear_ens = spearmanr(y_clean, oof_ensemble)[0]

    rae_ens = mae_ens / np.mean(np.abs(y_clean - np.mean(y_clean)))

    print(f"\n  Results:")
    print(f"    CatBoost:  MAE={mae_cat:.4f}, Spearman={spear_cat:.4f}")
    print(f"    RF:        MAE={mae_rf:.4f}, Spearman={spear_rf:.4f}")
    print(f"    Ensemble:  MAE={mae_ens:.4f}, Spearman={spear_ens:.4f}, RAE={rae_ens:.4f}")
    print(f"    Best weight (CatBoost): {best_weight:.2f}")

    # Train final models on all data
    print(f"  Training final models on all {len(y_clean)} samples...")

    final_cat = CatBoostRegressor(
        iterations=800,
        depth=8,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bylevel=0.8,
        l2_leaf_reg=3.0,
        verbose=False,
        random_seed=42
    )
    final_cat.fit(X_clean, y_clean, verbose=False)

    final_rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42
    )
    final_rf.fit(X_clean, y_clean)

    return {
        'catboost': final_cat,
        'rf': final_rf,
        'weight': best_weight,
        'metrics': {
            'MAE': mae_ens,
            'Spearman': spear_ens,
            'RAE': rae_ens,
            'n_samples': len(y_clean)
        }
    }


def main():
    print("="*70)
    print("ADMET PREDICTION - CatBoost + RandomForest Ensemble")
    print("="*70)

    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test_blinded.csv")

    print(f"Training: {len(train_df)} molecules")
    print(f"Test: {len(test_df)} molecules")

    # Compute features
    print("\n" + "="*70)
    print("COMPUTING FEATURES")
    print("="*70)

    all_smiles = pd.concat([train_df['SMILES'], test_df['SMILES']]).values
    all_features = compute_features(all_smiles)

    X_train = all_features[:len(train_df)]
    X_test = all_features[len(train_df):]

    print(f"\nFeature shape: {X_train.shape}")

    # Train models
    models = {}
    cv_results = {}

    for target in TARGETS:
        y = train_df[target].values
        result = train_model_for_target(X_train, y, target)

        if result is not None:
            models[target] = result
            cv_results[target] = result['metrics']

        gc.collect()

    # Print summary
    print("\n" + "="*70)
    print("CROSS-VALIDATION SUMMARY")
    print("="*70)

    rae_values = []
    for target, metrics in cv_results.items():
        print(f"  {target}: MAE={metrics['MAE']:.4f}, Spearman={metrics['Spearman']:.4f}, RAE={metrics['RAE']:.4f}")
        rae_values.append(metrics['RAE'])

    ma_rae = np.mean(rae_values)
    print(f"\n  MACRO-AVERAGED RAE: {ma_rae:.4f}")

    # Generate predictions
    print("\n" + "="*70)
    print("GENERATING PREDICTIONS")
    print("="*70)

    predictions = {}
    for target, model_info in models.items():
        cat_pred = model_info['catboost'].predict(X_test)
        rf_pred = model_info['rf'].predict(X_test)
        weight = model_info['weight']

        ensemble_pred = weight * cat_pred + (1 - weight) * rf_pred

        # Clip to valid ranges
        min_val, max_val = VALID_RANGES[target]
        clipped = np.clip(ensemble_pred, min_val, max_val)
        n_clipped = np.sum((ensemble_pred < min_val) | (ensemble_pred > max_val))

        predictions[target] = clipped
        print(f"  {target}: clipped {n_clipped} values")

    # Create submission
    submission = pd.DataFrame({'Molecule Name': test_df['Molecule Name']})
    for target in TARGETS:
        submission[target] = predictions[target]

    output_path = SUBMISSION_DIR / "catboost_rf_ensemble.csv"
    submission.to_csv(output_path, index=False)
    print(f"\nSubmission saved to: {output_path}")
    print(f"Shape: {submission.shape}")

    # Save models
    model_path = MODEL_DIR / "catboost_rf_ensemble.joblib"
    joblib.dump(models, model_path)
    print(f"Models saved to: {model_path}")

    print("\n" + "="*70)
    print("DONE!")
    print("="*70)


if __name__ == "__main__":
    main()
