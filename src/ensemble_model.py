"""
Ensemble Model for ADMET Prediction Challenge
Uses multiple CatBoost configurations with different hyperparameters
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem import MACCSkeys
from rdkit.DataStructs import cDataStructs

# ML
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
from catboost import CatBoostRegressor
import joblib

# Define paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
MODEL_DIR = BASE_DIR / "models"
SUBMIT_DIR = BASE_DIR / "submissions"

# Endpoints
ENDPOINTS = ['LogD', 'KSOL', 'HLM CLint', 'MLM CLint',
             'Caco-2 Permeability Papp A>B', 'Caco-2 Permeability Efflux',
             'MPPB', 'MBPB', 'MGMB']

# Valid ranges for clipping
VALID_RANGES = {
    'LogD': (-2.5, 6.0),
    'KSOL': (0.0, 400.0),
    'HLM CLint': (0.0, 3000.0),
    'MLM CLint': (0.0, 12000.0),
    'Caco-2 Permeability Papp A>B': (0.0, 60.0),
    'Caco-2 Permeability Efflux': (0.0, 120.0),
    'MPPB': (0.0, 100.0),
    'MBPB': (0.0, 100.0),
    'MGMB': (0.0, 100.0)
}


def smiles_to_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Generate Morgan fingerprint from SMILES"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.int8)
    cDataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def smiles_to_maccs(smiles: str) -> np.ndarray:
    """Generate MACCS keys from SMILES"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(167)
    fp = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros(167, dtype=np.int8)
    cDataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def smiles_to_descriptors(smiles: str) -> np.ndarray:
    """Generate molecular descriptors from SMILES"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(20)

    try:
        descriptors = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.NumHeteroatoms(mol),
            Descriptors.FractionCSP3(mol),
            rdMolDescriptors.CalcNumRings(mol),
            rdMolDescriptors.CalcNumHeavyAtoms(mol),
            Descriptors.NumAliphaticRings(mol),
            Descriptors.NumSaturatedRings(mol),
            Descriptors.RingCount(mol),
            Descriptors.HeavyAtomMolWt(mol),
            Descriptors.NumValenceElectrons(mol),
            Descriptors.NumRadicalElectrons(mol),
            Descriptors.MaxPartialCharge(mol) if Descriptors.MaxPartialCharge(mol) else 0,
            Descriptors.MinPartialCharge(mol) if Descriptors.MinPartialCharge(mol) else 0,
            Descriptors.MaxAbsPartialCharge(mol) if Descriptors.MaxAbsPartialCharge(mol) else 0,
        ]
        return np.array(descriptors, dtype=np.float32)
    except:
        return np.zeros(20)


def generate_features(smiles_list, verbose=True):
    """Generate combined features for all molecules"""
    from tqdm import tqdm

    features = []
    iterator = tqdm(smiles_list, desc="Generating features") if verbose else smiles_list

    for smiles in iterator:
        morgan = smiles_to_fingerprint(smiles)
        maccs = smiles_to_maccs(smiles)
        desc = smiles_to_descriptors(smiles)
        combined = np.concatenate([morgan, maccs, desc])
        features.append(combined)

    return np.array(features)


# Different CatBoost configurations for ensemble diversity
CATBOOST_CONFIGS = [
    {
        'name': 'deep',
        'params': {
            'iterations': 1000,
            'depth': 10,
            'learning_rate': 0.02,
            'subsample': 0.8,
            'colsample_bylevel': 0.8,
            'l2_leaf_reg': 3.0,
            'random_strength': 1.0,
            'verbose': False,
            'early_stopping_rounds': 100
        }
    },
    {
        'name': 'shallow_fast',
        'params': {
            'iterations': 1500,
            'depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.7,
            'colsample_bylevel': 0.7,
            'l2_leaf_reg': 1.0,
            'random_strength': 0.5,
            'verbose': False,
            'early_stopping_rounds': 100
        }
    },
    {
        'name': 'regularized',
        'params': {
            'iterations': 1000,
            'depth': 8,
            'learning_rate': 0.03,
            'subsample': 0.9,
            'colsample_bylevel': 0.6,
            'l2_leaf_reg': 10.0,
            'random_strength': 2.0,
            'verbose': False,
            'early_stopping_rounds': 100
        }
    }
]


def train_ensemble(X, y_dict, n_folds=5):
    """Train CatBoost ensemble with different configurations"""

    print("\n" + "="*70)
    print("TRAINING MULTI-CONFIG CATBOOST ENSEMBLE")
    print("="*70)

    models = {}
    cv_results = {}

    for target in ENDPOINTS:
        if target not in y_dict:
            continue

        y = y_dict[target]
        mask = ~np.isnan(y)
        X_clean = X[mask]
        y_clean = y[mask]

        if len(y_clean) < 50:
            print(f"\nSkipping {target}: only {len(y_clean)} samples")
            continue

        print(f"\n{'='*70}")
        print(f"Training models for: {target} ({len(y_clean)} samples)")
        print(f"{'='*70}")

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        # Store OOF predictions for each config
        oof_preds = {cfg['name']: np.zeros(len(y_clean)) for cfg in CATBOOST_CONFIGS}
        final_models = {cfg['name']: [] for cfg in CATBOOST_CONFIGS}

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_clean)):
            X_train, X_val = X_clean[train_idx], X_clean[val_idx]
            y_train, y_val = y_clean[train_idx], y_clean[val_idx]

            print(f"  Fold {fold+1}/{n_folds}...")

            for cfg in CATBOOST_CONFIGS:
                params = cfg['params'].copy()
                params['random_seed'] = 42 + fold

                model = CatBoostRegressor(**params)
                model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
                oof_preds[cfg['name']][val_idx] = model.predict(X_val)
                final_models[cfg['name']].append(model)

        # Find optimal weights for ensemble
        print("\n  Finding optimal ensemble weights...")

        best_weights = None
        best_mae = float('inf')

        # Grid search over weight combinations
        for w1 in np.arange(0.2, 0.6, 0.1):
            for w2 in np.arange(0.2, 0.6, 0.1):
                w3 = 1 - w1 - w2
                if w3 < 0.1 or w3 > 0.6:
                    continue

                weights = [w1, w2, w3]
                oof_ensemble = np.zeros(len(y_clean))
                for i, cfg in enumerate(CATBOOST_CONFIGS):
                    oof_ensemble += weights[i] * oof_preds[cfg['name']]

                mae = mean_absolute_error(y_clean, oof_ensemble)
                if mae < best_mae:
                    best_mae = mae
                    best_weights = weights

        # Use equal weights if optimization fails
        if best_weights is None:
            best_weights = [1/len(CATBOOST_CONFIGS)] * len(CATBOOST_CONFIGS)

        # Final ensemble prediction
        oof_ensemble = np.zeros(len(y_clean))
        for i, cfg in enumerate(CATBOOST_CONFIGS):
            oof_ensemble += best_weights[i] * oof_preds[cfg['name']]

        # Compute metrics
        mae_ens = mean_absolute_error(y_clean, oof_ensemble)
        spearman_ens = spearmanr(y_clean, oof_ensemble)[0]
        rae_ens = mae_ens / np.mean(np.abs(y_clean - np.mean(y_clean)))

        print(f"\n  Results for {target}:")
        for i, cfg in enumerate(CATBOOST_CONFIGS):
            mae = mean_absolute_error(y_clean, oof_preds[cfg['name']])
            spear = spearmanr(y_clean, oof_preds[cfg['name']])[0]
            print(f"    {cfg['name']:<15} - MAE: {mae:.4f}, Spearman: {spear:.4f}, weight: {best_weights[i]:.2f}")
        print(f"    {'ENSEMBLE':<15} - MAE: {mae_ens:.4f}, Spearman: {spearman_ens:.4f}, RAE: {rae_ens:.4f}")

        cv_results[target] = {
            'MAE': mae_ens,
            'Spearman': spearman_ens,
            'RAE': rae_ens,
            'n_samples': len(y_clean),
            'weights': best_weights
        }

        # Train final models on all data
        print(f"\n  Training final models on all {len(y_clean)} samples...")

        trained_models = {}
        for cfg in CATBOOST_CONFIGS:
            params = cfg['params'].copy()
            params['random_seed'] = 42
            params.pop('early_stopping_rounds', None)  # Remove for final training

            model = CatBoostRegressor(**params)
            model.fit(X_clean, y_clean, verbose=False)
            trained_models[cfg['name']] = model

        models[target] = {
            'models': trained_models,
            'weights': best_weights
        }

    # Print summary
    print("\n" + "="*70)
    print("CROSS-VALIDATION SUMMARY")
    print("="*70)
    print(f"{'Endpoint':<35} {'N':>6} {'MAE':>10} {'Spearman':>10} {'RAE':>10}")
    print("-"*70)

    total_rae = 0
    for target, metrics in cv_results.items():
        print(f"{target:<35} {metrics['n_samples']:>6} {metrics['MAE']:>10.4f} {metrics['Spearman']:>10.4f} {metrics['RAE']:>10.4f}")
        total_rae += metrics['RAE']

    ma_rae = total_rae / len(cv_results)
    print("-"*70)
    print(f"{'Macro-Averaged RAE':<35} {'':>6} {'':>10} {'':>10} {ma_rae:>10.4f}")

    return models, cv_results


def predict_ensemble(models, X):
    """Make predictions using the trained ensemble"""
    predictions = {}

    for target, model_info in models.items():
        ensemble_pred = np.zeros(len(X))

        for i, (name, model) in enumerate(model_info['models'].items()):
            pred = model.predict(X)
            ensemble_pred += model_info['weights'][i] * pred

        # Clip to valid ranges
        min_val, max_val = VALID_RANGES[target]
        predictions[target] = np.clip(ensemble_pred, min_val, max_val)

    return predictions


def main():
    print("="*70)
    print("ADMET ENSEMBLE PREDICTION MODEL")
    print("Multi-Config CatBoost Ensemble")
    print("="*70)

    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test_blinded.csv")

    print(f"Training: {len(train_df)} molecules")
    print(f"Test: {len(test_df)} molecules")

    # Generate features
    print("\nGenerating training features...")
    X_train = generate_features(train_df['SMILES'].values)
    print(f"Training features shape: {X_train.shape}")

    print("\nGenerating test features...")
    X_test = generate_features(test_df['SMILES'].values)

    # Prepare targets
    y_dict = {}
    for target in ENDPOINTS:
        if target in train_df.columns:
            y_dict[target] = train_df[target].values

    # Train ensemble
    models, cv_results = train_ensemble(X_train, y_dict)

    # Save models
    print("\nSaving models...")
    for target, model_info in models.items():
        joblib.dump(model_info, MODEL_DIR / f"ensemble_{target.replace(' ', '_').replace('-', '_')}.joblib")

    # Generate predictions
    print("\n" + "="*70)
    print("GENERATING TEST PREDICTIONS")
    print("="*70)

    predictions = predict_ensemble(models, X_test)

    # Create submission
    submission = pd.DataFrame({'Molecule Name': test_df['Molecule Name']})

    for target in ENDPOINTS:
        if target in predictions:
            submission[target] = predictions[target]
        else:
            # Use median from training for missing endpoints
            submission[target] = train_df[target].median()

    # Reorder columns
    columns = ['Molecule Name'] + ENDPOINTS
    submission = submission[columns]

    # Save
    output_path = SUBMIT_DIR / "ensemble_submission.csv"
    submission.to_csv(output_path, index=False)
    print(f"\nSaved submission to {output_path}")

    # Print summary statistics
    print("\nSUBMISSION STATISTICS:")
    for col in ENDPOINTS:
        print(f"  {col}: mean={submission[col].mean():.2f}, std={submission[col].std():.2f}")

    return models, cv_results


if __name__ == "__main__":
    main()
