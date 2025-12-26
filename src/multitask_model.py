"""
Multi-Task Learning Model for ADMET Prediction
Leverages correlations between endpoints (especially MBPB↔MGMB, LogD↔MPPB, HLM↔MLM)
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
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


def train_catboost_multitask(X, y_dict, n_folds=5):
    """
    Train CatBoost models with multi-task regularization via feature augmentation
    Uses auxiliary targets as additional features
    """
    from catboost import CatBoostRegressor

    print("\n" + "="*70)
    print("MULTI-TASK CATBOOST TRAINING")
    print("="*70)

    # Prepare auxiliary features (predictions from correlated endpoints)
    # For each endpoint, add mean predictions from other endpoints as features

    models = {}
    cv_results = {}

    # Define endpoint groups based on correlations
    correlation_groups = {
        'protein_binding': ['MPPB', 'MBPB', 'MGMB'],
        'metabolism': ['HLM CLint', 'MLM CLint'],
        'permeability': ['Caco-2 Permeability Papp A>B', 'Caco-2 Permeability Efflux'],
        'physicochemical': ['LogD', 'KSOL']
    }

    # First pass: train basic models for auxiliary features
    print("\nPhase 1: Training base models for auxiliary features...")
    base_predictions = {}

    for target in ENDPOINTS:
        y = y_dict[target]
        mask = ~np.isnan(y)
        X_clean = X[mask]
        y_clean = y[mask]

        if len(y_clean) < 50:
            print(f"  Skipping {target}: only {len(y_clean)} samples")
            continue

        print(f"  Training base model for {target} ({len(y_clean)} samples)...")

        # Simple CV to get OOF predictions
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        oof_preds = np.full(len(y), np.nan)

        for train_idx, val_idx in kf.split(X_clean):
            X_train, X_val = X_clean[train_idx], X_clean[val_idx]
            y_train, y_val = y_clean[train_idx], y_clean[val_idx]

            model = CatBoostRegressor(
                iterations=300,
                depth=6,
                learning_rate=0.05,
                subsample=0.8,
                verbose=False,
                random_seed=42
            )
            model.fit(X_train, y_train, verbose=False)

            # Map back to original indices
            original_indices = np.where(mask)[0][val_idx]
            oof_preds[original_indices] = model.predict(X_val)

        base_predictions[target] = oof_preds

    # Second pass: train enhanced models using auxiliary features
    print("\nPhase 2: Training enhanced models with auxiliary features...")

    for target in ENDPOINTS:
        y = y_dict[target]
        mask = ~np.isnan(y)

        if target not in base_predictions:
            print(f"  Skipping {target}")
            continue

        # Find related endpoints
        related = []
        for group_name, group_endpoints in correlation_groups.items():
            if target in group_endpoints:
                related = [ep for ep in group_endpoints if ep != target]
                break

        # Build augmented features
        aux_features = []
        for related_target in related:
            if related_target in base_predictions:
                pred = base_predictions[related_target].copy()
                # Fill NaN with mean
                pred[np.isnan(pred)] = np.nanmean(pred)
                aux_features.append(pred.reshape(-1, 1))

        if aux_features:
            X_augmented = np.hstack([X] + aux_features)
            print(f"  {target}: added {len(aux_features)} auxiliary features from {related}")
        else:
            X_augmented = X

        X_clean = X_augmented[mask]
        y_clean = y[mask]

        print(f"  Training enhanced model for {target} ({len(y_clean)} samples)...")

        # Cross-validation
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        oof_preds = np.zeros(len(y_clean))
        fold_models = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_clean)):
            X_train, X_val = X_clean[train_idx], X_clean[val_idx]
            y_train, y_val = y_clean[train_idx], y_clean[val_idx]

            model = CatBoostRegressor(
                iterations=500,
                depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bylevel=0.8,
                l2_leaf_reg=3.0,
                verbose=False,
                random_seed=42 + fold,
                early_stopping_rounds=50
            )

            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                verbose=False
            )

            oof_preds[val_idx] = model.predict(X_val)
            fold_models.append(model)

        # Compute metrics
        mae = mean_absolute_error(y_clean, oof_preds)
        spearman = spearmanr(y_clean, oof_preds)[0]
        rae = mae / np.mean(np.abs(y_clean - np.mean(y_clean)))

        cv_results[target] = {
            'MAE': mae,
            'Spearman': spearman,
            'RAE': rae,
            'n_samples': len(y_clean)
        }

        print(f"    MAE: {mae:.4f}, Spearman: {spearman:.4f}, RAE: {rae:.4f}")

        # Train final model on all data
        final_model = CatBoostRegressor(
            iterations=500,
            depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bylevel=0.8,
            l2_leaf_reg=3.0,
            verbose=False,
            random_seed=42
        )
        final_model.fit(X_clean, y_clean, verbose=False)

        # Store model with info about auxiliary features
        models[target] = {
            'model': final_model,
            'related_endpoints': related,
            'base_predictions': {ep: base_predictions.get(ep) for ep in related if ep in base_predictions}
        }

    # Print summary
    print("\n" + "="*70)
    print("CROSS-VALIDATION RESULTS")
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

    return models, cv_results, base_predictions


def predict_multitask(models, X, base_preds_for_test=None):
    """Make predictions using multi-task models"""
    from catboost import CatBoostRegressor

    predictions = {}

    # First, generate base predictions for test data if not provided
    if base_preds_for_test is None:
        base_preds_for_test = {}
        for target, model_info in models.items():
            model = model_info['model']
            # Use just X for base prediction (no augmentation)
            base_preds_for_test[target] = model.predict(X[:, :X.shape[1]])

    # Now make final predictions with auxiliary features
    for target, model_info in models.items():
        model = model_info['model']
        related = model_info['related_endpoints']

        # Build augmented features for test
        aux_features = []
        for related_target in related:
            if related_target in base_preds_for_test:
                aux_features.append(base_preds_for_test[related_target].reshape(-1, 1))

        if aux_features:
            X_augmented = np.hstack([X] + aux_features)
        else:
            X_augmented = X

        # Check if model expects augmented features
        expected_features = model.feature_count_
        if X_augmented.shape[1] != expected_features:
            # Fall back to base features
            X_augmented = X

        predictions[target] = model.predict(X_augmented)

        # Clip to valid range
        min_val, max_val = VALID_RANGES[target]
        predictions[target] = np.clip(predictions[target], min_val, max_val)

    return predictions


def main():
    print("="*70)
    print("MULTI-TASK ADMET PREDICTION MODEL")
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

    # Train multi-task models
    models, cv_results, base_predictions = train_catboost_multitask(X_train, y_dict)

    # Generate test predictions
    print("\n" + "="*70)
    print("GENERATING TEST PREDICTIONS")
    print("="*70)

    # First pass: base predictions
    base_test_preds = {}
    for target, model_info in models.items():
        model = model_info['model']
        # Predict with base features only
        base_test_preds[target] = model.predict(X_test)

    # Second pass: augmented predictions
    predictions = predict_multitask(models, X_test, base_test_preds)

    # Create submission
    submission = pd.DataFrame({'Molecule Name': test_df['Molecule Name']})

    for target in ENDPOINTS:
        if target in predictions:
            submission[target] = predictions[target]
        else:
            # Use median from training
            submission[target] = train_df[target].median()

    # Reorder columns to match expected format
    columns = ['Molecule Name'] + ENDPOINTS
    submission = submission[columns]

    # Save
    output_path = SUBMIT_DIR / "multitask_submission.csv"
    submission.to_csv(output_path, index=False)
    print(f"\nSaved submission to {output_path}")

    # Print summary statistics
    print("\nSUBMISSION STATISTICS:")
    for col in ENDPOINTS:
        print(f"  {col}: mean={submission[col].mean():.2f}, std={submission[col].std():.2f}")

    return models, cv_results


if __name__ == "__main__":
    main()
