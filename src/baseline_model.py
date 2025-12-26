"""
Baseline Model: XGBoost with Combined Fingerprints
Based on ADMETboost approach (ranked #1 in 18/22 TDC benchmarks)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# RDKit for fingerprints
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.Avalon.pyAvalonTools import GetAvalonFP

# ML
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
from catboost import CatBoostRegressor
from tqdm import tqdm
import joblib

# Constants
TARGETS = [
    'LogD', 'KSOL', 'HLM CLint', 'MLM CLint',
    'Caco-2 Permeability Papp A>B', 'Caco-2 Permeability Efflux',
    'MPPB', 'MBPB', 'MGMB'
]

DATA_DIR = Path("data/raw")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def smiles_to_mol(smiles: str):
    """Convert SMILES to RDKit mol object"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except:
        return None


def compute_fingerprints(smiles_list: List[str], verbose: bool = True) -> np.ndarray:
    """
    Compute combined fingerprints: ECFP4 + Avalon + MACCS + RDKit descriptors
    Based on ADMETboost optimal combination
    """
    features_list = []

    iterator = tqdm(smiles_list, desc="Computing fingerprints") if verbose else smiles_list

    for smi in iterator:
        mol = smiles_to_mol(smi)

        if mol is None:
            # Return zeros if molecule can't be parsed
            features_list.append(np.zeros(2048 + 512 + 167 + 200))
            continue

        # 1. ECFP4 (Morgan fingerprint, radius=2, 2048 bits)
        ecfp4 = GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        ecfp4_arr = np.array(ecfp4)

        # 2. Avalon fingerprint (512 bits)
        avalon = GetAvalonFP(mol, nBits=512)
        avalon_arr = np.array(avalon)

        # 3. MACCS keys (167 bits)
        maccs = MACCSkeys.GenMACCSKeys(mol)
        maccs_arr = np.array(maccs)

        # 4. RDKit 2D descriptors (200 most common)
        desc_dict = {
            'MolWt': Descriptors.MolWt(mol),
            'MolLogP': Descriptors.MolLogP(mol),
            'TPSA': Descriptors.TPSA(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),
            'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
            'NumAromaticRings': Descriptors.NumAromaticRings(mol),
            'NumHeavyAtoms': Descriptors.HeavyAtomCount(mol),
            'FractionCSP3': Descriptors.FractionCSP3(mol),
            'NumAliphaticRings': Descriptors.NumAliphaticRings(mol),
            'NumSaturatedRings': Descriptors.NumSaturatedRings(mol),
            'RingCount': Descriptors.RingCount(mol),
            'LabuteASA': Descriptors.LabuteASA(mol),
            'BalabanJ': Descriptors.BalabanJ(mol) if Descriptors.RingCount(mol) > 0 else 0,
            'BertzCT': Descriptors.BertzCT(mol),
            'Chi0': Descriptors.Chi0(mol),
            'Chi1': Descriptors.Chi1(mol),
            'HallKierAlpha': Descriptors.HallKierAlpha(mol),
            'Kappa1': Descriptors.Kappa1(mol),
            'Kappa2': Descriptors.Kappa2(mol),
            'MaxPartialCharge': Descriptors.MaxPartialCharge(mol),
            'MinPartialCharge': Descriptors.MinPartialCharge(mol),
            'NumValenceElectrons': Descriptors.NumValenceElectrons(mol),
            'MolMR': Descriptors.MolMR(mol),
        }

        # Pad to 200 with zeros
        desc_arr = np.array(list(desc_dict.values()))
        desc_arr = np.nan_to_num(desc_arr, nan=0.0, posinf=0.0, neginf=0.0)
        desc_padded = np.zeros(200)
        desc_padded[:len(desc_arr)] = desc_arr

        # Combine all features
        features = np.concatenate([ecfp4_arr, avalon_arr, maccs_arr, desc_padded])
        features_list.append(features)

    return np.array(features_list)


def train_catboost_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None
) -> CatBoostRegressor:
    """Train CatBoost regressor with optimized hyperparameters"""

    model = CatBoostRegressor(
        iterations=500,
        depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bylevel=0.8,
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=False,
        early_stopping_rounds=50 if X_val is not None else None,
    )

    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=False
        )
    else:
        model.fit(X_train, y_train)

    return model


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MAE and Spearman correlation"""
    mask = ~np.isnan(y_true)
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    spearman_r, _ = spearmanr(y_true_clean, y_pred_clean)

    return {'MAE': mae, 'Spearman_R': spearman_r}


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5
) -> Tuple[np.ndarray, Dict[str, float]]:
    """K-fold cross validation"""

    # Remove samples with missing target
    mask = ~np.isnan(y)
    X_clean = X[mask]
    y_clean = y[mask]

    if len(y_clean) < n_folds:
        return np.full(len(y), np.nan), {'MAE': np.nan, 'Spearman_R': np.nan}

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_preds = np.full(len(y_clean), np.nan)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_clean)):
        X_train, X_val = X_clean[train_idx], X_clean[val_idx]
        y_train, y_val = y_clean[train_idx], y_clean[val_idx]

        model = train_catboost_model(X_train, y_train, X_val, y_val)
        oof_preds[val_idx] = model.predict(X_val)

    metrics = compute_metrics(y_clean, oof_preds)

    # Map back to original indices
    full_preds = np.full(len(y), np.nan)
    full_preds[mask] = oof_preds

    return full_preds, metrics


def train_final_models(
    X: np.ndarray,
    y_dict: Dict[str, np.ndarray]
) -> Dict[str, CatBoostRegressor]:
    """Train final models on all data for each target"""

    models = {}

    for target in TARGETS:
        y = y_dict[target]
        mask = ~np.isnan(y)

        if mask.sum() < 10:
            print(f"  Skipping {target} - insufficient data ({mask.sum()} samples)")
            continue

        X_clean = X[mask]
        y_clean = y[mask]

        model = train_catboost_model(X_clean, y_clean)
        models[target] = model

    return models


def main():
    print("=" * 70)
    print("BASELINE MODEL: XGBoost + Combined Fingerprints")
    print("=" * 70)

    # Load data
    print("\n1. Loading data...")
    df_train = pd.read_csv(DATA_DIR / "train.csv")
    df_test = pd.read_csv(DATA_DIR / "test_blinded.csv")

    print(f"   Training samples: {len(df_train)}")
    print(f"   Test samples: {len(df_test)}")

    # Compute fingerprints
    print("\n2. Computing fingerprints for training data...")
    X_train = compute_fingerprints(df_train['SMILES'].tolist())
    print(f"   Feature shape: {X_train.shape}")

    print("\n3. Computing fingerprints for test data...")
    X_test = compute_fingerprints(df_test['SMILES'].tolist())

    # Prepare targets
    y_dict = {target: df_train[target].values for target in TARGETS}

    # Cross-validation
    print("\n4. Cross-validation (5-fold)...")
    print("-" * 70)

    cv_results = {}
    for target in TARGETS:
        y = y_dict[target]
        n_samples = (~np.isnan(y)).sum()

        oof_preds, metrics = cross_validate(X_train, y, n_folds=5)
        cv_results[target] = metrics

        print(f"   {target:35s} | N={n_samples:4d} | MAE={metrics['MAE']:.4f} | R={metrics['Spearman_R']:.4f}")

    # Train final models
    print("\n5. Training final models on all data...")
    models = train_final_models(X_train, y_dict)

    # Save models
    print("\n6. Saving models...")
    for target, model in models.items():
        safe_name = target.replace(' ', '_').replace('>', '').replace('-', '_')
        model_path = MODEL_DIR / f"xgb_{safe_name}.joblib"
        joblib.dump(model, model_path)
    print(f"   Saved {len(models)} models to {MODEL_DIR}")

    # Generate test predictions
    print("\n7. Generating test predictions...")
    predictions = pd.DataFrame({'Molecule Name': df_test['Molecule Name']})

    for target in TARGETS:
        if target in models:
            preds = models[target].predict(X_test)
            predictions[target] = preds
        else:
            predictions[target] = np.nan

    # Save predictions
    submission_path = Path("submissions") / "baseline_xgb_submission.csv"
    submission_path.parent.mkdir(exist_ok=True)
    predictions.to_csv(submission_path, index=False)
    print(f"   Saved predictions to {submission_path}")

    # Summary
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 70)

    avg_mae = np.nanmean([r['MAE'] for r in cv_results.values()])
    avg_r = np.nanmean([r['Spearman_R'] for r in cv_results.values()])

    print(f"\nAverage MAE: {avg_mae:.4f}")
    print(f"Average Spearman R: {avg_r:.4f}")

    return cv_results, predictions


if __name__ == "__main__":
    cv_results, predictions = main()
