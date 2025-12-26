"""
Simple CatBoost model - lightweight, memory-efficient
Uses only Morgan fingerprints + key descriptors
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import gc
warnings.filterwarnings('ignore')

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
from catboost import CatBoostRegressor

BASE_DIR = Path("/Users/tp53/Documents/code_macbook-air-m1__tp53/26dec2025_opeadmet-challenge")

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

def compute_features(smiles_list):
    """Morgan FP (512 bits) + 10 descriptors = 522 features"""
    print(f"Computing features for {len(smiles_list)} molecules...")
    features = []
    for i, smi in enumerate(smiles_list):
        if (i+1) % 1000 == 0:
            print(f"  {i+1}/{len(smiles_list)}")
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            features.append(np.zeros(522))
            continue
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)
            desc = [
                Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
                Descriptors.TPSA(mol), Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol), Descriptors.NumRotatableBonds(mol),
                Descriptors.NumAromaticRings(mol), Descriptors.FractionCSP3(mol),
                Descriptors.HeavyAtomCount(mol), Descriptors.RingCount(mol)
            ]
            features.append(np.concatenate([np.array(fp), desc]))
        except:
            features.append(np.zeros(522))
    return np.array(features, dtype=np.float32)

def main():
    print("="*60)
    print("SIMPLE CATBOOST MODEL")
    print("="*60)

    train_df = pd.read_csv(BASE_DIR / "data/raw/train.csv")
    test_df = pd.read_csv(BASE_DIR / "data/raw/test_blinded.csv")
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    # Compute features
    all_smiles = pd.concat([train_df['SMILES'], test_df['SMILES']]).values
    X_all = compute_features(all_smiles)
    X_train, X_test = X_all[:len(train_df)], X_all[len(train_df):]
    print(f"Feature shape: {X_train.shape}")

    predictions = {}
    results = {}

    for target in TARGETS:
        print(f"\n{'='*40}\n{target}\n{'='*40}")
        y = train_df[target].values
        mask = ~np.isnan(y)
        X_t, y_t = X_train[mask], y[mask]
        print(f"Samples: {len(y_t)}")

        # 5-fold CV
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        oof = np.zeros(len(y_t))

        for fold, (tr_idx, va_idx) in enumerate(kf.split(X_t)):
            model = CatBoostRegressor(
                iterations=500, depth=7, learning_rate=0.05,
                subsample=0.8, verbose=False, random_seed=42+fold,
                early_stopping_rounds=50
            )
            model.fit(X_t[tr_idx], y_t[tr_idx],
                     eval_set=(X_t[va_idx], y_t[va_idx]), verbose=False)
            oof[va_idx] = model.predict(X_t[va_idx])

        mae = mean_absolute_error(y_t, oof)
        spear = spearmanr(y_t, oof)[0]
        rae = mae / np.mean(np.abs(y_t - np.mean(y_t)))
        print(f"MAE={mae:.4f}, Spearman={spear:.4f}, RAE={rae:.4f}")
        results[target] = {'MAE': mae, 'Spearman': spear, 'RAE': rae}

        # Final model
        final = CatBoostRegressor(
            iterations=500, depth=7, learning_rate=0.05,
            subsample=0.8, verbose=False, random_seed=42
        )
        final.fit(X_t, y_t, verbose=False)

        pred = final.predict(X_test)
        pred = np.clip(pred, *VALID_RANGES[target])
        predictions[target] = pred
        gc.collect()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    raes = [r['RAE'] for r in results.values()]
    for t, r in results.items():
        print(f"{t}: RAE={r['RAE']:.4f}")
    print(f"\nMA-RAE: {np.mean(raes):.4f}")

    # Save submission
    sub = pd.DataFrame({'Molecule Name': test_df['Molecule Name']})
    for t in TARGETS:
        sub[t] = predictions[t]
    out_path = BASE_DIR / "submissions/simple_catboost.csv"
    sub.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
