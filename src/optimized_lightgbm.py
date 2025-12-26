"""
Optimized LightGBM model with tuned hyperparameters
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import gc
warnings.filterwarnings('ignore')

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
import lightgbm as lgb

BASE_DIR = Path(__file__).parent.parent

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

# Endpoint-specific hyperparameters
HYPERPARAMS = {
    'LogD': {'num_leaves': 64, 'lr': 0.03, 'n_estimators': 1000},
    'KSOL': {'num_leaves': 64, 'lr': 0.03, 'n_estimators': 800},
    'HLM CLint': {'num_leaves': 48, 'lr': 0.05, 'n_estimators': 600},
    'MLM CLint': {'num_leaves': 48, 'lr': 0.05, 'n_estimators': 600},
    'Caco-2 Permeability Papp A>B': {'num_leaves': 48, 'lr': 0.04, 'n_estimators': 700},
    'Caco-2 Permeability Efflux': {'num_leaves': 32, 'lr': 0.05, 'n_estimators': 500},
    'MPPB': {'num_leaves': 64, 'lr': 0.03, 'n_estimators': 800},
    'MBPB': {'num_leaves': 64, 'lr': 0.03, 'n_estimators': 800},
    'MGMB': {'num_leaves': 32, 'lr': 0.05, 'n_estimators': 400},
}

def compute_features(smiles_list):
    """Morgan FP (1024 bits) + 15 descriptors"""
    print(f"Computing features for {len(smiles_list)} molecules...")
    features = []
    for i, smi in enumerate(smiles_list):
        if (i+1) % 1000 == 0:
            print(f"  {i+1}/{len(smiles_list)}")
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            features.append(np.zeros(1039))
            continue
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            desc = [
                Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
                Descriptors.TPSA(mol), Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol), Descriptors.NumRotatableBonds(mol),
                Descriptors.NumAromaticRings(mol), Descriptors.FractionCSP3(mol),
                Descriptors.HeavyAtomCount(mol), Descriptors.RingCount(mol),
                rdMolDescriptors.CalcNumAliphaticRings(mol),
                rdMolDescriptors.CalcNumHeterocycles(mol),
                Descriptors.LabuteASA(mol),
                Descriptors.qed(mol),
                rdMolDescriptors.CalcNumAmideBonds(mol)
            ]
            features.append(np.concatenate([np.array(fp), desc]))
        except:
            features.append(np.zeros(1039))
    return np.array(features, dtype=np.float32)

def main():
    print("="*60)
    print("OPTIMIZED LIGHTGBM MODEL")
    print("="*60)

    train_df = pd.read_csv(BASE_DIR / "data/raw/train.csv")
    test_df = pd.read_csv(BASE_DIR / "data/raw/test_blinded.csv")
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    all_smiles = pd.concat([train_df['SMILES'], test_df['SMILES']]).values
    X_all = compute_features(all_smiles)
    X_train, X_test = X_all[:len(train_df)], X_all[len(train_df):]
    print(f"Feature shape: {X_train.shape}")

    predictions = {}
    results = {}

    for target in TARGETS:
        hp = HYPERPARAMS[target]
        print(f"\n{'='*40}\n{target} (leaves={hp['num_leaves']}, lr={hp['lr']}, n={hp['n_estimators']})\n{'='*40}")

        y = train_df[target].values
        mask = ~np.isnan(y)
        X_t, y_t = X_train[mask], y[mask]
        print(f"Samples: {len(y_t)}")

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        oof = np.zeros(len(y_t))

        for fold, (tr_idx, va_idx) in enumerate(kf.split(X_t)):
            model = lgb.LGBMRegressor(
                n_estimators=hp['n_estimators'],
                num_leaves=hp['num_leaves'],
                learning_rate=hp['lr'],
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=3.0,
                random_state=42+fold,
                n_jobs=-1,
                verbose=-1
            )
            model.fit(X_t[tr_idx], y_t[tr_idx],
                     eval_set=[(X_t[va_idx], y_t[va_idx])],
                     callbacks=[lgb.early_stopping(100, verbose=False)])
            oof[va_idx] = model.predict(X_t[va_idx])

        mae = mean_absolute_error(y_t, oof)
        spear = spearmanr(y_t, oof)[0]
        rae = mae / np.mean(np.abs(y_t - np.mean(y_t)))
        print(f"MAE={mae:.4f}, Spearman={spear:.4f}, RAE={rae:.4f}")
        results[target] = {'MAE': mae, 'Spearman': spear, 'RAE': rae}

        # Final model on all data
        final = lgb.LGBMRegressor(
            n_estimators=hp['n_estimators'],
            num_leaves=hp['num_leaves'],
            learning_rate=hp['lr'],
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=3.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        final.fit(X_t, y_t)

        pred = final.predict(X_test)
        pred = np.clip(pred, *VALID_RANGES[target])
        predictions[target] = pred
        gc.collect()

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    raes = [r['RAE'] for r in results.values()]
    for t, r in results.items():
        print(f"{t}: RAE={r['RAE']:.4f}")
    print(f"\nMA-RAE: {np.mean(raes):.4f}")

    sub = pd.DataFrame({'Molecule Name': test_df['Molecule Name']})
    for t in TARGETS:
        sub[t] = predictions[t]
    out_path = BASE_DIR / "submissions/optimized_lightgbm.csv"
    sub.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
