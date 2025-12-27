"""
Phase 2A Pipeline - Breaking into Top 10

Combines:
- Extended feature engineering (Morgan + MACCS + RDKit FP + Mordred)
- Stacking ensemble (XGBoost + CatBoost + LightGBM + RF with Ridge meta-learner)
- 5x5 Repeated CV for robust validation
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import gc
import argparse
import time
from datetime import datetime
warnings.filterwarnings('ignore')

from feature_engineering_v2 import FeatureEngineerV2, compute_phase2_features
from stacking_ensemble import StackingEnsemble, EndpointStackingEnsemble
from validation import RepeatedCVValidator, QuickValidator, validate_with_baselines

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

# Endpoint-specific hyperparameters (tuned from Phase 1)
HYPERPARAMS = {
    'LogD': {'max_depth': 8, 'lr': 0.03, 'n_estimators': 800},
    'KSOL': {'max_depth': 8, 'lr': 0.03, 'n_estimators': 700},
    'HLM CLint': {'max_depth': 7, 'lr': 0.05, 'n_estimators': 500},
    'MLM CLint': {'max_depth': 7, 'lr': 0.05, 'n_estimators': 500},
    'Caco-2 Permeability Papp A>B': {'max_depth': 7, 'lr': 0.04, 'n_estimators': 600},
    'Caco-2 Permeability Efflux': {'max_depth': 6, 'lr': 0.05, 'n_estimators': 400},
    'MPPB': {'max_depth': 8, 'lr': 0.03, 'n_estimators': 700},
    'MBPB': {'max_depth': 8, 'lr': 0.03, 'n_estimators': 700},
    'MGMB': {'max_depth': 6, 'lr': 0.05, 'n_estimators': 300},
}


def load_data():
    """Load train and test data"""
    train_path = BASE_DIR / "data/raw/train.csv"
    test_path = BASE_DIR / "data/raw/test_blinded.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")

    return train_df, test_df


def run_phase2_quick(train_df, test_df, feature_config='light'):
    """
    Quick Phase 2 run for rapid iteration

    Uses:
    - Light features (Morgan + MACCS only)
    - Single 5-fold CV
    - Faster training
    """
    print("\n" + "=" * 60)
    print("PHASE 2A - QUICK MODE")
    print("=" * 60)

    # Feature engineering
    print("\n[1/3] Computing features...")
    all_smiles = pd.concat([train_df['SMILES'], test_df['SMILES']]).values

    fe = FeatureEngineerV2(
        morgan_bits=1024,
        use_maccs=True,
        use_rdkit_fp=False,  # Skip for speed
        use_mordred=False,   # Skip for speed
        verbose=True
    )
    X_all = fe.compute_features(all_smiles)
    X_train = X_all[:len(train_df)]
    X_test = X_all[len(train_df):]
    print(f"Feature shape: {X_train.shape}")

    # Train stacking ensemble for each endpoint
    print("\n[2/3] Training stacking ensembles...")
    predictions = {}
    results = {}

    for target in TARGETS:
        print(f"\n{'='*40}")
        print(f"{target}")
        print(f"{'='*40}")

        y = train_df[target].values
        mask = ~np.isnan(y)
        X_t = X_train[mask]
        y_t = y[mask]
        print(f"Samples: {len(y_t)}")

        hp = HYPERPARAMS.get(target, {'max_depth': 7, 'lr': 0.05, 'n_estimators': 500})
        # Reduce iterations for quick mode
        hp_quick = {**hp, 'n_estimators': min(hp['n_estimators'], 300)}

        ensemble = StackingEnsemble(
            n_folds=5,
            use_xgb=True,
            use_catboost=True,
            use_lgb=True,
            use_rf=True,
            verbose=True
        )

        result = ensemble.fit(X_t, y_t, hp=hp_quick)
        results[target] = result

        # Predict on test set
        pred = ensemble.predict(X_test)
        pred = np.clip(pred, *VALID_RANGES[target])
        predictions[target] = pred

        # Show model weights
        weights = ensemble.get_model_weights()
        print(f"  Model weights: {weights}")

        gc.collect()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    raes = [r['RAE'] for r in results.values()]
    for t, r in results.items():
        print(f"{t}: RAE={r['RAE']:.4f}, Spearman={r['Spearman']:.4f}")
    print(f"\nMA-RAE: {np.mean(raes):.4f}")

    # Save submission
    sub = pd.DataFrame({'Molecule Name': test_df['Molecule Name']})
    for t in TARGETS:
        sub[t] = predictions[t]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = BASE_DIR / f"submissions/phase2_quick_{timestamp}.csv"
    sub.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    return results, out_path


def run_phase2_full(train_df, test_df):
    """
    Full Phase 2 run for best performance

    Uses:
    - Full features (Morgan + MACCS + RDKit FP + Mordred)
    - 5x5 repeated CV for validation
    - Full training iterations
    """
    print("\n" + "=" * 60)
    print("PHASE 2A - FULL MODE")
    print("=" * 60)

    # Feature engineering
    print("\n[1/3] Computing full features...")
    all_smiles = pd.concat([train_df['SMILES'], test_df['SMILES']]).values

    fe = FeatureEngineerV2(
        morgan_bits=1024,
        use_maccs=True,
        use_rdkit_fp=True,
        use_mordred=True,
        verbose=True
    )
    X_all = fe.compute_features(all_smiles)
    X_train = X_all[:len(train_df)]
    X_test = X_all[len(train_df):]
    print(f"Feature shape: {X_train.shape}")

    # Train stacking ensemble for each endpoint
    print("\n[2/3] Training stacking ensembles with 5x5 CV...")
    predictions = {}
    results = {}

    for target in TARGETS:
        print(f"\n{'='*40}")
        print(f"{target}")
        print(f"{'='*40}")

        y = train_df[target].values
        mask = ~np.isnan(y)
        X_t = X_train[mask]
        y_t = y[mask]
        print(f"Samples: {len(y_t)}")

        hp = HYPERPARAMS.get(target, {'max_depth': 7, 'lr': 0.05, 'n_estimators': 500})

        # Full stacking ensemble
        ensemble = StackingEnsemble(
            n_folds=5,
            use_xgb=True,
            use_catboost=True,
            use_lgb=True,
            use_rf=True,
            verbose=True
        )

        result = ensemble.fit(X_t, y_t, hp=hp)
        results[target] = result

        # Predict on test set
        pred = ensemble.predict(X_test)
        pred = np.clip(pred, *VALID_RANGES[target])
        predictions[target] = pred

        # Show model weights
        weights = ensemble.get_model_weights()
        print(f"  Model weights: {weights}")

        gc.collect()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    raes = [r['RAE'] for r in results.values()]
    for t, r in results.items():
        print(f"{t}: RAE={r['RAE']:.4f}, Spearman={r['Spearman']:.4f}")
    print(f"\nMA-RAE: {np.mean(raes):.4f}")

    # Save submission
    sub = pd.DataFrame({'Molecule Name': test_df['Molecule Name']})
    for t in TARGETS:
        sub[t] = predictions[t]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = BASE_DIR / f"submissions/phase2_full_{timestamp}.csv"
    sub.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    return results, out_path


def run_comparison_study(train_df, feature_config='light'):
    """
    Run comparison study between Phase 1 and Phase 2 approaches

    Uses 5x5 repeated CV for robust comparison
    """
    print("\n" + "=" * 60)
    print("COMPARISON STUDY: Phase 1 vs Phase 2")
    print("=" * 60)

    import xgboost as xgb
    from catboost import CatBoostRegressor

    # Compute features
    print("\n[1/4] Computing features...")
    all_smiles = train_df['SMILES'].values

    # Phase 1 features (Morgan only)
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors

    def compute_phase1_features(smiles_list):
        features = []
        for smi in smiles_list:
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
                    Descriptors.LabuteASA(mol), Descriptors.qed(mol),
                    Descriptors.BertzCT(mol), Descriptors.Chi0(mol),
                    Descriptors.Chi1(mol),
                ]
                features.append(np.concatenate([np.array(fp), desc]))
            except:
                features.append(np.zeros(1039))
        return np.array(features, dtype=np.float32)

    X_phase1 = compute_phase1_features(all_smiles)
    print(f"Phase 1 features: {X_phase1.shape}")

    # Phase 2 features
    fe = FeatureEngineerV2(
        morgan_bits=1024,
        use_maccs=True,
        use_rdkit_fp=False,
        use_mordred=False,
        verbose=False
    )
    X_phase2 = fe.compute_features(all_smiles)
    print(f"Phase 2 features: {X_phase2.shape}")

    # Compare on a few endpoints
    comparison_targets = ['LogD', 'KSOL', 'HLM CLint']

    print("\n[2/4] Running comparison...")

    for target in comparison_targets:
        print(f"\n{'='*40}")
        print(f"{target}")
        print(f"{'='*40}")

        y = train_df[target].values
        mask = ~np.isnan(y)

        # Phase 1: Single XGBoost
        def phase1_model_fn(X_train, y_train):
            model = xgb.XGBRegressor(
                n_estimators=300, max_depth=7, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, verbosity=0
            )
            model.fit(X_train, y_train)
            return model

        # Phase 2: Stacking ensemble
        def phase2_model_fn(X_train, y_train):
            ensemble = StackingEnsemble(n_folds=3, verbose=False)
            ensemble.fit(X_train, y_train, hp={'max_depth': 7, 'lr': 0.05, 'n_estimators': 300})
            return ensemble

        validator = RepeatedCVValidator(n_splits=5, n_repeats=3, verbose=True)

        print("  Phase 1 (XGBoost only):")
        validator.validate(phase1_model_fn, X_phase1[mask], y[mask], 'Phase1_XGB')

        print("  Phase 2 (Stacking + MACCS):")
        validator.validate(phase2_model_fn, X_phase2[mask], y[mask], 'Phase2_Stack')

        # Statistical comparison
        comparison = validator.compare_models('Phase1_XGB', 'Phase2_Stack')
        print(f"\n  Comparison:")
        for metric, stats in comparison.items():
            print(f"    {metric}: diff={stats['diff']:.4f}, p={stats['p_value']:.4f}, effect={stats['effect_size']}")

    print("\n" + "=" * 60)
    print(validator.summary_table())


def main():
    parser = argparse.ArgumentParser(description='Phase 2A Pipeline')
    parser.add_argument('--mode', choices=['quick', 'full', 'compare'], default='quick',
                       help='Running mode: quick, full, or compare')
    args = parser.parse_args()

    start_time = time.time()

    # Load data
    train_df, test_df = load_data()

    if args.mode == 'quick':
        results, out_path = run_phase2_quick(train_df, test_df)
    elif args.mode == 'full':
        results, out_path = run_phase2_full(train_df, test_df)
    elif args.mode == 'compare':
        run_comparison_study(train_df)
        out_path = None

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/60:.1f} minutes")

    if out_path:
        print(f"\nSubmission file: {out_path}")
        print("\nTo submit to leaderboard:")
        print(f"  Upload {out_path} to HuggingFace")


if __name__ == "__main__":
    main()
