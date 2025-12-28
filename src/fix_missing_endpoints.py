"""
Quick script to fill in missing endpoints in v4 submission.
Uses LightGBM with extended fingerprints for speed.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
import lightgbm as lgb
from tqdm import tqdm

# Setup paths
BASE_DIR = Path(__file__).parent.parent
TRAIN_PATH = BASE_DIR / 'data' / 'raw' / 'train.csv'
TEST_PATH = BASE_DIR / 'data' / 'raw' / 'test_blinded.csv'
SUBMISSION_PATH = BASE_DIR / 'submissions' / 'v4_submission_20251228_0729.csv'
OUTPUT_PATH = BASE_DIR / 'submissions' / 'v4_submission_fixed.csv'

# Import fingerprints
import sys
sys.path.insert(0, str(BASE_DIR / 'src'))
from extended_fingerprints import compute_all_features

# Missing endpoints (exact names from train.csv)
MISSING_ENDPOINTS = [
    'HLM CLint',
    'MLM CLint',
    'Caco-2 Permeability Papp A>B',
    'Caco-2 Permeability Efflux'
]

def train_lgb_ensemble(X_train, y_train, X_test, n_seeds=3, n_folds=5):
    """Train LightGBM ensemble with CV."""
    n_test = len(X_test)
    test_preds = np.zeros((n_test, n_seeds))

    for seed in range(n_seeds):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42 + seed * 100)
        fold_preds = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            model = lgb.LGBMRegressor(
                n_estimators=1000,
                max_depth=7,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42 + seed * 100 + fold,
                verbose=-1,
                n_jobs=-1
            )

            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )

            fold_preds.append(model.predict(X_test))

        test_preds[:, seed] = np.mean(fold_preds, axis=0)

    return np.mean(test_preds, axis=1)


def main():
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    submission = pd.read_csv(SUBMISSION_PATH)

    smiles_test = test_df['SMILES'].tolist()

    print(f"\nExisting submission columns: {list(submission.columns)}")
    print(f"Missing endpoints: {MISSING_ENDPOINTS}")

    # Compute features for test set once
    print("\nComputing test set features...")
    test_features = compute_all_features(smiles_test, verbose=True)
    X_test = test_features['comprehensive']

    # Train each missing endpoint
    for endpoint in MISSING_ENDPOINTS:
        print(f"\n{'='*50}")
        print(f"Training: {endpoint}")
        print(f"{'='*50}")

        if endpoint not in train_df.columns:
            print(f"  ERROR: {endpoint} not in training data!")
            continue

        # Get endpoint data
        mask = train_df[endpoint].notna()
        endpoint_smiles = train_df.loc[mask, 'SMILES'].tolist()
        endpoint_y = train_df.loc[mask, endpoint].values

        print(f"  Samples: {len(endpoint_smiles)}")

        # Compute features for training
        print("  Computing training features...")
        train_features = compute_all_features(endpoint_smiles, verbose=False)
        X_train = train_features['comprehensive']

        # Train
        print("  Training LightGBM ensemble...")
        predictions = train_lgb_ensemble(X_train, endpoint_y, X_test, n_seeds=3, n_folds=5)

        # Add to submission
        submission[endpoint] = predictions
        print(f"  Done! Predictions range: [{predictions.min():.3f}, {predictions.max():.3f}]")

    # Reorder columns to match expected order
    expected_order = [
        'SMILES', 'LogD', 'KSOL', 'HLM CLint', 'MLM CLint',
        'Caco-2 Permeability Papp A>B', 'Caco-2 Permeability Efflux',
        'MPPB', 'MBPB', 'MGMB'
    ]

    # Only include columns that exist
    final_columns = [c for c in expected_order if c in submission.columns]
    submission = submission[final_columns]

    # Save
    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"\n{'='*50}")
    print(f"Fixed submission saved to: {OUTPUT_PATH}")
    print(f"Columns: {list(submission.columns)}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
