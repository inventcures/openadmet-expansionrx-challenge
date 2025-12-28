"""
Quick script to fill in missing endpoints in v4 submission.
Uses LightGBM with extended fingerprints for speed.

Usage:
    python src/fix_missing_endpoints.py
    python src/fix_missing_endpoints.py --input submissions/v4_submission_20251228_0729.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
import lightgbm as lgb
from tqdm import tqdm
import sys

# Setup paths
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / 'src'))

from extended_fingerprints import compute_all_features

# All required endpoints (exact names from train.csv / competition)
ALL_ENDPOINTS = [
    'LogD',
    'KSOL',
    'HLM CLint',
    'MLM CLint',
    'Caco-2 Permeability Papp A>B',
    'Caco-2 Permeability Efflux',
    'MPPB',
    'MBPB',
    'MGMB'
]


def train_lgb_ensemble(X_train, y_train, X_test, n_seeds=3, n_folds=5, verbose=True):
    """Train LightGBM ensemble with CV."""
    n_test = len(X_test)
    n_train = len(y_train)

    test_preds = np.zeros((n_test, n_seeds))
    oof_preds = np.zeros(n_train)

    total_iters = n_seeds * n_folds
    pbar = tqdm(total=total_iters, desc="  Training", disable=not verbose)

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

            # OOF predictions (for first seed only)
            if seed == 0:
                oof_preds[val_idx] = model.predict(X_val)

            fold_preds.append(model.predict(X_test))
            pbar.update(1)

        test_preds[:, seed] = np.mean(fold_preds, axis=0)

    pbar.close()

    # Calculate OOF Spearman
    spearman = spearmanr(oof_preds, y_train)[0]

    return np.mean(test_preds, axis=1), spearman


def find_latest_submission(submissions_dir: Path) -> Path:
    """Find the most recent v4 submission file."""
    v4_files = list(submissions_dir.glob('v4_submission_*.csv'))
    if not v4_files:
        raise FileNotFoundError(f"No v4_submission_*.csv files found in {submissions_dir}")

    # Sort by modification time, newest first
    v4_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return v4_files[0]


def main():
    parser = argparse.ArgumentParser(description='Fix missing endpoints in V4 submission')
    parser.add_argument('--input', type=str, help='Input submission CSV (default: latest v4_submission_*.csv)')
    parser.add_argument('--output', type=str, help='Output submission CSV (default: v4_submission_fixed.csv)')
    parser.add_argument('--train', type=str, default=str(BASE_DIR / 'data' / 'raw' / 'train.csv'))
    parser.add_argument('--test', type=str, default=str(BASE_DIR / 'data' / 'raw' / 'test_blinded.csv'))
    args = parser.parse_args()

    # Paths
    train_path = Path(args.train)
    test_path = Path(args.test)
    submissions_dir = BASE_DIR / 'submissions'

    if args.input:
        submission_path = Path(args.input)
    else:
        submission_path = find_latest_submission(submissions_dir)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = submissions_dir / 'v4_submission_fixed.csv'

    print("="*60)
    print(" FIX MISSING ENDPOINTS")
    print("="*60)
    print(f"\nInput submission: {submission_path}")
    print(f"Output: {output_path}")

    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    submission = pd.read_csv(submission_path)

    smiles_test = test_df['SMILES'].tolist()
    n_test = len(smiles_test)

    print(f"Test samples: {n_test}")
    print(f"Submission rows: {len(submission)}")

    # Validate submission has correct number of rows
    if len(submission) != n_test:
        print(f"\n⚠️  WARNING: Submission has {len(submission)} rows but test set has {n_test}!")
        print("Creating new submission from scratch...")
        submission = pd.DataFrame({'SMILES': smiles_test})

    # Find missing endpoints
    existing_endpoints = [c for c in submission.columns if c in ALL_ENDPOINTS]
    missing_endpoints = [e for e in ALL_ENDPOINTS if e not in submission.columns]

    print(f"\nExisting endpoints ({len(existing_endpoints)}): {existing_endpoints}")
    print(f"Missing endpoints ({len(missing_endpoints)}): {missing_endpoints}")

    if not missing_endpoints:
        print("\n✓ All endpoints already present! No fix needed.")
        # Still reorder and save
        submission = submission[['SMILES'] + ALL_ENDPOINTS]
        submission.to_csv(output_path, index=False)
        print(f"Saved to: {output_path}")
        return

    # Compute features for test set once
    print("\n" + "="*60)
    print("Computing test set features (one time)...")
    print("="*60)
    test_features = compute_all_features(smiles_test, verbose=True)
    X_test = test_features['comprehensive']
    print(f"Test features shape: {X_test.shape}")

    # Train each missing endpoint
    results = {}

    for i, endpoint in enumerate(missing_endpoints):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(missing_endpoints)}] Training: {endpoint}")
        print(f"{'='*60}")

        if endpoint not in train_df.columns:
            print(f"  ✗ ERROR: '{endpoint}' not in training data!")
            print(f"  Available columns: {list(train_df.columns)}")
            continue

        # Get endpoint data
        mask = train_df[endpoint].notna()
        endpoint_smiles = train_df.loc[mask, 'SMILES'].tolist()
        endpoint_y = train_df.loc[mask, endpoint].values

        print(f"  Training samples: {len(endpoint_smiles)}")

        # Compute features for training
        print("  Computing training features...")
        train_features = compute_all_features(endpoint_smiles, verbose=False)
        X_train = train_features['comprehensive']

        # Train
        predictions, oof_spearman = train_lgb_ensemble(
            X_train, endpoint_y, X_test,
            n_seeds=3, n_folds=5, verbose=True
        )

        # Add to submission
        submission[endpoint] = predictions
        results[endpoint] = {
            'spearman': oof_spearman,
            'min': predictions.min(),
            'max': predictions.max(),
            'mean': predictions.mean()
        }

        print(f"  ✓ OOF Spearman: {oof_spearman:.4f}")
        print(f"  ✓ Predictions: [{predictions.min():.3f}, {predictions.max():.3f}]")

    # Final validation
    print(f"\n{'='*60}")
    print("VALIDATION")
    print(f"{'='*60}")

    final_columns = submission.columns.tolist()
    all_present = all(e in final_columns for e in ALL_ENDPOINTS)

    print(f"\nFinal columns: {final_columns}")
    print(f"All {len(ALL_ENDPOINTS)} endpoints present: {'✓ YES' if all_present else '✗ NO'}")

    if not all_present:
        still_missing = [e for e in ALL_ENDPOINTS if e not in final_columns]
        print(f"Still missing: {still_missing}")

    # Reorder columns to match expected order
    ordered_columns = ['SMILES'] + [e for e in ALL_ENDPOINTS if e in submission.columns]
    submission = submission[ordered_columns]

    # Validate row count
    print(f"\nFinal submission: {len(submission)} rows x {len(submission.columns)} columns")

    if len(submission) != n_test:
        print(f"✗ ERROR: Row count mismatch! Expected {n_test}, got {len(submission)}")
        return

    # Save
    submission.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"\n✓ Fixed submission saved to: {output_path}")
    print(f"\nEndpoint results:")
    for endpoint, stats in results.items():
        print(f"  {endpoint}: Spearman={stats['spearman']:.4f}, range=[{stats['min']:.3f}, {stats['max']:.3f}]")

    # Final column check
    print(f"\nFinal columns ({len(submission.columns)}):")
    for col in submission.columns:
        print(f"  ✓ {col}")


if __name__ == '__main__':
    main()
