"""
Create ensemble from multiple model predictions
Weighted average of CatBoost, XGBoost, LightGBM, and optionally Chemprop
"""
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
SUB_DIR = BASE_DIR / "submissions"

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

def load_submission(name):
    """Load a submission CSV"""
    path = SUB_DIR / name
    if path.exists():
        df = pd.read_csv(path)
        print(f"Loaded {name}: {df.shape}")
        return df
    else:
        print(f"Missing: {name}")
        return None

def create_ensemble(submissions, weights, output_name):
    """Create weighted ensemble"""
    print(f"\nCreating ensemble: {output_name}")
    print(f"Weights: {dict(zip([s[0] for s in submissions], weights))}")

    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()

    # Load submissions
    dfs = []
    valid_weights = []
    for (name, _), w in zip(submissions, weights):
        df = load_submission(name)
        if df is not None:
            dfs.append(df)
            valid_weights.append(w)

    if not dfs:
        print("No valid submissions found!")
        return None

    # Renormalize weights
    valid_weights = np.array(valid_weights)
    valid_weights = valid_weights / valid_weights.sum()
    print(f"Normalized weights: {valid_weights}")

    # Create ensemble
    result = dfs[0][['Molecule Name']].copy()

    for target in TARGETS:
        ensemble_pred = np.zeros(len(result))
        for df, w in zip(dfs, valid_weights):
            ensemble_pred += w * df[target].values

        # Clip to valid range
        ensemble_pred = np.clip(ensemble_pred, *VALID_RANGES[target])
        result[target] = ensemble_pred

    # Save
    out_path = SUB_DIR / output_name
    result.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    return result

def main():
    print("="*60)
    print("ENSEMBLE CREATION")
    print("="*60)

    # List available submissions
    print("\nAvailable submissions:")
    for f in sorted(SUB_DIR.glob("*.csv")):
        print(f"  - {f.name}")

    # Ensemble 1: CatBoost + XGBoost (if available)
    submissions_2way = [
        ("optimized_catboost.csv", 0.55),
        ("optimized_xgboost.csv", 0.45),
    ]
    create_ensemble(submissions_2way, [0.55, 0.45], "ensemble_cat_xgb.csv")

    # Ensemble 2: CatBoost + XGBoost + LightGBM
    submissions_3way = [
        ("optimized_catboost.csv", 0.40),
        ("optimized_xgboost.csv", 0.35),
        ("optimized_lightgbm.csv", 0.25),
    ]
    create_ensemble(submissions_3way, [0.40, 0.35, 0.25], "ensemble_3way.csv")

    # Ensemble 3: All 4 models (if Chemprop available)
    submissions_4way = [
        ("optimized_catboost.csv", 0.35),
        ("optimized_xgboost.csv", 0.25),
        ("optimized_lightgbm.csv", 0.20),
        ("chemprop_submission.csv", 0.20),
    ]
    create_ensemble(submissions_4way, [0.35, 0.25, 0.20, 0.20], "ensemble_4way.csv")

    # Ensemble 4: Equal weights
    submissions_equal = [
        ("optimized_catboost.csv", 1.0),
        ("optimized_xgboost.csv", 1.0),
        ("optimized_lightgbm.csv", 1.0),
    ]
    create_ensemble(submissions_equal, [1.0, 1.0, 1.0], "ensemble_equal_3way.csv")

    print("\n" + "="*60)
    print("DONE!")
    print("="*60)

if __name__ == "__main__":
    main()
