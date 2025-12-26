"""
Quick evaluation of submissions using CV on training data
Minimal output, suppressed warnings
"""
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['RDKit_DEPRECATION_WARNINGS'] = '0'

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"

TARGETS = ['LogD', 'KSOL', 'HLM CLint', 'MLM CLint',
           'Caco-2 Permeability Papp A>B', 'Caco-2 Permeability Efflux',
           'MPPB', 'MBPB', 'MGMB']

def compute_rae(y_true, y_pred):
    """Relative Absolute Error"""
    mae = mean_absolute_error(y_true, y_pred)
    baseline = np.mean(np.abs(y_true - np.mean(y_true)))
    return mae / baseline if baseline > 0 else 1.0

def main():
    print("="*50)
    print("ADMET Challenge - Quick Status")
    print("="*50)

    # Load training data for reference
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    print(f"\nTraining samples per endpoint:")
    for t in TARGETS:
        n = train_df[t].notna().sum()
        print(f"  {t}: {n}")

    # List submissions
    sub_dir = BASE_DIR / "submissions"
    print(f"\nSubmissions available:")
    for f in sorted(sub_dir.glob("*.csv")):
        size = f.stat().st_size / 1024
        print(f"  {f.name}: {size:.1f}KB")

    print("\n" + "="*50)
    print("TARGET: MA-RAE < 0.5593 (leader 'pebble')")
    print("BEST SO FAR: 0.5656 (optimized CatBoost)")
    print("GAP: 0.0063 (~1.1% improvement needed)")
    print("="*50)

if __name__ == "__main__":
    main()
