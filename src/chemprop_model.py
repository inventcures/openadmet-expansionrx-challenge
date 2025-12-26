"""
Chemprop D-MPNN model for ADMET prediction
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import gc
import tempfile
import shutil
warnings.filterwarnings('ignore')

from chemprop import data, featurizers, models, nn
from lightning import pytorch as pl
import torch

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

def train_chemprop_model(train_smiles, train_y, test_smiles, target_name, epochs=30):
    """Train a single Chemprop model"""
    print(f"\n{'='*40}")
    print(f"Training Chemprop for: {target_name}")
    print(f"{'='*40}")

    # Remove NaN
    mask = ~np.isnan(train_y)
    smiles_clean = [s for s, m in zip(train_smiles, mask) if m]
    y_clean = train_y[mask]
    print(f"Samples: {len(y_clean)}")

    # Create datasets
    train_data = [data.MoleculeDatapoint.from_smi(smi, y=[y])
                  for smi, y in zip(smiles_clean, y_clean)]
    test_data = [data.MoleculeDatapoint.from_smi(smi) for smi in test_smiles]

    # Use 80/20 split for validation
    n_train = int(0.8 * len(train_data))
    train_subset = train_data[:n_train]
    val_subset = train_data[n_train:]

    # Create datasets
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    train_dset = data.MoleculeDataset(train_subset, featurizer)
    val_dset = data.MoleculeDataset(val_subset, featurizer)
    test_dset = data.MoleculeDataset(test_data, featurizer)

    # Dataloaders
    train_loader = data.build_dataloader(train_dset, batch_size=64, shuffle=True)
    val_loader = data.build_dataloader(val_dset, batch_size=64, shuffle=False)
    test_loader = data.build_dataloader(test_dset, batch_size=64, shuffle=False)

    # Model
    mp = nn.BondMessagePassing()
    agg = nn.MeanAggregation()
    ffn = nn.RegressionFFN()

    mpnn = models.MPNN(mp, agg, ffn, batch_norm=True, metrics=[])

    # Trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator='cpu',
        enable_progress_bar=True,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=False
    )

    # Train
    trainer.fit(mpnn, train_loader, val_loader)

    # Predict
    preds = trainer.predict(mpnn, test_loader)
    predictions = np.concatenate([p.squeeze().numpy() for p in preds])

    return predictions

def main():
    print("="*60)
    print("CHEMPROP D-MPNN MODEL")
    print("="*60)

    train_df = pd.read_csv(BASE_DIR / "data/raw/train.csv")
    test_df = pd.read_csv(BASE_DIR / "data/raw/test_blinded.csv")
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    train_smiles = train_df['SMILES'].tolist()
    test_smiles = test_df['SMILES'].tolist()

    predictions = {}

    for target in TARGETS:
        y = train_df[target].values
        pred = train_chemprop_model(train_smiles, y, test_smiles, target, epochs=30)

        # Clip
        pred = np.clip(pred, *VALID_RANGES[target])
        predictions[target] = pred
        gc.collect()

    # Save
    sub = pd.DataFrame({'Molecule Name': test_df['Molecule Name']})
    for t in TARGETS:
        sub[t] = predictions[t]

    out_path = BASE_DIR / "submissions/chemprop_submission.csv"
    sub.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
