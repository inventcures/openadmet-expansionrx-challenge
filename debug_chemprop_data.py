#!/usr/bin/env python3
"""Debug why Chemprop fails with training data but works with test SMILES"""
import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 60)
print("CHEMPROP DATA DEBUGGING")
print("=" * 60)

# Load training data
train_df = pd.read_csv("data/raw/train.csv")
print(f"\nLoaded {len(train_df)} training samples")

# Get SMILES for first target
target = 'LogD'
y = train_df[target].values
mask = ~np.isnan(y)
smiles = train_df.loc[mask, 'SMILES'].tolist()
targets = y[mask]

print(f"Target: {target}, valid samples: {len(smiles)}")
print(f"First 5 SMILES: {smiles[:5]}")

# Test with Chemprop
from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader

print("\n--- Creating datapoints ---")
datapoints = []
failed = []

for i, smi in enumerate(smiles[:100]):  # Test first 100
    try:
        y_val = [float(targets[i])]
        dp = MoleculeDatapoint.from_smi(smi, y=y_val)
        datapoints.append(dp)
    except Exception as e:
        failed.append((i, smi, str(e)))

print(f"Created {len(datapoints)} datapoints, {len(failed)} failed")
if failed:
    print(f"First 5 failures: {failed[:5]}")

# Create dataset and dataloader
print("\n--- Creating dataset and dataloader ---")
dataset = MoleculeDataset(datapoints)
loader = build_dataloader(dataset, batch_size=32, shuffle=False)

print(f"Dataset size: {len(dataset)}")
print(f"Num batches: {len(loader)}")

# Check batches
print("\n--- Checking batches ---")
for i, batch in enumerate(loader):
    bmg = batch.bmg
    if bmg is None:
        print(f"Batch {i}: bmg is None!")
    elif getattr(bmg, 'V', None) is None:
        print(f"Batch {i}: bmg.V is None!")
    else:
        print(f"Batch {i}: OK - V.shape={bmg.V.shape}")

    if i >= 2:
        break

# Test model forward pass
print("\n--- Testing forward pass ---")
from chemprop import models, nn
import torch

mp = nn.BondMessagePassing(d_h=300, depth=3, dropout=0.0)
agg = nn.MeanAggregation()
ffn = nn.RegressionFFN(input_dim=300, hidden_dim=300, n_layers=1, dropout=0.0)
model = models.MPNN(message_passing=mp, agg=agg, predictor=ffn)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"Model on device: {device}")

for i, batch in enumerate(loader):
    try:
        bmg = batch.bmg
        if bmg is None:
            print(f"Batch {i}: Skipping - bmg is None")
            continue

        bmg = bmg.to(device)
        model.eval()
        with torch.no_grad():
            out = model(bmg)
        print(f"Batch {i}: Forward pass OK - output shape {out.shape}")
    except Exception as e:
        print(f"Batch {i}: Forward pass FAILED - {e}")

    if i >= 2:
        break

print("\n" + "=" * 60)
print("DEBUG COMPLETE")
print("=" * 60)
