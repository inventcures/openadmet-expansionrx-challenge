#!/usr/bin/env python3
"""Test script to debug Chemprop v2 API on RunPod"""
import sys
import numpy as np

print("=" * 60)
print("CHEMPROP API DIAGNOSTIC")
print("=" * 60)

try:
    import chemprop
    print(f"\n1. Chemprop version: {getattr(chemprop, '__version__', 'unknown')}")
except ImportError as e:
    print(f"FATAL: Cannot import chemprop: {e}")
    sys.exit(1)

# Check what's available
print("\n2. Checking available APIs...")

from chemprop import data
print(f"   data module: {[x for x in dir(data) if not x.startswith('_')][:10]}...")

from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
print(f"   MoleculeDatapoint methods: {[x for x in dir(MoleculeDatapoint) if not x.startswith('_')]}")

# Test datapoint creation
print("\n3. Testing MoleculeDatapoint creation...")
test_smiles = ["CCO", "CCCO", "c1ccccc1"]

# Method 1: from_smi
if hasattr(MoleculeDatapoint, 'from_smi'):
    print("   Trying from_smi()...")
    try:
        dp = MoleculeDatapoint.from_smi(test_smiles[0], y=[1.0])
        print(f"   SUCCESS: from_smi created {type(dp)}")
        print(f"   Datapoint attributes: {[x for x in dir(dp) if not x.startswith('_')]}")
    except Exception as e:
        print(f"   FAILED: {e}")
else:
    print("   from_smi not available")

# Method 2: direct init with SMILES
print("\n   Trying direct init with SMILES string...")
try:
    dp = MoleculeDatapoint(test_smiles[0], y=[1.0])
    print(f"   SUCCESS: direct init created {type(dp)}")
except Exception as e:
    print(f"   FAILED: {e}")

# Method 3: with mol object
print("\n   Trying init with RDKit mol...")
try:
    from rdkit import Chem
    mol = Chem.MolFromSmiles(test_smiles[0])
    dp = MoleculeDatapoint(mol=mol, y=[1.0])
    print(f"   SUCCESS: mol init created {type(dp)}")
except Exception as e:
    print(f"   FAILED: {e}")

# Test dataset and dataloader
print("\n4. Testing MoleculeDataset and dataloader...")
try:
    # Create datapoints
    datapoints = []
    for i, smi in enumerate(test_smiles):
        if hasattr(MoleculeDatapoint, 'from_smi'):
            dp = MoleculeDatapoint.from_smi(smi, y=[float(i)])
        else:
            dp = MoleculeDatapoint(smi, y=[float(i)])
        datapoints.append(dp)

    dataset = MoleculeDataset(datapoints)
    print(f"   Dataset created with {len(dataset)} samples")

    loader = build_dataloader(dataset, batch_size=2, shuffle=False)
    print(f"   Dataloader created")

    # Check batch structure
    for batch in loader:
        print(f"\n   Batch type: {type(batch)}")
        print(f"   Batch attributes: {[x for x in dir(batch) if not x.startswith('_')]}")

        # Check bmg
        bmg = getattr(batch, 'bmg', None)
        print(f"   batch.bmg = {bmg}")

        if bmg is not None:
            print(f"   bmg type: {type(bmg)}")
            print(f"   bmg.V = {getattr(bmg, 'V', 'NOT FOUND')}")
            print(f"   bmg.E = {getattr(bmg, 'E', 'NOT FOUND')}")
            if hasattr(bmg, 'V') and bmg.V is not None:
                print(f"   bmg.V.shape = {bmg.V.shape}")
        else:
            # Try other attribute names
            for attr in ['mg', 'molgraph', 'batch', 'graph']:
                val = getattr(batch, attr, None)
                if val is not None:
                    print(f"   Found batch.{attr} = {type(val)}")

        break  # Only check first batch

except Exception as e:
    import traceback
    print(f"   FAILED: {e}")
    traceback.print_exc()

# Test model creation and forward pass
print("\n5. Testing MPNN model...")
try:
    from chemprop import models, nn

    mp = nn.BondMessagePassing(d_h=300, depth=3, dropout=0.0)
    agg = nn.MeanAggregation()
    ffn = nn.RegressionFFN(input_dim=300, hidden_dim=300, n_layers=1, dropout=0.0)

    model = models.MPNN(message_passing=mp, agg=agg, predictor=ffn)
    print(f"   Model created: {type(model)}")

    # Try forward pass with a batch
    if bmg is not None and getattr(bmg, 'V', None) is not None:
        import torch
        model.eval()
        with torch.no_grad():
            out = model(bmg)
            print(f"   Forward pass SUCCESS: output shape = {out.shape}")
    else:
        print("   Skipping forward pass - no valid bmg")

except Exception as e:
    import traceback
    print(f"   FAILED: {e}")
    traceback.print_exc()

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
