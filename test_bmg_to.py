#!/usr/bin/env python3
"""Quick test: does bmg.to() return self or None?"""
from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
import torch

dp = MoleculeDatapoint.from_smi('CCO', y=[1.0])
dataset = MoleculeDataset([dp])
loader = build_dataloader(dataset, batch_size=1)

for batch in loader:
    bmg = batch.bmg
    print(f"Before: bmg.V.device = {bmg.V.device}")
    result = bmg.to('cuda')
    print(f"Return value of .to(): {result}")
    print(f"After: bmg.V.device = {bmg.V.device}")
    print(f"\nConclusion: .to() is IN-PLACE (returns None)" if result is None else "Returns self")
    break
