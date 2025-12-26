# OpenADMET ExpansionRx Challenge - Method Report

**Team:** inventcures
**Repository:** https://github.com/inventcures/openadmet-expansionrx-challenge

## Summary

Ensemble of gradient boosting models (XGBoost, CatBoost, LightGBM) with Morgan fingerprints and physicochemical descriptors for multi-task ADMET property prediction.

## Approach

### Features (2073 dimensions)
- **Morgan Fingerprints:** 2048-bit circular fingerprints (radius=2)
- **Molecular Descriptors (25):** MolWt, LogP, TPSA, HBD, HBA, RotatableBonds, AromaticRings, FractionCSP3, HeavyAtomCount, RingCount, AliphaticRings, Heterocycles, LabuteASA, QED, AmideBonds, BertzCT, Chi0, Chi1, Kappa1, Kappa2, HallKierAlpha, SpiroAtoms, BridgeheadAtoms, ValenceElectrons, HBA

### Models
1. **XGBoost** - Best single model performer
   - 600-1500 iterations, max_depth=7-8, learning_rate=0.03
   - GPU acceleration with `gpu_hist` tree method

2. **CatBoost** - Gradient boosting with ordered boosting
   - 600-1200 iterations, depth=7-8, learning_rate=0.03
   - Native GPU support

3. **LightGBM** - Fast gradient boosting
   - 600-1000 iterations, max_depth=7-8, learning_rate=0.03

4. **Chemprop D-MPNN** (optional) - Message passing neural network
   - 15-40 epochs, batch_size=64-256
   - Direct learning from SMILES

### Training Strategy
- **5-fold cross-validation** for each endpoint independently
- **Early stopping** (50-100 rounds) to prevent overfitting
- **Endpoint-specific clipping** to valid ranges
- **Weighted ensemble** based on inverse RAE per endpoint

### Ensemble
Final predictions are weighted averages where weights are inversely proportional to each model's RAE on that specific endpoint, giving more weight to models that perform better on each target.

## Results

| Model | MA-RAE (CV) |
|-------|-------------|
| XGBoost | 0.5730 |
| LightGBM | 0.6082 |
| CatBoost | 0.6545 |
| **Weighted Ensemble** | **TBD** |

## Code

Training scripts with checkpointing and resume capability:
- `run_local_m3.py` - Local training (Apple Silicon)
- `run_runpod.py` - GPU training (CUDA)

```bash
python run_local_m3.py --mode full
python run_runpod.py --mode full
```

## Dependencies
- RDKit, scikit-learn, XGBoost, CatBoost, LightGBM, Chemprop, PyTorch
