# OpenADMET ExpansionRx Challenge - Method Report

**Team:** inventcures
**Repository:** https://github.com/inventcures/openadmet-expansionrx-challenge
**Documentation:** https://deepwiki.com/inventcures/openadmet-expansionrx-challenge

---

## Summary

Tiered ensemble of gradient boosting models (XGBoost, CatBoost, LightGBM) with Morgan fingerprints and physicochemical descriptors for multi-task ADMET property prediction across nine endpoints.

## Architecture

We implement a **tiered modeling strategy** with progressive sophistication:

| Tier | Model | Features | Purpose |
|------|-------|----------|---------|
| 1 | Simple CatBoost | 522 | Baseline |
| 2 | Optimized GBDT (XGBoost, CatBoost, LightGBM) | 1,039-2,073 | Primary predictions |
| 3 | CatBoost + RandomForest Ensemble | 2,747 | Advanced ensemble |
| 4 | Chemprop D-MPNN | Graph-based | Deep learning |

## Feature Engineering

### Molecular Fingerprints
- **Morgan Fingerprints:** 2048-bit circular fingerprints (radius=2)
- Generated via RDKit from SMILES strings

### Physicochemical Descriptors (25 features)
MolWt, LogP, TPSA, HBD, HBA, RotatableBonds, AromaticRings, FractionCSP3, HeavyAtomCount, RingCount, AliphaticRings, Heterocycles, LabuteASA, QED, AmideBonds, BertzCT, Chi0, Chi1, Kappa1, Kappa2, HallKierAlpha, SpiroAtoms, BridgeheadAtoms, ValenceElectrons, HBA

**Total feature dimensionality:** 2,073 (2048 FP + 25 descriptors)

## Target Properties

| Endpoint | Range | Training Samples |
|----------|-------|------------------|
| LogD | [-3, 6] | 5,039 |
| KSOL | [0.001, 350] μM | 5,128 |
| HLM CLint | [0, 3000] μL/min/mg | 3,759 |
| MLM CLint | [0, 12000] μL/min/mg | 4,522 |
| Caco-2 Papp A>B | [0, 60] 10⁻⁶ cm/s | 2,157 |
| Caco-2 Efflux | [0.2, 120] | 2,161 |
| MPPB | [0, 100] % | 1,302 |
| MBPB | [0, 100] % | 975 |
| MGMB | [0, 100] % | 222 |

## Models

### XGBoost (Best Performer)
- 600-1500 iterations, max_depth=7-8, learning_rate=0.03
- GPU acceleration with `gpu_hist` tree method
- L1/L2 regularization (alpha=0.1, lambda=1.0)

### CatBoost
- 600-1200 iterations, depth=7-8, learning_rate=0.03
- Ordered boosting with symmetric trees
- Native GPU support with `task_type='GPU'`

### LightGBM
- 600-1000 iterations, max_depth=7-8, learning_rate=0.03
- Histogram-based learning with leaf-wise growth

### Chemprop D-MPNN (Optional)
- Message passing neural network
- 15-40 epochs, batch_size=64-256
- Direct learning from molecular graphs

## Training Strategy

1. **5-fold cross-validation** for each endpoint independently
2. **Per-target NaN filtering** (varying sample sizes per endpoint)
3. **Early stopping** (50-100 rounds) to prevent overfitting
4. **Endpoint-specific clipping** to valid ranges
5. **Weighted ensemble** based on inverse RAE per endpoint

## Ensemble Strategy

Final predictions use endpoint-specific weighted averaging:

```
weight[model][endpoint] = 1 / RAE[model][endpoint]
prediction[endpoint] = Σ (weight[m] × pred[m]) / Σ weight[m]
```

This gives higher influence to models performing better on each specific target.

## Results (Quick Mode - 200 iterations)

| Model | MA-RAE |
|-------|--------|
| **XGBoost** | **0.5730** |
| LightGBM | 0.6082 |
| CatBoost | 0.6545 |
| **Target (leader)** | **0.5593** |

## Infrastructure

- **Checkpointing:** Resumable training with automatic state recovery
- **GPU Support:** CUDA (RunPod) and MPS (Apple Silicon)
- **Training Modes:** Quick (~2 min), Medium (~60 min), Full (~2-3 hrs)

```bash
# Local training
python run_local_m3.py --mode full

# GPU training (RunPod)
python run_runpod.py --mode full
```

## Dependencies

RDKit ≥2023.9.1, XGBoost ≥2.0.0, CatBoost ≥1.2.0, LightGBM ≥4.0.0, Chemprop ≥2.0.0, PyTorch ≥2.0.0, scikit-learn ≥1.3.0
