# V4 TOP 5 Finish Strategy: Aggressive Performance Optimization

**Generated:** 2025-12-28
**Goal:** Secure TOP 5 finish on OpenADMET ExpansionRx leaderboard
**Status:** Deep research complete, implementation-ready

---

## Executive Summary

V3 achieved mid-tier results with ~0.93 Spearman on LogD but underperformed on other endpoints. **Root cause analysis:**

| Issue | Impact | V4 Solution |
|-------|--------|-------------|
| Low diversity (0.20) | Models too correlated | Add truly diverse architectures (GNN, 3D) |
| ChemBERTa underperformed (0.60) | Wasted ensemble slot | Replace with Chemprop-RDKit (TDC #1) |
| No external data | Limited generalization | Pre-train on TDC ADMET datasets |
| No SMILES augmentation | Overfitting | 20x SMILES enumeration |
| Weak ensemble | Suboptimal combination | Hill climbing + 3-level stacking |

**Expected improvement: 15-25% RAE reduction**

---

## 1. Critical Research Findings

### 1.1 TDC ADMET Leaderboard Analysis

The [TDC ADMET Benchmark](https://tdcommons.ai/benchmark/admet_group/overview/) is THE gold standard with 22 ADMET datasets. Top performers:

| Rank | Model | Architecture | Key Innovation |
|------|-------|--------------|----------------|
| **#1** | **Chemprop-RDKit** | D-MPNN + 200 RDKit features | Hybrid GNN+descriptors |
| #2 | XGBoost + features | GBDT | Ensemble fingerprints |
| #3 | NIST Meta-model | Ensemble | Multi-model meta-learning |

**Source:** [ADMET-AI Paper](https://academic.oup.com/bioinformatics/article/40/7/btae416/7698030)

### 1.2 NeurIPS 2024 BELKA 1st Place Solution

The [1st place solution](https://zenn.dev/yuto_mo/articles/13fc8fc77c5147) used:

```
Architecture: Self-Attention Transformer (4 layers, 8 heads)
Tokenizer: atomInSmiles (character-level)
Pre-training: 2-stage
  1. MLM (Masked Language Model) on SMILES
  2. SMILES → ECFP prediction task
Final: Change head to dense + sigmoid
```

**Key insight:** Simple architecture + smart pre-training beats complex models.

### 1.3 Uni-Mol: 3D Molecular Representation (SOTA)

[Uni-Mol](https://github.com/deepmodeling/Uni-Mol) outperforms all 2D methods:
- Pre-trained on 209M 3D conformations
- SE(3)-equivariant transformer
- **Outperforms SOTA in 14/15 molecular property tasks**
- Uni-Mol2 (NeurIPS 2024): 1.1B parameters, 800M conformations

### 1.4 SMILES Augmentation

[SMILES Enumeration](https://arxiv.org/abs/1703.07076) provides massive gains:
- Same molecule can have multiple valid SMILES
- **20x augmentation improves R² from 0.56 → 0.66** (18% relative improvement)
- Reduces overfitting on small datasets

### 1.5 Multi-Task Learning: MTGL-ADMET

[MTGL-ADMET](https://www.sciencedirect.com/science/article/pii/S2589004223023623) shows:
- "One primary, multiple auxiliaries" paradigm
- Status theory + maximum flow for auxiliary task selection
- Task-specific + task-shared modules
- **Prevents negative transfer** between unrelated tasks

---

## 2. V4 Architecture: Maximum Diversity Ensemble

### 2.1 Model Portfolio (12+ Diverse Models)

```
TIER 1: MUST IMPLEMENT (Proven SOTA)
═══════════════════════════════════
├── Chemprop-RDKit
│   ├── D-MPNN with 200 RDKit descriptors
│   ├── THE TDC leaderboard #1 model
│   ├── Multi-task variant for related endpoints
│   └── Expected: +10-15% over current best
│
├── Uni-Mol (3D Pre-trained)
│   ├── 3D conformer-aware representations
│   ├── Pre-trained on 209M molecules
│   ├── pip install unimol-tools
│   └── Expected: +5-10% on binding/metabolism
│
└── AttentiveFP
    ├── Graph attention mechanism
    ├── Atom + molecule level attention
    ├── Available in DGL-LifeSci
    └── Expected: +3-5% diversity boost

TIER 2: HIGH VALUE
══════════════════
├── SMILES Transformer (BELKA-style)
│   ├── 2-stage pre-training (MLM + ECFP)
│   ├── Character-level tokenization
│   └── Self-attention (4 layers)
│
├── LightGBM + Extended Features
│   ├── ECFP4 + ECFP6 + FCFP4 + MACCS
│   ├── 200 RDKit descriptors
│   ├── Fragment counts
│   └── CURRENTLY BEST IN V3
│
└── XGBoost + Tanimoto Kernel Features
    ├── Kernel-transformed fingerprints
    └── Different representation space

TIER 3: DIVERSITY BOOSTERS
══════════════════════════
├── 1D CNN + SMILES Augmentation (20x)
│   ├── Current implementation enhanced
│   └── Apply enumeration to training data
│
├── Random Forest + Morgan FP
│   └── Simple but decorrelated predictions
│
├── CatBoost + Scaffold Features
│   └── Bemis-Murcko scaffold embeddings
│
└── Ridge + PCA(ChemBERTa)
    └── Linear model for regularization
```

### 2.2 Expected Diversity Matrix

Target correlation structure after V4:
```
                ChempropRDKit  UniMol  AttentiveFP  LGB  XGB  CNN
ChempropRDKit         1.00     0.70      0.75     0.60 0.55 0.45
UniMol                0.70     1.00      0.65     0.50 0.45 0.40
AttentiveFP           0.75     0.65      1.00     0.55 0.50 0.45
LightGBM              0.60     0.50      0.55     1.00 0.85 0.50
XGBoost               0.55     0.45      0.50     0.85 1.00 0.55
1D CNN                0.45     0.40      0.45     0.50 0.55 1.00

Mean pairwise diversity: 0.45 (vs 0.20 in V3) → 125% improvement
```

---

## 3. External Data Strategy

### 3.1 TDC ADMET Datasets for Pre-training

```python
from tdc.single_pred import ADME, Tox

# ABSORPTION
PRETRAINING_DATASETS = {
    # Lipophilicity (related to LogD)
    'Lipophilicity_AstraZeneca': ADME(name='Lipophilicity_AstraZeneca'),  # 4,200 compounds

    # Solubility (related to KSOL)
    'Solubility_AqSolDB': ADME(name='Solubility_AqSolDB'),  # 9,982 compounds
    'ESOL': ADME(name='ESOL'),  # 1,128 compounds

    # Metabolism (related to HLM/MLM CLint)
    'CYP2C9_Substrate': ADME(name='CYP2C9_Substrate_CarbonMangels'),
    'CYP2D6_Substrate': ADME(name='CYP2D6_Substrate_CarbonMangels'),
    'CYP3A4_Substrate': ADME(name='CYP3A4_Substrate_CarbonMangels'),
    'Half_Life': ADME(name='Half_Life_Obach'),  # 667 compounds
    'Clearance_Hepatocyte': ADME(name='Clearance_Hepatocyte_AZ'),  # 1,213 compounds

    # Permeability (related to Caco-2)
    'Caco2_Wang': ADME(name='Caco2_Wang'),  # 906 compounds
    'HIA_Hou': ADME(name='HIA_Hou'),  # 578 compounds
    'Pgp_Broccatelli': ADME(name='Pgp_Broccatelli'),  # 1,212 compounds

    # Protein Binding (related to MPPB/MBPB/MGMB)
    'PPBR_AZ': ADME(name='PPBR_AZ'),  # 1,797 compounds
}

# Total: ~25,000 additional labeled compounds
```

### 3.2 Pre-training Strategy

```
Phase 1: Self-Supervised Pre-training (All ~25K compounds)
├── Masked Language Modeling on SMILES
├── SMILES → ECFP prediction
└── 3D coordinate denoising (Uni-Mol style)

Phase 2: Multi-Task Pre-training (Grouped datasets)
├── Lipophilicity group → LogD head
├── Solubility group → KSOL head
├── Metabolism group → CLint heads
├── Permeability group → Caco-2 heads
└── Binding group → PPB heads

Phase 3: Fine-tuning (Competition data only)
├── Freeze encoder (first 5 epochs)
├── Unfreeze + lower LR (next 20 epochs)
└── Final ensemble selection
```

---

## 4. SMILES Augmentation Pipeline

### 4.1 Implementation

```python
from rdkit import Chem
import random

def enumerate_smiles(smiles: str, n_variants: int = 20) -> list:
    """
    Generate multiple valid SMILES for same molecule.

    Each molecule → 20 representations for training.
    Test time: predict on all 20, average predictions.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles]

    variants = set()
    variants.add(smiles)  # Keep canonical

    attempts = 0
    while len(variants) < n_variants and attempts < 100:
        # Random atom ordering
        atom_order = list(range(mol.GetNumAtoms()))
        random.shuffle(atom_order)

        new_mol = Chem.RenumberAtoms(mol, atom_order)
        new_smiles = Chem.MolToSmiles(new_mol, canonical=False)
        variants.add(new_smiles)
        attempts += 1

    return list(variants)

# Training: Each molecule appears 20x with different SMILES
# Inference: Predict 20x and average → reduces variance
```

### 4.2 Expected Impact

| Dataset Size | Without Aug | With 20x Aug | Improvement |
|--------------|-------------|--------------|-------------|
| 5,000 | 0.75 R² | 0.82 R² | +9.3% |
| 10,000 | 0.80 R² | 0.85 R² | +6.3% |
| 20,000 | 0.83 R² | 0.87 R² | +4.8% |

---

## 5. Advanced Ensemble Strategy

### 5.1 Three-Level Stacking (Kaggle Grandmaster Technique)

```
LEVEL 0: Base Model OOF Predictions (5-fold CV each)
═════════════════════════════════════════════════════
Model                    │ Seeds │ Features        │ OOF Predictions
─────────────────────────┼───────┼─────────────────┼────────────────
Chemprop-RDKit           │   5   │ Graph + RDKit   │ 5 columns
Chemprop-RDKit-MTL       │   3   │ Graph + RDKit   │ 3 columns
Uni-Mol                  │   3   │ 3D Conformer    │ 3 columns
AttentiveFP              │   3   │ Graph Attention │ 3 columns
LightGBM-Extended        │   5   │ FP + Desc       │ 5 columns
XGBoost-ECFP             │   3   │ ECFP4/6         │ 3 columns
CatBoost-Scaffold        │   3   │ Scaffold FP     │ 3 columns
1D-CNN-Augmented         │   3   │ SMILES tokens   │ 3 columns
RandomForest-Morgan      │   3   │ Morgan FP       │ 3 columns
─────────────────────────┴───────┴─────────────────┴────────────────
Total: 34 base predictions per sample

LEVEL 1: Meta-Models on OOF Predictions
═══════════════════════════════════════
├── Ridge (α=1.0)
├── ElasticNet (α=0.5, l1_ratio=0.5)
├── LightGBM (shallow: depth=3, n_est=100)
├── XGBoost (shallow: depth=2, n_est=50)
└── Neural Network (34 → 64 → 32 → 1)

Each produces new OOF predictions → 5 columns

LEVEL 2: Final Blender
══════════════════════
├── Weighted average (weights from hill climbing)
├── Optuna-tuned weights
└── Output: Final prediction
```

### 5.2 Hill Climbing Ensemble Selection

```python
def hill_climbing_ensemble(oof_preds: np.ndarray, y_true: np.ndarray,
                            metric='spearman', max_models=15):
    """
    Greedy forward selection with weight optimization.

    1. Start with best single model
    2. Add model that maximizes ensemble metric
    3. Optimize weights for selected models
    4. Repeat until no improvement
    """
    n_models = oof_preds.shape[1]
    selected = []
    weights = []

    # Find best single model
    best_idx = max(range(n_models),
                   key=lambda i: spearmanr(oof_preds[:, i], y_true)[0])
    selected.append(best_idx)
    weights = [1.0]
    best_score = spearmanr(oof_preds[:, best_idx], y_true)[0]

    # Greedy addition
    for _ in range(max_models - 1):
        best_improvement = 0
        best_add = None
        best_new_weights = None

        for idx in range(n_models):
            if idx in selected:
                continue

            # Try adding this model
            trial_selected = selected + [idx]
            trial_preds = oof_preds[:, trial_selected]

            # Optimize weights via scipy
            result = minimize(
                lambda w: -spearmanr(trial_preds @ w / w.sum(), y_true)[0],
                x0=np.ones(len(trial_selected)),
                bounds=[(0, 1)] * len(trial_selected)
            )

            trial_weights = result.x / result.x.sum()
            trial_score = -result.fun

            improvement = trial_score - best_score
            if improvement > best_improvement:
                best_improvement = improvement
                best_add = idx
                best_new_weights = trial_weights

        if best_improvement < 0.0001:  # Convergence threshold
            break

        selected.append(best_add)
        weights = best_new_weights
        best_score += best_improvement

    return selected, weights, best_score
```

### 5.3 Diversity-Aware Selection

```python
def diversity_hill_climbing(oof_preds, y_true, diversity_bonus=0.1):
    """
    Modified hill climbing that rewards diversity.

    Score = accuracy + diversity_bonus * (1 - max_correlation_with_selected)
    """
    # Penalize adding highly correlated models
    # Prefer models that make different errors
```

---

## 6. Hyperparameter Optimization

### 6.1 Optuna Configuration

```python
import optuna

def objective_chemprop(trial):
    params = {
        'hidden_size': trial.suggest_int('hidden_size', 300, 600, step=50),
        'depth': trial.suggest_int('depth', 3, 6),
        'dropout': trial.suggest_float('dropout', 0.0, 0.3, step=0.05),
        'ffn_hidden_size': trial.suggest_int('ffn_hidden_size', 200, 500, step=50),
        'ffn_num_layers': trial.suggest_int('ffn_num_layers', 2, 4),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        'lr': trial.suggest_float('lr', 1e-4, 1e-3, log=True),
    }

    # 3-fold CV for speed
    cv_scores = cross_validate(params, n_folds=3)
    return np.mean(cv_scores)

# Run optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective_chemprop, n_trials=100, n_jobs=-1)
```

### 6.2 Endpoint-Specific Tuning

Different endpoints need different hyperparameters:

| Endpoint | Optimal Model | Key Hyperparameters |
|----------|--------------|---------------------|
| LogD | Chemprop-RDKit | depth=4, hidden=500 |
| KSOL | LightGBM | num_leaves=127 (complex) |
| HLM CLint | Uni-Mol | 3D features important |
| Caco-2 | AttentiveFP | Attention for transport |
| PPB | Multi-task | Share info across MPPB/MBPB/MGMB |

---

## 7. Implementation Plan

### Phase 1: Infrastructure (Day 1)

```bash
# Install dependencies
pip install PyTDC unimol-tools dgl-life chemprop optuna

# Download TDC datasets
python -c "from tdc.single_pred import ADME; ADME(name='Lipophilicity_AstraZeneca')"

# Setup Uni-Mol
pip install unimol-tools
```

### Phase 2: New Model Integration (Days 2-3)

```
□ Chemprop-RDKit integration
  ├── Use chemprop v2.0+ with RDKit features
  ├── Train per-endpoint and multi-task variants
  └── Implement 5-seed ensemble

□ Uni-Mol integration
  ├── pip install unimol-tools
  ├── Generate 3D conformers (RDKit)
  └── Fine-tune on competition data

□ AttentiveFP integration
  ├── Use DGL-LifeSci implementation
  └── Hyperparameter search with Optuna

□ SMILES augmentation
  ├── Implement enumerate_smiles()
  ├── Apply 20x augmentation to training
  └── Test-time averaging
```

### Phase 3: External Data Pre-training (Day 4)

```
□ Download all TDC ADMET datasets (~25K compounds)
□ Pre-train Chemprop-RDKit on combined data
□ Fine-tune on competition data
□ Compare: with vs without pre-training
```

### Phase 4: Advanced Ensembling (Day 5)

```
□ Collect all OOF predictions (34 columns)
□ Implement hill climbing selection
□ Implement 3-level stacking
□ Optuna weight optimization
□ Final submission generation
```

---

## 8. Expected Performance

### 8.1 Component-wise Improvement Estimates

| Component | Expected RAE Improvement | Confidence |
|-----------|-------------------------|------------|
| Chemprop-RDKit | 10-15% | HIGH (TDC #1) |
| Uni-Mol (3D) | 5-10% | MEDIUM |
| SMILES Augmentation | 3-5% | HIGH (proven) |
| External Pre-training | 5-8% | MEDIUM |
| Better Ensemble | 5-10% | HIGH |
| **Combined** | **25-40%** | MEDIUM |

### 8.2 Target Metrics

| Endpoint | V3 Spearman | V4 Target | Improvement |
|----------|-------------|-----------|-------------|
| LogD | 0.93 | 0.96+ | +3% |
| KSOL | 0.85 | 0.90+ | +6% |
| HLM CLint | 0.80 | 0.88+ | +10% |
| MLM CLint | 0.78 | 0.86+ | +10% |
| Caco-2 Papp | 0.75 | 0.83+ | +11% |
| Caco-2 Efflux | 0.72 | 0.80+ | +11% |
| MPPB | 0.78 | 0.85+ | +9% |
| MBPB | 0.76 | 0.84+ | +11% |
| MGMB | 0.80 | 0.87+ | +9% |

---

## 9. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Uni-Mol slow on CPU | Use GPU, batch processing |
| Chemprop training unstable | Use warmup LR, gradient clipping |
| Overfitting on small data | SMILES augmentation, dropout |
| Negative transfer in MTL | Task grouping, gradient surgery |
| Ensemble overfitting | Strict OOF only, no leakage |

---

## 10. Files to Create

```
src/
├── v4_pipeline.py           # Main orchestration
├── chemprop_rdkit.py        # Chemprop-RDKit wrapper
├── unimol_wrapper.py        # Uni-Mol integration
├── attentivefp_wrapper.py   # AttentiveFP wrapper
├── smiles_augmentation.py   # SMILES enumeration
├── tdc_pretraining.py       # External data pre-training
├── hill_climbing.py         # Ensemble optimization
└── three_level_stacking.py  # Advanced stacking
```

---

## References

1. [ADMET-AI: TDC Leaderboard #1](https://academic.oup.com/bioinformatics/article/40/7/btae416/7698030)
2. [TDC ADMET Benchmark](https://tdcommons.ai/benchmark/admet_group/overview/)
3. [Uni-Mol: 3D Molecular Representation](https://github.com/deepmodeling/Uni-Mol)
4. [BELKA 1st Place Solution](https://zenn.dev/yuto_mo/articles/13fc8fc77c5147)
5. [SMILES Enumeration](https://arxiv.org/abs/1703.07076)
6. [MTGL-ADMET: Multi-task Learning](https://www.sciencedirect.com/science/article/pii/S2589004223023623)
7. [Kaggle Stacking Guide](https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-a-kaggle-competition-with-stacking-using-cuml/)
8. [Chemprop Documentation](https://chemprop.readthedocs.io/)
