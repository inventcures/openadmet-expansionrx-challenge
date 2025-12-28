# V3 Deep Research Specs: TOP 3 Strategy for OpenADMET ExpansionRx Challenge

**Generated:** 2025-12-28
**Goal:** Break into TOP 3 on the leaderboard
**Status:** Implementation-ready specification

---

## Executive Summary

Based on deep research of 3 papers in `docs/` and latest 2024-2025 ADMET AI literature, this document outlines a comprehensive strategy to achieve TOP 3 performance. The key insight is that **no single model architecture dominates** - TOP performers use **diverse ensembles with proper pre-training, multi-task learning, and rigorous cross-validation**.

---

## 1. Literature Key Findings

### 1.1 From "Practically Significant Method Comparison Protocols" (ChemRxiv)

**Critical for Robust Evaluation:**
- **5x5 Repeated Cross-Validation**: 5 repeats × 5 folds = 25 samples for statistical power
- **Scaffold-based Splitting**: Bemis-Murcko scaffolds ensure no train/test scaffold overlap
- **Tukey HSD Testing**: For statistically significant method comparison
- **Cohen's D Effect Size**: d ≥ 0.8 = large effect, d ≥ 0.5 = medium
- **Post-hoc Classification**: Convert regression to classification for decisional metrics

### 1.2 From "ML ADME Models in Practice: Four Guidelines"

**Proven Industry Strategies:**
1. **Fine-tuned Global+Local Models** outperform either alone (5-15% improvement)
2. **Weekly Retraining** adapts to activity cliffs
3. **GNNs are the architecture of choice** for ADME prediction
4. **Data quality > model complexity** - curation matters

### 1.3 From Web Research (2024-2025 State-of-the-Art)

| Model | Architecture | Key Innovation | Source |
|-------|-------------|----------------|--------|
| **ADMET-AI** | Chemprop-RDKit | Best on TDC ADMET leaderboard | Oxford Academic |
| **ChemXTree** | GNN + Neural Decision Tree | Interpretable ADMET | ACS JCIM 2024 |
| **MSformer-ADMET** | Transformer | Multiscale fragment pretraining | Oxford Briefings 2025 |
| **HimNet** | Hierarchical GNN | Atom-bond-molecule hierarchy | Research Square 2025 |
| **MTGL-ADMET** | Multi-task GNN | Adaptive auxiliary task selection | ScienceDirect |

**NeurIPS 2024 BELKA Competition Winners Used:**
- 1D CNNs on SMILES sequences
- Graph Neural Networks
- Transformers with various tokenizers (SMILES, SELFIES, DeepSMILES)
- Heavy ensembling of diverse architectures

---

## 2. Architecture Strategy for TOP 3

### 2.1 Multi-Architecture Ensemble (Priority: CRITICAL)

The winning strategy is **NOT** finding the single best model, but combining complementary architectures:

```
Level 1: Base Models (8-10 diverse models)
├── GNN-based
│   ├── Chemprop D-MPNN (current, keep)
│   ├── Multi-task Chemprop (shared MPNN, implemented)
│   ├── Chemprop with RDKit features (hybrid)
│   └── AttentiveFP (attention-based GNN)
├── Transformer-based
│   ├── ChemBERTa embeddings → XGBoost (current, keep)
│   ├── MolBERT embeddings → Ridge
│   └── Uni-Mol (3D pre-trained transformer)
├── Fingerprint-based
│   ├── ECFP4 + LightGBM (current, keep)
│   ├── ECFP6 + XGBoost
│   ├── MACCS + CatBoost
│   └── RDKit descriptors + RandomForest
└── Sequence-based (NEW)
    ├── 1D CNN on SMILES (BELKA winner technique)
    └── LSTM on SMILES

Level 2: Meta-Learner Stacking
├── Per-endpoint Ridge regression (current)
├── Per-endpoint Gradient Boosting (add)
└── Weighted average by OOF performance
```

### 2.2 Pre-training Strategy (Priority: HIGH)

**Why Pre-training Matters:**
- Models pre-trained on large molecular datasets generalize better
- Transfer learning from related ADMET tasks improves performance

**Implementation:**
```python
# Pre-trained models to integrate:
PRETRAINED_MODELS = {
    'chemberta': 'seyonec/ChemBERTa-zinc-base-v1',  # Current
    'molbert': 'seyonec/PubChem10M_SMILES_BPE_450k',
    'chemgpt': 'ncfrey/ChemGPT-19M',  # Generative pre-training
    'unimol': 'dptech/Uni-Mol',  # 3D pre-trained
}

# External ADMET datasets for transfer learning:
EXTERNAL_DATASETS = [
    'TDC ADMET Benchmark',  # ~20 datasets
    'ChEMBL ADMET subset',
    'PubChem BioAssay ADME',
]
```

### 2.3 Multi-Task Learning Architecture (Priority: HIGH)

**Current:** Multi-task Chemprop with shared MPNN (implemented)

**Enhanced Design:**
```
Input: SMILES
    ↓
Shared Encoder (Message Passing Neural Network)
    ├── Hidden size: 500-600
    ├── Depth: 4-5 layers
    └── Aggregation: Mean + Max pooling
    ↓
Shared Backbone FFN
    ├── Hidden: 512 → 256
    └── Dropout: 0.15
    ↓
Task-Specific Heads (9 endpoints)
    ├── LogD head (physico-chem)
    ├── KSOL head (physico-chem)
    ├── HLM CLint head (metabolism)
    ├── MLM CLint head (metabolism)
    ├── Caco-2 Papp head (permeability)
    ├── Caco-2 Efflux head (permeability)
    ├── MPPB head (binding)
    ├── MBPB head (binding)
    └── MGMB head (binding)
```

**Task Grouping (exploit correlations):**
```python
TASK_GROUPS = {
    'physicochemical': ['LogD', 'KSOL'],
    'metabolism': ['HLM CLint', 'MLM CLint'],
    'permeability': ['Caco-2 Permeability Papp A>B', 'Caco-2 Permeability Efflux'],
    'binding': ['MPPB', 'MBPB', 'MGMB'],
}
```

---

## 3. Feature Engineering Strategy

### 3.1 Multi-Scale Molecular Representations

```python
FEATURE_SETS = {
    # Atom-level (for GNNs)
    'atom_features': ['atomic_num', 'chirality', 'hybridization', 'aromatic',
                      'ring_size', 'degree', 'formal_charge', 'num_Hs'],

    # Bond-level (for GNNs)
    'bond_features': ['bond_type', 'conjugated', 'ring', 'stereo'],

    # Molecule-level (for fingerprint models)
    'fingerprints': {
        'ECFP4_1024': {'radius': 2, 'bits': 1024},
        'ECFP6_2048': {'radius': 3, 'bits': 2048},
        'FCFP4_1024': {'radius': 2, 'bits': 1024, 'features': True},
        'MACCS_167': {'keys': 167},
        'RDKit_2048': {'bits': 2048},
    },

    # Descriptor-level
    'rdkit_descriptors': [
        'MolWt', 'LogP', 'TPSA', 'NumHDonors', 'NumHAcceptors',
        'NumRotatableBonds', 'NumAromaticRings', 'FractionCSP3',
        'MolMR', 'NumHeteroatoms', 'NumAliphaticRings',
    ],

    # Pre-trained embeddings
    'embeddings': {
        'chemberta_768': 768,  # Full
        'chemberta_256': 256,  # PCA reduced
        'molbert_768': 768,
    },
}
```

### 3.2 Fragment-Aware Features (MSformer-ADMET inspiration)

```python
def extract_fragment_features(smiles):
    """
    Extract Bemis-Murcko scaffolds and functional group counts
    for fragment-aware ADMET prediction.
    """
    mol = Chem.MolFromSmiles(smiles)

    features = {
        # Scaffold features
        'scaffold_smiles': get_murcko_scaffold(mol),
        'generic_scaffold': get_generic_scaffold(mol),
        'scaffold_fp': compute_scaffold_fingerprint(mol),

        # Functional groups (ADMET-relevant)
        'n_carboxylic_acids': count_pattern(mol, '[CX3](=O)[OX2H1]'),
        'n_amines': count_pattern(mol, '[NX3;H2,H1;!$(NC=O)]'),
        'n_amides': count_pattern(mol, '[NX3][CX3](=[OX1])[#6]'),
        'n_esters': count_pattern(mol, '[#6][CX3](=O)[OX2H0][#6]'),
        'n_sulfonamides': count_pattern(mol, '[#16X4]([NX3])(=[OX1])(=[OX1])'),
        'n_halogens': count_pattern(mol, '[F,Cl,Br,I]'),
        # ... more ADMET-relevant groups
    }
    return features
```

---

## 4. Training Strategy

### 4.1 Cross-Validation Protocol (5x5 Repeated Scaffold CV)

```python
CV_CONFIG = {
    'n_repeats': 5,
    'n_folds': 5,
    'split_type': 'scaffold',  # Bemis-Murcko scaffold-based
    'stratify_by': None,  # Regression, no stratification
    'random_seeds': [42, 123, 456, 789, 1024],  # One per repeat
}

# This gives 25 CV folds for robust performance estimation
# Statistical power for Tukey HSD testing
```

### 4.2 Hyperparameter Optimization

```python
# Bayesian optimization with Optuna
OPTUNA_CONFIG = {
    'n_trials': 100,
    'pruning': True,  # Early stopping bad trials
    'sampler': 'TPESampler',
    'objective': 'minimize_rae',  # Competition metric
}

# Search space per model type
SEARCH_SPACES = {
    'chemprop': {
        'hidden_size': [300, 400, 500, 600],
        'depth': [3, 4, 5],
        'dropout': [0.0, 0.1, 0.2, 0.3],
        'ffn_hidden_size': [200, 300, 400],
        'ffn_num_layers': [2, 3],
        'epochs': [30, 50, 75, 100],
        'lr': [1e-4, 5e-4, 1e-3],
    },
    'lightgbm': {
        'n_estimators': [100, 200, 500, 1000],
        'max_depth': [3, 5, 7, 10, -1],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [31, 63, 127],
        'min_child_samples': [5, 10, 20],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [0, 0.1, 1],
    },
}
```

### 4.3 Training Schedule

```python
TRAINING_SCHEDULE = {
    # Phase 1: Feature extraction (parallelizable)
    'phase1_features': {
        'molecular_features': True,  # RDKit descriptors, fingerprints
        'chemberta_embeddings': True,  # Pre-trained embeddings
        'fragment_features': True,  # Scaffold, functional groups
        'estimated_time_gpu': '10 min',
    },

    # Phase 2: Base model training (per-endpoint)
    'phase2_base_models': {
        'chemprop_singletask': {'n_models': 5, 'per_endpoint': True},
        'chemprop_multitask': {'n_models': 5, 'all_endpoints': True},
        'lightgbm_ecfp': {'n_models': 10, 'per_endpoint': True},
        'xgboost_chemberta': {'n_models': 5, 'per_endpoint': True},
        'catboost_rdkit': {'n_models': 5, 'per_endpoint': True},
        'estimated_time_gpu': '2-3 hours',
    },

    # Phase 3: Meta-learner stacking
    'phase3_stacking': {
        'ridge_meta': True,
        'gbm_meta': True,
        'weighted_average': True,
        'estimated_time': '15 min',
    },
}
```

---

## 5. Ensemble Strategy

### 5.1 Two-Level Stacking Architecture

```
Level 0: Out-of-Fold Predictions
─────────────────────────────────
For each endpoint, collect OOF predictions from:
  - Chemprop Single-task (5 seeds) → 5 columns
  - Chemprop Multi-task (5 seeds) → 5 columns
  - LightGBM ECFP4 (10 seeds) → 10 columns
  - XGBoost ChemBERTa (5 seeds) → 5 columns
  - CatBoost RDKit (5 seeds) → 5 columns
  - RandomForest descriptors → 5 columns
  = 35 base predictions per endpoint

Level 1: Meta-Learner
─────────────────────
Input: 35 base predictions + original features (optional)
Output: Final prediction

Meta-learner options (ensemble them too):
  1. Ridge regression (α optimized via CV)
  2. ElasticNet (α, l1_ratio optimized)
  3. LightGBM (shallow, regularized)
  4. Simple weighted average (weights from OOF R²)
```

### 5.2 Diversity-Weighted Ensemble

```python
def compute_ensemble_weights(oof_predictions, y_true, method='diversity'):
    """
    Compute ensemble weights that maximize diversity and performance.

    Methods:
    - 'performance': Weight by individual R² or Spearman
    - 'diversity': Maximize prediction diversity while maintaining accuracy
    - 'greedy': Forward selection to minimize ensemble error
    """
    if method == 'performance':
        scores = [spearmanr(pred, y_true)[0] for pred in oof_predictions]
        weights = softmax(scores)

    elif method == 'diversity':
        # Negative correlation ensemble (NCL)
        # Weight models that are accurate AND diverse
        n_models = len(oof_predictions)
        corr_matrix = np.corrcoef(oof_predictions)

        weights = []
        for i in range(n_models):
            accuracy = spearmanr(oof_predictions[i], y_true)[0]
            diversity = 1 - np.mean(corr_matrix[i])  # Low correlation with others
            weights.append(accuracy * diversity)
        weights = softmax(weights)

    elif method == 'greedy':
        # Greedy forward selection
        weights = greedy_ensemble_selection(oof_predictions, y_true)

    return weights
```

### 5.3 Uncertainty-Weighted Predictions

```python
def uncertainty_weighted_ensemble(predictions, uncertainties):
    """
    Weight predictions by inverse uncertainty (epistemic).

    Models more confident in their predictions get higher weight.
    """
    # Uncertainty from:
    # - Ensemble variance (multiple seeds)
    # - MC Dropout variance
    # - Distance to training data (applicability domain)

    weights = 1.0 / (uncertainties + 1e-8)
    weights = weights / weights.sum()

    return np.average(predictions, weights=weights)
```

---

## 6. External Data Strategy

### 6.1 Pre-training on Public ADMET Datasets

```python
EXTERNAL_DATASETS = {
    # TDC ADMET Benchmark (Therapeutics Data Commons)
    'tdc_lipophilicity': {'task': 'LogD', 'n_samples': 4200},
    'tdc_solubility_aqsol': {'task': 'KSOL', 'n_samples': 9982},
    'tdc_hia': {'task': 'permeability', 'n_samples': 578},
    'tdc_ppbr': {'task': 'binding', 'n_samples': 1797},
    'tdc_cyp_substrates': {'task': 'metabolism', 'n_samples': 1000+},

    # ChEMBL curated
    'chembl_logd': {'task': 'LogD', 'n_samples': ~50000},
    'chembl_solubility': {'task': 'KSOL', 'n_samples': ~20000},

    # PubChem BioAssay
    'pubchem_caco2': {'task': 'permeability', 'n_samples': ~5000},
}

# Strategy: Pre-train on large external, fine-tune on competition data
TRANSFER_LEARNING_STRATEGY = {
    'phase1_pretrain': {
        'datasets': ['chembl_logd', 'tdc_solubility_aqsol', ...],
        'epochs': 100,
        'freeze_encoder': False,
    },
    'phase2_finetune': {
        'dataset': 'competition_train',
        'epochs': 50,
        'freeze_encoder_epochs': 10,  # Freeze initially
        'unfreeze_lr_mult': 0.1,  # Lower LR for encoder
    },
}
```

### 6.2 Data Augmentation

```python
DATA_AUGMENTATION = {
    # SMILES enumeration (same molecule, different SMILES)
    'smiles_enumeration': {
        'n_augmentations': 5,
        'use_for': ['transformer_models'],
    },

    # Test-time augmentation
    'tta': {
        'n_augmentations': 10,
        'aggregation': 'mean',
    },
}
```

---

## 7. Implementation Plan

### Phase 1: Foundation (Current → +2 days)

```
[x] Multi-task Chemprop (implemented)
[x] Scaffold-based CV (implemented)
[x] ChemBERTa embeddings (implemented)
[x] Deep stacking ensemble (implemented)
[x] Checkpointing system (implemented)

[ ] Fix any remaining bugs
[ ] Run full v2 pipeline, establish baseline
```

### Phase 2: Architecture Diversity (+3 days)

```
[ ] Add AttentiveFP GNN model
[ ] Add 1D CNN on SMILES (BELKA winner technique)
[ ] Add MolBERT embeddings as alternative to ChemBERTa
[ ] Add CatBoost as alternative to LightGBM
[ ] Implement ECFP6 fingerprints (currently ECFP4)
```

### Phase 3: Pre-training & Transfer Learning (+4 days)

```
[ ] Download and curate TDC ADMET datasets
[ ] Pre-train Chemprop on ChEMBL LogD/Solubility
[ ] Fine-tune pre-trained models on competition data
[ ] Implement Uni-Mol 3D pre-trained features (if time)
```

### Phase 4: Advanced Ensembling (+2 days)

```
[ ] Implement diversity-weighted ensemble
[ ] Implement greedy ensemble selection
[ ] Add uncertainty quantification (ensemble variance)
[ ] Test-time augmentation for transformers
```

### Phase 5: Hyperparameter Optimization (+3 days)

```
[ ] Set up Optuna study per endpoint
[ ] Run 100-trial Bayesian optimization
[ ] Re-train best configs with full CV
[ ] Final ensemble with optimized models
```

---

## 8. Expected Improvements

| Component | Expected RAE Improvement | Confidence |
|-----------|-------------------------|------------|
| Multi-task Chemprop | 3-5% | High |
| Scaffold-based CV | Better generalization | High |
| Pre-training on external data | 5-10% | Medium |
| Architecture diversity | 3-7% | High |
| Hyperparameter optimization | 2-5% | High |
| Advanced ensembling | 2-4% | Medium |
| **Combined** | **15-25%** | Medium |

---

## 9. Code Structure

```
src/
├── v3_pipeline.py           # Main orchestration
├── models/
│   ├── chemprop_models.py   # Single-task, multi-task Chemprop
│   ├── transformer_models.py # ChemBERTa, MolBERT, Uni-Mol
│   ├── gnn_models.py        # AttentiveFP, custom GNNs
│   ├── fingerprint_models.py # LightGBM, XGBoost, CatBoost
│   └── sequence_models.py   # 1D CNN, LSTM on SMILES
├── features/
│   ├── fingerprints.py      # ECFP, FCFP, MACCS, RDKit
│   ├── descriptors.py       # RDKit descriptors
│   ├── embeddings.py        # Pre-trained embeddings
│   └── fragments.py         # Scaffold, functional groups
├── ensemble/
│   ├── stacking.py          # Two-level stacking
│   ├── blending.py          # OOF blending
│   └── selection.py         # Greedy selection, diversity
├── cv/
│   ├── scaffold_cv.py       # Scaffold-based splits
│   └── repeated_cv.py       # 5x5 repeated CV
├── transfer/
│   ├── pretrain.py          # Pre-training on external data
│   └── finetune.py          # Fine-tuning strategy
└── utils/
    ├── checkpointing.py     # Save/resume functionality
    ├── logging.py           # Detailed logging
    └── metrics.py           # RAE, Spearman, MAE
```

---

## 10. Critical Success Factors

1. **Diverse Model Architectures**: No single model wins - diversity is key
2. **Proper Cross-Validation**: 5x5 repeated scaffold CV for robust estimates
3. **Pre-training**: Leverage external ADMET data for transfer learning
4. **Multi-task Learning**: Exploit correlations between related endpoints
5. **Ensemble Diversity**: Weight by both accuracy AND diversity
6. **Hyperparameter Optimization**: Systematic search, not guessing
7. **Rigorous Evaluation**: Statistical testing (Tukey HSD) for significance

---

## 11. Sources

### Papers (in docs/)
1. "Practically Significant Method Comparison Protocols for ML in Small Molecule Drug Discovery" - ChemRxiv 2024
2. "ML ADME Models in Practice: Four Guidelines from a Successful Lead Optimization Case Study"

### Web Research (2024-2025)
- [ADMET-AI: ML ADMET Platform](https://academic.oup.com/bioinformatics/article/40/7/btae416/7698030) - Oxford Academic
- [ChemXTree: GNN-NDT for ADMET](https://pubs.acs.org/doi/10.1021/acs.jcim.4c01186) - ACS JCIM 2024
- [MSformer-ADMET: Multiscale Fragment Pretraining](https://academic.oup.com/bib/article/26/5/bbaf506/8267887) - Oxford Briefings 2025
- [MTGL-ADMET: Multi-task Graph Learning](https://www.sciencedirect.com/science/article/pii/S2589004223023623) - ScienceDirect
- [Chemprop v2 Paper](https://pubs.acs.org/doi/10.1021/acs.jcim.3c01250) - ACS JCIM 2023
- [NeurIPS 2024 BELKA Competition](https://www.kaggle.com/competitions/leash-BELKA) - Kaggle
- [BELKA 11th Place Solution](https://github.com/dangnh0611/kaggle_leash_belka) - GitHub

---

## 12. Implementation Command

When you (Claude Code) receive this document, execute:

```bash
# Start with current v2 pipeline as baseline
python src/top10_pipeline.py --status

# Then implement v3 enhancements step by step
# Priority order: pre-training > architecture diversity > ensemble improvements
```

Focus on **incremental improvements** - each component should be testable independently before combining into the final ensemble.

---

*Document generated by Claude Code for OpenADMET ExpansionRx Challenge TOP 3 strategy.*
