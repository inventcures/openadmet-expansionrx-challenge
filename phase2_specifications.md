# Phase 2 Specifications: Breaking into Top 10

---

## User Prompt (27 Dec 2025)

> i am not even in the top 100 of the leaderboard. do a deep research using the current arcitecture, current comprehension of the code https://deepwiki.com/inventcures/fda_crl_analysis and all the .md files, as the base, to come up with : 1. how to break into the top 10 on the leaderboard 2. also use @docs/chemrxiv_27dec2025_1229ist.pdf and @docs/machine-learning-adme-models-in-practice-four-guidelines-from-a-successful-lead-optimization-case-study.pdf as sthe starting input to the deep research write the results and strategy, plan & specifications to the @phase2_specifications.md write this prompt & your next response(in the next turn) to the top of the @phase2_specifications.md

---

## Claude's Response

### Executive Summary

After analyzing both PDF documents, the current codebase, and DeepWiki documentation, I've identified **critical gaps** between our current approach and what top-performing ADMET models require. The key insight: **we're likely overfitting to CV metrics** while the leaderboard uses a different evaluation strategy.

### Key Findings from Research

#### From ChemRxiv Paper: "Practically Significant Method Comparison Protocols"

1. **5x5 Repeated Cross-Validation**: For datasets of 500-100k samples (our range: 222-5,128), use 5-fold CV repeated 5 times with different random seeds
2. **Statistical Testing**: Repeated Measures ANOVA + Tukey HSD post-hoc tests
3. **Practical Significance**: Cohen's D effect size (|d| > 0.2 for small, > 0.5 for medium, > 0.8 for large effects)
4. **Performance Bounds**:
   - **Lower bound**: Null models (mean/median predictor)
   - **Upper bound**: Experimental variability of the assays

#### From ACS Med Chem Paper: "Four Guidelines from Successful Lead Optimization"

1. **Guideline 1 - Time-based Evaluation**: Random/scaffold splits overestimate performance. Time-based splits (train on older, test on newer) are more realistic
2. **Guideline 2 - Global + Local Fine-tuning**: Models trained on global data then fine-tuned on local/project-specific data outperform either approach alone (GNN architecture recommended)
3. **Guideline 3 - Weekly Retraining**: Spearman R drops from 0.65 → 0.55 → 0.49 over 2 months without retraining
4. **Guideline 4 - Interactive/Interpretable/Integrated**: End-user tools matter

---

## Gap Analysis: Current vs. Required

| Aspect | Current Implementation | Required for Top 10 |
|--------|----------------------|---------------------|
| **Fingerprints** | Morgan 2048-bit only | Multiple: ECFP4, ECFP6, MACCS, RDKit FP, AtomPair, TopologicalTorsion |
| **Descriptors** | 25 physicochemical | 200+ Mordred descriptors + 3D conformer features |
| **Pre-training** | None | Uni-Mol, ChemBERTa, MolBERT embeddings |
| **Multi-task** | Independent models per endpoint | Shared representation learning for related endpoints |
| **Ensemble** | Weighted average | Stacking with meta-learner (Ridge/ElasticNet) |
| **Validation** | Single 5-fold CV | 5x5 repeated CV with statistical testing |
| **Graph Models** | Chemprop basic | Chemprop with hyperparameter tuning + AttentiveFP + GIN |
| **Uncertainty** | None | Conformal prediction (MAPIE) |

---

## Phase 2 Strategy: 6 Pillars for Top 10

### Pillar 1: Feature Engineering Expansion

**Goal**: Increase feature diversity from 2,073 to 5,000+ features

```python
# Feature types to add:
features = {
    'fingerprints': {
        'morgan_1024_r2': 1024,   # Existing (reduce from 2048)
        'morgan_1024_r3': 1024,   # Radius 3
        'maccs_keys': 167,        # Substructure keys
        'rdkit_fp': 2048,         # RDKit fingerprint
        'atompair_fp': 1024,      # Atom pair fingerprint
        'torsion_fp': 1024,       # Topological torsion
    },
    'descriptors': {
        'mordred_2d': 1613,       # Full Mordred 2D
        'mordred_3d': 200,        # 3D descriptors (requires conformers)
        'rdkit_basic': 25,        # Current descriptors
    },
    'embeddings': {
        'chemberta': 768,         # Pre-trained transformer
        'unimol': 512,            # 3D conformer embeddings
    }
}
```

**Implementation**: `src/feature_engineering_v2.py`

### Pillar 2: Multi-Task Learning Architecture

**Rationale**: Endpoints are related (e.g., HLM/MLM clearance, MPPB/MBPB binding)

**Endpoint Groupings**:
- **Group A (Metabolic)**: HLM_CLint, MLM_CLint
- **Group B (Binding)**: MPPB, MBPB, MGMB
- **Group C (Permeability)**: Caco2_Papp, Caco2_Efflux
- **Group D (Solubility)**: LogD, KSOL

**Architecture**: Shared trunk with task-specific heads

```python
class MultiTaskADMET(nn.Module):
    def __init__(self, input_dim, shared_dim=512, task_dims=128):
        self.shared_trunk = nn.Sequential(
            nn.Linear(input_dim, shared_dim),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(shared_dim, shared_dim),
            nn.ReLU(), nn.Dropout(0.2),
        )
        self.task_heads = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(shared_dim, task_dims),
                nn.ReLU(),
                nn.Linear(task_dims, 1)
            ) for task in ENDPOINTS
        })
```

**Implementation**: `src/multitask_model.py`

### Pillar 3: Advanced Ensemble Strategy (Stacking)

**Current**: Simple weighted average
**Required**: Two-level stacking ensemble

**Level 1 (Base Models)**:
- XGBoost (Morgan FP)
- CatBoost (Mordred descriptors)
- LightGBM (Combined features)
- RandomForest (MACCS keys)
- Chemprop D-MPNN (Graph)
- Multi-task NN (Shared representations)

**Level 2 (Meta-Learner)**:
- Ridge Regression with per-endpoint optimization
- Input: Out-of-fold predictions from all base models
- Cross-validated to prevent leakage

```python
from sklearn.linear_model import RidgeCV

class StackingEnsemble:
    def __init__(self, base_models, meta_alphas=[0.1, 1.0, 10.0]):
        self.base_models = base_models
        self.meta_model = RidgeCV(alphas=meta_alphas)

    def fit(self, X, y, n_folds=5):
        # Generate OOF predictions
        oof_preds = np.zeros((len(X), len(self.base_models)))
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X)):
            for model_idx, model in enumerate(self.base_models):
                model.fit(X[train_idx], y[train_idx])
                oof_preds[val_idx, model_idx] = model.predict(X[val_idx])

        # Train meta-learner on OOF predictions
        self.meta_model.fit(oof_preds, y)
```

**Implementation**: `src/stacking_ensemble.py`

### Pillar 4: Chemprop Optimization

**Current Issues**: Default hyperparameters, single model
**Required**: Optimized ensemble of D-MPNNs

**Hyperparameter Search Space**:
```python
chemprop_space = {
    'hidden_size': [300, 500, 700, 1000],
    'depth': [3, 4, 5, 6],
    'dropout': [0.0, 0.1, 0.2],
    'ffn_hidden_size': [300, 500, 700],
    'ffn_num_layers': [2, 3],
    'epochs': [30, 50, 100],
    'batch_size': [32, 64, 128],
    'aggregation': ['mean', 'sum', 'norm'],
}
```

**Ensemble Strategy**: Train 5 models with different seeds, average predictions

**Implementation**: `src/chemprop_optimized.py`

### Pillar 5: Validation Strategy Upgrade

**Current**: Single 5-fold CV
**Required**: 5x5 Repeated CV with statistical testing

```python
from sklearn.model_selection import RepeatedKFold

def robust_cv_evaluation(model, X, y, n_splits=5, n_repeats=5):
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    scores = []

    for train_idx, val_idx in rkf.split(X):
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[val_idx])
        rae = np.mean(np.abs(preds - y[val_idx]) / np.abs(y[val_idx].mean() - y[val_idx]))
        scores.append(rae)

    return {
        'mean': np.mean(scores),
        'std': np.std(scores),
        'ci_95': (np.percentile(scores, 2.5), np.percentile(scores, 97.5))
    }
```

**Implementation**: `src/validation.py`

### Pillar 6: 3D Conformer Features (Uni-Mol)

**Rationale**: 3D molecular shape affects ADMET properties (especially binding, permeability)

**Approach**:
1. Generate 3D conformers with RDKit (ETKDG)
2. Use pre-trained Uni-Mol to extract embeddings
3. Concatenate with 2D features

```python
from unimol import UniMolModel

def get_3d_embeddings(smiles_list):
    model = UniMolModel.from_pretrained('unimol_base')
    embeddings = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        emb = model.encode(mol)
        embeddings.append(emb)
    return np.array(embeddings)
```

**Implementation**: `src/conformer_features.py`

---

## Implementation Plan

### Phase 2A: Quick Wins (1-2 days)

| Task | Expected Impact | Priority |
|------|-----------------|----------|
| Add MACCS keys fingerprint | +0.01-0.02 MA-RAE | P0 |
| Add Mordred 2D descriptors | +0.02-0.03 MA-RAE | P0 |
| Implement stacking ensemble | +0.02-0.04 MA-RAE | P0 |
| 5x5 repeated CV validation | Better model selection | P0 |

### Phase 2B: Medium Effort (3-5 days)

| Task | Expected Impact | Priority |
|------|-----------------|----------|
| Multi-task learning for related endpoints | +0.02-0.05 MA-RAE | P1 |
| Chemprop hyperparameter optimization | +0.01-0.03 MA-RAE | P1 |
| Add ChemBERTa embeddings | +0.01-0.02 MA-RAE | P1 |
| Endpoint-specific feature selection | +0.01-0.02 MA-RAE | P1 |

### Phase 2C: High Effort (1 week+)

| Task | Expected Impact | Priority |
|------|-----------------|----------|
| Uni-Mol 3D conformer features | +0.02-0.04 MA-RAE | P2 |
| Neural network ensemble member | +0.01-0.02 MA-RAE | P2 |
| Uncertainty quantification | Better calibration | P2 |

---

## Expected Outcome

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| MA-RAE | ~0.65+ | <0.50 | >0.15 |
| Leaderboard Rank | >100 | Top 10 | Significant |

**Key Success Factors**:
1. **Ensemble diversity** - Different model types, different features
2. **Proper validation** - Avoid overfitting to CV metrics
3. **Multi-task learning** - Leverage endpoint relationships
4. **Feature richness** - Multiple fingerprints + descriptors + embeddings

---

## File Structure for Phase 2

```
src/
├── feature_engineering_v2.py    # Extended feature generation
├── multitask_model.py           # Multi-task neural network
├── stacking_ensemble.py         # Two-level stacking
├── chemprop_optimized.py        # Tuned Chemprop models
├── conformer_features.py        # 3D Uni-Mol embeddings
├── validation.py                # 5x5 repeated CV
└── phase2_pipeline.py           # Main Phase 2 training script
```

---

## Next Steps

1. **Immediate**: Implement `feature_engineering_v2.py` with MACCS + Mordred
2. **Then**: Build stacking ensemble framework
3. **Then**: Add multi-task learning for endpoint groups
4. **Then**: Optimize Chemprop with proper hyperparameter search
5. **Finally**: Add 3D features if time permits

---

*Document created: 27 Dec 2025*
*Last updated: 27 Dec 2025*
