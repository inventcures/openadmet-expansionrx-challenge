# OpenADMET ExpansionRx Blind Challenge - Winning Strategy

## Challenge Overview

**Deadline:** January 19, 2026
**Current Leader:** "pebble" with 0.5593 MA-RAE (136 submissions total)
**Evaluation Metric:** Macro-averaged Relative Absolute Error (MA-RAE)

### Nine ADMET Endpoints to Predict:
1. **LogD** - Lipophilicity (log scale)
2. **KSOL** - Kinetic Solubility
3. **HLM** - Human Liver Microsomal stability
4. **MLM** - Mouse Liver Microsomal stability
5. **Caco-2 Papp A>B** - Intestinal permeability
6. **Caco-2 Efflux Ratio** - Active vs passive transport
7. **MPPB** - Mouse Plasma Protein Binding
8. **MBPB** - Mouse Brain Protein Binding
9. **MGMB** - Mouse Gastrocnemius Muscle Binding

### Key Challenge Characteristics:
- **Time-split validation**: Training on earlier campaign data, predicting later molecules
- Real-world drug discovery data from Expansion Therapeutics
- External data integration is permitted
- Log-scale transformation applied to non-log endpoints

---

## Recommended Model Architecture Strategy

### Primary Recommendation: Ensemble of Diverse Models

Based on extensive research, the optimal approach combines multiple model types:

```
┌─────────────────────────────────────────────────────────────────┐
│                    FINAL ENSEMBLE (Stacking)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   XGBoost/   │  │   Chemprop   │  │   Uni-Mol    │          │
│  │   CatBoost   │  │   D-MPNN     │  │   (3D)       │          │
│  │  (FP+Desc)   │  │              │  │              │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│         │                 │                 │                   │
│         └─────────────────┴─────────────────┘                   │
│                           │                                      │
│                    ┌──────▼──────┐                              │
│                    │ Meta-Learner │                              │
│                    │   (Ridge)    │                              │
│                    └─────────────┘                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tier 1: Core Models (Highest Priority)

### 1. Gradient Boosted Trees with Combined Fingerprints
**Why:** Benchmark studies consistently show XGBoost/CatBoost with fingerprints achieves SOTA on ADMET tasks, ranking 1st in 18/22 TDC benchmarks.

**Configuration:**
```python
# Optimal fingerprint combination (validated by ADMETboost)
features = {
    "ECFP4_counts": {"radius": 2, "nBits": 1024},  # Extended connectivity
    "Avalon_counts": {"nBits": 1024},              # Substructure patterns
    "ErG": {"length": 315},                         # Extended reduced graph
    "RDKit_descriptors": 200,                       # Molecular properties
    # Optional: Add GIN fingerprint for +5% boost
}

# Model: CatBoost (handles missing values, categorical features)
from catboost import CatBoostRegressor
model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=3,
    early_stopping_rounds=50
)
```

### 2. Chemprop (D-MPNN)
**Why:** State-of-the-art message passing neural network with proven track record (discovered Halicin antibiotic). v2 offers 2x speed, 3x memory efficiency.

**Configuration:**
```bash
# Install
pip install chemprop

# Train with multi-task learning for related endpoints
chemprop_train \
    --data_path train.csv \
    --dataset_type regression \
    --save_dir models/chemprop \
    --epochs 100 \
    --ensemble_size 5 \
    --features_generator rdkit_2d_normalized \
    --num_folds 5
```

**Multi-task groupings (based on endpoint relationships):**
- Group A: LogD, Caco-2 Papp, Efflux Ratio (membrane permeability related)
- Group B: HLM, MLM (metabolic stability)
- Group C: MPPB, MBPB, MGMB (tissue binding)

### 3. Uni-Mol / Uni-Mol2 (3D Pretrained)
**Why:** Leverages 3D molecular conformations; pretrained on 209M conformations. Achieves #1 on OGB-LSC benchmark.

**Configuration:**
```python
# Use pretrained Uni-Mol and fine-tune on challenge data
from unimol import UniMolModel

model = UniMolModel.from_pretrained("unimol_base")
# Fine-tune with low learning rate
model.fine_tune(train_data, lr=1e-5, epochs=50)
```

---

## Tier 2: Supplementary Models

### 4. Random Forest Baseline
**Why:** Robust, interpretable, handles small datasets well. Serves as sanity check.

```python
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=1,
    n_jobs=-1
)
```

### 5. Graph Attention Networks (optional)
**Why:** Attention mechanisms can capture critical molecular substructures.

---

## Feature Engineering Strategy

### Molecular Fingerprints (Primary)
| Fingerprint | Type | Why Include |
|------------|------|-------------|
| ECFP4 (radius=2) | Circular | Best general performance |
| Avalon | Substructure | Complementary to ECFP |
| MACCS Keys | Structural | Interpretable, pharmacophoric |
| ErG | Reduced graph | Captures scaffold topology |

### Molecular Descriptors (Secondary)
```python
from rdkit import Chem
from rdkit.Chem import Descriptors, Descriptors3D
from mordred import Calculator, descriptors

# Key descriptors for ADMET
key_descriptors = [
    # Lipophilicity (critical for LogD, Caco-2)
    'MolLogP', 'TPSA', 'MolWt',
    # H-bonding (affects permeability, binding)
    'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds',
    # Electrostatics (affects binding)
    'NumAromaticRings', 'FractionCSP3',
    # Size/Shape
    'LabuteASA', 'BalabanJ',
]
```

### 3D Conformer Features (for Uni-Mol)
```python
from rdkit.Chem import AllChem

def generate_3d_conformer(mol, n_conformers=10):
    mol = Chem.AddHs(mol)
    AllChem.EmbedMultipleConfs(mol, n_conformers, randomSeed=42)
    AllChem.MMFFOptimizeMoleculeConfs(mol)
    return mol
```

---

## Multi-Task Learning Strategy

### Rationale
Research shows MTL outperforms STL by 40-60% reduction in error when tasks are related. ADMET endpoints share underlying biophysical properties.

### Implementation with MTGL-ADMET approach:
```python
# Task relationship matrix (empirical correlations)
task_correlations = {
    ('LogD', 'Caco2_Papp'): 0.65,    # Lipophilicity → permeability
    ('LogD', 'PPB'): 0.58,           # Lipophilicity → protein binding
    ('HLM', 'MLM'): 0.82,            # Cross-species metabolic stability
    ('MPPB', 'MBPB'): 0.71,          # Tissue binding similarities
}

# Use adaptive auxiliary task weighting
# Weight tasks by gradient magnitude normalization
```

---

## Ensemble Strategy

### Stacking Meta-Learner
```python
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict

# Get OOF predictions from each base model
base_models = [xgb_model, chemprop_model, unimol_model, rf_model]

meta_features = np.column_stack([
    cross_val_predict(model, X_train, y_train, cv=5)
    for model in base_models
])

# Train meta-learner
meta_model = RidgeCV(alphas=[0.1, 1.0, 10.0])
meta_model.fit(meta_features, y_train)
```

### Weighted Averaging (simpler alternative)
```python
# Weights based on validation performance
weights = {
    'xgboost': 0.35,
    'chemprop': 0.30,
    'unimol': 0.25,
    'rf': 0.10
}

final_pred = sum(w * pred for w, pred in zip(weights.values(), predictions))
```

---

## Uncertainty Quantification

### Conformal Prediction (Recommended)
```python
from mapie.regression import MapieRegressor

# Wrap your model with conformal prediction
mapie = MapieRegressor(model, cv=5, method="plus")
mapie.fit(X_train, y_train)

# Get predictions with confidence intervals
y_pred, y_intervals = mapie.predict(X_test, alpha=0.1)  # 90% CI
```

### Benefits for this challenge:
- Identifies predictions likely to be wrong
- Can weight ensemble by confidence
- May correlate with RAE performance

---

## Data Augmentation Strategies

### 1. SMILES Augmentation
```python
from rdkit import Chem

def smiles_augmentation(smiles, n_variants=5):
    mol = Chem.MolFromSmiles(smiles)
    variants = [Chem.MolToSmiles(mol, doRandom=True) for _ in range(n_variants)]
    return list(set(variants))
```

### 2. External Data Integration (Permitted by challenge)
**High-quality sources:**
- ChEMBL (curated ADMET data)
- OpenADMET datasets
- Polaris benchmarks
- AIRCHECK datasets

**Warning from Pat Walters:** Avoid MoleculeNet and TDC - they contain fundamental flaws.

---

## Validation Strategy

### Time-Split Cross-Validation (Critical)
The challenge uses time-split, so mimic this in validation:

```python
# Sort by date if available, otherwise use scaffold split
from sklearn.model_selection import TimeSeriesSplit

# If no dates, use scaffold clustering as proxy
from rdkit.Chem.Scaffolds import MurckoScaffold

def get_scaffold_split(smiles_list, n_splits=5):
    scaffolds = [MurckoScaffold.MurckoScaffoldSmiles(smi) for smi in smiles_list]
    # Cluster by scaffold similarity
    # Use later scaffolds as test set
```

### Avoid Overly Optimistic Random Splits
Random CV gives ~20% better metrics than time-split but doesn't reflect real performance.

### Series-Level Evaluation (From Rich et al. 2024)
**Critical insight**: Model performance varies by chemical series!

```python
# Stratify evaluation by scaffold/series
def evaluate_by_series(y_true, y_pred, series_labels):
    """Report metrics stratified by chemical series"""
    results = {}
    for series in np.unique(series_labels):
        mask = series_labels == series
        results[series] = {
            'MAE': mean_absolute_error(y_true[mask], y_pred[mask]),
            'Spearman_R': spearmanr(y_true[mask], y_pred[mask])[0],
            'n_compounds': mask.sum()
        }
    return results
```

### Progressive Validation (Recommended)
Following the Nested/Inductive approach:
1. **Initial validation**: Use first 100 compounds as training, next 100 as test
2. **Ongoing validation**: Weekly re-evaluation with time-based splits
3. **Track model degradation**: Monitor if performance drops as chemistry evolves

---

## Endpoint-Specific Tips

### LogD (Lipophilicity)
- Add S+ logP/logD as helper features
- Consider pH-dependent calculations
- Strong correlation with other endpoints - good MTL anchor

### Caco-2 Permeability
- Include 3D descriptors (15% MAE reduction)
- TPSA and H-bond descriptors critical
- Consider efflux ratio as related task

### Microsomal Stability (HLM/MLM)
- Cross-species correlation is high (~0.82)
- Train jointly for transfer learning
- Consider MetaboGNN architecture

### Protein Binding (MPPB/MBPB/MGMB)
- Lipophilicity is primary driver
- Consider tissue-specific features
- Train as multi-output regression

---

## Implementation Roadmap

### Week 1-2: Baseline Models
1. ☐ Download and explore challenge data from HuggingFace
2. ☐ Implement XGBoost + combined fingerprints baseline
3. ☐ Set up proper time-split/scaffold-split CV
4. ☐ Submit baseline to leaderboard

### Week 2-3: Advanced Models
5. ☐ Train Chemprop D-MPNN with multi-task learning
6. ☐ Fine-tune Uni-Mol on challenge data
7. ☐ Compare single-task vs multi-task performance

### Week 3-4: Ensemble & Optimization
8. ☐ Build stacking ensemble
9. ☐ Implement uncertainty-weighted averaging
10. ☐ Hyperparameter optimization (Optuna)
11. ☐ Final submission

---

## Code Resources

### Essential Libraries
```bash
pip install rdkit chemprop catboost xgboost scikit-learn
pip install torch torch-geometric  # For GNNs
pip install mapie  # Conformal prediction
pip install optuna  # Hyperparameter optimization
```

### Recommended Tutorials
- [Practical Cheminformatics](https://github.com/PatWalters/practical_cheminformatics_tutorials)
- [Chemprop Documentation](https://github.com/chemprop/chemprop)
- [Uni-Mol Repository](https://github.com/deepmodeling/Uni-Mol)

---

## Four Guidelines from Industry (Rich et al., ACS Med Chem Lett 2024)

This paper from Nested Therapeutics/Inductive Bio provides critical real-world guidance:

### Guideline 1: Time-Based + Series-Level Evaluation
> "Time-based splits simulate real usage... more rigorous than random or scaffold splitting, which can overestimate performance"

**Key findings:**
- Random/scaffold splits are **overly optimistic**
- Stratify metrics by chemical series (performance varies by chemotype)
- Re-evaluate weekly using time-based splits
- Build initial trust with historical program data before deployment

### Guideline 2: Combine Global + Local Data (Fine-tuning)
> "Fine-tuned models trained with combined local and global data perform better than those trained with local or global data alone"

**Their approach:**
- Start with curated global proprietary dataset
- Fine-tune by adding project-specific data
- Use Graph Neural Network architecture

**Critical insight from their case study:**
- RLM clearance was **8× higher** than HLM in their program
- Global-only model couldn't predict this species divergence
- Fine-tuned model captured it successfully
- **Lesson: Always validate on early program data**

### Guideline 3: Frequent Model Retraining (Weekly)
> "Weekly retraining aligns well with the weekly cycle of design meetings"

**Performance degradation with stale models (HLM):**
| Model Age | Spearman R |
|-----------|------------|
| Current week | 0.65 |
| 1 month old | 0.55 |
| 2 months old | 0.49 |

**Activity cliff handling:**
- Models can't predict activity cliffs in advance
- Weekly retraining allows rapid adjustment to newly discovered SAR
- Example: substitution position change caused several-fold jump in clearance

### Guideline 4: Interactive, Interpretable, Integrated
> "The best ML ADME model will not have an impact unless it is actively used"

**Three I's for adoption:**
1. **Integrated**: Available within existing computational workflows
2. **Interactive**: Real-time predictions during ideation (not batch scoring)
3. **Interpretable**: Atom-level visualizations, metabolism site predictions

**Their success story:**
- Compound 1 → Compound 5 progression
- Co-optimized permeability + metabolic stability + potency
- Achieved development candidate with excellent cross-species PK

---

## Key Takeaways from Research

1. **Fingerprints still win**: ECFP + Avalon + descriptors with XGBoost/CatBoost matches or beats deep learning in most ADMET tasks

2. **Multi-task learning helps**: 40-60% error reduction when tasks are properly related

3. **3D matters for some endpoints**: Uni-Mol with conformers improves permeability predictions

4. **Ensemble is essential**: Stacking diverse models (tree-based + neural) provides best results

5. **Time-split is critical**: Random CV is overly optimistic; use scaffold or temporal splits

6. **Data quality > quantity**: Be selective about external data sources

7. **Uncertainty helps**: Conformal prediction can identify unreliable predictions

8. **Fine-tune global models**: Combining external data with local program data outperforms either alone (Guideline 2)

9. **Retrain frequently**: Weekly retraining maintains accuracy as chemistry evolves (Guideline 3)

---

## References

- [Rich et al. - Machine Learning ADME Models in Practice: Four Guidelines (2024)](https://pubs.acs.org/doi/10.1021/acsmedchemlett.4c00290) **← Critical paper for this challenge**
- [ADMETboost Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9903341/)
- [Chemprop Paper](https://pubs.acs.org/doi/10.1021/acs.jcim.3c01250)
- [ChemXTree](https://pubs.acs.org/doi/10.1021/acs.jcim.4c01186)
- [Fingerprint Combinations for ADMET](https://arxiv.org/pdf/2310.00174)
- [Pat Walters Resources](https://github.com/PatWalters/resources_2025)
- [Multi-Task ADMET Prediction](https://www.sciencedirect.com/science/article/pii/S2589004223023623)
- [Uni-Mol](https://github.com/deepmodeling/Uni-Mol)
- [Time-Split Cross-Validation (Sheridan 2013)](https://pubs.acs.org/doi/abs/10.1021/ci400084k)
- [Conformal ADMET Prediction](https://github.com/peiyaoli/Conformal-ADMET-Prediction)
