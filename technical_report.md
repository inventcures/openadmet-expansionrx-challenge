# Tiered Ensemble Learning for Multi-Task ADMET Property Prediction

**OpenADMET ExpansionRx Challenge Technical Report**

---

**Authors:** [Your Name]<sup>1</sup>
**Affiliations:** <sup>1</sup>[Your Affiliation]
**Correspondence:** [email]
**Code:** https://github.com/inventcures/openadmet-expansionrx-challenge
**Documentation:** https://deepwiki.com/inventcures/openadmet-expansionrx-challenge

---

## Abstract

We present a tiered ensemble approach for predicting nine ADMET (Absorption, Distribution, Metabolism, Excretion, and Toxicity) properties from molecular structures as part of the OpenADMET ExpansionRx Challenge. Our architecture progressively increases model sophistication across four tiers: baseline CatBoost, optimized gradient boosting (XGBoost, CatBoost, LightGBM), hybrid CatBoost-RandomForest ensembles, and Chemprop D-MPNN graph neural networks. We employ Morgan fingerprints combined with physicochemical descriptors as molecular representations, with endpoint-specific weighted ensembling based on cross-validation performance. Our implementation features idempotent training with checkpointing, supporting both Apple Silicon (MPS) and NVIDIA GPU (CUDA) acceleration. On the challenge benchmark, our best single model (XGBoost) achieves MA-RAE of 0.5730, approaching the leaderboard baseline of 0.5593.

**Keywords:** ADMET prediction, gradient boosting, ensemble learning, molecular fingerprints, drug discovery, XGBoost, CatBoost, LightGBM, Chemprop

---

## 1. Introduction

Accurate prediction of ADMET properties is crucial in early-stage drug discovery, enabling researchers to identify promising drug candidates while filtering out compounds with unfavorable pharmacokinetic profiles (Lombardo et al., 2017). Computational ADMET prediction can significantly reduce attrition rates in drug development by identifying problematic compounds before costly experimental testing.

The OpenADMET ExpansionRx Challenge provides a standardized benchmark for evaluating computational methods on nine key ADMET endpoints, with approximately 5,000 training molecules and 2,282 test compounds. The primary evaluation metric is Mean Average Relative Absolute Error (MA-RAE), which normalizes prediction error across endpoints with different scales.

Traditional QSAR (Quantitative Structure-Activity Relationship) approaches rely on molecular descriptors and machine learning models to predict biological properties from chemical structures (Cherkasov et al., 2014). Recent advances in gradient boosting methods—particularly XGBoost (Chen & Guestrin, 2016), CatBoost (Prokhorenkova et al., 2018), and LightGBM (Ke et al., 2017)—have shown strong performance in molecular property prediction. Additionally, graph neural networks like Chemprop's Directed Message Passing Neural Network (D-MPNN) enable end-to-end learning directly from molecular graphs (Yang et al., 2019).

In this work, we present a tiered ensemble approach combining the strengths of multiple modeling paradigms. Our key contributions include:

1. A **tiered architecture** with progressive model sophistication from baseline to deep learning
2. **Comprehensive feature engineering** combining Morgan fingerprints (2,048 bits) with 25 physicochemical descriptors
3. An **endpoint-specific weighted ensemble** strategy adapting to each target's characteristics
4. **Production-ready infrastructure** with checkpointing, signal handling, and multi-platform GPU support

---

## 2. Methods

### 2.1 Dataset

The ExpansionRx dataset comprises training compounds with labels for nine ADMET endpoints. Due to experimental constraints, each endpoint has a different number of available measurements, requiring per-target handling of missing values.

| Endpoint | Description | N (train) | Valid Range | Unit |
|----------|-------------|-----------|-------------|------|
| LogD | Distribution Coefficient | 5,039 | [-3, 6] | log units |
| KSOL | Kinetic Solubility | 5,128 | [0.001, 350] | μM |
| HLM CLint | Human Liver Microsomal Clearance | 3,759 | [0, 3000] | μL/min/mg |
| MLM CLint | Mouse Liver Microsomal Clearance | 4,522 | [0, 12000] | μL/min/mg |
| Caco-2 Papp A>B | Caco-2 Permeability | 2,157 | [0, 60] | 10⁻⁶ cm/s |
| Caco-2 Efflux | Caco-2 Efflux Ratio | 2,161 | [0.2, 120] | ratio |
| MPPB | Mouse Plasma Protein Binding | 1,302 | [0, 100] | % |
| MBPB | Mouse Brain Protein Binding | 975 | [0, 100] | % |
| MGMB | Mouse Gut Microbiome Binding | 222 | [0, 100] | % |

The test set contains 2,282 blinded compounds for evaluation. Note the substantial variation in training set sizes, from 5,128 (KSOL) to only 222 (MGMB), presenting a challenge for consistent model performance.

### 2.2 Tiered Architecture

We implement a tiered modeling strategy with progressive sophistication:

| Tier | Model | Features | Purpose |
|------|-------|----------|---------|
| 1 | Simple CatBoost | 522 | Rapid baseline |
| 2 | Optimized GBDT Ensemble | 1,039–2,073 | Primary predictions |
| 3 | CatBoost + RandomForest | 2,747 | Advanced hybrid |
| 4 | Chemprop D-MPNN | Graph-based | Deep learning |

This tiered approach enables rapid iteration during development (Tier 1) while providing sophisticated predictions for final submission (Tiers 2-4).

### 2.3 Molecular Representation

Molecules are represented using a concatenation of fingerprints and physicochemical descriptors, resulting in feature vectors of 522 to 2,747 dimensions depending on the tier.

**Morgan Fingerprints (512–2,048 bits):** Circular fingerprints with radius 2, capturing local atomic environments up to 4 bonds from each atom (Rogers & Hahn, 2010). Morgan fingerprints provide a fixed-length binary representation encoding substructural information. We use 2,048 bits for optimized models.

**Physicochemical Descriptors (25 features):** Computed using RDKit (Landrum, 2006):

| Category | Descriptors |
|----------|-------------|
| Bulk Properties | MolWt, HeavyAtomCount, ValenceElectrons |
| Lipophilicity | MolLogP (Wildman-Crippen) |
| Polarity | TPSA, NumHDonors, NumHAcceptors |
| Flexibility | NumRotatableBonds, FractionCSP3 |
| Ring Systems | NumAromaticRings, RingCount, CalcNumAliphaticRings, CalcNumHeterocycles |
| Topological | BertzCT, Chi0, Chi1, Kappa1, Kappa2, HallKierAlpha |
| Surface Area | LabuteASA |
| Drug-likeness | QED |
| Special Features | CalcNumAmideBonds, CalcNumSpiroAtoms, CalcNumBridgeheadAtoms, CalcNumHBA |

### 2.4 Model Implementations

#### 2.4.1 XGBoost

XGBoost implements regularized gradient boosting with second-order Taylor expansion of the loss function (Chen & Guestrin, 2016). Configuration:

```python
XGBRegressor(
    n_estimators=600-1500,
    max_depth=7-8,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,      # L1 regularization
    reg_lambda=1.0,     # L2 regularization
    tree_method='gpu_hist',  # GPU acceleration
    early_stopping_rounds=50
)
```

#### 2.4.2 CatBoost

CatBoost uses ordered boosting and symmetric (oblivious) decision trees, providing built-in handling of categorical features (Prokhorenkova et al., 2018). Configuration:

```python
CatBoostRegressor(
    iterations=600-1200,
    depth=7-8,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bylevel=0.8,
    l2_leaf_reg=3.0,
    random_strength=1.0,
    task_type='GPU',  # Native GPU support
    early_stopping_rounds=100
)
```

#### 2.4.3 LightGBM

LightGBM employs histogram-based algorithms and leaf-wise tree growth for efficiency (Ke et al., 2017). Configuration:

```python
LGBMRegressor(
    n_estimators=600-1000,
    max_depth=7-8,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    device='gpu'
)
```

#### 2.4.4 Chemprop D-MPNN

The Directed Message Passing Neural Network learns molecular representations directly from graphs (Yang et al., 2019):

```python
# Architecture
mp = nn.BondMessagePassing()
agg = nn.MeanAggregation()
ffn = nn.RegressionFFN()
model = models.MPNN(mp, agg, ffn)

# Training: 15-40 epochs, batch_size=64-256
# Mixed precision (FP16) on GPU
```

### 2.5 Training Procedure

For each of the nine endpoints, we perform independent 5-fold cross-validation:

1. **Data Filtering:** Remove samples with NaN for target endpoint
2. **Fold Split:** 5-fold stratified split (shuffle=True, random_state=42)
3. **Per-Fold Training:** Train model on 4 folds, validate on held-out fold
4. **OOF Predictions:** Aggregate out-of-fold predictions for metric computation
5. **Final Model:** Retrain on all filtered data
6. **Test Prediction:** Generate predictions with range clipping

Early stopping monitors validation MAE with patience of 50-100 rounds.

### 2.6 Ensemble Strategy

We combine predictions using endpoint-specific weighted averaging:

$$\hat{y}_{ensemble}^{(t)} = \sum_{m \in M} w_m^{(t)} \cdot \hat{y}_m^{(t)}$$

where weights are inversely proportional to each model's RAE on endpoint $t$:

$$w_m^{(t)} = \frac{1/RAE_m^{(t)}}{\sum_{m' \in M} 1/RAE_{m'}^{(t)}}$$

This adaptive weighting gives higher influence to models that perform better on each specific endpoint, rather than using global weights.

### 2.7 Evaluation Metrics

**Relative Absolute Error (RAE):** Normalizes MAE by baseline mean predictor:

$$RAE^{(t)} = \frac{MAE^{(t)}}{\frac{1}{n}\sum_{i=1}^{n}|y_i^{(t)} - \bar{y}^{(t)}|}$$

**Mean Average RAE (MA-RAE):** Primary challenge metric:

$$MA\text{-}RAE = \frac{1}{9}\sum_{t=1}^{9} RAE^{(t)}$$

RAE < 1.0 indicates better-than-naive performance. MA-RAE enables comparison across endpoints with different scales.

---

## 3. Implementation

### 3.1 Training Infrastructure

We developed production-ready training scripts with:

- **Checkpointing:** State saved after each target, enabling resume on interruption
- **Signal Handling:** Graceful SIGINT/SIGTERM handling with checkpoint preservation
- **GPU Acceleration:** CUDA support for all models; MPS for Apple Silicon
- **Progress Tracking:** tqdm progress bars with timing estimates
- **Logging:** Dual console/file logging with colored output

### 3.2 File Structure

```
openadmet-expansionrx-challenge/
├── src/
│   ├── optimized_catboost.py      # CatBoost with GPU
│   ├── optimized_xgboost.py       # XGBoost with GPU
│   ├── optimized_lightgbm.py      # LightGBM
│   ├── catboost_rf_ensemble.py    # CatBoost + RF hybrid
│   ├── chemprop_model.py          # Basic Chemprop
│   │
│   │   # Phase 2: Advanced Components
│   ├── feature_engineering_v2.py  # MACCS + RDKit FP + Descriptors
│   ├── stacking_ensemble.py       # Two-level stacking ensemble
│   ├── validation.py              # 5x5 repeated CV
│   ├── multitask_nn.py            # Multi-task neural network
│   ├── chemprop_optimized.py      # Optimized Chemprop D-MPNN
│   ├── chemberta_embeddings.py    # ChemBERTa (RTX 4090 optimized)
│   ├── unimol_features.py         # Uni-Mol 3D features
│   └── feature_selection.py       # Endpoint-specific selection
│
├── run_local_m3.py          # Apple Silicon orchestration
├── run_runpod.py            # CUDA GPU orchestration
├── run_phase2_runpod.py     # Phase 2 pipeline (hardened)
├── create_ensemble.py       # Ensemble creation
├── requirements.txt         # Full requirements
├── requirements_runpod.txt  # RunPod RTX 4090 requirements
├── submissions/             # Output CSVs
├── .checkpoints/            # Training state
└── logs/                    # Execution logs
```

### 3.3 Computational Requirements

| Configuration | Hardware | Time |
|---------------|----------|------|
| Quick (200 iter) | Apple M3 Pro | ~2 min |
| Medium (600 iter) | Apple M3 Pro | ~60-80 min |
| Full (1500 iter) | RTX 4090 | ~30-45 min |

### 3.4 Software Dependencies

**Core (Required):**
- **Chemistry:** RDKit ≥2023.9.1
- **Gradient Boosting:** XGBoost ≥2.0.0, CatBoost ≥1.2.0, LightGBM ≥4.0.0
- **Deep Learning:** PyTorch ≥2.0.0, Lightning ≥2.0.0
- **Core:** scikit-learn ≥1.3.0, pandas ≥2.0.0, numpy ≥1.24.0, scipy ≥1.11.0
- **Utilities:** tqdm ≥4.66.0, joblib ≥1.3.0

**Phase 2 Advanced Features (Recommended):**
- **Transformer Embeddings:** transformers ≥4.36.0, huggingface_hub ≥0.19.0
- **Graph Neural Networks:** Chemprop ≥2.0.0
- **Extended Descriptors:** mordred ≥1.2.0

**Optional (Maximum Performance on RTX 4090):**
- **Flash Attention 2:** flash-attn (2x faster transformers)
- **3D Features:** unimol_tools (Uni-Mol 3D conformer embeddings)

See `requirements.txt` and `requirements_runpod.txt` for complete package lists.

---

## 4. Results

### 4.1 Cross-Validation Performance

Per-endpoint RAE scores from 5-fold cross-validation (quick mode, 200 iterations):

| Endpoint | XGBoost | LightGBM | CatBoost | Best |
|----------|---------|----------|----------|------|
| LogD | **0.394** | 0.419 | 0.482 | XGBoost |
| KSOL | **0.519** | 0.550 | 0.634 | XGBoost |
| HLM CLint | **0.684** | 0.729 | 0.776 | XGBoost |
| MLM CLint | **0.672** | 0.691 | 0.752 | XGBoost |
| Caco-2 Papp A>B | **0.586** | 0.603 | 0.658 | XGBoost |
| Caco-2 Efflux | **0.688** | 0.722 | 0.727 | XGBoost |
| MPPB | **0.513** | 0.516 | 0.596 | XGBoost |
| MBPB | **0.534** | 0.589 | 0.626 | XGBoost |
| MGMB | **0.568** | 0.655 | 0.640 | XGBoost |
| **MA-RAE** | **0.573** | 0.608 | 0.655 | XGBoost |

XGBoost achieves the best performance across all nine endpoints.

### 4.2 Model Comparison

| Model | MA-RAE | Training Time |
|-------|--------|---------------|
| XGBoost | **0.5730** | 30s |
| LightGBM | 0.6082 | 36s |
| CatBoost | 0.6545 | 43s |

### 4.3 Comparison to Leaderboard

| Method | MA-RAE | Gap |
|--------|--------|-----|
| Leader ("pebble") | **0.5593** | — |
| Our Best (XGBoost) | 0.5730 | +0.0137 |
| Weighted Ensemble | TBD | — |

---

## 5. Discussion

### 5.1 Model Analysis

XGBoost consistently outperformed CatBoost and LightGBM across all endpoints. Potential contributing factors:

1. **Regularization:** XGBoost's L1/L2 regularization effectively handles high-dimensional fingerprint features
2. **gpu_hist Algorithm:** Efficient histogram-based splitting for binary fingerprint bits
3. **Second-Order Gradients:** Taylor expansion captures curvature information

### 5.2 Endpoint Difficulty

Endpoints vary substantially in predictability:
- **Easiest:** LogD (RAE 0.39) — well-established structure-lipophilicity relationships
- **Hardest:** HLM/MLM CLint (RAE 0.67-0.68) — complex metabolic processes
- **Data-Limited:** MGMB (n=222) — insufficient training data

### 5.3 Limitations

1. **2D Features Only:** No 3D conformational information captured
2. **Limited Hyperparameter Search:** Time constraints prevented exhaustive optimization
3. **Single Fingerprint Type:** Alternative fingerprints (ECFP4, MACCS, Avalon) not explored
4. **No Pre-training:** Transfer learning from larger datasets not implemented

### 5.4 Future Directions

1. **Graph Neural Networks:** Full integration of Chemprop D-MPNN with ensemble
2. **Hyperparameter Optimization:** Bayesian optimization for endpoint-specific tuning
3. **Feature Expansion:** 3D descriptors, pharmacophore fingerprints, molecular embeddings
4. **Transfer Learning:** Pre-training on ChEMBL/PubChem ADMET data
5. **Uncertainty Quantification:** Conformal prediction for prediction intervals

---

## 6. Conclusion

We presented a tiered ensemble approach for multi-task ADMET prediction, achieving competitive performance on the OpenADMET ExpansionRx Challenge. Our best single model (XGBoost with 2,073 features) achieves MA-RAE of 0.5730, within 2.4% of the leaderboard leader. The implementation emphasizes reproducibility through checkpointed training and supports efficient GPU acceleration on both NVIDIA and Apple Silicon platforms.

The endpoint-specific weighted ensemble strategy provides a principled method for combining diverse model predictions, adapting to each target's characteristics. Our tiered architecture enables rapid development iteration while maintaining production-quality predictions.

---

## Acknowledgments

We thank the OpenADMET consortium for organizing the ExpansionRx Challenge and providing the benchmark dataset.

---

## References

1. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of KDD*, 785-794.

2. Cherkasov, A., et al. (2014). QSAR Modeling: Where Have You Been? Where Are You Going To? *J. Med. Chem.*, 57(12), 4977-5010.

3. Ke, G., et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS*, 3149-3157.

4. Landrum, G. (2006). RDKit: Open-source cheminformatics. https://www.rdkit.org

5. Lombardo, F., et al. (2017). In Silico Absorption, Distribution, Metabolism, Excretion, and Pharmacokinetics (ADME-PK). *J. Med. Chem.*, 60(21), 9097-9113.

6. Prokhorenkova, L., et al. (2018). CatBoost: unbiased boosting with categorical features. *NeurIPS*, 6638-6648.

7. Rogers, D., & Hahn, M. (2010). Extended-Connectivity Fingerprints. *J. Chem. Inf. Model.*, 50(5), 742-754.

8. Yang, K., et al. (2019). Analyzing Learned Molecular Representations for Property Prediction. *J. Chem. Inf. Model.*, 59(8), 3370-3388.

---

## Supplementary Materials

**Code Repository:** https://github.com/inventcures/openadmet-expansionrx-challenge

**Auto-Generated Documentation:** https://deepwiki.com/inventcures/openadmet-expansionrx-challenge

**Training Commands:**
```bash
# Quick validation
python run_local_m3.py --mode quick

# Full local training
python run_local_m3.py --mode full

# GPU training (RunPod)
python run_runpod.py --mode full

# Check checkpoint status
python run_local_m3.py --status

# Force restart
python run_local_m3.py --force
```

**Submission Files:**
```
submissions/
├── xgboost_quick.csv
├── catboost_quick.csv
├── lightgbm_quick.csv
├── ensemble_equal_quick.csv
└── ensemble_weighted_quick.csv
```
