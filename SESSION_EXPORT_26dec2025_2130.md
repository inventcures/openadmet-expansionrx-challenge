# ADMET Challenge Session Export - Dec 26, 2025 @ 21:30 IST

## RESUME INSTRUCTIONS
Copy this to start new session:
```
Resume ADMET challenge work from @SESSION_EXPORT_26dec2025_2130.md
```

---

## CHALLENGE OVERVIEW

**Competition:** OpenADMET ExpansionRx Blind Challenge
- **Deadline:** January 19, 2026
- **Leader:** "pebble" with MA-RAE 0.5593 (136 submissions)
- **Metric:** Macro-averaged Relative Absolute Error (MA-RAE) - LOWER is better
- **HuggingFace:** https://huggingface.co/spaces/openadmet/OpenADMET-ExpansionRx-Challenge

**9 Endpoints to Predict:**
1. LogD - Lipophilicity
2. KSOL - Kinetic Solubility
3. HLM CLint - Human Liver Microsomal clearance
4. MLM CLint - Mouse Liver Microsomal clearance
5. Caco-2 Permeability Papp A>B
6. Caco-2 Permeability Efflux
7. MPPB - Mouse Plasma Protein Binding
8. MBPB - Mouse Brain Protein Binding
9. MGMB - Mouse Gastrocnemius Muscle Binding

---

## DATA SUMMARY

**Training Data:** `/data/raw/train.csv`
- 5326 molecules with SMILES
- Varying samples per endpoint (222-5128)

**Test Data:** `/data/raw/test_blinded.csv`
- 2282 molecules to predict

**Sample Counts per Endpoint:**
| Endpoint | Samples | Notes |
|----------|---------|-------|
| LogD | 5039 | Best data |
| KSOL | 5128 | Best data |
| HLM CLint | 3759 | |
| MLM CLint | 4522 | |
| Caco-2 Papp | 2157 | |
| Caco-2 Efflux | 2161 | |
| MPPB | 1302 | Limited |
| MBPB | 975 | Limited |
| MGMB | 222 | Very limited! |

---

## PROJECT STRUCTURE

```
/Users/tp53/Documents/code_macbook-air-m1__tp53/26dec2025_opeadmet-challenge/
├── admet_challenge/          # Python virtual environment
├── data/raw/                 # Training and test CSVs
├── models/                   # Saved model files (XGBoost .joblib)
├── submissions/              # CSV files for leaderboard
├── src/                      # Python scripts
│   ├── baseline_model.py     # Original XGBoost baseline
│   ├── simple_catboost.py    # Simple CatBoost (MA-RAE: 0.5710)
│   ├── optimized_catboost.py # Tuned CatBoost (MA-RAE: 0.5656) ← BEST
│   ├── chemprop_model.py     # D-MPNN model (was running)
│   ├── ensemble_model.py     # Multi-config ensemble (OOM failed)
│   └── multitask_model.py    # MTL model (stuck/killed)
├── docs/                     # Reference PDFs
├── ADMET_CHALLENGE_STRATEGY.md  # Comprehensive strategy doc
└── requirements.txt
```

---

## CURRENT SUBMISSIONS & RESULTS

| File | MA-RAE | Status |
|------|--------|--------|
| **optimized_catboost.csv** | **0.5656** | ✅ Best single model |
| simple_catboost.csv | 0.5710 | ✅ Complete |
| baseline_xgb_clipped.csv | ~0.58 | ✅ Complete |
| ensemble_v1.csv | TBD | ✅ Created (weighted avg) |
| ensemble_equal.csv | TBD | ✅ Created (equal avg) |
| chemprop_submission.csv | TBD | ⏳ Was running |

**Target to Beat:** 0.5593 (leader "pebble")

---

## OPTIMIZED CATBOOST RESULTS (BEST MODEL)

Cross-validation results from `optimized_catboost.py`:

| Endpoint | RAE | Notes |
|----------|-----|-------|
| LogD | 0.3461 | Excellent |
| KSOL | 0.5216 | Good |
| HLM CLint | 0.6843 | Needs improvement |
| MLM CLint | 0.6749 | Needs improvement |
| Caco-2 Papp | 0.5814 | Moderate |
| Caco-2 Efflux | 0.6877 | Needs improvement |
| MPPB | 0.4976 | Good |
| MBPB | 0.5209 | Good |
| MGMB | 0.5758 | Limited data |
| **MA-RAE** | **0.5656** | |

**Hyperparameters used (endpoint-specific):**
```python
HYPERPARAMS = {
    'LogD': {'depth': 8, 'lr': 0.03, 'iters': 1000},
    'KSOL': {'depth': 8, 'lr': 0.03, 'iters': 800},
    'HLM CLint': {'depth': 7, 'lr': 0.05, 'iters': 600},
    'MLM CLint': {'depth': 7, 'lr': 0.05, 'iters': 600},
    'Caco-2 Papp': {'depth': 7, 'lr': 0.04, 'iters': 700},
    'Caco-2 Efflux': {'depth': 6, 'lr': 0.05, 'iters': 500},
    'MPPB': {'depth': 8, 'lr': 0.03, 'iters': 800},
    'MBPB': {'depth': 8, 'lr': 0.03, 'iters': 800},
    'MGMB': {'depth': 6, 'lr': 0.05, 'iters': 400},
}
```

**Features used:** Morgan FP (1024 bits) + 15 RDKit descriptors = 1039 features

---

## ENVIRONMENT SETUP

**Virtual Environment:**
```bash
source /Users/tp53/Documents/code_macbook-air-m1__tp53/26dec2025_opeadmet-challenge/admet_challenge/bin/activate
```

**Working Packages:**
- pandas, numpy, sklearn, scipy ✅
- catboost ✅
- rdkit ✅
- chemprop ✅ (newly installed)
- joblib ✅

**Broken Packages (need libomp):**
- xgboost ❌ (libomp issue)
- lightgbm ❌ (libomp issue)

**To fix XGBoost/LightGBM:**
```bash
# libomp was installed via brew
export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"
uv pip uninstall xgboost lightgbm
uv pip install xgboost lightgbm
```

---

## WHAT FAILED & WHY

1. **Multi-config CatBoost ensemble** - Exit code 137 (OOM/killed)
   - Too many models in memory simultaneously

2. **CatBoost + XGBoost ensemble** - libomp missing for XGBoost

3. **CatBoost + LightGBM ensemble** - libomp missing for LightGBM

4. **Multi-task learning model** - Got stuck, killed

5. **Heavy feature engineering** - 2747 features caused OOM

---

## NEXT STEPS TO IMPROVE

### Priority 1: Fix XGBoost
```bash
export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"
source admet_challenge/bin/activate
uv pip install --force-reinstall xgboost
```

### Priority 2: Complete Chemprop
Run `src/chemprop_model.py` - D-MPNN is state-of-the-art for molecular properties

### Priority 3: Better Ensemble
Combine predictions from:
- optimized_catboost.csv (weight: 0.4)
- XGBoost (if fixed) (weight: 0.3)
- Chemprop (weight: 0.3)

### Priority 4: Target Weak Endpoints
Focus on improving:
- HLM CLint (RAE: 0.6843)
- MLM CLint (RAE: 0.6749)
- Caco-2 Efflux (RAE: 0.6877)

### Ideas to Try:
1. **Multi-task learning** for correlated endpoints:
   - HLM + MLM (metabolic stability)
   - MPPB + MBPB + MGMB (protein binding)

2. **External data** from ChEMBL (permitted by challenge rules)

3. **Avalon + MACCS fingerprints** (partially implemented in `catboost_rf_ensemble.py`)

4. **Conformer-based 3D features** (Uni-Mol style)

---

## KEY FILES TO READ

1. **Strategy:** `ADMET_CHALLENGE_STRATEGY.md` - Comprehensive winning strategy
2. **Best model:** `src/optimized_catboost.py` - Current best (0.5656)
3. **Chemprop:** `src/chemprop_model.py` - D-MPNN implementation
4. **Reference PDF:** `docs/machine-learning-adme-models-in-practice-four-guidelines-from-a-successful-lead-optimization-case-study.pdf`

---

## QUICK START COMMANDS

```bash
cd /Users/tp53/Documents/code_macbook-air-m1__tp53/26dec2025_opeadmet-challenge
source admet_challenge/bin/activate

# Run best model
python3 src/optimized_catboost.py

# Run Chemprop
python3 src/chemprop_model.py

# Check submissions
ls -la submissions/
```

---

## VALID PREDICTION RANGES (for clipping)

```python
VALID_RANGES = {
    'LogD': (-3.0, 6.0),
    'KSOL': (0.001, 350.0),
    'HLM CLint': (0.0, 3000.0),
    'MLM CLint': (0.0, 12000.0),
    'Caco-2 Permeability Papp A>B': (0.0, 60.0),
    'Caco-2 Permeability Efflux': (0.2, 120.0),
    'MPPB': (0.0, 100.0),
    'MBPB': (0.0, 100.0),
    'MGMB': (0.0, 100.0)
}
```

---

## SESSION HISTORY

1. ✅ Downloaded challenge data from HuggingFace
2. ✅ Built XGBoost baseline (~0.58 MA-RAE)
3. ✅ Built simple CatBoost (0.5710 MA-RAE)
4. ✅ Built optimized CatBoost (0.5656 MA-RAE) ← CURRENT BEST
5. ✅ Created ensemble submissions
6. ⏳ Chemprop was training when session ended
7. ❌ Multi-config ensemble failed (OOM)
8. ❌ XGBoost/LightGBM broken (libomp)

---

**Gap to leader: 0.5656 - 0.5593 = 0.0063**

Need ~1% relative improvement to reach #1!
