"""
Phase 2A Stacking Ensemble
Two-level stacking with Ridge meta-learner
"""
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import gc
warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
import xgboost as xgb
from catboost import CatBoostRegressor
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except:
    LGB_AVAILABLE = False
from sklearn.ensemble import RandomForestRegressor


class StackingEnsemble:
    """
    Two-level stacking ensemble for ADMET prediction

    Level 1: Diverse base models (XGBoost, CatBoost, LightGBM, RF)
    Level 2: Ridge regression meta-learner on OOF predictions
    """

    def __init__(self,
                 n_folds=5,
                 use_xgb=True,
                 use_catboost=True,
                 use_lgb=True,
                 use_rf=True,
                 meta_model='ridge',
                 random_state=42,
                 verbose=True):
        self.n_folds = n_folds
        self.use_xgb = use_xgb
        self.use_catboost = use_catboost
        self.use_lgb = use_lgb and LGB_AVAILABLE
        self.use_rf = use_rf
        self.meta_model_type = meta_model
        self.random_state = random_state
        self.verbose = verbose

        self.base_models = {}
        self.meta_model = None
        self.oof_predictions = None

    def _get_base_models(self, hp=None):
        """Get base models with hyperparameters"""
        models = {}

        if hp is None:
            hp = {
                'max_depth': 7,
                'lr': 0.05,
                'n_estimators': 500,
            }

        if self.use_xgb:
            models['xgb'] = xgb.XGBRegressor(
                n_estimators=hp['n_estimators'],
                max_depth=hp['max_depth'],
                learning_rate=hp['lr'],
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=3.0,
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=0,
            )

        if self.use_catboost:
            models['catboost'] = CatBoostRegressor(
                iterations=hp['n_estimators'],
                depth=min(hp['max_depth'], 8),
                learning_rate=hp['lr'],
                subsample=0.8,
                random_seed=self.random_state,
                verbose=False,
            )

        if self.use_lgb:
            models['lgb'] = lgb.LGBMRegressor(
                n_estimators=hp['n_estimators'],
                max_depth=hp['max_depth'],
                learning_rate=hp['lr'],
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=3.0,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1,
            )

        if self.use_rf:
            models['rf'] = RandomForestRegressor(
                n_estimators=min(hp['n_estimators'], 300),
                max_depth=hp['max_depth'] + 2,
                min_samples_leaf=5,
                random_state=self.random_state,
                n_jobs=-1,
            )

        return models

    def fit(self, X, y, hp=None):
        """
        Fit stacking ensemble using OOF predictions

        Args:
            X: Features (n_samples, n_features)
            y: Target values (n_samples,)
            hp: Hyperparameters dict
        """
        n_samples = len(y)
        base_models = self._get_base_models(hp)
        n_models = len(base_models)

        if self.verbose:
            print(f"Fitting stacking ensemble with {n_models} base models...")

        # Generate OOF predictions for each base model
        oof_preds = np.zeros((n_samples, n_models))
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        # Store trained models for each fold
        self.base_models = {name: [] for name in base_models.keys()}

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            if self.verbose:
                print(f"  Fold {fold_idx + 1}/{self.n_folds}")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            for model_idx, (name, model) in enumerate(base_models.items()):
                # Clone model for this fold
                if name == 'xgb':
                    fold_model = xgb.XGBRegressor(**model.get_params())
                    fold_model.fit(X_train, y_train,
                                  eval_set=[(X_val, y_val)],
                                  verbose=False)
                elif name == 'catboost':
                    # CatBoost requires get_params() not get_all_params()
                    params = model.get_params()
                    fold_model = CatBoostRegressor(**params)
                    fold_model.fit(X_train, y_train,
                                  eval_set=(X_val, y_val),
                                  early_stopping_rounds=50,
                                  verbose=False)
                elif name == 'lgb':
                    fold_model = lgb.LGBMRegressor(**model.get_params())
                    fold_model.fit(X_train, y_train,
                                  eval_set=[(X_val, y_val)],
                                  callbacks=[lgb.early_stopping(50, verbose=False)])
                elif name == 'rf':
                    fold_model = RandomForestRegressor(**model.get_params())
                    fold_model.fit(X_train, y_train)
                else:
                    fold_model = model.__class__(**model.get_params())
                    fold_model.fit(X_train, y_train)

                # Store model and get OOF predictions
                self.base_models[name].append(fold_model)
                oof_preds[val_idx, model_idx] = fold_model.predict(X_val)

            gc.collect()

        self.oof_predictions = oof_preds

        # Fit meta-learner on OOF predictions
        if self.verbose:
            print("  Fitting meta-learner...")

        if self.meta_model_type == 'ridge':
            self.meta_model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        else:
            self.meta_model = ElasticNetCV(alphas=[0.01, 0.1, 1.0, 10.0],
                                           l1_ratio=[0.1, 0.5, 0.9])

        self.meta_model.fit(oof_preds, y)

        # Report OOF performance
        oof_final = self.meta_model.predict(oof_preds)
        oof_mae = mean_absolute_error(y, oof_final)
        oof_rae = oof_mae / np.mean(np.abs(y - np.mean(y)))
        oof_spear = spearmanr(y, oof_final)[0]

        if self.verbose:
            print(f"  OOF MAE: {oof_mae:.4f}, RAE: {oof_rae:.4f}, Spearman: {oof_spear:.4f}")

        return {'MAE': oof_mae, 'RAE': oof_rae, 'Spearman': oof_spear}

    def predict(self, X):
        """
        Make predictions using the stacking ensemble

        Args:
            X: Features (n_samples, n_features)

        Returns:
            predictions: (n_samples,)
        """
        n_samples = len(X)
        n_models = len(self.base_models)
        base_preds = np.zeros((n_samples, n_models))

        # Get predictions from all base models (average across folds)
        for model_idx, (name, fold_models) in enumerate(self.base_models.items()):
            fold_preds = np.zeros((n_samples, len(fold_models)))
            for fold_idx, model in enumerate(fold_models):
                fold_preds[:, fold_idx] = model.predict(X)
            base_preds[:, model_idx] = fold_preds.mean(axis=1)

        # Meta-learner prediction
        return self.meta_model.predict(base_preds)

    def get_model_weights(self):
        """Get the weights assigned by the meta-learner to each base model"""
        if self.meta_model is None:
            return None

        model_names = list(self.base_models.keys())
        weights = self.meta_model.coef_

        return dict(zip(model_names, weights))


class EndpointStackingEnsemble:
    """
    Endpoint-specific stacking ensemble

    Trains separate stacking ensembles for each endpoint
    """

    def __init__(self, n_folds=5, verbose=True, **kwargs):
        self.n_folds = n_folds
        self.verbose = verbose
        self.kwargs = kwargs
        self.ensembles = {}
        self.results = {}

    def fit(self, X, y_dict, hp_dict=None):
        """
        Fit stacking ensemble for each endpoint

        Args:
            X: Features (n_samples, n_features)
            y_dict: Dict of {endpoint: target_values}
            hp_dict: Dict of {endpoint: hyperparameters}
        """
        for endpoint, y in y_dict.items():
            if self.verbose:
                print(f"\n{'='*50}")
                print(f"Endpoint: {endpoint}")
                print(f"{'='*50}")

            # Filter valid samples
            mask = ~np.isnan(y)
            X_ep = X[mask]
            y_ep = y[mask]

            if self.verbose:
                print(f"  Samples: {len(y_ep)}")

            # Get endpoint-specific hyperparameters
            hp = hp_dict.get(endpoint) if hp_dict else None

            # Create and fit stacking ensemble
            ensemble = StackingEnsemble(
                n_folds=self.n_folds,
                verbose=self.verbose,
                **self.kwargs
            )

            result = ensemble.fit(X_ep, y_ep, hp=hp)
            self.ensembles[endpoint] = ensemble
            self.results[endpoint] = result

            gc.collect()

        return self.results

    def predict(self, X, endpoint):
        """Make predictions for a specific endpoint"""
        return self.ensembles[endpoint].predict(X)

    def predict_all(self, X):
        """Make predictions for all endpoints"""
        return {ep: self.predict(X, ep) for ep in self.ensembles.keys()}

    def summary(self):
        """Print summary of results"""
        print("\n" + "=" * 60)
        print("STACKING ENSEMBLE SUMMARY")
        print("=" * 60)

        raes = []
        for ep, res in self.results.items():
            print(f"{ep}: RAE={res['RAE']:.4f}, Spearman={res['Spearman']:.4f}")
            raes.append(res['RAE'])

        print(f"\nMA-RAE: {np.mean(raes):.4f}")
        return np.mean(raes)


if __name__ == "__main__":
    # Quick test
    from sklearn.datasets import make_regression

    print("Testing Stacking Ensemble")
    print("=" * 50)

    X, y = make_regression(n_samples=500, n_features=100, noise=10, random_state=42)

    ensemble = StackingEnsemble(n_folds=3, verbose=True)
    result = ensemble.fit(X, y)

    print(f"\nModel weights: {ensemble.get_model_weights()}")

    preds = ensemble.predict(X[:10])
    print(f"Sample predictions: {preds[:5]}")
