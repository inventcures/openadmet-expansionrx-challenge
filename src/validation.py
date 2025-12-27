"""
Phase 2A Validation Module
5x5 Repeated Cross-Validation with statistical testing
Based on ChemRxiv paper recommendations
"""
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import RepeatedKFold, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr, pearsonr
from scipy import stats


def compute_rae(y_true, y_pred):
    """Compute Relative Absolute Error"""
    mae = np.mean(np.abs(y_true - y_pred))
    baseline_mae = np.mean(np.abs(y_true - np.mean(y_true)))
    return mae / baseline_mae if baseline_mae > 0 else 1.0


def compute_metrics(y_true, y_pred):
    """Compute comprehensive metrics for ADMET prediction"""
    metrics = {}

    # Core metrics
    metrics['MAE'] = mean_absolute_error(y_true, y_pred)
    metrics['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics['R2'] = r2_score(y_true, y_pred)
    metrics['RAE'] = compute_rae(y_true, y_pred)

    # Correlation metrics
    metrics['Spearman'] = spearmanr(y_true, y_pred)[0]
    metrics['Pearson'] = pearsonr(y_true, y_pred)[0]

    return metrics


class RepeatedCVValidator:
    """
    5x5 Repeated Cross-Validation with statistical analysis

    Implements recommendations from ChemRxiv paper:
    - Multiple repetitions to assess variance
    - Statistical testing between methods
    - Practical significance via Cohen's D
    """

    def __init__(self, n_splits=5, n_repeats=5, random_state=42, verbose=True):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.verbose = verbose
        self.cv_results = {}

    def validate(self, model_fn, X, y, model_name='model'):
        """
        Run 5x5 repeated CV for a model

        Args:
            model_fn: Function that returns a fitted model: model_fn(X_train, y_train) -> model
            X: Features
            y: Target values
            model_name: Name for logging

        Returns:
            dict with mean, std, CI for each metric
        """
        rkf = RepeatedKFold(
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            random_state=self.random_state
        )

        all_metrics = {
            'MAE': [], 'RMSE': [], 'R2': [],
            'RAE': [], 'Spearman': [], 'Pearson': []
        }

        total_folds = self.n_splits * self.n_repeats

        for fold_idx, (train_idx, val_idx) in enumerate(rkf.split(X)):
            if self.verbose and (fold_idx + 1) % 5 == 0:
                print(f"  {model_name}: Fold {fold_idx + 1}/{total_folds}")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Fit model
            model = model_fn(X_train, y_train)

            # Predict and compute metrics
            preds = model.predict(X_val)
            fold_metrics = compute_metrics(y_val, preds)

            for metric, value in fold_metrics.items():
                all_metrics[metric].append(value)

        # Compute summary statistics
        results = {}
        for metric, values in all_metrics.items():
            values = np.array(values)
            results[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'ci_lower': np.percentile(values, 2.5),
                'ci_upper': np.percentile(values, 97.5),
                'values': values
            }

        self.cv_results[model_name] = results

        if self.verbose:
            print(f"  {model_name} Summary:")
            print(f"    RAE: {results['RAE']['mean']:.4f} +/- {results['RAE']['std']:.4f}")
            print(f"    Spearman: {results['Spearman']['mean']:.4f} +/- {results['Spearman']['std']:.4f}")

        return results

    def compare_models(self, model_a, model_b):
        """
        Statistical comparison between two models

        Implements:
        - Paired t-test
        - Wilcoxon signed-rank test
        - Cohen's D effect size
        """
        if model_a not in self.cv_results or model_b not in self.cv_results:
            raise ValueError("Both models must be validated first")

        results = {}

        for metric in ['RAE', 'MAE', 'Spearman']:
            values_a = self.cv_results[model_a][metric]['values']
            values_b = self.cv_results[model_b][metric]['values']

            # Paired t-test
            t_stat, p_value = stats.ttest_rel(values_a, values_b)

            # Wilcoxon signed-rank test (non-parametric)
            try:
                w_stat, w_pvalue = stats.wilcoxon(values_a, values_b)
            except:
                w_stat, w_pvalue = np.nan, np.nan

            # Cohen's D effect size
            pooled_std = np.sqrt((np.var(values_a) + np.var(values_b)) / 2)
            cohens_d = (np.mean(values_a) - np.mean(values_b)) / pooled_std if pooled_std > 0 else 0

            results[metric] = {
                'mean_a': np.mean(values_a),
                'mean_b': np.mean(values_b),
                'diff': np.mean(values_a) - np.mean(values_b),
                't_statistic': t_stat,
                'p_value': p_value,
                'wilcoxon_p': w_pvalue,
                'cohens_d': cohens_d,
                'effect_size': self._interpret_cohens_d(cohens_d)
            }

        return results

    def _interpret_cohens_d(self, d):
        """Interpret Cohen's D effect size"""
        d = abs(d)
        if d < 0.2:
            return 'negligible'
        elif d < 0.5:
            return 'small'
        elif d < 0.8:
            return 'medium'
        else:
            return 'large'

    def summary_table(self):
        """Generate summary table of all validated models"""
        rows = []
        for model_name, results in self.cv_results.items():
            row = {
                'Model': model_name,
                'RAE': f"{results['RAE']['mean']:.4f} +/- {results['RAE']['std']:.4f}",
                'Spearman': f"{results['Spearman']['mean']:.4f} +/- {results['Spearman']['std']:.4f}",
                'R2': f"{results['R2']['mean']:.4f} +/- {results['R2']['std']:.4f}",
            }
            rows.append(row)

        return pd.DataFrame(rows)


class QuickValidator:
    """
    Faster single-round CV for quick iterations
    Use RepeatedCVValidator for final model selection
    """

    def __init__(self, n_splits=5, random_state=42, verbose=True):
        self.n_splits = n_splits
        self.random_state = random_state
        self.verbose = verbose

    def validate(self, model_fn, X, y):
        """Quick 5-fold CV validation"""
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        oof_preds = np.zeros(len(y))

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = model_fn(X_train, y_train)
            oof_preds[val_idx] = model.predict(X_val)

        metrics = compute_metrics(y, oof_preds)

        if self.verbose:
            print(f"RAE: {metrics['RAE']:.4f}, Spearman: {metrics['Spearman']:.4f}")

        return metrics, oof_preds


class NullModelBaseline:
    """
    Null model baselines for performance bounds

    From ChemRxiv paper:
    - Lower bound: null models (mean/median predictor)
    """

    @staticmethod
    def mean_predictor(y_train, y_test):
        """Mean predictor baseline"""
        pred = np.full(len(y_test), np.mean(y_train))
        return compute_metrics(y_test, pred)

    @staticmethod
    def median_predictor(y_train, y_test):
        """Median predictor baseline"""
        pred = np.full(len(y_test), np.median(y_train))
        return compute_metrics(y_test, pred)

    @staticmethod
    def random_predictor(y_train, y_test, random_state=42):
        """Random predictor baseline (sample from training distribution)"""
        np.random.seed(random_state)
        pred = np.random.choice(y_train, size=len(y_test), replace=True)
        return compute_metrics(y_test, pred)


def validate_with_baselines(model_fn, X, y, n_splits=5, verbose=True):
    """
    Validate model against null baselines

    Returns performance relative to baselines
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    model_raes = []
    mean_raes = []
    median_raes = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Model performance
        model = model_fn(X_train, y_train)
        preds = model.predict(X_val)
        model_raes.append(compute_rae(y_val, preds))

        # Baseline performance
        mean_raes.append(NullModelBaseline.mean_predictor(y_train, y_val)['RAE'])
        median_raes.append(NullModelBaseline.median_predictor(y_train, y_val)['RAE'])

    results = {
        'model_rae': np.mean(model_raes),
        'mean_baseline_rae': np.mean(mean_raes),
        'median_baseline_rae': np.mean(median_raes),
        'improvement_vs_mean': 1 - np.mean(model_raes) / np.mean(mean_raes),
        'improvement_vs_median': 1 - np.mean(model_raes) / np.mean(median_raes),
    }

    if verbose:
        print(f"Model RAE: {results['model_rae']:.4f}")
        print(f"Mean baseline RAE: {results['mean_baseline_rae']:.4f}")
        print(f"Improvement: {results['improvement_vs_mean']*100:.1f}%")

    return results


if __name__ == "__main__":
    # Test validation module
    from sklearn.datasets import make_regression
    from sklearn.linear_model import Ridge
    import xgboost as xgb

    print("Testing Validation Module")
    print("=" * 50)

    X, y = make_regression(n_samples=500, n_features=50, noise=10, random_state=42)

    # Define model functions
    def ridge_fn(X_train, y_train):
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        return model

    def xgb_fn(X_train, y_train):
        model = xgb.XGBRegressor(n_estimators=100, max_depth=5, verbosity=0)
        model.fit(X_train, y_train)
        return model

    # Test 5x5 repeated CV
    print("\n5x5 Repeated CV:")
    validator = RepeatedCVValidator(n_splits=5, n_repeats=3, verbose=True)  # 3 repeats for quick test
    validator.validate(ridge_fn, X, y, model_name='Ridge')
    validator.validate(xgb_fn, X, y, model_name='XGBoost')

    # Compare models
    print("\nModel Comparison:")
    comparison = validator.compare_models('Ridge', 'XGBoost')
    for metric, stats in comparison.items():
        print(f"  {metric}: diff={stats['diff']:.4f}, p={stats['p_value']:.4f}, effect={stats['effect_size']}")

    # Summary table
    print("\nSummary Table:")
    print(validator.summary_table())

    # Test baselines
    print("\nBaseline Comparison:")
    validate_with_baselines(xgb_fn, X, y)
