"""
Hill Climbing Ensemble Selection for V4 Pipeline.

Greedy forward selection with weight optimization:
1. Start with best single model
2. Add model that maximizes ensemble metric
3. Optimize weights for selected models
4. Repeat until no improvement

This is a proven Kaggle Grandmaster technique for ensemble optimization.
"""

import numpy as np
from typing import List, Tuple, Optional, Callable, Dict
from scipy.optimize import minimize
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings


def spearman_metric(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Spearman correlation (higher is better)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr, _ = spearmanr(y_pred, y_true)
    return corr if not np.isnan(corr) else 0.0


def pearson_metric(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Pearson correlation (higher is better)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr, _ = pearsonr(y_pred, y_true)
    return corr if not np.isnan(corr) else 0.0


def neg_rmse_metric(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Negative RMSE (higher is better)."""
    return -np.sqrt(mean_squared_error(y_true, y_pred))


def neg_mae_metric(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Negative MAE (higher is better)."""
    return -mean_absolute_error(y_true, y_pred)


METRICS = {
    'spearman': spearman_metric,
    'pearson': pearson_metric,
    'neg_rmse': neg_rmse_metric,
    'neg_mae': neg_mae_metric,
}


def optimize_weights(
    predictions: np.ndarray,
    y_true: np.ndarray,
    metric_fn: Callable,
    init_weights: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, float]:
    """
    Optimize ensemble weights using scipy.minimize.

    Args:
        predictions: Array of shape (n_samples, n_models)
        y_true: Target values
        metric_fn: Metric function (higher is better)
        init_weights: Initial weights (uniform if None)

    Returns:
        Tuple of (optimized_weights, best_score)
    """
    n_models = predictions.shape[1]

    if init_weights is None:
        init_weights = np.ones(n_models) / n_models

    def objective(weights):
        # Normalize weights to sum to 1
        w = np.abs(weights) / np.sum(np.abs(weights))
        ensemble_pred = predictions @ w
        return -metric_fn(ensemble_pred, y_true)  # Negate for minimization

    result = minimize(
        objective,
        x0=init_weights,
        method='SLSQP',
        bounds=[(0, 1)] * n_models,
        constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    )

    optimized_weights = np.abs(result.x) / np.sum(np.abs(result.x))
    best_score = -result.fun

    return optimized_weights, best_score


def hill_climbing_ensemble(
    oof_predictions: np.ndarray,
    y_true: np.ndarray,
    model_names: Optional[List[str]] = None,
    metric: str = 'spearman',
    max_models: int = 15,
    improvement_threshold: float = 0.0001,
    verbose: bool = True
) -> Dict:
    """
    Greedy forward selection with weight optimization.

    Algorithm:
    1. Start with best single model
    2. For each remaining model:
       - Try adding it to ensemble
       - Optimize weights
       - Keep if improves score
    3. Stop when no improvement or max_models reached

    Args:
        oof_predictions: OOF predictions, shape (n_samples, n_models)
        y_true: True target values
        model_names: Optional list of model names
        metric: Metric to optimize ('spearman', 'pearson', 'neg_rmse', 'neg_mae')
        max_models: Maximum models to select
        improvement_threshold: Minimum improvement to continue adding
        verbose: Print progress

    Returns:
        Dictionary with:
        - 'selected_indices': List of selected model indices
        - 'selected_names': List of selected model names
        - 'weights': Optimized weights for selected models
        - 'score': Final ensemble score
        - 'history': List of (model_added, score) tuples
    """
    n_models = oof_predictions.shape[1]

    if model_names is None:
        model_names = [f"model_{i}" for i in range(n_models)]

    metric_fn = METRICS[metric]

    # Find best single model
    single_scores = [metric_fn(oof_predictions[:, i], y_true) for i in range(n_models)]
    best_idx = int(np.argmax(single_scores))
    best_score = single_scores[best_idx]

    selected = [best_idx]
    weights = np.array([1.0])
    history = [(model_names[best_idx], best_score)]

    if verbose:
        print(f"Starting with {model_names[best_idx]}: {best_score:.6f}")

    # Greedy addition
    while len(selected) < max_models:
        best_improvement = 0
        best_add_idx = None
        best_new_weights = None
        best_new_score = best_score

        for idx in range(n_models):
            if idx in selected:
                continue

            # Try adding this model
            trial_indices = selected + [idx]
            trial_preds = oof_predictions[:, trial_indices]

            # Optimize weights
            trial_weights, trial_score = optimize_weights(
                trial_preds, y_true, metric_fn
            )

            improvement = trial_score - best_score

            if improvement > best_improvement:
                best_improvement = improvement
                best_add_idx = idx
                best_new_weights = trial_weights
                best_new_score = trial_score

        # Check if improvement is significant
        if best_improvement < improvement_threshold:
            if verbose:
                print(f"\nStopping: best improvement ({best_improvement:.6f}) below threshold")
            break

        # Add the best model
        selected.append(best_add_idx)
        weights = best_new_weights
        best_score = best_new_score
        history.append((model_names[best_add_idx], best_score))

        if verbose:
            print(f"Added {model_names[best_add_idx]}: {best_score:.6f} (+{best_improvement:.6f})")

    selected_names = [model_names[i] for i in selected]

    return {
        'selected_indices': selected,
        'selected_names': selected_names,
        'weights': weights,
        'score': best_score,
        'history': history
    }


def diversity_hill_climbing(
    oof_predictions: np.ndarray,
    y_true: np.ndarray,
    model_names: Optional[List[str]] = None,
    metric: str = 'spearman',
    max_models: int = 15,
    diversity_bonus: float = 0.1,
    improvement_threshold: float = 0.0001,
    verbose: bool = True
) -> Dict:
    """
    Hill climbing with diversity bonus.

    Modified scoring: score + diversity_bonus * (1 - max_correlation_with_selected)

    This encourages selection of models that make different errors,
    which often leads to better generalization.

    Args:
        Same as hill_climbing_ensemble, plus:
        diversity_bonus: Weight for diversity term (0-1)

    Returns:
        Same as hill_climbing_ensemble
    """
    n_models = oof_predictions.shape[1]

    if model_names is None:
        model_names = [f"model_{i}" for i in range(n_models)]

    metric_fn = METRICS[metric]

    # Precompute correlation matrix
    corr_matrix = np.corrcoef(oof_predictions.T)

    # Find best single model
    single_scores = [metric_fn(oof_predictions[:, i], y_true) for i in range(n_models)]
    best_idx = int(np.argmax(single_scores))
    best_score = single_scores[best_idx]

    selected = [best_idx]
    weights = np.array([1.0])
    history = [(model_names[best_idx], best_score)]

    if verbose:
        print(f"Starting with {model_names[best_idx]}: {best_score:.6f}")

    while len(selected) < max_models:
        best_adjusted_score = 0
        best_add_idx = None
        best_new_weights = None
        best_raw_score = best_score

        for idx in range(n_models):
            if idx in selected:
                continue

            # Calculate diversity bonus
            max_corr = max(abs(corr_matrix[idx, s]) for s in selected)
            diversity = 1 - max_corr

            # Try adding this model
            trial_indices = selected + [idx]
            trial_preds = oof_predictions[:, trial_indices]

            trial_weights, trial_score = optimize_weights(
                trial_preds, y_true, metric_fn
            )

            improvement = trial_score - best_score

            # Adjusted score includes diversity bonus
            adjusted_improvement = improvement + diversity_bonus * diversity

            if adjusted_improvement > best_adjusted_score:
                best_adjusted_score = adjusted_improvement
                best_add_idx = idx
                best_new_weights = trial_weights
                best_raw_score = trial_score

        # Use raw improvement for stopping criterion
        raw_improvement = best_raw_score - best_score

        if raw_improvement < improvement_threshold and best_adjusted_score < improvement_threshold:
            if verbose:
                print(f"\nStopping: no significant improvement")
            break

        selected.append(best_add_idx)
        weights = best_new_weights
        best_score = best_raw_score
        history.append((model_names[best_add_idx], best_score))

        diversity = 1 - max(abs(corr_matrix[best_add_idx, s]) for s in selected[:-1])
        if verbose:
            print(f"Added {model_names[best_add_idx]}: {best_score:.6f} "
                  f"(+{raw_improvement:.6f}, div={diversity:.3f})")

    return {
        'selected_indices': selected,
        'selected_names': [model_names[i] for i in selected],
        'weights': weights,
        'score': best_score,
        'history': history
    }


class HillClimbingEnsemble:
    """
    Sklearn-like wrapper for hill climbing ensemble.

    Usage:
        ensemble = HillClimbingEnsemble(metric='spearman', max_models=10)
        ensemble.fit(oof_predictions, y_true, model_names)
        test_predictions = ensemble.predict(test_predictions_matrix)
    """

    def __init__(
        self,
        metric: str = 'spearman',
        max_models: int = 15,
        diversity_bonus: float = 0.0,
        improvement_threshold: float = 0.0001,
        verbose: bool = True
    ):
        self.metric = metric
        self.max_models = max_models
        self.diversity_bonus = diversity_bonus
        self.improvement_threshold = improvement_threshold
        self.verbose = verbose

        self.selected_indices_ = None
        self.selected_names_ = None
        self.weights_ = None
        self.score_ = None
        self.history_ = None

    def fit(
        self,
        oof_predictions: np.ndarray,
        y_true: np.ndarray,
        model_names: Optional[List[str]] = None
    ) -> 'HillClimbingEnsemble':
        """Fit the ensemble using hill climbing selection."""

        if self.diversity_bonus > 0:
            result = diversity_hill_climbing(
                oof_predictions, y_true, model_names,
                metric=self.metric,
                max_models=self.max_models,
                diversity_bonus=self.diversity_bonus,
                improvement_threshold=self.improvement_threshold,
                verbose=self.verbose
            )
        else:
            result = hill_climbing_ensemble(
                oof_predictions, y_true, model_names,
                metric=self.metric,
                max_models=self.max_models,
                improvement_threshold=self.improvement_threshold,
                verbose=self.verbose
            )

        self.selected_indices_ = result['selected_indices']
        self.selected_names_ = result['selected_names']
        self.weights_ = result['weights']
        self.score_ = result['score']
        self.history_ = result['history']

        return self

    def predict(self, predictions: np.ndarray) -> np.ndarray:
        """
        Predict using the selected ensemble.

        Args:
            predictions: Array of shape (n_samples, n_models)
                         Models must be in same order as training

        Returns:
            Weighted ensemble predictions
        """
        if self.selected_indices_ is None:
            raise ValueError("Ensemble not fitted. Call fit() first.")

        selected_preds = predictions[:, self.selected_indices_]
        return selected_preds @ self.weights_

    def get_summary(self) -> str:
        """Get a summary of the ensemble."""
        if self.selected_indices_ is None:
            return "Ensemble not fitted."

        lines = [
            f"Hill Climbing Ensemble Summary",
            f"=" * 40,
            f"Metric: {self.metric}",
            f"Final score: {self.score_:.6f}",
            f"Selected {len(self.selected_indices_)} models:",
        ]

        for name, weight in zip(self.selected_names_, self.weights_):
            lines.append(f"  {name}: {weight:.4f}")

        lines.append("\nSelection history:")
        for name, score in self.history_:
            lines.append(f"  {name}: {score:.6f}")

        return "\n".join(lines)


def cross_validate_ensemble(
    oof_predictions: np.ndarray,
    y_true: np.ndarray,
    model_names: Optional[List[str]] = None,
    n_splits: int = 5,
    metric: str = 'spearman',
    max_models: int = 15,
    verbose: bool = True
) -> Dict:
    """
    Cross-validate the hill climbing ensemble selection.

    This helps assess how stable the selection is and provides
    a more robust estimate of ensemble performance.

    Args:
        oof_predictions: OOF predictions, shape (n_samples, n_models)
        y_true: True target values
        model_names: Optional list of model names
        n_splits: Number of CV folds
        metric: Metric to optimize
        max_models: Maximum models to select
        verbose: Print progress

    Returns:
        Dictionary with CV results
    """
    from sklearn.model_selection import KFold

    n_samples = len(y_true)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    cv_scores = []
    cv_selections = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(range(n_samples))):
        if verbose:
            print(f"\nFold {fold + 1}/{n_splits}")

        result = hill_climbing_ensemble(
            oof_predictions[train_idx],
            y_true[train_idx],
            model_names=model_names,
            metric=metric,
            max_models=max_models,
            verbose=False
        )

        # Evaluate on held-out data
        ensemble = HillClimbingEnsemble(metric=metric, max_models=max_models, verbose=False)
        ensemble.selected_indices_ = result['selected_indices']
        ensemble.weights_ = result['weights']

        val_pred = ensemble.predict(oof_predictions[val_idx])
        val_score = METRICS[metric](val_pred, y_true[val_idx])

        cv_scores.append(val_score)
        cv_selections.append(result['selected_names'])

        if verbose:
            print(f"  Validation score: {val_score:.6f}")
            print(f"  Selected: {result['selected_names'][:5]}...")

    return {
        'cv_scores': cv_scores,
        'mean_score': np.mean(cv_scores),
        'std_score': np.std(cv_scores),
        'selections': cv_selections
    }


if __name__ == "__main__":
    # Quick test
    np.random.seed(42)

    n_samples = 1000
    n_models = 10

    # Simulate OOF predictions
    y_true = np.random.randn(n_samples)

    # Create correlated predictions with varying quality
    oof_preds = np.zeros((n_samples, n_models))
    for i in range(n_models):
        noise = np.random.randn(n_samples) * (0.3 + i * 0.1)
        oof_preds[:, i] = y_true + noise

    model_names = [f"Model_{i}" for i in range(n_models)]

    print("=" * 60)
    print("Testing Hill Climbing Ensemble")
    print("=" * 60)

    result = hill_climbing_ensemble(
        oof_preds, y_true,
        model_names=model_names,
        metric='spearman',
        max_models=5,
        verbose=True
    )

    print(f"\nFinal ensemble: {result['selected_names']}")
    print(f"Weights: {result['weights']}")
    print(f"Score: {result['score']:.6f}")

    print("\n" + "=" * 60)
    print("Testing Diversity Hill Climbing")
    print("=" * 60)

    result2 = diversity_hill_climbing(
        oof_preds, y_true,
        model_names=model_names,
        metric='spearman',
        max_models=5,
        diversity_bonus=0.1,
        verbose=True
    )

    print(f"\nFinal ensemble: {result2['selected_names']}")
    print(f"Score: {result2['score']:.6f}")
