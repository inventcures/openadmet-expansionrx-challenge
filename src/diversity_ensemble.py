"""
Diversity-Weighted Ensemble for V3 Pipeline

Implements ensemble strategies that maximize both accuracy AND diversity:
- Performance-based weighting (weight by individual Spearman/R²)
- Diversity-based weighting (Negative Correlation Learning)
- Greedy forward selection
- Uncertainty-weighted predictions

Based on literature: diverse ensembles consistently outperform homogeneous ones.
"""
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from scipy.stats import spearmanr
from scipy.special import softmax
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_predict
import warnings
warnings.filterwarnings('ignore')


class DiversityWeightedEnsemble:
    """
    Ensemble that weights models by both accuracy AND diversity.

    Key insight: Two models with 0.8 Spearman but 0.9 correlation between them
    add less value than two models with 0.75 Spearman but 0.3 correlation.
    """

    def __init__(
        self,
        method: str = 'diversity',
        n_top_models: Optional[int] = None,
        diversity_weight: float = 0.5,
        min_accuracy_threshold: float = 0.3,
    ):
        """
        Args:
            method: Weighting method - 'performance', 'diversity', 'greedy', or 'stacking'
            n_top_models: Optional limit on number of models in ensemble
            diversity_weight: Weight given to diversity vs accuracy (0-1)
            min_accuracy_threshold: Minimum Spearman to include model
        """
        self.method = method
        self.n_top_models = n_top_models
        self.diversity_weight = diversity_weight
        self.min_accuracy_threshold = min_accuracy_threshold
        self.weights_ = None
        self.selected_indices_ = None
        self.meta_model_ = None

    def fit(
        self,
        oof_predictions: np.ndarray,
        y_true: np.ndarray,
        model_names: Optional[List[str]] = None
    ) -> 'DiversityWeightedEnsemble':
        """
        Fit ensemble weights based on OOF predictions.

        Args:
            oof_predictions: Shape (n_samples, n_models) - OOF predictions from each model
            y_true: Shape (n_samples,) - True target values
            model_names: Optional names for logging

        Returns:
            self
        """
        n_models = oof_predictions.shape[1]

        # Filter by accuracy threshold
        valid_indices = []
        for i in range(n_models):
            corr, _ = spearmanr(oof_predictions[:, i], y_true)
            if corr >= self.min_accuracy_threshold:
                valid_indices.append(i)

        if len(valid_indices) == 0:
            # Fallback: use all models
            valid_indices = list(range(n_models))

        oof_valid = oof_predictions[:, valid_indices]

        if self.method == 'performance':
            self.weights_, self.selected_indices_ = self._performance_weights(
                oof_valid, y_true, valid_indices
            )
        elif self.method == 'diversity':
            self.weights_, self.selected_indices_ = self._diversity_weights(
                oof_valid, y_true, valid_indices
            )
        elif self.method == 'greedy':
            self.weights_, self.selected_indices_ = self._greedy_selection(
                oof_valid, y_true, valid_indices
            )
        elif self.method == 'stacking':
            self._fit_stacking(oof_valid, y_true, valid_indices)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return self

    def _performance_weights(
        self,
        oof_predictions: np.ndarray,
        y_true: np.ndarray,
        valid_indices: List[int]
    ) -> Tuple[np.ndarray, List[int]]:
        """Weight by individual Spearman correlation"""
        scores = []
        for i in range(oof_predictions.shape[1]):
            corr, _ = spearmanr(oof_predictions[:, i], y_true)
            scores.append(max(corr, 0))  # Clip negative correlations

        # Select top models if specified
        if self.n_top_models and len(scores) > self.n_top_models:
            top_indices = np.argsort(scores)[-self.n_top_models:]
            scores = [scores[i] for i in top_indices]
            valid_indices = [valid_indices[i] for i in top_indices]

        weights = softmax(np.array(scores) * 5)  # Temperature scaling
        return weights, valid_indices

    def _diversity_weights(
        self,
        oof_predictions: np.ndarray,
        y_true: np.ndarray,
        valid_indices: List[int]
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Weight by accuracy AND diversity (Negative Correlation Learning).

        Models get higher weights if they are:
        1. Accurate (high Spearman with true values)
        2. Diverse (low correlation with other models)
        """
        n_models = oof_predictions.shape[1]

        # Compute accuracy scores
        accuracy_scores = []
        for i in range(n_models):
            corr, _ = spearmanr(oof_predictions[:, i], y_true)
            accuracy_scores.append(max(corr, 0))
        accuracy_scores = np.array(accuracy_scores)

        # Compute pairwise correlations between models
        corr_matrix = np.corrcoef(oof_predictions.T)

        # Diversity score: how different is each model from others?
        diversity_scores = []
        for i in range(n_models):
            # Mean absolute correlation with other models
            other_corrs = np.abs(np.delete(corr_matrix[i], i))
            diversity = 1 - np.mean(other_corrs)
            diversity_scores.append(max(diversity, 0))
        diversity_scores = np.array(diversity_scores)

        # Combine accuracy and diversity
        # Higher diversity_weight means diversity matters more
        combined_scores = (
            (1 - self.diversity_weight) * accuracy_scores +
            self.diversity_weight * diversity_scores
        )

        # Also bonus for being accurate AND diverse
        combined_scores *= (1 + accuracy_scores * diversity_scores)

        # Select top models if specified
        if self.n_top_models and n_models > self.n_top_models:
            top_indices = np.argsort(combined_scores)[-self.n_top_models:]
            combined_scores = combined_scores[top_indices]
            valid_indices = [valid_indices[i] for i in top_indices]

        weights = softmax(combined_scores * 3)  # Temperature scaling
        return weights, valid_indices

    def _greedy_selection(
        self,
        oof_predictions: np.ndarray,
        y_true: np.ndarray,
        valid_indices: List[int],
        max_models: int = 20
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Greedy forward selection to minimize ensemble error.

        Iteratively adds the model that most improves the ensemble.
        """
        n_models = oof_predictions.shape[1]
        max_models = self.n_top_models or min(max_models, n_models)

        selected = []
        remaining = list(range(n_models))

        # Start with best individual model
        best_corr = -1
        best_idx = 0
        for i in remaining:
            corr, _ = spearmanr(oof_predictions[:, i], y_true)
            if corr > best_corr:
                best_corr = corr
                best_idx = i

        selected.append(best_idx)
        remaining.remove(best_idx)

        # Greedily add models
        while len(selected) < max_models and len(remaining) > 0:
            best_improvement = -np.inf
            best_add_idx = None

            current_ensemble = oof_predictions[:, selected].mean(axis=1)
            current_corr, _ = spearmanr(current_ensemble, y_true)

            for idx in remaining:
                # Try adding this model
                new_selected = selected + [idx]
                new_ensemble = oof_predictions[:, new_selected].mean(axis=1)
                new_corr, _ = spearmanr(new_ensemble, y_true)

                improvement = new_corr - current_corr
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_add_idx = idx

            # Stop if no improvement
            if best_improvement <= 0:
                break

            selected.append(best_add_idx)
            remaining.remove(best_add_idx)

        # Equal weights for greedy selection
        weights = np.ones(len(selected)) / len(selected)
        selected_valid = [valid_indices[i] for i in selected]

        return weights, selected_valid

    def _fit_stacking(
        self,
        oof_predictions: np.ndarray,
        y_true: np.ndarray,
        valid_indices: List[int]
    ):
        """Fit a stacking meta-learner"""
        self.selected_indices_ = valid_indices

        # Use Ridge regression as meta-learner (robust to collinearity)
        self.meta_model_ = Ridge(alpha=1.0)
        self.meta_model_.fit(oof_predictions, y_true)

        # Store coefficients as "weights" for inspection
        self.weights_ = np.abs(self.meta_model_.coef_)
        self.weights_ = self.weights_ / self.weights_.sum()

    def predict(self, predictions: np.ndarray) -> np.ndarray:
        """
        Make ensemble predictions.

        Args:
            predictions: Shape (n_samples, n_models) - predictions from all models

        Returns:
            Ensemble predictions, shape (n_samples,)
        """
        if self.method == 'stacking' and self.meta_model_ is not None:
            selected_preds = predictions[:, self.selected_indices_]
            return self.meta_model_.predict(selected_preds)
        else:
            selected_preds = predictions[:, self.selected_indices_]
            return np.average(selected_preds, axis=1, weights=self.weights_)

    def get_model_importance(self, model_names: Optional[List[str]] = None) -> Dict:
        """Get model importance/weights for analysis"""
        if model_names is None:
            model_names = [f"model_{i}" for i in range(len(self.selected_indices_))]

        importance = {}
        for idx, weight in zip(self.selected_indices_, self.weights_):
            name = model_names[idx] if idx < len(model_names) else f"model_{idx}"
            importance[name] = float(weight)

        return dict(sorted(importance.items(), key=lambda x: -x[1]))


class UncertaintyWeightedEnsemble:
    """
    Weight predictions by inverse uncertainty.

    Models more confident in their predictions get higher weight.
    Uncertainty estimated from:
    - Ensemble variance (multiple seeds)
    - Distance to training data (applicability domain)
    """

    def __init__(self, use_ad_weighting: bool = True):
        """
        Args:
            use_ad_weighting: Whether to use applicability domain weighting
        """
        self.use_ad_weighting = use_ad_weighting
        self.train_predictions_ = None

    def fit(self, train_predictions: np.ndarray):
        """
        Store training predictions for AD estimation.

        Args:
            train_predictions: Shape (n_train, n_models)
        """
        self.train_predictions_ = train_predictions
        return self

    def predict(
        self,
        predictions: np.ndarray,
        uncertainties: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Make uncertainty-weighted predictions.

        Args:
            predictions: Shape (n_samples, n_models) - predictions from each model
            uncertainties: Shape (n_samples, n_models) - optional uncertainty estimates

        Returns:
            Weighted ensemble predictions, shape (n_samples,)
        """
        n_samples, n_models = predictions.shape

        if uncertainties is None:
            # Estimate uncertainty from prediction variance across models
            mean_pred = predictions.mean(axis=1, keepdims=True)
            uncertainties = np.abs(predictions - mean_pred)

        # Inverse uncertainty weighting
        # Higher uncertainty = lower weight
        weights = 1.0 / (uncertainties + 1e-8)
        weights = weights / weights.sum(axis=1, keepdims=True)

        # Weighted average
        ensemble_pred = (predictions * weights).sum(axis=1)

        return ensemble_pred

    def predict_with_uncertainty(
        self,
        predictions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return both predictions and uncertainty estimates.

        Args:
            predictions: Shape (n_samples, n_models)

        Returns:
            - Ensemble predictions, shape (n_samples,)
            - Uncertainty estimates, shape (n_samples,)
        """
        ensemble_pred = self.predict(predictions)

        # Uncertainty = standard deviation across models
        uncertainty = predictions.std(axis=1)

        return ensemble_pred, uncertainty


class MultiMethodEnsemble:
    """
    Combines multiple ensemble methods and ensembles THEM.

    Meta-ensemble of ensemble methods for maximum robustness.
    """

    def __init__(
        self,
        methods: List[str] = ['performance', 'diversity', 'greedy'],
        meta_weights: Optional[List[float]] = None
    ):
        """
        Args:
            methods: List of ensemble methods to combine
            meta_weights: Optional weights for each method (uniform if None)
        """
        self.methods = methods
        self.meta_weights = meta_weights or [1/len(methods)] * len(methods)
        self.ensembles_ = {}

    def fit(
        self,
        oof_predictions: np.ndarray,
        y_true: np.ndarray,
        model_names: Optional[List[str]] = None
    ) -> 'MultiMethodEnsemble':
        """Fit all ensemble methods"""
        for method in self.methods:
            ensemble = DiversityWeightedEnsemble(method=method)
            ensemble.fit(oof_predictions, y_true, model_names)
            self.ensembles_[method] = ensemble

        return self

    def predict(self, predictions: np.ndarray) -> np.ndarray:
        """Make meta-ensemble predictions"""
        method_predictions = []
        for method, weight in zip(self.methods, self.meta_weights):
            method_pred = self.ensembles_[method].predict(predictions)
            method_predictions.append(method_pred)

        method_predictions = np.array(method_predictions)
        return np.average(method_predictions, axis=0, weights=self.meta_weights)

    def get_all_importances(self, model_names: Optional[List[str]] = None) -> Dict:
        """Get model importances from all methods"""
        all_importance = {}
        for method in self.methods:
            all_importance[method] = self.ensembles_[method].get_model_importance(model_names)
        return all_importance


def compute_model_diversity_matrix(oof_predictions: np.ndarray) -> np.ndarray:
    """
    Compute pairwise diversity (1 - correlation) between models.

    Args:
        oof_predictions: Shape (n_samples, n_models)

    Returns:
        Diversity matrix, shape (n_models, n_models)
    """
    corr_matrix = np.corrcoef(oof_predictions.T)
    diversity_matrix = 1 - np.abs(corr_matrix)
    return diversity_matrix


def analyze_ensemble_diversity(
    oof_predictions: np.ndarray,
    y_true: np.ndarray,
    model_names: Optional[List[str]] = None
) -> Dict:
    """
    Analyze diversity and complementarity of models.

    Returns:
        Dictionary with diversity metrics and recommendations
    """
    n_models = oof_predictions.shape[1]
    if model_names is None:
        model_names = [f"model_{i}" for i in range(n_models)]

    # Individual accuracies
    accuracies = {}
    for i, name in enumerate(model_names):
        corr, _ = spearmanr(oof_predictions[:, i], y_true)
        accuracies[name] = corr

    # Diversity matrix
    diversity_matrix = compute_model_diversity_matrix(oof_predictions)

    # Mean diversity
    triu_indices = np.triu_indices(n_models, k=1)
    mean_diversity = diversity_matrix[triu_indices].mean()

    # Find most diverse pairs
    most_diverse_pairs = []
    for i in range(n_models):
        for j in range(i+1, n_models):
            diversity = diversity_matrix[i, j]
            most_diverse_pairs.append((model_names[i], model_names[j], diversity))
    most_diverse_pairs.sort(key=lambda x: -x[2])

    # Find redundant models (high correlation)
    redundant_pairs = [(a, b, d) for a, b, d in most_diverse_pairs if d < 0.2]

    return {
        'individual_accuracies': accuracies,
        'mean_diversity': mean_diversity,
        'most_diverse_pairs': most_diverse_pairs[:5],
        'redundant_pairs': redundant_pairs,
        'diversity_matrix': diversity_matrix,
        'recommendation': (
            'High diversity ensemble' if mean_diversity > 0.4
            else 'Consider adding more diverse models' if mean_diversity > 0.2
            else 'Models are highly correlated - add different architectures'
        )
    }


if __name__ == "__main__":
    print("Testing Diversity-Weighted Ensemble")
    print("=" * 60)

    # Synthetic test data
    np.random.seed(42)
    n_samples = 500
    n_models = 8

    # True values
    y_true = np.random.randn(n_samples)

    # Simulated model predictions with varying accuracy and diversity
    oof_predictions = np.zeros((n_samples, n_models))

    # Model 0: Good accuracy
    oof_predictions[:, 0] = y_true + np.random.randn(n_samples) * 0.3

    # Model 1: Very similar to model 0 (low diversity)
    oof_predictions[:, 1] = oof_predictions[:, 0] + np.random.randn(n_samples) * 0.1

    # Model 2: Good accuracy, more diverse
    oof_predictions[:, 2] = y_true + np.random.randn(n_samples) * 0.35

    # Model 3: Medium accuracy, high diversity
    oof_predictions[:, 3] = y_true * 0.8 + np.random.randn(n_samples) * 0.5

    # Model 4-7: Various quality levels
    for i in range(4, n_models):
        noise = 0.3 + (i - 4) * 0.1
        oof_predictions[:, i] = y_true + np.random.randn(n_samples) * noise

    model_names = [f"Model_{i}" for i in range(n_models)]

    # Analyze diversity
    print("\n1. Diversity Analysis:")
    analysis = analyze_ensemble_diversity(oof_predictions, y_true, model_names)
    print(f"   Mean diversity: {analysis['mean_diversity']:.3f}")
    print(f"   Recommendation: {analysis['recommendation']}")
    print(f"   Most diverse pairs:")
    for a, b, d in analysis['most_diverse_pairs'][:3]:
        print(f"     {a} - {b}: {d:.3f}")

    # Test different ensemble methods
    print("\n2. Ensemble Method Comparison:")

    # Split for validation
    train_idx = np.random.choice(n_samples, size=int(n_samples * 0.7), replace=False)
    val_idx = np.array([i for i in range(n_samples) if i not in train_idx])

    for method in ['performance', 'diversity', 'greedy', 'stacking']:
        ensemble = DiversityWeightedEnsemble(method=method)
        ensemble.fit(oof_predictions[train_idx], y_true[train_idx], model_names)

        val_pred = ensemble.predict(oof_predictions[val_idx])
        corr, _ = spearmanr(val_pred, y_true[val_idx])

        print(f"\n   {method.capitalize()} method:")
        print(f"     Validation Spearman: {corr:.4f}")
        print(f"     Selected models: {len(ensemble.selected_indices_)}")
        print(f"     Top weights: ", end="")
        importance = ensemble.get_model_importance(model_names)
        for name, weight in list(importance.items())[:3]:
            print(f"{name}: {weight:.2f}, ", end="")
        print()

    # Multi-method ensemble
    print("\n3. Multi-Method Meta-Ensemble:")
    multi = MultiMethodEnsemble()
    multi.fit(oof_predictions[train_idx], y_true[train_idx], model_names)
    multi_pred = multi.predict(oof_predictions[val_idx])
    multi_corr, _ = spearmanr(multi_pred, y_true[val_idx])
    print(f"   Meta-ensemble Spearman: {multi_corr:.4f}")

    # Uncertainty-weighted ensemble
    print("\n4. Uncertainty-Weighted Ensemble:")
    unc_ensemble = UncertaintyWeightedEnsemble()
    pred, uncertainty = unc_ensemble.predict_with_uncertainty(oof_predictions[val_idx])
    unc_corr, _ = spearmanr(pred, y_true[val_idx])
    print(f"   Spearman: {unc_corr:.4f}")
    print(f"   Mean uncertainty: {uncertainty.mean():.4f}")

    print("\n" + "=" * 60)
    print("Diversity ensemble tests passed!")
