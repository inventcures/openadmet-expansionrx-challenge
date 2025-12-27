"""
Phase 2B: Endpoint-Specific Feature Selection

Selects optimal features for each endpoint using:
- Mutual information
- Feature importance from tree models
- Correlation-based filtering
"""
import numpy as np
from sklearn.feature_selection import mutual_info_regression, SelectKBest
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb


class EndpointFeatureSelector:
    """
    Select optimal features for each ADMET endpoint

    Different endpoints may benefit from different feature subsets.
    """

    def __init__(self, method='importance', n_features=500):
        """
        Args:
            method: 'importance' (tree-based), 'mutual_info', or 'hybrid'
            n_features: Number of features to select per endpoint
        """
        self.method = method
        self.n_features = n_features
        self.selected_features = {}
        self.feature_scores = {}

    def _select_by_importance(self, X, y):
        """Select features using XGBoost importance"""
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            n_jobs=-1,
            verbosity=0
        )
        model.fit(X, y)

        importance = model.feature_importances_
        top_idx = np.argsort(importance)[-self.n_features:]
        return top_idx, importance

    def _select_by_mutual_info(self, X, y):
        """Select features using mutual information"""
        mi_scores = mutual_info_regression(X, y, random_state=42)
        top_idx = np.argsort(mi_scores)[-self.n_features:]
        return top_idx, mi_scores

    def _select_hybrid(self, X, y):
        """Combine importance and mutual info"""
        # Get both scores
        _, imp_scores = self._select_by_importance(X, y)
        _, mi_scores = self._select_by_mutual_info(X, y)

        # Normalize and combine
        imp_norm = imp_scores / (imp_scores.max() + 1e-8)
        mi_norm = mi_scores / (mi_scores.max() + 1e-8)
        combined = 0.6 * imp_norm + 0.4 * mi_norm

        top_idx = np.argsort(combined)[-self.n_features:]
        return top_idx, combined

    def fit(self, X, y, endpoint_name):
        """
        Fit feature selector for an endpoint

        Args:
            X: Features (n_samples, n_features)
            y: Target values
            endpoint_name: Name of the endpoint

        Returns:
            self
        """
        if self.method == 'importance':
            idx, scores = self._select_by_importance(X, y)
        elif self.method == 'mutual_info':
            idx, scores = self._select_by_mutual_info(X, y)
        else:
            idx, scores = self._select_hybrid(X, y)

        self.selected_features[endpoint_name] = idx
        self.feature_scores[endpoint_name] = scores

        return self

    def transform(self, X, endpoint_name):
        """
        Transform features using selected subset

        Args:
            X: Features (n_samples, n_features)
            endpoint_name: Name of the endpoint

        Returns:
            X_selected: (n_samples, n_selected_features)
        """
        if endpoint_name not in self.selected_features:
            raise ValueError(f"No selection fitted for {endpoint_name}")

        idx = self.selected_features[endpoint_name]
        return X[:, idx]

    def fit_transform(self, X, y, endpoint_name):
        """Fit and transform in one step"""
        self.fit(X, y, endpoint_name)
        return self.transform(X, endpoint_name)


class MultiEndpointFeatureSelector:
    """
    Manage feature selection across all endpoints
    """

    def __init__(self, method='hybrid', n_features=500):
        self.method = method
        self.n_features = n_features
        self.selectors = {}

    def fit(self, X, y_dict, verbose=True):
        """
        Fit feature selectors for all endpoints

        Args:
            X: Features
            y_dict: Dict of {endpoint: target_values}
        """
        for endpoint, y in y_dict.items():
            mask = ~np.isnan(y)
            if mask.sum() < 100:
                if verbose:
                    print(f"  {endpoint}: Skipping (too few samples)")
                continue

            X_valid = X[mask]
            y_valid = y[mask]

            selector = EndpointFeatureSelector(
                method=self.method,
                n_features=self.n_features
            )
            selector.fit(X_valid, y_valid, endpoint)
            self.selectors[endpoint] = selector

            if verbose:
                print(f"  {endpoint}: Selected {self.n_features} features")

        return self

    def transform(self, X, endpoint):
        """Transform features for a specific endpoint"""
        if endpoint not in self.selectors:
            return X  # Return original if no selector
        return self.selectors[endpoint].transform(X, endpoint)

    def get_common_features(self, min_endpoints=5):
        """
        Get features selected by multiple endpoints

        Args:
            min_endpoints: Minimum number of endpoints that must select the feature

        Returns:
            Array of feature indices
        """
        if not self.selectors:
            return np.array([])

        # Count feature occurrences
        all_features = []
        for selector in self.selectors.values():
            for endpoint, idx in selector.selected_features.items():
                all_features.extend(idx)

        unique, counts = np.unique(all_features, return_counts=True)
        common = unique[counts >= min_endpoints]

        return common


def remove_correlated_features(X, threshold=0.95):
    """
    Remove highly correlated features

    Args:
        X: Features (n_samples, n_features)
        threshold: Correlation threshold

    Returns:
        X_reduced, selected_indices
    """
    n_features = X.shape[1]

    # Compute correlation matrix
    corr_matrix = np.corrcoef(X.T)
    corr_matrix = np.nan_to_num(corr_matrix)

    # Find features to drop
    to_drop = set()
    for i in range(n_features):
        if i in to_drop:
            continue
        for j in range(i + 1, n_features):
            if j in to_drop:
                continue
            if abs(corr_matrix[i, j]) > threshold:
                to_drop.add(j)

    # Select remaining features
    selected = [i for i in range(n_features) if i not in to_drop]
    return X[:, selected], np.array(selected)


if __name__ == "__main__":
    print("Testing Feature Selection")
    print("=" * 50)

    # Synthetic data
    np.random.seed(42)
    n_samples, n_features = 500, 1000
    X = np.random.randn(n_samples, n_features).astype(np.float32)

    # Create target with some informative features
    informative_idx = [0, 5, 10, 50, 100]
    y = sum(X[:, i] * (i + 1) for i in informative_idx) + np.random.randn(n_samples) * 0.5

    print(f"X shape: {X.shape}")
    print(f"Informative features: {informative_idx}")

    # Test selector
    selector = EndpointFeatureSelector(method='hybrid', n_features=50)
    X_selected = selector.fit_transform(X, y, 'test')

    print(f"\nSelected shape: {X_selected.shape}")
    print(f"Top 10 selected indices: {selector.selected_features['test'][-10:]}")

    # Check if informative features are in top selected
    selected = set(selector.selected_features['test'])
    found = [i for i in informative_idx if i in selected]
    print(f"Informative features found: {found}")
