"""
Scaffold-based Cross-Validation

More realistic evaluation than random splits:
- Groups molecules by Bemis-Murcko scaffold
- Ensures train/test splits don't share scaffolds
- Better simulates real-world performance on novel chemistry

Based on paper recommendations for robust model evaluation.
"""
import numpy as np
from collections import defaultdict
from sklearn.model_selection import BaseCrossValidator
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def get_scaffold(smiles):
    """
    Get Bemis-Murcko scaffold for a SMILES string

    Returns:
        Canonical SMILES of the scaffold, or None if invalid
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Get core scaffold
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)

        # Get generic scaffold (removing side chains)
        try:
            generic = MurckoScaffold.MakeScaffoldGeneric(scaffold)
            return Chem.MolToSmiles(generic)
        except:
            return Chem.MolToSmiles(scaffold)

    except Exception:
        return None


def group_by_scaffold(smiles_list):
    """
    Group molecules by their Bemis-Murcko scaffold

    Args:
        smiles_list: List of SMILES strings

    Returns:
        Dict of {scaffold: list of indices}
    """
    scaffold_to_indices = defaultdict(list)

    for i, smi in enumerate(smiles_list):
        scaffold = get_scaffold(smi)
        if scaffold is None:
            scaffold = f"INVALID_{i}"  # Each invalid mol gets its own group
        scaffold_to_indices[scaffold].append(i)

    return scaffold_to_indices


class ScaffoldKFold(BaseCrossValidator):
    """
    Scaffold-based K-Fold cross-validation

    Groups molecules by scaffold and ensures that train/test splits
    don't share scaffolds. This provides more realistic evaluation
    of model performance on novel chemistry.

    Parameters:
        n_splits: Number of folds
        shuffle: Whether to shuffle scaffolds before splitting
        random_state: Random seed for reproducibility
    """

    def __init__(self, n_splits=5, shuffle=True, random_state=42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def _iter_test_indices(self, X=None, y=None, groups=None, smiles=None):
        """
        Generate test indices for each fold

        Args:
            X: Ignored (for sklearn compatibility)
            y: Ignored
            groups: Ignored
            smiles: List of SMILES strings (required!)
        """
        if smiles is None:
            raise ValueError("smiles parameter is required for ScaffoldKFold")

        # Group by scaffold
        scaffold_to_indices = group_by_scaffold(smiles)
        scaffolds = list(scaffold_to_indices.keys())

        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(scaffolds)

        # Distribute scaffolds into folds, trying to balance sample counts
        n_samples = len(smiles)
        target_size = n_samples // self.n_splits

        folds = [[] for _ in range(self.n_splits)]
        fold_sizes = [0] * self.n_splits

        # Sort scaffolds by size (larger first) for more balanced distribution
        scaffolds_sorted = sorted(scaffolds,
                                  key=lambda s: len(scaffold_to_indices[s]),
                                  reverse=True)

        for scaffold in scaffolds_sorted:
            indices = scaffold_to_indices[scaffold]

            # Add to smallest fold
            min_fold = np.argmin(fold_sizes)
            folds[min_fold].extend(indices)
            fold_sizes[min_fold] += len(indices)

        return folds

    def split(self, X, y=None, groups=None, smiles=None):
        """
        Generate train/test indices for each fold

        Args:
            X: Feature matrix (for sklearn compatibility)
            y: Target values (ignored)
            groups: Ignored
            smiles: List of SMILES strings (REQUIRED!)

        Yields:
            train_indices, test_indices for each fold
        """
        n_samples = len(X) if X is not None else len(smiles)
        all_indices = set(range(n_samples))

        folds = self._iter_test_indices(X, y, groups, smiles)

        for fold_indices in folds:
            test_indices = np.array(fold_indices)
            train_indices = np.array(list(all_indices - set(fold_indices)))

            # Sort for reproducibility
            train_indices.sort()
            test_indices.sort()

            yield train_indices, test_indices


class RepeatedScaffoldKFold(BaseCrossValidator):
    """
    Repeated Scaffold K-Fold cross-validation

    Runs ScaffoldKFold multiple times with different random seeds.
    Provides more robust estimates of model performance.

    Based on paper recommendation: 5x5 repeated CV for datasets of 500-100,000 samples.

    Parameters:
        n_splits: Number of folds per repeat
        n_repeats: Number of times to repeat
        random_state: Base random seed
    """

    def __init__(self, n_splits=5, n_repeats=5, random_state=42):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits * self.n_repeats

    def split(self, X, y=None, groups=None, smiles=None):
        """Generate train/test indices for all folds across all repeats"""
        for repeat in range(self.n_repeats):
            cv = ScaffoldKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_state + repeat * 1000
            )

            for train_idx, test_idx in cv.split(X, y, groups, smiles=smiles):
                yield train_idx, test_idx


def analyze_scaffold_distribution(smiles_list):
    """
    Analyze scaffold distribution in a dataset

    Returns:
        Dict with statistics about scaffold distribution
    """
    scaffold_to_indices = group_by_scaffold(smiles_list)

    scaffold_sizes = [len(indices) for indices in scaffold_to_indices.values()]

    stats = {
        'n_molecules': len(smiles_list),
        'n_scaffolds': len(scaffold_to_indices),
        'scaffold_ratio': len(scaffold_to_indices) / len(smiles_list),
        'mean_scaffold_size': np.mean(scaffold_sizes),
        'median_scaffold_size': np.median(scaffold_sizes),
        'max_scaffold_size': max(scaffold_sizes),
        'min_scaffold_size': min(scaffold_sizes),
        'singletons': sum(1 for s in scaffold_sizes if s == 1),
    }

    return stats


if __name__ == "__main__":
    print("Testing Scaffold-based Cross-Validation")
    print("=" * 50)

    # Test SMILES
    test_smiles = [
        # Benzene derivatives
        'c1ccccc1',
        'Cc1ccccc1',
        'CCc1ccccc1',
        'c1ccc(O)cc1',
        'c1ccc(N)cc1',
        # Naphthalene derivatives
        'c1ccc2ccccc2c1',
        'Cc1ccc2ccccc2c1',
        'c1ccc2c(O)cccc2c1',
        # Pyridine derivatives
        'c1ccncc1',
        'Cc1ccncc1',
        'c1cc(O)ncc1',
        # Unique scaffolds
        'CC(C)C',
        'CCCC',
        'C1CCCCC1',
        'C1CCNCC1',
    ]

    # Analyze distribution
    stats = analyze_scaffold_distribution(test_smiles)
    print(f"\nDataset statistics:")
    for key, val in stats.items():
        print(f"  {key}: {val}")

    # Test ScaffoldKFold
    print("\n\nScaffoldKFold (5 folds):")
    cv = ScaffoldKFold(n_splits=5, shuffle=True, random_state=42)

    X = np.random.randn(len(test_smiles), 10)  # Dummy features

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, smiles=test_smiles)):
        train_scaffolds = set(get_scaffold(test_smiles[i]) for i in train_idx)
        test_scaffolds = set(get_scaffold(test_smiles[i]) for i in test_idx)
        overlap = train_scaffolds & test_scaffolds

        print(f"  Fold {fold + 1}: train={len(train_idx)}, test={len(test_idx)}, "
              f"scaffold overlap={len(overlap)}")

    # Test RepeatedScaffoldKFold
    print("\n\nRepeatedScaffoldKFold (3x3 = 9 folds):")
    cv = RepeatedScaffoldKFold(n_splits=3, n_repeats=3, random_state=42)

    fold_count = 0
    for train_idx, test_idx in cv.split(X, smiles=test_smiles):
        fold_count += 1

    print(f"  Total folds: {fold_count}")
