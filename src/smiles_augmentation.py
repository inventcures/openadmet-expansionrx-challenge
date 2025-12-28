"""
SMILES Augmentation Module for V4 Pipeline.

Key technique: Generate multiple valid SMILES for the same molecule via
random atom ordering (enumeration). This provides:
- 20x data augmentation for training
- Test-time averaging for reduced variance
- Proven 18% R² improvement in literature

Reference: https://arxiv.org/abs/1703.07076
"""

import random
from typing import List, Tuple, Optional
import numpy as np
from rdkit import Chem
from tqdm import tqdm


def enumerate_smiles(smiles: str, n_variants: int = 20, seed: Optional[int] = None) -> List[str]:
    """
    Generate multiple valid SMILES for the same molecule.

    Each molecule can be represented by many valid SMILES strings depending
    on the atom ordering. This diversity helps models learn invariant features.

    Args:
        smiles: Input canonical SMILES
        n_variants: Target number of variants (default 20)
        seed: Random seed for reproducibility

    Returns:
        List of unique SMILES variants (including canonical)
    """
    if seed is not None:
        random.seed(seed)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles]

    n_atoms = mol.GetNumAtoms()
    if n_atoms == 0:
        return [smiles]

    variants = set()
    variants.add(smiles)  # Always include canonical

    max_attempts = n_variants * 10  # Limit attempts to avoid infinite loops
    attempts = 0

    while len(variants) < n_variants and attempts < max_attempts:
        try:
            # Random atom ordering
            atom_order = list(range(n_atoms))
            random.shuffle(atom_order)

            new_mol = Chem.RenumberAtoms(mol, atom_order)
            new_smiles = Chem.MolToSmiles(new_mol, canonical=False)

            # Verify the new SMILES is valid
            if Chem.MolFromSmiles(new_smiles) is not None:
                variants.add(new_smiles)
        except Exception:
            pass

        attempts += 1

    return list(variants)


def augment_dataset(
    smiles_list: List[str],
    labels: np.ndarray,
    n_augmentations: int = 20,
    seed: int = 42,
    verbose: bool = True
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Augment training dataset with SMILES enumeration.

    Each molecule gets up to n_augmentations SMILES variants,
    all sharing the same label.

    Args:
        smiles_list: List of canonical SMILES
        labels: Array of target values
        n_augmentations: Number of SMILES variants per molecule
        seed: Random seed
        verbose: Show progress bar

    Returns:
        Tuple of (augmented_smiles, augmented_labels, original_indices)
        - original_indices maps each augmented sample to its original index
    """
    random.seed(seed)
    np.random.seed(seed)

    augmented_smiles = []
    augmented_labels = []
    original_indices = []

    iterator = tqdm(enumerate(smiles_list), total=len(smiles_list),
                    desc="Augmenting SMILES", disable=not verbose)

    for idx, smi in iterator:
        variants = enumerate_smiles(smi, n_variants=n_augmentations, seed=seed + idx)

        for variant in variants:
            augmented_smiles.append(variant)
            augmented_labels.append(labels[idx])
            original_indices.append(idx)

    return (
        augmented_smiles,
        np.array(augmented_labels),
        np.array(original_indices)
    )


def predict_with_augmentation(
    predict_fn,
    smiles_list: List[str],
    n_augmentations: int = 20,
    aggregation: str = 'mean',
    seed: int = 42,
    verbose: bool = True
) -> np.ndarray:
    """
    Test-time augmentation: predict on multiple SMILES variants and aggregate.

    This reduces prediction variance by averaging over different representations.

    Args:
        predict_fn: Function that takes list of SMILES and returns predictions
        smiles_list: List of canonical SMILES to predict
        n_augmentations: Number of SMILES variants per molecule
        aggregation: 'mean' or 'median'
        seed: Random seed
        verbose: Show progress bar

    Returns:
        Array of aggregated predictions (one per input molecule)
    """
    random.seed(seed)

    all_predictions = []

    iterator = tqdm(range(len(smiles_list)), desc="TTA predictions", disable=not verbose)

    for idx in iterator:
        smi = smiles_list[idx]
        variants = enumerate_smiles(smi, n_variants=n_augmentations, seed=seed + idx)

        # Get predictions for all variants
        variant_preds = predict_fn(variants)

        # Aggregate
        if aggregation == 'mean':
            pred = np.mean(variant_preds)
        elif aggregation == 'median':
            pred = np.median(variant_preds)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        all_predictions.append(pred)

    return np.array(all_predictions)


def batch_predict_with_augmentation(
    predict_fn,
    smiles_list: List[str],
    n_augmentations: int = 20,
    batch_size: int = 1000,
    aggregation: str = 'mean',
    seed: int = 42,
    verbose: bool = True
) -> np.ndarray:
    """
    Batch version of TTA for efficiency with large datasets.

    Generates all augmented SMILES first, predicts in batches, then aggregates.
    More efficient for models with high per-call overhead.

    Args:
        predict_fn: Function that takes list of SMILES and returns predictions
        smiles_list: List of canonical SMILES to predict
        n_augmentations: Number of SMILES variants per molecule
        batch_size: Batch size for prediction
        aggregation: 'mean' or 'median'
        seed: Random seed
        verbose: Show progress bar

    Returns:
        Array of aggregated predictions (one per input molecule)
    """
    random.seed(seed)

    # Generate all augmented SMILES with tracking
    all_augmented = []
    molecule_ids = []

    iterator = tqdm(range(len(smiles_list)), desc="Generating variants", disable=not verbose)

    for idx in iterator:
        variants = enumerate_smiles(smiles_list[idx], n_variants=n_augmentations, seed=seed + idx)
        for v in variants:
            all_augmented.append(v)
            molecule_ids.append(idx)

    molecule_ids = np.array(molecule_ids)

    # Predict in batches
    all_preds = []
    n_total = len(all_augmented)

    for start in tqdm(range(0, n_total, batch_size), desc="Predicting", disable=not verbose):
        end = min(start + batch_size, n_total)
        batch_smiles = all_augmented[start:end]
        batch_preds = predict_fn(batch_smiles)
        all_preds.extend(batch_preds)

    all_preds = np.array(all_preds)

    # Aggregate by molecule
    final_preds = np.zeros(len(smiles_list))

    for idx in range(len(smiles_list)):
        mask = molecule_ids == idx
        if aggregation == 'mean':
            final_preds[idx] = np.mean(all_preds[mask])
        elif aggregation == 'median':
            final_preds[idx] = np.median(all_preds[mask])

    return final_preds


class SmilesAugmenter:
    """
    Wrapper class for SMILES augmentation with sklearn-like interface.

    Usage:
        augmenter = SmilesAugmenter(n_augmentations=20)

        # Training: augment data
        aug_X, aug_y, indices = augmenter.fit_transform(smiles_train, y_train)
        model.fit(aug_X, aug_y)

        # Prediction: TTA
        predictions = augmenter.predict(model.predict, smiles_test)
    """

    def __init__(
        self,
        n_augmentations: int = 20,
        aggregation: str = 'mean',
        seed: int = 42,
        verbose: bool = True
    ):
        self.n_augmentations = n_augmentations
        self.aggregation = aggregation
        self.seed = seed
        self.verbose = verbose

    def fit_transform(
        self,
        smiles: List[str],
        labels: np.ndarray
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """Augment training data."""
        return augment_dataset(
            smiles, labels,
            n_augmentations=self.n_augmentations,
            seed=self.seed,
            verbose=self.verbose
        )

    def predict(
        self,
        predict_fn,
        smiles: List[str],
        batch_mode: bool = True,
        batch_size: int = 1000
    ) -> np.ndarray:
        """Predict with test-time augmentation."""
        if batch_mode:
            return batch_predict_with_augmentation(
                predict_fn, smiles,
                n_augmentations=self.n_augmentations,
                batch_size=batch_size,
                aggregation=self.aggregation,
                seed=self.seed,
                verbose=self.verbose
            )
        else:
            return predict_with_augmentation(
                predict_fn, smiles,
                n_augmentations=self.n_augmentations,
                aggregation=self.aggregation,
                seed=self.seed,
                verbose=self.verbose
            )


def get_augmentation_stats(smiles_list: List[str], n_augmentations: int = 20) -> dict:
    """
    Analyze augmentation potential for a dataset.

    Returns stats on how many unique SMILES can be generated per molecule.
    Useful for understanding if augmentation will be effective.
    """
    counts = []

    for smi in tqdm(smiles_list[:min(1000, len(smiles_list))], desc="Analyzing"):
        variants = enumerate_smiles(smi, n_variants=n_augmentations * 2)
        counts.append(len(variants))

    counts = np.array(counts)

    return {
        'mean_variants': np.mean(counts),
        'median_variants': np.median(counts),
        'min_variants': np.min(counts),
        'max_variants': np.max(counts),
        'pct_at_target': np.mean(counts >= n_augmentations) * 100,
        'effective_augmentation_factor': np.mean(np.minimum(counts, n_augmentations))
    }


if __name__ == "__main__":
    # Quick test
    test_smiles = [
        "CCO",  # Ethanol
        "CC(=O)O",  # Acetic acid
        "c1ccccc1",  # Benzene
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # Ibuprofen
    ]

    print("Testing SMILES enumeration:")
    for smi in test_smiles:
        variants = enumerate_smiles(smi, n_variants=5)
        print(f"\n{smi} ({len(variants)} variants):")
        for v in variants[:3]:
            print(f"  {v}")

    # Test augmentation stats
    print("\n\nAugmentation stats:")
    stats = get_augmentation_stats(test_smiles, n_augmentations=20)
    for k, v in stats.items():
        print(f"  {k}: {v:.2f}")
