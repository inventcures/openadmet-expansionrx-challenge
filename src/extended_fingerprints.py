"""
Extended Fingerprints for V3 Pipeline

Multiple fingerprint types for maximum model diversity:
- ECFP4 (Morgan radius 2, 1024/2048 bits) - standard
- ECFP6 (Morgan radius 3, 2048 bits) - larger context
- FCFP4 (feature-based, radius 2) - pharmacophore-like
- MACCS keys (166 structural keys)
- RDKit fingerprints
- Atom pair fingerprints
- Topological torsion fingerprints

Plus fragment-aware features (MSformer-ADMET inspired):
- Bemis-Murcko scaffolds
- Functional group counts
- ADMET-relevant substructures
"""
import numpy as np
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, MACCSkeys
from rdkit.Chem import Fragments
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import DataStructs
from tqdm import tqdm


# ADMET-relevant SMARTS patterns for functional group counting
ADMET_SMARTS = {
    # Acidic groups
    'carboxylic_acid': '[CX3](=O)[OX2H1]',
    'sulfonamide': '[#16X4]([NX3])(=[OX1])(=[OX1])',
    'sulfonic_acid': '[#16X4](=[OX1])(=[OX1])[OX2H]',
    'phosphate': '[PX4](=[OX1])([OX2])([OX2])[OX2]',

    # Basic groups
    'primary_amine': '[NX3;H2;!$(NC=O)]',
    'secondary_amine': '[NX3;H1;!$(NC=O)]',
    'tertiary_amine': '[NX3;H0;!$(NC=O)]',
    'guanidine': '[NX3][CX3]([NX3])=[NX2]',
    'amidine': '[NX3][CX3]=[NX2]',

    # Hydrogen bond donors/acceptors
    'hydroxyl': '[OX2H]',
    'amide': '[NX3][CX3](=[OX1])[#6]',
    'urea': '[NX3][CX3](=[OX1])[NX3]',
    'ester': '[#6][CX3](=O)[OX2H0][#6]',
    'ether': '[OD2]([#6])[#6]',

    # Metabolism-relevant
    'cyp_substrate_aromatic': 'a1aaaa1',  # 5-membered aromatic
    'cyp_substrate_phenyl': 'c1ccccc1',
    'halogen': '[F,Cl,Br,I]',
    'fluorine': '[F]',
    'chlorine': '[Cl]',
    'nitro': '[N+](=O)[O-]',
    'nitrile': '[CX2]#[NX1]',

    # Permeability-relevant
    'rotatable_chain': '[CH2][CH2][CH2]',
    'aromatic_nitrogen': '[nR]',
    'aliphatic_ring': '[R;!a]',

    # Toxicity alerts (simplified)
    'michael_acceptor': '[CX3]=[CX3][CX3]=O',
    'epoxide': 'C1OC1',
    'aziridine': 'C1NC1',
}


def count_smarts_pattern(mol, smarts: str) -> int:
    """Count occurrences of a SMARTS pattern in a molecule"""
    try:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            return 0
        matches = mol.GetSubstructMatches(pattern)
        return len(matches)
    except:
        return 0


class ExtendedFingerprinter:
    """
    Compute multiple fingerprint types for maximum diversity.
    """

    def __init__(
        self,
        use_ecfp4: bool = True,
        use_ecfp6: bool = True,
        use_fcfp4: bool = True,
        use_maccs: bool = True,
        use_rdkit_fp: bool = True,
        use_atom_pair: bool = False,
        use_torsion: bool = False,
        use_descriptors: bool = True,
        use_fragments: bool = True,
        ecfp4_bits: int = 1024,
        ecfp6_bits: int = 2048,
        fcfp4_bits: int = 1024,
        rdkit_fp_bits: int = 2048,
        verbose: bool = True,
    ):
        self.use_ecfp4 = use_ecfp4
        self.use_ecfp6 = use_ecfp6
        self.use_fcfp4 = use_fcfp4
        self.use_maccs = use_maccs
        self.use_rdkit_fp = use_rdkit_fp
        self.use_atom_pair = use_atom_pair
        self.use_torsion = use_torsion
        self.use_descriptors = use_descriptors
        self.use_fragments = use_fragments

        self.ecfp4_bits = ecfp4_bits
        self.ecfp6_bits = ecfp6_bits
        self.fcfp4_bits = fcfp4_bits
        self.rdkit_fp_bits = rdkit_fp_bits
        self.verbose = verbose

        # Calculate expected feature dimension
        self._calc_feature_dim()

    def _calc_feature_dim(self):
        """Calculate total feature dimension"""
        dim = 0
        if self.use_ecfp4:
            dim += self.ecfp4_bits
        if self.use_ecfp6:
            dim += self.ecfp6_bits
        if self.use_fcfp4:
            dim += self.fcfp4_bits
        if self.use_maccs:
            dim += 167
        if self.use_rdkit_fp:
            dim += self.rdkit_fp_bits
        if self.use_atom_pair:
            dim += 2048
        if self.use_torsion:
            dim += 2048
        if self.use_descriptors:
            dim += 20  # RDKit descriptors
        if self.use_fragments:
            dim += len(ADMET_SMARTS)

        self.feature_dim = dim

    def _compute_single(self, smiles: str) -> np.ndarray:
        """Compute features for a single molecule"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(self.feature_dim, dtype=np.float32)

        features = []

        try:
            # ECFP4 (Morgan radius 2)
            if self.use_ecfp4:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=self.ecfp4_bits)
                arr = np.zeros(self.ecfp4_bits, dtype=np.float32)
                DataStructs.ConvertToNumpyArray(fp, arr)
                features.append(arr)

            # ECFP6 (Morgan radius 3)
            if self.use_ecfp6:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=self.ecfp6_bits)
                arr = np.zeros(self.ecfp6_bits, dtype=np.float32)
                DataStructs.ConvertToNumpyArray(fp, arr)
                features.append(arr)

            # FCFP4 (feature-based Morgan)
            if self.use_fcfp4:
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, 2, nBits=self.fcfp4_bits, useFeatures=True
                )
                arr = np.zeros(self.fcfp4_bits, dtype=np.float32)
                DataStructs.ConvertToNumpyArray(fp, arr)
                features.append(arr)

            # MACCS keys
            if self.use_maccs:
                fp = MACCSkeys.GenMACCSKeys(mol)
                arr = np.zeros(167, dtype=np.float32)
                DataStructs.ConvertToNumpyArray(fp, arr)
                features.append(arr)

            # RDKit fingerprint
            if self.use_rdkit_fp:
                fp = Chem.RDKFingerprint(mol, fpSize=self.rdkit_fp_bits)
                arr = np.zeros(self.rdkit_fp_bits, dtype=np.float32)
                DataStructs.ConvertToNumpyArray(fp, arr)
                features.append(arr)

            # Atom pair fingerprint
            if self.use_atom_pair:
                fp = AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=2048)
                arr = np.zeros(2048, dtype=np.float32)
                DataStructs.ConvertToNumpyArray(fp, arr)
                features.append(arr)

            # Topological torsion fingerprint
            if self.use_torsion:
                fp = AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=2048)
                arr = np.zeros(2048, dtype=np.float32)
                DataStructs.ConvertToNumpyArray(fp, arr)
                features.append(arr)

            # RDKit descriptors
            if self.use_descriptors:
                desc = [
                    Descriptors.MolWt(mol),
                    Descriptors.MolLogP(mol),
                    Descriptors.TPSA(mol),
                    Descriptors.NumHDonors(mol),
                    Descriptors.NumHAcceptors(mol),
                    Descriptors.NumRotatableBonds(mol),
                    Descriptors.NumAromaticRings(mol),
                    Descriptors.FractionCSP3(mol),
                    Descriptors.HeavyAtomCount(mol),
                    Descriptors.RingCount(mol),
                    rdMolDescriptors.CalcNumAliphaticRings(mol),
                    rdMolDescriptors.CalcNumHeterocycles(mol),
                    Descriptors.LabuteASA(mol),
                    Descriptors.NumRadicalElectrons(mol),
                    rdMolDescriptors.CalcNumAmideBonds(mol),
                    rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
                    rdMolDescriptors.CalcNumSpiroAtoms(mol),
                    Descriptors.qed(mol),
                    rdMolDescriptors.CalcNumRings(mol),
                    rdMolDescriptors.CalcNumAromaticHeterocycles(mol),
                ]
                features.append(np.array(desc, dtype=np.float32))

            # Fragment counts (ADMET-relevant)
            if self.use_fragments:
                frag_counts = []
                for name, smarts in ADMET_SMARTS.items():
                    count = count_smarts_pattern(mol, smarts)
                    frag_counts.append(count)
                features.append(np.array(frag_counts, dtype=np.float32))

        except Exception as e:
            # Return zeros on error
            return np.zeros(self.feature_dim, dtype=np.float32)

        return np.concatenate(features)

    def compute_features(self, smiles_list: List[str]) -> np.ndarray:
        """
        Compute features for a list of SMILES.

        Returns:
            np.ndarray of shape (n_molecules, feature_dim)
        """
        features = []

        if self.verbose:
            iterator = tqdm(smiles_list, desc="Computing fingerprints", unit="mol")
        else:
            iterator = smiles_list

        for smi in iterator:
            feat = self._compute_single(smi)
            features.append(feat)

        return np.array(features, dtype=np.float32)

    def get_feature_names(self) -> List[str]:
        """Get feature names for interpretability"""
        names = []

        if self.use_ecfp4:
            names.extend([f'ecfp4_{i}' for i in range(self.ecfp4_bits)])
        if self.use_ecfp6:
            names.extend([f'ecfp6_{i}' for i in range(self.ecfp6_bits)])
        if self.use_fcfp4:
            names.extend([f'fcfp4_{i}' for i in range(self.fcfp4_bits)])
        if self.use_maccs:
            names.extend([f'maccs_{i}' for i in range(167)])
        if self.use_rdkit_fp:
            names.extend([f'rdkit_fp_{i}' for i in range(self.rdkit_fp_bits)])
        if self.use_atom_pair:
            names.extend([f'atom_pair_{i}' for i in range(2048)])
        if self.use_torsion:
            names.extend([f'torsion_{i}' for i in range(2048)])
        if self.use_descriptors:
            names.extend([
                'MolWt', 'MolLogP', 'TPSA', 'NumHDonors', 'NumHAcceptors',
                'NumRotatableBonds', 'NumAromaticRings', 'FractionCSP3',
                'HeavyAtomCount', 'RingCount', 'NumAliphaticRings',
                'NumHeterocycles', 'LabuteASA', 'NumRadicalElectrons',
                'NumAmideBonds', 'NumBridgeheadAtoms', 'NumSpiroAtoms',
                'QED', 'NumRings', 'NumAromaticHeterocycles',
            ])
        if self.use_fragments:
            names.extend([f'frag_{name}' for name in ADMET_SMARTS.keys()])

        return names


class ScaffoldFeatures:
    """
    Extract scaffold-based features for fragment-aware modeling.
    """

    def __init__(self, max_scaffold_fp_bits: int = 512):
        self.max_scaffold_fp_bits = max_scaffold_fp_bits

    def get_scaffold(self, smiles: str) -> Optional[str]:
        """Get Bemis-Murcko scaffold"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            try:
                generic = MurckoScaffold.MakeScaffoldGeneric(scaffold)
                return Chem.MolToSmiles(generic)
            except:
                return Chem.MolToSmiles(scaffold)
        except:
            return None

    def compute_scaffold_fingerprint(self, smiles: str) -> np.ndarray:
        """Compute fingerprint of the molecular scaffold"""
        scaffold_smi = self.get_scaffold(smiles)

        if scaffold_smi is None:
            return np.zeros(self.max_scaffold_fp_bits, dtype=np.float32)

        mol = Chem.MolFromSmiles(scaffold_smi)
        if mol is None:
            return np.zeros(self.max_scaffold_fp_bits, dtype=np.float32)

        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, 2, nBits=self.max_scaffold_fp_bits
            )
            arr = np.zeros(self.max_scaffold_fp_bits, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            return arr
        except:
            return np.zeros(self.max_scaffold_fp_bits, dtype=np.float32)

    def compute_features(self, smiles_list: List[str], verbose: bool = True) -> np.ndarray:
        """Compute scaffold fingerprints for all molecules"""
        if verbose:
            iterator = tqdm(smiles_list, desc="Computing scaffold features", unit="mol")
        else:
            iterator = smiles_list

        features = [self.compute_scaffold_fingerprint(smi) for smi in iterator]
        return np.array(features, dtype=np.float32)


def compute_all_features(
    smiles_list: List[str],
    use_scaffold: bool = True,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Compute all feature types for maximum ensemble diversity.

    Returns dict of feature arrays, allowing different models to use different features.
    """
    results = {}

    # Standard fingerprints (ECFP4 + descriptors)
    if verbose:
        print("Computing standard features (ECFP4 + descriptors)...")
    fp_standard = ExtendedFingerprinter(
        use_ecfp4=True, use_ecfp6=False, use_fcfp4=False,
        use_maccs=True, use_rdkit_fp=False,
        use_descriptors=True, use_fragments=True,
        verbose=verbose
    )
    results['standard'] = fp_standard.compute_features(smiles_list)

    # Extended fingerprints (ECFP6 + FCFP4)
    if verbose:
        print("Computing extended features (ECFP6 + FCFP4)...")
    fp_extended = ExtendedFingerprinter(
        use_ecfp4=False, use_ecfp6=True, use_fcfp4=True,
        use_maccs=False, use_rdkit_fp=True,
        use_descriptors=False, use_fragments=False,
        verbose=verbose
    )
    results['extended'] = fp_extended.compute_features(smiles_list)

    # All fingerprints (comprehensive)
    if verbose:
        print("Computing comprehensive features (all fingerprints)...")
    fp_all = ExtendedFingerprinter(
        use_ecfp4=True, use_ecfp6=True, use_fcfp4=True,
        use_maccs=True, use_rdkit_fp=True,
        use_descriptors=True, use_fragments=True,
        verbose=verbose
    )
    results['comprehensive'] = fp_all.compute_features(smiles_list)

    # Scaffold features
    if use_scaffold:
        if verbose:
            print("Computing scaffold features...")
        scaffold = ScaffoldFeatures()
        results['scaffold'] = scaffold.compute_features(smiles_list, verbose=verbose)

    if verbose:
        print("\nFeature dimensions:")
        for name, arr in results.items():
            print(f"  {name}: {arr.shape}")

    return results


if __name__ == "__main__":
    print("Testing Extended Fingerprints")
    print("=" * 60)

    test_smiles = [
        'CCO',
        'CC(=O)Oc1ccccc1C(=O)O',  # Aspirin
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
        'CC(C)Cc1ccc(cc1)C(C)C(=O)O',  # Ibuprofen
    ]

    # Test single fingerprinter
    print("\nTesting ExtendedFingerprinter:")
    fp = ExtendedFingerprinter(verbose=False)
    features = fp.compute_features(test_smiles)
    print(f"Features shape: {features.shape}")
    print(f"Feature dimension: {fp.feature_dim}")

    # Test scaffold features
    print("\nTesting ScaffoldFeatures:")
    scaffold = ScaffoldFeatures()
    for smi in test_smiles:
        scaf = scaffold.get_scaffold(smi)
        print(f"  {smi[:30]:<30} -> {scaf}")

    scaffold_fp = scaffold.compute_features(test_smiles, verbose=False)
    print(f"Scaffold fingerprints shape: {scaffold_fp.shape}")

    # Test compute_all_features
    print("\n" + "=" * 60)
    print("Testing compute_all_features:")
    all_features = compute_all_features(test_smiles, verbose=True)
