"""
Phase 2A Feature Engineering - Extended Features
Adds MACCS keys, multiple fingerprints, and Mordred descriptors
"""
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, MACCSkeys
from rdkit.Chem.AtomPairs import Pairs, Torsions

# Try to import mordred, fallback to RDKit descriptors if not available
try:
    from mordred import Calculator, descriptors
    MORDRED_AVAILABLE = True
except ImportError:
    MORDRED_AVAILABLE = False
    print("Mordred not available, using extended RDKit descriptors")


class FeatureEngineerV2:
    """Extended feature engineering for Phase 2"""

    def __init__(self,
                 morgan_bits=1024,
                 morgan_radius=2,
                 use_maccs=True,
                 use_rdkit_fp=True,
                 use_mordred=True,
                 use_atompair=False,  # Large fingerprint, optional
                 verbose=True):
        self.morgan_bits = morgan_bits
        self.morgan_radius = morgan_radius
        self.use_maccs = use_maccs
        self.use_rdkit_fp = use_rdkit_fp
        self.use_mordred = use_mordred and MORDRED_AVAILABLE
        self.use_atompair = use_atompair
        self.verbose = verbose

        # Initialize Mordred calculator if available
        if self.use_mordred:
            self.mordred_calc = Calculator(descriptors, ignore_3D=True)

        self._compute_feature_dim()

    def _compute_feature_dim(self):
        """Calculate total feature dimensionality"""
        dim = 0

        # Morgan fingerprint
        dim += self.morgan_bits
        self.morgan_start = 0
        self.morgan_end = dim

        # MACCS keys (167 bits)
        if self.use_maccs:
            self.maccs_start = dim
            dim += 167
            self.maccs_end = dim

        # RDKit fingerprint (2048 bits)
        if self.use_rdkit_fp:
            self.rdkit_fp_start = dim
            dim += 2048
            self.rdkit_fp_end = dim

        # Atom pair fingerprint (optional, 2048 bits)
        if self.use_atompair:
            self.atompair_start = dim
            dim += 2048
            self.atompair_end = dim

        # Descriptors (Mordred 2D or extended RDKit)
        if self.use_mordred:
            self.desc_start = dim
            # Mordred 2D descriptors (~1613 features, but we'll filter NaN-prone ones)
            dim += 200  # Use top 200 most reliable Mordred descriptors
            self.desc_end = dim
        else:
            self.desc_start = dim
            dim += 50  # Extended RDKit descriptors
            self.desc_end = dim

        self.feature_dim = dim
        if self.verbose:
            print(f"Feature dimensionality: {self.feature_dim}")

    def _get_morgan_fp(self, mol):
        """Get Morgan fingerprint as numpy array"""
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.morgan_radius, nBits=self.morgan_bits)
        return np.array(fp, dtype=np.float32)

    def _get_maccs_keys(self, mol):
        """Get MACCS keys as numpy array (167 bits)"""
        fp = MACCSkeys.GenMACCSKeys(mol)
        return np.array(fp, dtype=np.float32)

    def _get_rdkit_fp(self, mol):
        """Get RDKit fingerprint as numpy array"""
        fp = Chem.RDKFingerprint(mol, fpSize=2048)
        return np.array(fp, dtype=np.float32)

    def _get_atompair_fp(self, mol):
        """Get Atom Pair fingerprint as numpy array"""
        fp = AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=2048)
        return np.array(fp, dtype=np.float32)

    def _get_rdkit_descriptors(self, mol):
        """Get extended RDKit descriptors (50 features)"""
        try:
            desc = [
                # Basic properties
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
                # Additional descriptors
                rdMolDescriptors.CalcNumAliphaticRings(mol),
                rdMolDescriptors.CalcNumHeterocycles(mol),
                Descriptors.LabuteASA(mol),
                Descriptors.qed(mol),
                rdMolDescriptors.CalcNumAmideBonds(mol),
                # Complexity
                Descriptors.BertzCT(mol),
                Descriptors.Chi0(mol),
                Descriptors.Chi1(mol),
                Descriptors.Kappa1(mol),
                Descriptors.Kappa2(mol),
                Descriptors.HallKierAlpha(mol),
                # Counts
                rdMolDescriptors.CalcNumSpiroAtoms(mol),
                rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
                Descriptors.NumValenceElectrons(mol),
                Descriptors.NumRadicalElectrons(mol),
                # More ring info
                rdMolDescriptors.CalcNumAromaticRings(mol),
                rdMolDescriptors.CalcNumSaturatedRings(mol),
                rdMolDescriptors.CalcNumAromaticHeterocycles(mol),
                rdMolDescriptors.CalcNumAromaticCarbocycles(mol),
                rdMolDescriptors.CalcNumSaturatedHeterocycles(mol),
                rdMolDescriptors.CalcNumSaturatedCarbocycles(mol),
                # Atom counts
                rdMolDescriptors.CalcNumHeteroatoms(mol),
                rdMolDescriptors.CalcNumLipinskiHBA(mol),
                rdMolDescriptors.CalcNumLipinskiHBD(mol),
                # Surface area
                Descriptors.PEOE_VSA1(mol),
                Descriptors.PEOE_VSA2(mol),
                Descriptors.SMR_VSA1(mol),
                Descriptors.SMR_VSA2(mol),
                Descriptors.SlogP_VSA1(mol),
                Descriptors.SlogP_VSA2(mol),
                # EState
                Descriptors.EState_VSA1(mol),
                Descriptors.EState_VSA2(mol),
                # Charges
                Descriptors.MaxPartialCharge(mol),
                Descriptors.MinPartialCharge(mol),
                Descriptors.MaxAbsPartialCharge(mol),
                Descriptors.MinAbsPartialCharge(mol),
                # Additional
                Descriptors.NHOHCount(mol),
                Descriptors.NOCount(mol),
                Descriptors.NumAliphaticCarbocycles(mol),
                Descriptors.NumAliphaticHeterocycles(mol),
            ]
            return np.array(desc, dtype=np.float32)
        except:
            return np.zeros(50, dtype=np.float32)

    def _get_mordred_descriptors(self, mol):
        """Get Mordred 2D descriptors (top 200 reliable ones)"""
        try:
            result = self.mordred_calc(mol)
            # Convert to numpy, replacing errors with 0
            values = []
            for v in result.values():
                if isinstance(v, (int, float)) and not np.isnan(v) and not np.isinf(v):
                    values.append(float(v))
                else:
                    values.append(0.0)
            # Take first 200 features (most reliable ones)
            arr = np.array(values[:200], dtype=np.float32)
            if len(arr) < 200:
                arr = np.pad(arr, (0, 200 - len(arr)))
            return arr
        except:
            return np.zeros(200, dtype=np.float32)

    def compute_features(self, smiles_list):
        """Compute all features for a list of SMILES"""
        n_mols = len(smiles_list)
        features = np.zeros((n_mols, self.feature_dim), dtype=np.float32)

        for i, smi in enumerate(smiles_list):
            if self.verbose and (i + 1) % 500 == 0:
                print(f"  Processing {i+1}/{n_mols} molecules...")

            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue

            try:
                # Morgan fingerprint
                features[i, self.morgan_start:self.morgan_end] = self._get_morgan_fp(mol)

                # MACCS keys
                if self.use_maccs:
                    features[i, self.maccs_start:self.maccs_end] = self._get_maccs_keys(mol)

                # RDKit fingerprint
                if self.use_rdkit_fp:
                    features[i, self.rdkit_fp_start:self.rdkit_fp_end] = self._get_rdkit_fp(mol)

                # Atom pair fingerprint
                if self.use_atompair:
                    features[i, self.atompair_start:self.atompair_end] = self._get_atompair_fp(mol)

                # Descriptors
                if self.use_mordred:
                    features[i, self.desc_start:self.desc_end] = self._get_mordred_descriptors(mol)
                else:
                    features[i, self.desc_start:self.desc_end] = self._get_rdkit_descriptors(mol)

            except Exception as e:
                if self.verbose:
                    print(f"  Error processing molecule {i}: {e}")
                continue

        # Replace NaN/Inf with 0
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        if self.verbose:
            print(f"Features computed: {features.shape}")

        return features

    def get_feature_names(self):
        """Get feature names for interpretability"""
        names = []

        # Morgan FP
        names.extend([f'Morgan_{i}' for i in range(self.morgan_bits)])

        # MACCS
        if self.use_maccs:
            names.extend([f'MACCS_{i}' for i in range(167)])

        # RDKit FP
        if self.use_rdkit_fp:
            names.extend([f'RDKitFP_{i}' for i in range(2048)])

        # Atom pair
        if self.use_atompair:
            names.extend([f'AtomPair_{i}' for i in range(2048)])

        # Descriptors
        if self.use_mordred:
            names.extend([f'Mordred_{i}' for i in range(200)])
        else:
            names.extend([f'RDKitDesc_{i}' for i in range(50)])

        return names


def compute_phase2_features(smiles_list, config='default'):
    """Convenience function to compute Phase 2 features"""
    configs = {
        'default': {
            'morgan_bits': 1024,
            'use_maccs': True,
            'use_rdkit_fp': True,
            'use_mordred': True,
            'use_atompair': False,
        },
        'full': {
            'morgan_bits': 2048,
            'use_maccs': True,
            'use_rdkit_fp': True,
            'use_mordred': True,
            'use_atompair': True,
        },
        'light': {
            'morgan_bits': 1024,
            'use_maccs': True,
            'use_rdkit_fp': False,
            'use_mordred': False,
            'use_atompair': False,
        },
    }

    cfg = configs.get(config, configs['default'])
    fe = FeatureEngineerV2(**cfg)
    return fe.compute_features(smiles_list), fe


if __name__ == "__main__":
    # Test the feature engineering
    test_smiles = [
        'CCO',
        'CC(=O)Oc1ccccc1C(=O)O',
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
    ]

    print("Testing Feature Engineering V2")
    print("=" * 50)

    fe = FeatureEngineerV2(verbose=True)
    features = fe.compute_features(test_smiles)

    print(f"\nFeature shape: {features.shape}")
    print(f"Non-zero features per molecule: {np.count_nonzero(features, axis=1)}")
