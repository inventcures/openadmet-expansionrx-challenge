"""
Phase 2C: Uni-Mol 3D Conformer Features (RTX 4090 Optimized)

Extracts 3D molecular representations using Uni-Mol pretrained model.
Requires: unimol_tools (pip install unimol_tools)
"""
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import torch
from rdkit import Chem
from rdkit.Chem import AllChem
import gc

DEVICE = torch.device('cuda')  # RTX 4090

# Try to import Uni-Mol
UNIMOL_AVAILABLE = False
try:
    from unimol_tools import UniMolRepr
    UNIMOL_AVAILABLE = True
except ImportError:
    try:
        # Alternative import path
        from unimol import UniMolModel
        UNIMOL_AVAILABLE = True
    except ImportError:
        pass


class UniMolEmbedder:
    """
    Generate 3D molecular embeddings using Uni-Mol

    Uni-Mol is pretrained on 209M conformers and captures 3D structure.
    Produces 512-dimensional embeddings per molecule.
    """

    def __init__(self, device='cuda'):
        self.device = device
        self.model = None
        self.embedding_dim = 512

    def load_model(self):
        """Load Uni-Mol model"""
        if not UNIMOL_AVAILABLE:
            raise ImportError("unimol_tools not installed. Run: pip install unimol_tools")

        print("Loading Uni-Mol model...")
        self.model = UniMolRepr(data_type='molecule', remove_hs=False)
        print(f"Uni-Mol loaded on {self.device}")
        return self

    def embed_batch(self, smiles_list, batch_size=32, show_progress=True):
        """
        Get 3D embeddings for SMILES list

        Args:
            smiles_list: List of SMILES strings
            batch_size: Batch size for GPU processing
            show_progress: Print progress

        Returns:
            np.array of shape (n_molecules, 512)
        """
        if self.model is None:
            self.load_model()

        n_molecules = len(smiles_list)
        embeddings = np.zeros((n_molecules, self.embedding_dim), dtype=np.float32)

        for i in range(0, n_molecules, batch_size):
            if show_progress and (i // batch_size) % 10 == 0:
                print(f"  Uni-Mol: {i}/{n_molecules}")

            batch = smiles_list[i:i + batch_size]

            try:
                # Uni-Mol expects list of SMILES
                reprs = self.model.get_repr(batch, return_atomic_reprs=False)
                # reprs['cls_repr'] is the molecular representation
                batch_emb = reprs['cls_repr']
                embeddings[i:i + len(batch)] = batch_emb
            except Exception as e:
                print(f"  Warning: Batch {i} failed: {e}")
                # Fill with zeros for failed molecules
                continue

            # Clear GPU memory periodically
            if i % (batch_size * 10) == 0:
                torch.cuda.empty_cache()

        return embeddings


class Conformer3DFeatures:
    """
    Generate 3D conformer-based features using RDKit

    Fallback when Uni-Mol is not available.
    Computes 3D descriptors from generated conformers.
    """

    def __init__(self, n_conformers=1):
        self.n_conformers = n_conformers
        self.feature_dim = 20  # Number of 3D descriptors

    def _generate_conformer(self, mol):
        """Generate 3D conformer for a molecule"""
        mol = Chem.AddHs(mol)
        try:
            # Use ETKDG for conformer generation
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            result = AllChem.EmbedMolecule(mol, params)
            if result == 0:
                AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
                return mol
        except:
            pass
        return None

    def _compute_3d_descriptors(self, mol):
        """Compute 3D descriptors from conformer"""
        try:
            from rdkit.Chem import Descriptors3D, rdMolDescriptors

            desc = [
                Descriptors3D.Asphericity(mol),
                Descriptors3D.Eccentricity(mol),
                Descriptors3D.InertialShapeFactor(mol),
                Descriptors3D.NPR1(mol),
                Descriptors3D.NPR2(mol),
                Descriptors3D.PMI1(mol),
                Descriptors3D.PMI2(mol),
                Descriptors3D.PMI3(mol),
                Descriptors3D.RadiusOfGyration(mol),
                Descriptors3D.SpherocityIndex(mol),
                rdMolDescriptors.CalcPBF(mol),
                # Plane of best fit
            ]
            # Pad to feature_dim
            desc = desc[:self.feature_dim]
            while len(desc) < self.feature_dim:
                desc.append(0.0)

            return np.array(desc, dtype=np.float32)
        except:
            return np.zeros(self.feature_dim, dtype=np.float32)

    def compute_features(self, smiles_list, show_progress=True):
        """Compute 3D features for SMILES list"""
        n_molecules = len(smiles_list)
        features = np.zeros((n_molecules, self.feature_dim), dtype=np.float32)

        for i, smi in enumerate(smiles_list):
            if show_progress and (i + 1) % 500 == 0:
                print(f"  3D features: {i+1}/{n_molecules}")

            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue

            mol_3d = self._generate_conformer(mol)
            if mol_3d is not None:
                features[i] = self._compute_3d_descriptors(mol_3d)

        return features


def compute_3d_features(smiles_list, use_unimol=True, batch_size=32):
    """
    Compute 3D molecular features

    Uses Uni-Mol if available, otherwise falls back to RDKit 3D descriptors.

    Args:
        smiles_list: List of SMILES strings
        use_unimol: Try to use Uni-Mol (requires GPU)
        batch_size: Batch size for processing

    Returns:
        np.array of 3D features
    """
    if use_unimol and UNIMOL_AVAILABLE:
        print("Using Uni-Mol for 3D features (512 dims)")
        embedder = UniMolEmbedder()
        return embedder.embed_batch(smiles_list, batch_size)
    else:
        print("Using RDKit 3D descriptors (20 dims)")
        extractor = Conformer3DFeatures()
        return extractor.compute_features(smiles_list)


if __name__ == "__main__":
    print("Testing 3D Features (RTX 4090)")
    print("=" * 50)
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Uni-Mol available: {UNIMOL_AVAILABLE}")

    test_smiles = [
        'CCO',
        'CC(=O)Oc1ccccc1C(=O)O',
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
        'CC(C)Cc1ccc(cc1)C(C)C(=O)O',
    ]

    print(f"\nTest SMILES: {len(test_smiles)}")

    features = compute_3d_features(test_smiles, use_unimol=UNIMOL_AVAILABLE)
    print(f"Features shape: {features.shape}")
    print(f"Sample features: {features[0, :10]}")
