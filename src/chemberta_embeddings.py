"""
Phase 2B: ChemBERTa Molecular Embeddings

Pre-trained transformer embeddings for molecules.
Uses DeepChem's ChemBERTa or HuggingFace transformers.
"""
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import torch
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Try different ChemBERTa sources
CHEMBERTA_AVAILABLE = False
CHEMBERTA_SOURCE = None

try:
    from transformers import AutoTokenizer, AutoModel
    CHEMBERTA_AVAILABLE = True
    CHEMBERTA_SOURCE = 'huggingface'
except ImportError:
    pass


class ChemBERTaEmbedder:
    """
    Generate molecular embeddings using ChemBERTa

    ChemBERTa is a BERT-like model pre-trained on ~77M SMILES strings.
    Produces 768-dimensional embeddings per molecule.
    """

    def __init__(self, model_name='seyonec/ChemBERTa-zinc-base-v1', device=None):
        self.model_name = model_name
        self.device = device or DEVICE
        self.tokenizer = None
        self.model = None
        self.embedding_dim = 768

    def load_model(self):
        """Load ChemBERTa model and tokenizer"""
        if not CHEMBERTA_AVAILABLE:
            raise ImportError("transformers not installed. Run: pip install transformers")

        print(f"Loading ChemBERTa: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"ChemBERTa loaded on {self.device}")

        return self

    def embed_single(self, smiles):
        """Get embedding for a single SMILES"""
        inputs = self.tokenizer(
            smiles,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding (first token)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        return embedding.flatten()

    def embed_batch(self, smiles_list, batch_size=32, show_progress=True):
        """
        Get embeddings for a list of SMILES

        Args:
            smiles_list: List of SMILES strings
            batch_size: Batch size for processing
            show_progress: Show progress bar

        Returns:
            np.array of shape (n_molecules, 768)
        """
        if self.model is None:
            self.load_model()

        n_molecules = len(smiles_list)
        embeddings = np.zeros((n_molecules, self.embedding_dim), dtype=np.float32)

        iterator = range(0, n_molecules, batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="ChemBERTa embeddings")

        for i in iterator:
            batch = smiles_list[i:i + batch_size]

            inputs = self.tokenizer(
                batch,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            embeddings[i:i + len(batch)] = batch_embeddings

        return embeddings


class ChemBERTaFeatureExtractor:
    """
    Extract ChemBERTa features with optional dimensionality reduction
    """

    def __init__(self, n_components=256, model_name='seyonec/ChemBERTa-zinc-base-v1'):
        self.n_components = n_components
        self.model_name = model_name
        self.embedder = None
        self.pca = None

    def fit_transform(self, smiles_list, batch_size=32):
        """
        Compute embeddings and optionally reduce dimensionality

        Args:
            smiles_list: List of SMILES strings
            batch_size: Batch size

        Returns:
            np.array of shape (n_molecules, n_components)
        """
        # Get raw embeddings
        self.embedder = ChemBERTaEmbedder(self.model_name)
        embeddings = self.embedder.embed_batch(smiles_list, batch_size)

        # Optionally reduce dimensionality
        if self.n_components and self.n_components < embeddings.shape[1]:
            from sklearn.decomposition import PCA
            print(f"Reducing ChemBERTa from 768 to {self.n_components} dims with PCA")
            self.pca = PCA(n_components=self.n_components, random_state=42)
            embeddings = self.pca.fit_transform(embeddings)

        return embeddings.astype(np.float32)

    def transform(self, smiles_list, batch_size=32):
        """
        Transform new SMILES using fitted model

        Args:
            smiles_list: List of SMILES strings
            batch_size: Batch size

        Returns:
            np.array of shape (n_molecules, n_components)
        """
        embeddings = self.embedder.embed_batch(smiles_list, batch_size)

        if self.pca is not None:
            embeddings = self.pca.transform(embeddings)

        return embeddings.astype(np.float32)


def compute_chemberta_features(smiles_list, n_components=256, batch_size=32):
    """
    Convenience function to compute ChemBERTa features

    Args:
        smiles_list: List of SMILES strings
        n_components: Output dimensionality (None for full 768)
        batch_size: Batch size for processing

    Returns:
        np.array of embeddings
    """
    if not CHEMBERTA_AVAILABLE:
        print("ChemBERTa not available - returning zeros")
        return np.zeros((len(smiles_list), n_components or 768), dtype=np.float32)

    extractor = ChemBERTaFeatureExtractor(n_components=n_components)
    return extractor.fit_transform(smiles_list, batch_size)


if __name__ == "__main__":
    print("Testing ChemBERTa Embeddings")
    print("=" * 50)
    print(f"Device: {DEVICE}")
    print(f"ChemBERTa available: {CHEMBERTA_AVAILABLE}")

    if CHEMBERTA_AVAILABLE:
        test_smiles = [
            'CCO',
            'CC(=O)Oc1ccccc1C(=O)O',
            'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
            'CC(C)Cc1ccc(cc1)C(C)C(=O)O',
        ]

        print(f"\nTest SMILES: {len(test_smiles)}")

        embedder = ChemBERTaEmbedder()
        embedder.load_model()

        embeddings = embedder.embed_batch(test_smiles, show_progress=False)
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Sample embedding (first 10): {embeddings[0, :10]}")

        # Test with PCA reduction
        print("\nWith PCA reduction to 128 dims:")
        extractor = ChemBERTaFeatureExtractor(n_components=128)
        reduced = extractor.fit_transform(test_smiles)
        print(f"Reduced shape: {reduced.shape}")
    else:
        print("Install transformers: pip install transformers")
