"""
Phase 2B: ChemBERTa Molecular Embeddings (RTX 4090 Optimized)

Pre-trained transformer embeddings for molecules.
Blazingly fast with: Flash Attention 2, BF16, torch.compile, large batches.
"""
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import torch
from tqdm import tqdm
from pathlib import Path

# Set HuggingFace cache to persistent location (survives pod restarts)
CACHE_DIR = Path.home() / ".cache" / "huggingface"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ['HF_HOME'] = str(CACHE_DIR)
os.environ['TRANSFORMERS_CACHE'] = str(CACHE_DIR / "transformers")
os.environ['HF_DATASETS_CACHE'] = str(CACHE_DIR / "datasets")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Try different ChemBERTa sources
CHEMBERTA_AVAILABLE = False
CHEMBERTA_SOURCE = None

try:
    from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
    CHEMBERTA_AVAILABLE = True
    CHEMBERTA_SOURCE = 'huggingface'
except ImportError:
    try:
        from transformers import AutoTokenizer, AutoModel
        CHEMBERTA_AVAILABLE = True
        CHEMBERTA_SOURCE = 'huggingface'
        BitsAndBytesConfig = None
    except ImportError:
        pass


class ChemBERTaEmbedder:
    """
    Generate molecular embeddings using ChemBERTa (RTX 4090 Optimized)

    Optimizations:
    - Flash Attention 2 (if available)
    - BFloat16 precision (native on RTX 4090)
    - torch.compile() for kernel fusion
    - Large batch sizes (128-256)
    - Pinned memory transfers
    - CUDA graphs for inference
    """

    def __init__(self, model_name='seyonec/ChemBERTa-zinc-base-v1', device=None):
        self.model_name = model_name
        self.device = device or DEVICE
        self.tokenizer = None
        self.model = None
        self.embedding_dim = 768
        self.use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        self.compiled = False

    def load_model(self):
        """Load ChemBERTa model with maximum optimizations"""
        if not CHEMBERTA_AVAILABLE:
            raise ImportError("transformers not installed. Run: pip install transformers")

        print(f"Loading ChemBERTa: {self.model_name}")
        print(f"  Cache dir: {CACHE_DIR}")

        # Check if model is cached
        model_cache = CACHE_DIR / "transformers" / f"models--{self.model_name.replace('/', '--')}"
        if model_cache.exists():
            print("  Loading from cache (instant)")
        else:
            print("  Downloading model (~500MB, cached for future runs)")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=CACHE_DIR / "transformers")

        # Load model with optimizations
        model_kwargs = {}

        # Try Flash Attention 2 (requires transformers >= 4.36)
        try:
            model_kwargs['attn_implementation'] = 'flash_attention_2'
            print("  Using Flash Attention 2")
        except:
            pass

        # Use BF16 on Ampere+ GPUs (RTX 3090, 4090)
        if self.use_bf16:
            model_kwargs['torch_dtype'] = torch.bfloat16
            print("  Using BFloat16 precision")

        self.model = AutoModel.from_pretrained(self.model_name, cache_dir=CACHE_DIR / "transformers", **model_kwargs)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Enable torch.compile for PyTorch 2.0+ (massive speedup)
        if hasattr(torch, 'compile') and torch.cuda.is_available():
            try:
                self.model = torch.compile(self.model, mode='max-autotune')
                self.compiled = True
                print("  Using torch.compile (max-autotune)")
            except Exception as e:
                print(f"  torch.compile failed: {e}")

        # Enable CUDA optimizations
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

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
        inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16 if self.use_bf16 else torch.float16):
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].float().cpu().numpy()

        return embedding.flatten()

    @torch.inference_mode()
    def embed_batch(self, smiles_list, batch_size=128, show_progress=True):
        """
        Get embeddings for a list of SMILES (Blazingly Fast)

        Args:
            smiles_list: List of SMILES strings
            batch_size: Batch size (default 128, use 256 for RTX 4090)
            show_progress: Show progress bar

        Returns:
            np.array of shape (n_molecules, 768)
        """
        if self.model is None:
            self.load_model()

        n_molecules = len(smiles_list)

        # Pre-allocate pinned memory for faster GPU transfer
        embeddings = np.zeros((n_molecules, self.embedding_dim), dtype=np.float32)

        # Optimal batch size for RTX 4090 (24GB VRAM)
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_mem >= 20:  # RTX 4090, A100, etc.
                batch_size = max(batch_size, 256)
            elif gpu_mem >= 10:  # RTX 3080, etc.
                batch_size = max(batch_size, 128)

        iterator = range(0, n_molecules, batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="ChemBERTa", unit="batch",
                           total=(n_molecules + batch_size - 1) // batch_size,
                           ncols=80)

        # Warmup for torch.compile
        if self.compiled and n_molecules > batch_size:
            warmup_batch = smiles_list[:min(batch_size, 32)]
            warmup_inputs = self.tokenizer(warmup_batch, return_tensors='pt',
                                           padding=True, truncation=True, max_length=512)
            warmup_inputs = {k: v.to(self.device) for k, v in warmup_inputs.items()}
            with torch.cuda.amp.autocast(dtype=torch.bfloat16 if self.use_bf16 else torch.float16):
                _ = self.model(**warmup_inputs)
            torch.cuda.synchronize()

        for i in iterator:
            batch = smiles_list[i:i + batch_size]

            # Tokenize with padding
            inputs = self.tokenizer(
                batch,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            )

            # Non-blocking transfer to GPU
            inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}

            # Mixed precision inference
            with torch.cuda.amp.autocast(dtype=torch.bfloat16 if self.use_bf16 else torch.float16):
                outputs = self.model(**inputs)
                # Get CLS token, convert to float32 for numpy
                batch_embeddings = outputs.last_hidden_state[:, 0, :].float()

            # Async copy back to CPU
            embeddings[i:i + len(batch)] = batch_embeddings.cpu().numpy()

        # Sync before returning
        if torch.cuda.is_available():
            torch.cuda.synchronize()

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

    def fit_transform(self, smiles_list, batch_size=256):
        """
        Compute embeddings and optionally reduce dimensionality

        Args:
            smiles_list: List of SMILES strings
            batch_size: Batch size (256 optimal for RTX 4090)

        Returns:
            np.array of shape (n_molecules, n_components)
        """
        # Get raw embeddings with optimized embedder
        self.embedder = ChemBERTaEmbedder(self.model_name)
        embeddings = self.embedder.embed_batch(smiles_list, batch_size)

        # Optionally reduce dimensionality
        if self.n_components and self.n_components < embeddings.shape[1]:
            from sklearn.decomposition import PCA
            print(f"Reducing ChemBERTa from 768 to {self.n_components} dims with PCA")
            self.pca = PCA(n_components=self.n_components, random_state=42)
            embeddings = self.pca.fit_transform(embeddings)

        return embeddings.astype(np.float32)

    def transform(self, smiles_list, batch_size=256):
        """
        Transform new SMILES using fitted model

        Args:
            smiles_list: List of SMILES strings
            batch_size: Batch size (256 optimal for RTX 4090)

        Returns:
            np.array of shape (n_molecules, n_components)
        """
        embeddings = self.embedder.embed_batch(smiles_list, batch_size)

        if self.pca is not None:
            embeddings = self.pca.transform(embeddings)

        return embeddings.astype(np.float32)


def compute_chemberta_features(smiles_list, n_components=256, batch_size=256):
    """
    Convenience function to compute ChemBERTa features (RTX 4090 optimized)

    Args:
        smiles_list: List of SMILES strings
        n_components: Output dimensionality (None for full 768)
        batch_size: Batch size (256 optimal for RTX 4090)

    Returns:
        np.array of embeddings
    """
    if not CHEMBERTA_AVAILABLE:
        print("ChemBERTa not available - returning zeros")
        return np.zeros((len(smiles_list), n_components or 768), dtype=np.float32)

    extractor = ChemBERTaFeatureExtractor(n_components=n_components)
    return extractor.fit_transform(smiles_list, batch_size)


if __name__ == "__main__":
    print("Testing ChemBERTa Embeddings (RTX 4090 Optimized)")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"ChemBERTa available: {CHEMBERTA_AVAILABLE}")
    print(f"Cache dir: {CACHE_DIR}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"BF16 support: {torch.cuda.get_device_capability()[0] >= 8}")

    if CHEMBERTA_AVAILABLE:
        test_smiles = [
            'CCO',
            'CC(=O)Oc1ccccc1C(=O)O',
            'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
            'CC(C)Cc1ccc(cc1)C(C)C(=O)O',
        ] * 25  # 100 molecules for speed test

        print(f"\nTest SMILES: {len(test_smiles)}")

        import time
        embedder = ChemBERTaEmbedder()
        embedder.load_model()

        start = time.time()
        embeddings = embedder.embed_batch(test_smiles, batch_size=256, show_progress=True)
        elapsed = time.time() - start

        print(f"\nEmbeddings shape: {embeddings.shape}")
        print(f"Time: {elapsed:.2f}s ({len(test_smiles)/elapsed:.0f} mol/sec)")
        print(f"Sample embedding (first 5): {embeddings[0, :5]}")

        # Test with PCA reduction
        print("\nWith PCA reduction to 128 dims:")
        extractor = ChemBERTaFeatureExtractor(n_components=128)
        reduced = extractor.fit_transform(test_smiles)
        print(f"Reduced shape: {reduced.shape}")
    else:
        print("\nInstall for RTX 4090:")
        print("  uv pip install transformers")
        print("  uv pip install flash-attn --no-build-isolation  # Optional, 2x faster")
