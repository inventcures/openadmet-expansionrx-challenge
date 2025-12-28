"""
1D CNN on SMILES - BELKA Competition Winner Technique

Treats SMILES as a character sequence and uses 1D convolutions
to extract molecular features. Simple but surprisingly effective.

Based on NeurIPS 2024 BELKA competition winning solutions.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
from tqdm import tqdm
import gc
import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# SMILES character vocabulary
SMILES_CHARS = [
    'PAD', 'UNK',  # Special tokens
    'C', 'c', 'N', 'n', 'O', 'o', 'S', 's', 'P', 'F', 'Cl', 'Br', 'I',
    'H', 'B', 'Si', 'Se', 'Te', 'As',
    '(', ')', '[', ']', '=', '#', '-', '+', '\\', '/', '@', '.',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    '%',  # For ring numbers > 9
]

CHAR_TO_IDX = {char: idx for idx, char in enumerate(SMILES_CHARS)}
VOCAB_SIZE = len(SMILES_CHARS)


def tokenize_smiles(smiles: str, max_len: int = 150) -> np.ndarray:
    """
    Convert SMILES string to integer tokens.

    Handles multi-character tokens like 'Cl', 'Br', etc.
    """
    tokens = []
    i = 0
    while i < len(smiles):
        # Check for two-character tokens first
        if i + 1 < len(smiles):
            two_char = smiles[i:i+2]
            if two_char in CHAR_TO_IDX:
                tokens.append(CHAR_TO_IDX[two_char])
                i += 2
                continue

        # Single character
        char = smiles[i]
        if char in CHAR_TO_IDX:
            tokens.append(CHAR_TO_IDX[char])
        else:
            tokens.append(CHAR_TO_IDX['UNK'])
        i += 1

    # Pad or truncate
    if len(tokens) < max_len:
        tokens = tokens + [CHAR_TO_IDX['PAD']] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]

    return np.array(tokens, dtype=np.int64)


class SMILESDataset(Dataset):
    """Dataset for SMILES sequences"""

    def __init__(self, smiles_list, targets=None, max_len=150):
        self.smiles = smiles_list
        self.targets = targets
        self.max_len = max_len

        # Pre-tokenize all SMILES
        self.tokens = np.array([tokenize_smiles(s, max_len) for s in smiles_list])

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        x = torch.LongTensor(self.tokens[idx])

        if self.targets is not None:
            y = torch.FloatTensor([self.targets[idx]])
            return x, y
        return x


class SMILES1DCNN(nn.Module):
    """
    1D CNN for SMILES strings.

    Architecture:
    - Embedding layer for character tokens
    - Multiple 1D conv layers with different kernel sizes
    - Global max pooling
    - Dense layers for prediction
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = 64,
        num_filters: int = 128,
        kernel_sizes: list = [3, 5, 7, 9],
        dropout: float = 0.2,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Multiple conv layers with different kernel sizes (parallel)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])

        # Batch normalization for each conv
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(num_filters) for _ in kernel_sizes
        ])

        # Dense layers
        total_filters = num_filters * len(kernel_sizes)
        self.fc1 = nn.Linear(total_filters, hidden_dim)
        self.bn_fc = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch_size, seq_len)
        x = self.embedding(x)  # (batch, seq_len, embed_dim)
        x = x.transpose(1, 2)  # (batch, embed_dim, seq_len)

        # Apply convolutions and pool
        conv_outs = []
        for conv, bn in zip(self.convs, self.bns):
            h = conv(x)  # (batch, num_filters, seq_len)
            h = bn(h)
            h = F.relu(h)
            h = F.adaptive_max_pool1d(h, 1).squeeze(-1)  # (batch, num_filters)
            conv_outs.append(h)

        # Concatenate all conv outputs
        x = torch.cat(conv_outs, dim=1)  # (batch, total_filters)

        # Dense layers
        x = self.dropout(x)
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x.squeeze(-1)


class SMILES1DCNNRegressor:
    """
    Wrapper class for training and prediction with 1D CNN on SMILES.
    """

    def __init__(
        self,
        max_len: int = 150,
        embed_dim: int = 64,
        num_filters: int = 128,
        kernel_sizes: list = None,
        dropout: float = 0.2,
        hidden_dim: int = 256,
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: torch.device = None,
    ):
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes or [3, 5, 7, 9]
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device or DEVICE

        self.model = None
        self.scaler_mean = None
        self.scaler_std = None

    def _create_model(self):
        return SMILES1DCNN(
            vocab_size=VOCAB_SIZE,
            embed_dim=self.embed_dim,
            num_filters=self.num_filters,
            kernel_sizes=self.kernel_sizes,
            dropout=self.dropout,
            hidden_dim=self.hidden_dim,
        ).to(self.device)

    def fit(self, smiles_train, y_train, smiles_val=None, y_val=None, verbose=True):
        """Train the model"""
        # Normalize targets
        self.scaler_mean = np.mean(y_train)
        self.scaler_std = np.std(y_train) + 1e-8
        y_train_scaled = (y_train - self.scaler_mean) / self.scaler_std

        # Create datasets
        train_dataset = SMILESDataset(smiles_train, y_train_scaled, self.max_len)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=0, pin_memory=True, drop_last=True  # Avoid batch size 1 for BatchNorm
        )

        if smiles_val is not None and y_val is not None:
            y_val_scaled = (y_val - self.scaler_mean) / self.scaler_std
            val_dataset = SMILESDataset(smiles_val, y_val_scaled, self.max_len)
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False,
                num_workers=0, pin_memory=True
            )
        else:
            val_loader = None

        # Create model and optimizer
        self.model = self._create_model()
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        # OneCycleLR scheduler
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.lr * 10,
            epochs=self.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
        )

        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0.0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device).squeeze()

                optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device).squeeze()

                        preds = self.model(batch_x)
                        loss = criterion(preds, batch_y)
                        val_loss += loss.item()

                val_loss /= len(val_loader)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    if verbose:
                        print(f"    Early stopping at epoch {epoch+1}")
                    break

            if verbose and (epoch + 1) % 10 == 0:
                msg = f"    Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}"
                if val_loader:
                    msg += f", Val Loss: {val_loss:.4f}"
                print(msg)

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self

    def predict(self, smiles_list):
        """Make predictions"""
        self.model.eval()

        dataset = SMILESDataset(smiles_list, max_len=self.max_len)
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=0, pin_memory=True
        )

        predictions = []

        with torch.no_grad():
            for batch_x in loader:
                if isinstance(batch_x, tuple):
                    batch_x = batch_x[0]
                batch_x = batch_x.to(self.device)
                preds = self.model(batch_x)
                predictions.extend(preds.cpu().numpy())

        # Inverse transform
        predictions = np.array(predictions) * self.scaler_std + self.scaler_mean

        return predictions


class SMILES1DCNNEnsemble:
    """Ensemble of 1D CNN models with different seeds and CV support"""

    def __init__(self, n_models=3, n_folds=5, **kwargs):
        self.n_models = n_models
        self.n_folds = n_folds
        self.kwargs = kwargs
        self.models = []

    def fit(self, smiles_train, y_train, smiles_val=None, y_val=None, verbose=True):
        """Train ensemble of models"""
        self.models = []

        for i in range(self.n_models):
            if verbose:
                print(f"  Training 1D CNN model {i+1}/{self.n_models}")

            # Set different seed for each model
            torch.manual_seed(42 + i)
            np.random.seed(42 + i)

            model = SMILES1DCNNRegressor(**self.kwargs)
            model.fit(smiles_train, y_train, smiles_val, y_val, verbose=verbose)
            self.models.append(model)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return self

    def predict(self, smiles_list):
        """Average predictions from all models"""
        all_preds = [model.predict(smiles_list) for model in self.models]
        return np.mean(all_preds, axis=0)

    def fit_predict(self, smiles_train, y_train, smiles_test, verbose=True):
        """
        Train with CV and return OOF + test predictions.

        Returns:
            oof_pred: np.ndarray of shape (n_train,)
            test_pred: np.ndarray of shape (n_test,)
        """
        n_train = len(smiles_train)
        n_test = len(smiles_test)

        oof_pred = np.zeros(n_train)
        test_preds = []

        smiles_train = np.array(smiles_train)
        y_train = np.array(y_train)

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(kf.split(smiles_train)):
            if verbose:
                print(f"\nFold {fold+1}/{self.n_folds}")

            # Split data
            fold_train_smiles = smiles_train[train_idx].tolist()
            fold_val_smiles = smiles_train[val_idx].tolist()
            fold_train_y = y_train[train_idx]
            fold_val_y = y_train[val_idx]

            # Train models for this fold
            fold_models = []
            for i in range(self.n_models):
                if verbose:
                    print(f"  Training model {i+1}/{self.n_models}")

                torch.manual_seed(42 + fold * 100 + i)
                np.random.seed(42 + fold * 100 + i)

                model = SMILES1DCNNRegressor(**self.kwargs)
                model.fit(fold_train_smiles, fold_train_y, fold_val_smiles, fold_val_y, verbose=False)
                fold_models.append(model)

            # OOF predictions (average across models)
            fold_oof_preds = [m.predict(fold_val_smiles) for m in fold_models]
            oof_pred[val_idx] = np.mean(fold_oof_preds, axis=0)

            # Test predictions (average across models)
            fold_test_preds = [m.predict(smiles_test) for m in fold_models]
            test_preds.append(np.mean(fold_test_preds, axis=0))

            # Cleanup
            del fold_models
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Average test predictions across folds
        test_pred = np.mean(test_preds, axis=0)

        return oof_pred, test_pred


def train_smiles_cnn_cv(smiles, y_dict, n_folds=5, n_models=3, verbose=True):
    """
    Train 1D CNN models with cross-validation.

    Returns OOF predictions and metrics.
    """
    from pathlib import Path

    TARGETS = ['LogD', 'KSOL', 'HLM CLint', 'MLM CLint',
               'Caco-2 Permeability Papp A>B', 'Caco-2 Permeability Efflux',
               'MPPB', 'MBPB', 'MGMB']

    VALID_RANGES = {
        'LogD': (-3.0, 6.0), 'KSOL': (0.001, 350.0),
        'HLM CLint': (0.0, 3000.0), 'MLM CLint': (0.0, 12000.0),
        'Caco-2 Permeability Papp A>B': (0.0, 60.0),
        'Caco-2 Permeability Efflux': (0.2, 120.0),
        'MPPB': (0.0, 100.0), 'MBPB': (0.0, 100.0), 'MGMB': (0.0, 100.0)
    }

    n_samples = len(smiles)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    oof_preds = {t: np.zeros(n_samples) for t in TARGETS}
    results = {}

    for target in TARGETS:
        if verbose:
            print(f"\n{'='*50}")
            print(f"1D CNN: {target}")
            print(f"{'='*50}")

        y = y_dict[target]
        mask = ~np.isnan(y)
        valid_idx = np.where(mask)[0]
        valid_smiles = [smiles[i] for i in valid_idx]
        valid_y = y[mask]

        if verbose:
            print(f"Samples: {len(valid_y)}")

        fold_preds = np.zeros(len(valid_y))

        for fold, (train_idx, val_idx) in enumerate(kf.split(valid_smiles)):
            if verbose:
                print(f"\n  Fold {fold+1}/{n_folds}")

            train_smiles = [valid_smiles[i] for i in train_idx]
            val_smiles_fold = [valid_smiles[i] for i in val_idx]
            train_y = valid_y[train_idx]
            val_y = valid_y[val_idx]

            # Train ensemble
            ensemble = SMILES1DCNNEnsemble(
                n_models=n_models,
                epochs=50,
                batch_size=64,
                lr=1e-3,
                dropout=0.2,
            )
            ensemble.fit(train_smiles, train_y, val_smiles_fold, val_y, verbose=False)

            # Predict
            preds = ensemble.predict(val_smiles_fold)
            preds = np.clip(preds, *VALID_RANGES[target])
            fold_preds[val_idx] = preds

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Store OOF predictions
        oof_preds[target][valid_idx] = fold_preds

        # Calculate metrics
        mae = mean_absolute_error(valid_y, fold_preds)
        rae = mae / np.mean(np.abs(valid_y - np.mean(valid_y)))
        spear = spearmanr(valid_y, fold_preds)[0]

        results[target] = {'MAE': mae, 'RAE': rae, 'Spearman': spear}

        if verbose:
            print(f"\n  {target}: RAE={rae:.4f}, Spearman={spear:.4f}")

    return oof_preds, results


if __name__ == "__main__":
    print("Testing 1D CNN on SMILES")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Vocabulary size: {VOCAB_SIZE}")

    # Test tokenization
    test_smiles = [
        'CCO',
        'CC(=O)Oc1ccccc1C(=O)O',  # Aspirin
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
    ]

    print("\nTokenization test:")
    for smi in test_smiles:
        tokens = tokenize_smiles(smi, max_len=50)
        print(f"  {smi[:30]:<30} -> {tokens[:15]}...")

    # Test model
    print("\nModel architecture test:")
    model = SMILES1DCNN()
    print(model)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable parameters: {n_params:,}")

    # Test forward pass
    batch_tokens = torch.LongTensor([tokenize_smiles(s) for s in test_smiles])
    print(f"\nInput shape: {batch_tokens.shape}")

    model.eval()
    with torch.no_grad():
        output = model(batch_tokens)
    print(f"Output shape: {output.shape}")
    print(f"Output values: {output.numpy()}")

    # Test with synthetic data
    print("\n" + "=" * 60)
    print("Training test with synthetic data")
    print("=" * 60)

    n_train = 100
    train_smiles = test_smiles * 34  # 102 samples
    train_y = np.random.randn(len(train_smiles)) * 2 + 5

    regressor = SMILES1DCNNRegressor(epochs=10, batch_size=16)
    regressor.fit(train_smiles[:80], train_y[:80],
                  train_smiles[80:], train_y[80:], verbose=True)

    preds = regressor.predict(train_smiles[80:])
    print(f"\nPredictions shape: {preds.shape}")
    print(f"Sample predictions: {preds[:5]}")
