"""
Download OpenADMET ExpansionRx Challenge Data from HuggingFace
"""

import os
import pandas as pd
from pathlib import Path

# Set data directory
DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def download_training_data():
    """Download training dataset"""
    print("Downloading training data...")

    # Main training data (cleaned)
    train_url = "hf://datasets/openadmet/openadmet-expansionrx-challenge-train-data/expansion_data_train.csv"
    df_train = pd.read_csv(train_url)

    output_path = DATA_DIR / "train.csv"
    df_train.to_csv(output_path, index=False)
    print(f"Training data saved to {output_path}")
    print(f"Shape: {df_train.shape}")
    print(f"Columns: {list(df_train.columns)}")

    return df_train

def download_training_data_raw():
    """Download raw training dataset (includes out-of-bounds measurements)"""
    print("\nDownloading raw training data...")

    raw_url = "hf://datasets/openadmet/openadmet-expansionrx-challenge-train-data/expansion_data_train_raw.csv"
    df_raw = pd.read_csv(raw_url)

    output_path = DATA_DIR / "train_raw.csv"
    df_raw.to_csv(output_path, index=False)
    print(f"Raw training data saved to {output_path}")
    print(f"Shape: {df_raw.shape}")

    return df_raw

def download_test_data():
    """Download blinded test dataset"""
    print("\nDownloading blinded test data...")

    try:
        from datasets import load_dataset
        ds = load_dataset("openadmet/openadmet-expansionrx-challenge-test-data-blinded")
        df_test = ds["test"].to_pandas()
    except:
        # Fallback to direct URL
        test_url = "hf://datasets/openadmet/openadmet-expansionrx-challenge-test-data-blinded/expansion_data_test_blinded.csv"
        df_test = pd.read_csv(test_url)

    output_path = DATA_DIR / "test_blinded.csv"
    df_test.to_csv(output_path, index=False)
    print(f"Test data saved to {output_path}")
    print(f"Shape: {df_test.shape}")
    print(f"Columns: {list(df_test.columns)}")

    return df_test

if __name__ == "__main__":
    print("=" * 60)
    print("OpenADMET ExpansionRx Challenge - Data Download")
    print("=" * 60)

    df_train = download_training_data()
    df_raw = download_training_data_raw()
    df_test = download_test_data()

    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)

    # Summary
    print("\nDataset Summary:")
    print(f"  Training samples: {len(df_train)}")
    print(f"  Raw training samples: {len(df_raw)}")
    print(f"  Test samples: {len(df_test)}")
