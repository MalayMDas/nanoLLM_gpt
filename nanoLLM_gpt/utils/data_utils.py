"""
Data preparation and loading utilities for GPT training.

This module provides comprehensive data handling functionality for GPT model training,
including data acquisition, tokenization, splitting, and efficient batch loading.

## Key Components:
1. **DataPreparer**: Downloads/loads text data and converts to tokenized binary format
2. **DataLoader**: Efficiently loads tokenized data during training with memory mapping

## Data Flow:
1. Raw text (file/URL) → DataPreparer.prepare_data()
2. Text → Tokenization (tiktoken) → Binary format (.bin files)
3. Binary files → DataLoader → Training batches

## Features:
- Automatic data downloading from URLs
- Efficient tokenization with tiktoken (GPT-2 compatible)
- Memory-mapped data loading for large datasets
- Automatic train/validation splitting
- Support for custom and predefined datasets (tiny shakespeare, openwebtext)
- Metadata tracking (vocab size, token counts)

## File Structure:
```
data_dir/
├── train.bin      # Training tokens (uint16 array)
├── val.bin        # Validation tokens (uint16 array)
└── meta.pkl       # Metadata (vocab_size, tokenizer info)
```

## Usage Example:
```python
# Prepare data
preparer = DataPreparer()
data_dir = preparer.prepare_data(
    data_path='input.txt',
    dataset_name='my_dataset',
    train_val_split=0.1
)

# Load batches during training
loader = DataLoader(
    data_dir=data_dir,
    block_size=1024,
    batch_size=12,
    device='cuda',
    device_type='cuda'
)

x, y = loader.get_batch('train')
```
"""

import os
import pickle
import requests
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from tqdm import tqdm

import numpy as np
import torch
import tiktoken


class DataPreparer:
    """
    Handles data downloading, tokenization, and preparation.

    This class manages the entire data preparation pipeline from raw text
    to tokenized binary files ready for training. It supports multiple
    data sources and handles large datasets efficiently.

    The tokenization uses tiktoken (OpenAI's BPE tokenizer) which is
    compatible with GPT-2 and provides efficient encoding/decoding.

    Attributes:
        tokenizer: Tiktoken encoder instance for text tokenization
        num_proc (int): Number of processes for parallel processing

    Methods:
        prepare_data(): Main entry point for data preparation
        _load_text_data(): Load text from file or URL
        _tokenize_and_save(): Tokenize and save as binary
        _prepare_openwebtext(): Special handler for OpenWebText dataset

    Output Format:
        Creates train.bin and val.bin files containing uint16 token arrays
        that can be efficiently memory-mapped during training.

    Called by:
        - train.Trainer.setup_data() during training initialization
        - Standalone scripts for data preprocessing
    """

    def __init__(self, tokenizer_name: str = "gpt2", num_proc: int = 8):
        """
        Initialize data preparer with tokenizer.

        Creates a tiktoken encoder for the specified tokenizer. The encoder
        handles text-to-token conversion using byte-pair encoding (BPE).

        Args:
            tokenizer_name (str): Name of tokenizer to use. Options:
                - 'gpt2': Standard GPT-2 tokenizer (50257 tokens)
                - 'r50k_base': Base tokenizer (50257 tokens)
                - 'p50k_base': Codex tokenizer (50281 tokens)
                - 'cl100k_base': GPT-4 tokenizer (100256 tokens)
            num_proc (int): Number of processes for parallel data processing
                           (used for datasets like OpenWebText)

        Raises:
            ValueError: If tokenizer_name is not recognized by tiktoken

        Example:
            >>> preparer = DataPreparer(tokenizer_name='gpt2')
            >>> preparer.tokenizer.encode("Hello world")
            [15496, 995]
        """
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        self.num_proc = num_proc

    def prepare_data(
        self,
        data_path: Optional[str] = None,
        dataset_name: str = "custom",
        output_dir: str = "data",
        train_val_split: float = 0.0005,
    ) -> str:
        """
        Prepare data for training by downloading/loading and tokenizing.

        Main entry point that orchestrates the entire data preparation pipeline.
        Checks for existing prepared data to avoid redundant processing.

        Args:
            data_path (Optional[str]): Path or URL to text data
                - File path: Local text file (UTF-8 encoded)
                - URL: HTTP/HTTPS URL to download text from
                - None: Uses tiny shakespeare dataset as default
            dataset_name (str): Name for organizing output files
                - 'custom': For user-provided data
                - 'openwebtext': Triggers special OpenWebText handling
                - Any string: Creates subdirectory with this name
            output_dir (str): Base directory for saving processed data
            train_val_split (float): Fraction of data for validation
                - E.g., 0.1 means 10% validation, 90% training
                - Default 0.0005 suitable for large datasets

        Returns:
            str: Path to data directory containing:
                - train.bin: Training tokens
                - val.bin: Validation tokens
                - meta.pkl: Metadata (vocab_size, etc.)

        Raises:
            RuntimeError: If data loading or tokenization fails

        Process Flow:
            1. Check if dataset already prepared (skip if exists)
            2. Load raw text from source
            3. Tokenize using tiktoken
            4. Split into train/validation
            5. Save as binary files

        Called by:
            - train.Trainer.setup_data() during training setup

        Example:
            >>> preparer = DataPreparer()
            >>> data_dir = preparer.prepare_data(
            ...     data_path='corpus.txt',
            ...     dataset_name='my_corpus',
            ...     train_val_split=0.1
            ... )
            >>> print(data_dir)  # 'data/my_corpus'
        """
        # Handle pre-existing datasets
        if dataset_name != "custom" and not data_path:
            data_dir = os.path.join(output_dir, dataset_name)
            if os.path.exists(os.path.join(data_dir, "train.bin")):
                print(f"Using existing dataset: {dataset_name}")
                return data_dir
            elif dataset_name == "openwebtext":
                return self._prepare_openwebtext(output_dir)

        # Load raw text data
        if data_path:
            text_data = self._load_text_data(data_path)
        else:
            # Default to tiny shakespeare
            print("No data path provided, using tiny shakespeare dataset")
            text_data = self._load_text_data(
                "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            )

        # Create data directory
        data_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(data_dir, exist_ok=True)

        # Tokenize and save
        self._tokenize_and_save(text_data, data_dir, train_val_split)

        return data_dir

    def _load_text_data(self, path: str) -> str:
        """
        Load text data from file path or URL.

        Handles both local files and remote URLs with appropriate error handling
        and progress reporting. For URLs, includes timeout and retry logic.

        Args:
            path (str): Source path
                - Local file: Absolute or relative path
                - URL: Must start with http:// or https://

        Returns:
            str: Loaded text content (UTF-8 decoded)

        Raises:
            RuntimeError: If loading fails (network error, file not found, etc.)

        Features:
            - 5-minute timeout for URL downloads
            - UTF-8 encoding for file reading
            - Progress reporting (character count)
            - Detailed error messages

        Called by:
            - prepare_data() for loading raw text

        Calls:
            - requests.get() for URL downloads
            - Built-in open() for file reading
        """
        if path.startswith(("http://", "https://")):
            print(f"Downloading data from {path}")
            try:
                response = requests.get(path, timeout=300)  # 5 minute timeout
                response.raise_for_status()
                print(f"Downloaded {len(response.text):,} characters")
                return response.text
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"Failed to download data from {path}: {e}")
        else:
            print(f"Loading data from {path}")
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                print(f"Loaded {len(text):,} characters")
                return text
            except IOError as e:
                raise RuntimeError(f"Failed to load data from {path}: {e}")

    def _tokenize_and_save(self, text: str, data_dir: str, train_val_split: float):
        """
        Tokenize text and save as binary files.

        Converts raw text to token IDs and saves as efficient binary format.
        Appends EOT (end-of-text) token and splits into train/validation sets.

        Args:
            text (str): Raw text to tokenize
            data_dir (str): Directory to save output files
            train_val_split (float): Validation fraction (0.0 to 1.0)

        File Format:
            - Binary files contain uint16 arrays (2 bytes per token)
            - Tokens are stored sequentially with no delimiters
            - EOT token appended to mark text boundary

        Process:
            1. Tokenize entire text using tiktoken
            2. Append EOT token for proper sequence ending
            3. Convert to numpy uint16 array (saves space)
            4. Split based on train_val_split ratio
            5. Save train/val splits as .bin files
            6. Save metadata (vocab_size, token counts)

        Output Files:
            - train.bin: Training tokens (binary uint16)
            - val.bin: Validation tokens (binary uint16)
            - meta.pkl: Metadata dictionary with:
                - vocab_size: Maximum token value + 1
                - train_tokens: Number of training tokens
                - val_tokens: Number of validation tokens
                - tokenizer: Tokenizer name

        Called by:
            - prepare_data() after loading text

        Calls:
            - tokenizer.encode_ordinary(): Convert text to tokens
            - numpy.array.tofile(): Save binary data
            - pickle.dump(): Save metadata
        """
        print("Tokenizing data...")

        # Tokenize entire text
        tokens = self.tokenizer.encode_ordinary(text)
        tokens.append(self.tokenizer.eot_token)

        # Convert to numpy array
        tokens = np.array(tokens, dtype=np.uint16)

        # Split into train and validation
        n = len(tokens)
        train_n = int(n * (1 - train_val_split))

        train_tokens = tokens[:train_n]
        val_tokens = tokens[train_n:]

        # Save train split
        train_path = os.path.join(data_dir, "train.bin")
        print(f"Saving {len(train_tokens):,} training tokens to {train_path}")
        train_tokens.tofile(train_path)

        # Save validation split
        val_path = os.path.join(data_dir, "val.bin")
        print(f"Saving {len(val_tokens):,} validation tokens to {val_path}")
        val_tokens.tofile(val_path)

        # Save metadata
        meta = {
            "vocab_size": self.tokenizer.max_token_value + 1,
            "train_tokens": len(train_tokens),
            "val_tokens": len(val_tokens),
            "tokenizer": self.tokenizer.name,
        }
        meta_path = os.path.join(data_dir, "meta.pkl")
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)

        print(f"Data preparation complete. Vocab size: {meta['vocab_size']}")

    def _prepare_openwebtext(self, output_dir: str) -> str:
        """
        Prepare OpenWebText dataset using HuggingFace datasets.

        Special handler for the OpenWebText dataset, a large corpus of
        web text (40GB) commonly used for training language models.
        Uses HuggingFace datasets library for efficient processing.

        Args:
            output_dir (str): Base directory for output

        Returns:
            str: Path to openwebtext data directory

        Raises:
            ImportError: If datasets library not installed

        Features:
            - Automatic download and caching via HuggingFace
            - Parallel tokenization using multiple processes
            - Memory-mapped output for handling large data
            - Progress bars for long-running operations

        Dataset Info:
            - Size: ~40GB raw text, ~17B tokens
            - Source: Reddit submissions with 3+ karma
            - Split: 99.95% train, 0.05% validation

        Memory Requirements:
            - Downloads ~40GB compressed data
            - Requires ~80GB disk space for processing
            - Uses memory mapping to avoid RAM limitations

        Called by:
            - prepare_data() when dataset_name='openwebtext'

        Calls:
            - datasets.load_dataset(): Download/load data
            - Dataset.map(): Parallel tokenization
            - numpy.memmap(): Memory-mapped file writing
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")

        data_dir = os.path.join(output_dir, "openwebtext")
        os.makedirs(data_dir, exist_ok=True)

        # Check if already prepared
        if os.path.exists(os.path.join(data_dir, "train.bin")):
            print("OpenWebText dataset already prepared")
            return data_dir

        print("Preparing OpenWebText dataset...")

        # Load dataset
        dataset = load_dataset("openwebtext", num_proc=self.num_proc)

        # Create train/val split
        split_dataset = dataset["train"].train_test_split(
            test_size=0.0005, seed=2357, shuffle=True  # Small validation set
        )
        split_dataset["val"] = split_dataset.pop("test")

        # Tokenize function
        def process(example):
            ids = self.tokenizer.encode_ordinary(example["text"])
            ids.append(self.tokenizer.eot_token)
            return {"ids": ids, "len": len(ids)}

        # Tokenize dataset
        tokenized = split_dataset.map(
            process,
            remove_columns=["text"],
            desc="tokenizing the splits",
            num_proc=self.num_proc,
        )

        # Save tokenized data
        for split, dset in tokenized.items():
            arr_len = np.sum(dset["len"], dtype=np.uint64)
            filename = os.path.join(data_dir, f"{split}.bin")
            dtype = np.uint16
            arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))

            total_batches = 1024
            idx = 0

            for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
                batch = dset.shard(
                    num_shards=total_batches, index=batch_idx, contiguous=True
                ).with_format("numpy")
                arr_batch = np.concatenate(batch["ids"])
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)

            arr.flush()

        # Save metadata
        meta = {
            "vocab_size": self.tokenizer.max_token_value + 1,
            "tokenizer": self.tokenizer.name,
        }
        meta_path = os.path.join(data_dir, "meta.pkl")
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)

        print("OpenWebText preparation complete")
        return data_dir


class DataLoader:
    """
    Efficient data loader for pre-tokenized binary files.

    Provides memory-mapped access to tokenized data with random sampling
    for training batches. Designed for minimal memory usage even with
    very large datasets.

    The loader uses numpy memory mapping to access data without loading
    entire files into RAM, enabling training on datasets larger than
    available memory.

    Attributes:
        data_dir (Path): Directory containing data files
        block_size (int): Sequence length for each sample
        batch_size (int): Number of sequences per batch
        device (str): Target device ('cuda:0', 'cpu', etc.)
        device_type (str): Device type ('cuda' or 'cpu')
        meta (dict): Metadata loaded from meta.pkl

    Methods:
        get_batch(): Load a random batch for training
        get_vocab_size(): Return vocabulary size from metadata
        get_data_stats(): Return statistics about the dataset

    Data Format:
        Expects .bin files containing uint16 token arrays created
        by DataPreparer. Uses memory mapping for efficient access.

    Called by:
        - train.Trainer during training loop
        - Evaluation scripts needing data access
    """

    def __init__(
        self,
        data_dir: str,
        block_size: int,
        batch_size: int,
        device: str,
        device_type: str,
    ):
        """
        Initialize data loader with configuration.

        Sets up memory-mapped access to tokenized data files and validates
        that files exist with sufficient data for the requested block size.

        Args:
            data_dir (str): Directory containing prepared data files
                Must contain train.bin and val.bin from DataPreparer
            block_size (int): Context length for each training sequence
                Each sample will be block_size tokens
            batch_size (int): Number of sequences per batch
                Total tokens per batch = batch_size * block_size
            device (str): PyTorch device string
                Examples: 'cuda', 'cuda:0', 'cpu', 'mps'
            device_type (str): Device category for optimizations
                Either 'cuda' or 'cpu' (mps maps to 'cpu')

        Raises:
            FileNotFoundError: If train.bin or val.bin not found
            ValueError: If data files too small for block_size

        Validation:
            - Checks both train.bin and val.bin exist
            - Ensures each file has >= block_size + 1 tokens
            - Loads metadata if meta.pkl exists

        Memory Usage:
            Only loads batch_size * block_size tokens at a time,
            regardless of total dataset size.
        """
        self.data_dir = Path(data_dir)
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.device_type = device_type

        # Verify data files exist
        for split in ["train", "val"]:
            data_file = self.data_dir / f"{split}.bin"
            if not data_file.exists():
                raise FileNotFoundError(f"Data file not found: {data_file}")

            # Check if file has enough tokens for at least one sequence
            file_size = data_file.stat().st_size
            num_tokens = file_size // 2  # uint16 = 2 bytes per token
            if num_tokens < block_size + 1:  # Need block_size + 1 for targets
                raise ValueError(
                    f"{split} data has only {num_tokens} tokens, but needs at least "
                    f"{block_size + 1} tokens (block_size + 1). "
                    f"Either reduce block_size or increase the amount of {split} data."
                )

        # Load metadata if available
        meta_path = self.data_dir / "meta.pkl"
        if meta_path.exists():
            with open(meta_path, "rb") as f:
                self.meta = pickle.load(f)
        else:
            self.meta = {}

    def get_batch(self, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load a random batch of sequences for training or validation.

        Samples random positions from the data and extracts sequences
        for next-token prediction. Uses memory mapping for efficiency.

        Args:
            split (str): Data split to load from
                - 'train': Training data (random positions)
                - 'val': Validation data (random positions)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - x: Input sequences (batch_size, block_size)
                     Contains tokens [i, i+1, ..., i+block_size-1]
                - y: Target sequences (batch_size, block_size)
                     Contains tokens [i+1, i+2, ..., i+block_size]

        Raises:
            ValueError: If insufficient data for block_size

        Memory Optimization:
            - Uses memory mapping to avoid loading entire dataset
            - Employs pinned memory for async GPU transfer
            - Non-blocking transfer when using CUDA

        Sampling Strategy:
            - Randomly samples batch_size positions
            - Each position must have block_size + 1 tokens available
            - Positions are independent (may overlap)

        Called by:
            - train.Trainer.train_step() during training
            - train.Trainer.estimate_loss() during evaluation
            - generate.py for getting initial context

        Example:
            >>> loader = DataLoader(data_dir, block_size=1024, batch_size=4, ...)
            >>> x, y = loader.get_batch('train')
            >>> assert x.shape == (4, 1024)
            >>> assert y[0, 0] == x[0, 1]  # y is x shifted by 1
        """
        # Load data as memory-mapped array
        data_file = self.data_dir / f"{split}.bin"
        data = np.memmap(data_file, dtype=np.uint16, mode="r")

        # Ensure we have enough data
        if len(data) <= self.block_size:
            raise ValueError(
                f"{split} data has only {len(data)} tokens, but needs at least "
                f"{self.block_size + 1} tokens for block_size={self.block_size}. "
                f"This error often occurs when validation split is too small. "
                f"Try increasing train_val_split ratio or reducing block_size."
            )

        # Sample random starting positions
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))

        # Extract sequences
        x = torch.stack(
            [
                torch.from_numpy((data[i : i + self.block_size]).astype(np.int64))
                for i in ix
            ]
        )
        y = torch.stack(
            [
                torch.from_numpy(
                    (data[i + 1 : i + 1 + self.block_size]).astype(np.int64)
                )
                for i in ix
            ]
        )

        # Move to device with pinned memory for async transfer
        if self.device_type == "cuda":
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)

        return x, y

    def get_vocab_size(self) -> Optional[int]:
        """
        Get vocabulary size from metadata if available.

        Returns the vocabulary size stored during data preparation.
        This is used to set the model's vocab_size parameter.

        Returns:
            Optional[int]: Vocabulary size (max token value + 1)
                          Returns None if metadata not found

        Called by:
            - train.Trainer.setup_data() to configure model
        """
        return self.meta.get("vocab_size")

    def get_data_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the loaded dataset.

        Provides information about data sizes, token counts, and metadata
        for monitoring and debugging purposes.

        Returns:
            Dict[str, Any]: Statistics dictionary containing:
                - 'train': Training data stats
                    - 'file_size_mb': Size in megabytes
                    - 'num_tokens': Total token count
                    - 'num_tokens_millions': Tokens in millions
                - 'val': Validation data stats (same structure)
                - 'metadata': Original metadata from preparation

        Usage:
            Helpful for verifying data preparation and understanding
            training dynamics (tokens per epoch, etc.)

        Called by:
            - Training scripts for logging
            - Debugging utilities

        Example:
            >>> stats = loader.get_data_stats()
            >>> print(f"Training on {stats['train']['num_tokens_millions']:.1f}M tokens")
        """
        stats = {}

        for split in ["train", "val"]:
            data_file = self.data_dir / f"{split}.bin"
            if data_file.exists():
                file_size = data_file.stat().st_size
                num_tokens = file_size // 2  # uint16 = 2 bytes
                stats[split] = {
                    "file_size_mb": file_size / 1024 / 1024,
                    "num_tokens": num_tokens,
                    "num_tokens_millions": num_tokens / 1e6,
                }

        if self.meta:
            stats["metadata"] = self.meta

        return stats
