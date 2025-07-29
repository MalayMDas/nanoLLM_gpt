"""
Centralized model loading utilities for the GPT project.

This module provides a unified interface for model management, handling all
aspects of loading, creating, and configuring GPT models from various sources.

## Key Features:
- Load models from local checkpoints or HuggingFace
- Create new models from configuration
- Handle device placement and dtype conversion
- PyTorch 2.0 compilation support
- Automatic mixed precision setup

## Model Sources:
1. **Local Checkpoints**: Saved model states from training
2. **HuggingFace Models**: Pretrained GPT-2 variants (gpt2, gpt2-medium, etc.)
3. **New Models**: Created from ModelConfig specifications

## Device and Precision Handling:
- Automatic CUDA availability checking with CPU fallback
- Smart dtype selection based on hardware capabilities
- TF32 enablement for Ampere GPUs
- Mixed precision inference contexts

## Usage Examples:
```python
# Load pretrained model
model = ModelLoader.load_model(
    model_type='gpt2-medium',
    device='cuda',
    dtype='auto',
    compile=True
)

# Load from checkpoint
model = ModelLoader.load_model(
    checkpoint_path='out/ckpt.pt',
    device='cuda'
)

# Create new model
config = ModelConfig(n_layer=12, n_head=12, n_embd=768)
model = ModelLoader.create_model(config, device='cuda')
```
"""

from typing import Optional, Union, Dict, Any
from pathlib import Path
from contextlib import nullcontext

import torch

from ..model import GPT
from ..config import ModelConfig


class ModelLoader:
    """
    Handles all model loading operations in a consistent manner.

    This class provides static methods for model management, ensuring
    consistent handling of devices, data types, and compilation across
    the entire project.

    The ModelLoader abstracts away the complexity of:
    - Device selection and validation
    - Data type conversions
    - Model compilation with PyTorch 2.0
    - Loading from different sources

    All methods are static, so no instantiation is required.

    Methods:
        load_model(): Load from checkpoint or HuggingFace
        create_model(): Create new model from config
        get_model_context(): Get autocast context for inference
        get_model_info(): Extract model information

    Called by:
        - train.Trainer.setup_model() for training
        - generate.main() for text generation
        - server.ModelManager for API serving
    """

    @staticmethod
    def load_model(
        checkpoint_path: Optional[Union[str, Path]] = None,
        model_type: str = "gpt2",
        device: str = "cuda",
        dtype: str = "auto",
        compile: bool = False,
        override_args: Optional[Dict[str, Any]] = None,
    ) -> GPT:
        """
        Load a GPT model from checkpoint or HuggingFace.

        This is the primary method for loading models, supporting both
        local checkpoints and pretrained models from HuggingFace.

        Args:
            checkpoint_path (Optional[Union[str, Path]]): Path to local checkpoint
                - If provided and exists: loads from checkpoint
                - If None or doesn't exist: loads from HuggingFace
            model_type (str): HuggingFace model identifier
                - 'gpt2': 124M parameters
                - 'gpt2-medium': 350M parameters
                - 'gpt2-large': 774M parameters
                - 'gpt2-xl': 1.5B parameters
            device (str): Target device for model
                - 'cuda' or 'cuda:N': GPU device
                - 'cpu': CPU device
                - 'mps': Apple Metal Performance Shaders
            dtype (str): Model precision
                - 'auto': Automatically select best dtype
                - 'float32': Full precision
                - 'float16': Half precision
                - 'bfloat16': Brain float (better for training)
            compile (bool): Enable PyTorch 2.0 compilation
                - True: Compile with torch.compile() for faster inference
                - False: Standard eager execution
            override_args (Optional[Dict[str, Any]]): Config overrides
                - E.g., {'dropout': 0.1} to add dropout for fine-tuning

        Returns:
            GPT: Loaded model ready for use

        Process Flow:
            1. Setup device and validate availability
            2. Determine dtype based on device capabilities
            3. Load model from checkpoint or HuggingFace
            4. Apply compilation if requested
            5. Convert to specified dtype

        Called by:
            - train.Trainer.setup_model() when init_from != 'scratch'
            - generate.main() for text generation
            - server.ModelManager.load_model() for API

        Calls:
            - GPT.load_from_checkpoint() for local checkpoints
            - GPT.from_pretrained() for HuggingFace models
            - torch.compile() for model compilation

        Example:
            >>> # Load GPT-2 medium with bfloat16
            >>> model = ModelLoader.load_model(
            ...     model_type='gpt2-medium',
            ...     device='cuda',
            ...     dtype='bfloat16',
            ...     compile=True
            ... )
        """
        # Setup device and precision
        device = ModelLoader._setup_device(device)
        dtype_torch = ModelLoader._get_torch_dtype(dtype, device)

        # Load model
        if checkpoint_path and Path(checkpoint_path).exists():
            model = GPT.load_from_checkpoint(checkpoint_path, device, compile)
        else:
            model = GPT.from_pretrained(model_type, override_args)
            model.eval()
            model.to(device)

            if compile and hasattr(torch, "compile"):
                print("Compiling model with PyTorch 2.0...")
                model = torch.compile(model)

        # Convert to specified dtype if needed
        if dtype_torch != torch.float32:
            model = model.to(dtype_torch)

        return model

    @staticmethod
    def create_model(
        config: ModelConfig, device: str = "cuda", compile: bool = False
    ) -> GPT:
        """
        Create a new GPT model from configuration.

        Instantiates a fresh GPT model with random weights based on
        the provided configuration. Used for training from scratch.

        Args:
            config (ModelConfig): Model architecture specification
                - n_layer: Number of transformer blocks
                - n_head: Number of attention heads
                - n_embd: Embedding dimension
                - block_size: Maximum sequence length
                - vocab_size: Vocabulary size
                - dropout: Dropout probability
                - bias: Whether to use biases
            device (str): Device to create model on
                - 'cuda': Default GPU
                - 'cpu': CPU device
            compile (bool): Whether to compile with PyTorch 2.0
                - Provides ~30% speedup on compatible hardware

        Returns:
            GPT: New model with randomly initialized weights

        Weight Initialization:
            - Embeddings: Normal(0, 0.02)
            - Linear layers: Normal(0, 0.02/sqrt(2*n_layer))
            - LayerNorm: weight=1, bias=0

        Called by:
            - train.Trainer.setup_model() when init_from='scratch'
            - Testing utilities for model creation

        Calls:
            - GPT.__init__() for model instantiation
            - torch.compile() if compilation requested

        Example:
            >>> config = ModelConfig(
            ...     n_layer=6,
            ...     n_head=6,
            ...     n_embd=384,
            ...     block_size=256
            ... )
            >>> model = ModelLoader.create_model(config, device='cuda')
        """
        model = GPT(config)
        model.to(device)

        if compile and hasattr(torch, "compile"):
            print("Compiling model...")
            model = torch.compile(model)

        return model

    @staticmethod
    def get_model_context(device_type: str, dtype: str = "auto"):
        """
        Get the appropriate context manager for model inference.

        Creates a context manager for automatic mixed precision (AMP)
        inference. On CUDA devices, enables autocast for faster inference
        with reduced precision. On CPU, returns nullcontext (no-op).

        Args:
            device_type (str): Device category
                - 'cuda': GPU device (enables autocast)
                - 'cpu': CPU device (no autocast)
            dtype (str): Precision for autocast
                - 'auto': Automatically select based on hardware
                - 'float16': Force FP16
                - 'bfloat16': Force BF16

        Returns:
            ContextManager: Either autocast or nullcontext

        Autocast Benefits:
            - Faster inference on GPU (~2x speedup)
            - Lower memory usage
            - Automatic op-level precision selection

        Hardware Requirements:
            - FP16: All modern GPUs
            - BF16: Ampere (A100, RTX 30xx) or newer

        Called by:
            - generate.py for inference context
            - server.py for API inference
            - train.Trainer for mixed precision training

        Example:
            >>> ctx = ModelLoader.get_model_context('cuda', 'auto')
            >>> with ctx:
            ...     logits = model(input_ids)
        """
        if device_type == "cpu":
            return nullcontext()

        dtype_torch = ModelLoader._get_torch_dtype(dtype, device_type)
        return torch.amp.autocast(device_type=device_type, dtype=dtype_torch)

    @staticmethod
    def _setup_device(device: str) -> str:
        """
        Setup and validate device selection.

        Validates the requested device is available and configures
        hardware-specific optimizations. Falls back to CPU if
        requested device is unavailable.

        Args:
            device (str): Requested device string

        Returns:
            str: Validated device string (may differ if fallback)

        Side Effects:
            - Enables TF32 on CUDA devices
            - Prints warning if falling back to CPU

        TF32 Note:
            TensorFloat-32 provides ~10x speedup for matmuls
            on Ampere GPUs with minimal accuracy loss.

        Called by:
            - load_model() for device setup
        """
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            return "cpu"

        # Enable TF32 for better performance on Ampere GPUs
        if "cuda" in device:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        return device

    @staticmethod
    def _get_torch_dtype(dtype: str, device: str) -> torch.dtype:
        """
        Convert string dtype to torch dtype with hardware awareness.

        Intelligently selects the best dtype based on device capabilities
        when 'auto' is specified. Otherwise converts string to torch type.

        Args:
            dtype (str): Requested dtype
                - 'auto': Automatically select best dtype
                - 'float32': Full precision
                - 'float16': Half precision
                - 'bfloat16': Brain float
            device (str): Target device (affects auto selection)

        Returns:
            torch.dtype: PyTorch dtype object

        Auto Selection Logic:
            - CUDA with BF16 support: torch.bfloat16 (better for training)
            - CUDA without BF16: torch.float16 (better compatibility)
            - CPU: torch.float32 (no half precision on CPU)

        Hardware Support:
            - BF16: Ampere (compute capability 8.0+)
            - FP16: All CUDA GPUs

        Called by:
            - load_model() for dtype conversion
            - get_model_context() for autocast setup
        """
        if dtype == "auto":
            if "cuda" in device:
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                    return torch.bfloat16
                else:
                    return torch.float16
            else:
                return torch.float32

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }

        return dtype_map.get(dtype, torch.float32)

    @staticmethod
    def get_model_info(model: GPT) -> Dict[str, Any]:
        """
        Get comprehensive information about a loaded model.

        Extracts model architecture details, parameter counts, and
        runtime information for logging and debugging purposes.

        Args:
            model (GPT): Model instance to analyze

        Returns:
            Dict[str, Any]: Model information containing:
                - num_parameters: Total parameter count
                - num_parameters_millions: Parameters in millions
                - architecture: Dict with architectural details
                    - n_layer: Number of transformer blocks
                    - n_head: Number of attention heads
                    - n_embd: Embedding dimension
                    - block_size: Maximum sequence length
                    - vocab_size: Vocabulary size
                    - bias: Whether biases are used
                    - dropout: Dropout probability
                - device: Current device placement
                - dtype: Current data type

        Use Cases:
            - Logging model details during training
            - API responses for model information
            - Debugging and verification

        Called by:
            - Training scripts for logging
            - API endpoints for model info
            - Testing utilities

        Example:
            >>> info = ModelLoader.get_model_info(model)
            >>> print(f"Model has {info['num_parameters_millions']:.1f}M parameters")
            >>> print(f"Running on {info['device']} with {info['dtype']}")
        """
        config = model.config
        return {
            "num_parameters": model.get_num_params(),
            "num_parameters_millions": model.get_num_params() / 1e6,
            "architecture": {
                "n_layer": config.n_layer,
                "n_head": config.n_head,
                "n_embd": config.n_embd,
                "block_size": config.block_size,
                "vocab_size": config.vocab_size,
                "bias": config.bias,
                "dropout": config.dropout,
            },
            "device": str(next(model.parameters()).device),
            "dtype": str(next(model.parameters()).dtype),
        }
