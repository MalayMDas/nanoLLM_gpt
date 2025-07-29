"""
Full definition of a GPT Language Model with enhanced modularity and documentation.

This module implements a complete GPT (Generative Pre-trained Transformer) model following
the architecture introduced in "Attention is All You Need" and refined in GPT-2/GPT-3.

## Architecture Overview:
The GPT model consists of:
1. Token embeddings: Convert token IDs to vectors
2. Position embeddings: Add positional information to tokens
3. Transformer blocks: Multiple layers of self-attention and feed-forward networks
4. Layer normalization: Applied before attention and feed-forward layers
5. Output projection: Convert hidden states back to vocabulary logits

## Key Features:
- Multiple GPT-2 model sizes (base, medium, large, xl)
- Flash Attention for improved performance on modern GPUs
- Weight tying between input embeddings and output layer
- Configurable architecture parameters
- Checkpoint loading and saving utilities
- Support for both training and inference modes

## Usage Examples:
```python
# Create a small GPT model
from nanoLLM_gpt.model import GPT
from nanoLLM_gpt.config import ModelConfig

config = ModelConfig(
    n_layer=12,      # Number of transformer layers
    n_head=12,       # Number of attention heads
    n_embd=768,      # Embedding dimension
    vocab_size=50257 # GPT-2 vocabulary size
)
model = GPT(config)

# Forward pass for training (with targets)
logits, loss = model(input_ids, targets=target_ids)

# Forward pass for inference (without targets)
logits, _ = model(input_ids)

# Generate text
generated_ids = model.generate(input_ids, max_new_tokens=100)
```

## References:
1) Official GPT-2 TensorFlow implementation: https://github.com/openai/gpt-2/blob/master/src/model.py
2) HuggingFace PyTorch implementation: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
3) "Attention is All You Need" paper: https://arxiv.org/abs/1706.03762
4) "Language Models are Unsupervised Multitask Learners" (GPT-2): https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
"""

import math
import inspect
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Union
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F


# Re-export GPTConfig for backward compatibility
from .config import ModelConfig as GPTConfig


class LayerNorm(nn.Module):
    """
    LayerNorm with optional bias parameter.

    Layer normalization normalizes the inputs across the features for each data sample.
    This helps stabilize the learning process and allows for faster training.

    PyTorch's native LayerNorm doesn't support bias=False, so this custom
    implementation allows for more flexibility in model configuration.

    Mathematical formula:
        y = γ * (x - μ) / σ + β
    where:
        - x: input
        - μ: mean of x
        - σ: standard deviation of x
        - γ: learned scale parameter (weight)
        - β: learned shift parameter (bias)

    Called by:
        - CausalSelfAttention (before attention)
        - MLP (before feed-forward)
        - GPT (final layer norm before output)
    """

    def __init__(self, ndim: int, bias: bool = True):
        """
        Initialize LayerNorm module.

        Args:
            ndim (int): Dimension of the layer normalization (typically n_embd)
            bias (bool): Whether to include bias parameter β. Default: True

        Attributes:
            weight (nn.Parameter): Scale parameter γ, shape (ndim,)
            bias (nn.Parameter or None): Shift parameter β, shape (ndim,) if bias=True
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization to input tensor.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, seq_len, ndim)
                                 or any shape where the last dimension is ndim

        Returns:
            torch.Tensor: Normalized tensor of same shape as input

        Note:
            Uses epsilon=1e-5 for numerical stability in the normalization
        """
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """
    Multi-head masked self-attention layer with causal mask.

    This layer implements the self-attention mechanism where each position
    can only attend to previous positions (causal masking). This ensures
    that predictions for position i can depend only on positions < i,
    maintaining the autoregressive property required for language modeling.

    Key concepts:
    - Multi-head: Attention is computed h times in parallel with different projections
    - Causal mask: Upper triangular mask prevents attending to future tokens
    - Scaled dot-product attention: Attention scores are scaled by sqrt(d_k)

    Called by:
        - Block.forward() (as self.attn)

    Calls:
        - F.scaled_dot_product_attention (if Flash Attention available)
        - F.softmax (for attention scores)
        - nn.Dropout (for regularization)
    """

    def __init__(self, config):
        """
        Initialize the causal self-attention layer.

        Args:
            config: Model configuration object with attributes:
                - n_embd (int): Embedding dimension
                - n_head (int): Number of attention heads
                - block_size (int): Maximum sequence length
                - bias (bool): Whether to use bias in linear layers
                - dropout (float): Dropout probability

        Attributes:
            c_attn (nn.Linear): Combined QKV projection, maps n_embd -> 3*n_embd
            c_proj (nn.Linear): Output projection, maps n_embd -> n_embd
            attn_dropout (nn.Dropout): Dropout for attention weights
            resid_dropout (nn.Dropout): Dropout for residual connection
            bias (torch.Tensor): Causal mask buffer (if not using Flash Attention)
        """
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"

        # Combined projection for query, key, and value
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection after attention
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Regularization layers
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Store configuration parameters
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # Check for Flash Attention availability (PyTorch >= 2.0)
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            # Create causal mask for standard attention
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-head causal self-attention.

        This method implements the core attention mechanism:
        1. Project input to Q, K, V matrices
        2. Split into multiple heads
        3. Compute scaled dot-product attention with causal masking
        4. Concatenate heads and project output

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, n_embd)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, n_embd)
                         after applying self-attention

        Attention formula:
            Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
        where d_k = n_embd / n_head (dimension per head)
        """
        B, T, C = x.size()  # batch size, sequence length, embedding dimension

        # Calculate Q, K, V for all heads in parallel
        # Split the 3*n_embd output into three n_embd tensors
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # Reshape to separate heads: (B, T, n_head, head_size)
        # Then transpose to (B, n_head, T, head_size) for attention computation
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Compute attention
        if self.flash:
            # Use optimized Flash Attention when available
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            # Manual attention computation
            # Attention scores: (B, n_head, T, T)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # Apply causal mask
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            # Convert to probabilities
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            # Apply attention to values: (B, n_head, T, head_size)
            y = att @ v

        # Reassemble heads: transpose back and concatenate
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Final output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (Feed-Forward Network) used in transformer blocks.

    Implements a 2-layer neural network with GELU activation:
        h = GELU(xW1 + b1)
        y = hW2 + b2 + dropout(hW2 + b2)

    The hidden layer typically has 4x the dimension of the input/output,
    following the transformer architecture convention.

    Called by:
        - Block.forward() (as self.mlp)

    Calls:
        - F.gelu() for non-linear activation
        - nn.Dropout for regularization
    """

    def __init__(self, config):
        """Initialize MLP module.

        Args:
            config: Model configuration with attributes:
                - n_embd (int): Input/output dimension
                - bias (bool): Whether to use bias in linear layers
                - dropout (float): Dropout probability

        Attributes:
            c_fc (nn.Linear): First layer projection n_embd -> 4*n_embd
            gelu (nn.GELU): Gaussian Error Linear Unit activation
            c_proj (nn.Linear): Second layer projection 4*n_embd -> n_embd
            dropout (nn.Dropout): Dropout for regularization
        """

        super().__init__()
        # Expand to 4x hidden dimension (following GPT-2 architecture)
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        # Project back to original dimension
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply two-layer feedforward network with GELU activation.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, n_embd)

        Returns:
            torch.Tensor: Output tensor of same shape as input

        Processing steps:
            1. Linear projection to 4*n_embd dimension
            2. GELU activation function
            3. Linear projection back to n_embd dimension
            4. Dropout for regularization
        """
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    Transformer block consisting of self-attention and MLP layers.

    This implements a standard transformer decoder block with:
    1. Multi-head self-attention with residual connection
    2. Position-wise feedforward network with residual connection
    3. Layer normalization before each sub-layer (pre-norm architecture)

    The block follows the architecture:
        x' = x + Attention(LayerNorm(x))
        y = x' + MLP(LayerNorm(x'))

    Called by:
        - GPT.__init__() (creates n_layer blocks)

    Calls:
        - LayerNorm (for normalization)
        - CausalSelfAttention (for self-attention)
        - MLP (for feed-forward)
    """

    def __init__(self, config):
        """
        Initialize a transformer block.

        Args:
            config: Model configuration containing all necessary parameters

        Attributes:
            ln_1 (LayerNorm): Layer norm before attention
            attn (CausalSelfAttention): Multi-head self-attention module
            ln_2 (LayerNorm): Layer norm before MLP
            mlp (MLP): Feed-forward network
        """
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process input through attention and MLP with residual connections.

        Uses pre-normalization: LayerNorm is applied before attention/MLP.
        This is different from the original transformer which uses post-norm.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, n_embd)

        Returns:
            torch.Tensor: Output tensor of same shape as input

        Processing flow:
            1. Apply LayerNorm, then self-attention, then add residual
            2. Apply LayerNorm, then MLP, then add residual
        """
        # Attention block with residual connection
        x = x + self.attn(self.ln_1(x))
        # MLP block with residual connection
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """
    GPT Language Model implementation.

    This is the main model class that orchestrates the entire GPT architecture:
    1. Token embeddings: Convert token IDs to dense vectors
    2. Positional embeddings: Add position information to tokens
    3. Transformer blocks: Stack of self-attention and feed-forward layers
    4. Output projection: Convert hidden states to vocabulary logits

    Key features:
    - Weight tying between input embeddings and output layer
    - Configurable architecture (layers, heads, dimensions)
    - Support for both training (with targets) and inference
    - Autoregressive generation capability

    This is the entry point for:
    - Training: Called by Trainer class
    - Inference: Called by InferencePipeline
    - Generation: Via generate() method

    Calls:
        - nn.Embedding (for token/position embeddings)
        - Block (transformer layers)
        - LayerNorm (final normalization)
        - nn.Linear (output projection)
    """

    def __init__(self, config: GPTConfig):
        """
        Initialize GPT model.

        Args:
            config (GPTConfig): Model configuration with attributes:
                - vocab_size (int): Size of token vocabulary
                - block_size (int): Maximum sequence length
                - n_layer (int): Number of transformer blocks
                - n_head (int): Number of attention heads
                - n_embd (int): Embedding dimension
                - dropout (float): Dropout probability
                - bias (bool): Whether to use bias in layers

        The initialization process:
        1. Create embeddings and transformer blocks
        2. Set up weight tying between input/output embeddings
        3. Initialize all weights using GPT-2 scheme
        4. Apply special initialization to residual projections
        """
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # Build the transformer architecture
        self.transformer = nn.ModuleDict(
            dict(
                # Token embeddings: convert token indices to vectors
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                # Positional embeddings: encode position information
                wpe=nn.Embedding(config.block_size, config.n_embd),
                # Dropout for regularization
                drop=nn.Dropout(config.dropout),
                # Stack of transformer blocks
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                # Final layer normalization
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )

        # Language modeling head: project to vocabulary size
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share weights between token embeddings and output layer
        # This reduces parameters and improves performance
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize all weights
        self.apply(self._init_weights)

        # Apply special scaled initialization to residual projections
        # This helps with training stability in deep networks
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        # Report model size
        print(f"Number of parameters: {self.get_num_params()/1e6:.2f}M")

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Calculate total number of parameters in the model.

        Args:
            non_embedding (bool): If True, exclude position embeddings from count.
                                 This gives a more accurate parameter count since
                                 position embeddings aren't involved in the output
                                 layer due to weight tying. Default: True

        Returns:
            int: Total number of parameters

        Example:
            >>> model = GPT(config)
            >>> print(f"Parameters: {model.get_num_params()/1e6:.2f}M")
            Parameters: 124.44M

        Called by:
            - __init__ (for reporting)
            - ModelLoader.get_model_info()
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # Subtract positional embeddings as they're not used in computation
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module: nn.Module) -> None:
        """
        Initialize weights using GPT-2 scheme.

        This method is called on every module during model initialization
        via self.apply(self._init_weights).

        Args:
            module (nn.Module): The module to initialize

        Initialization scheme:
            - Linear layers: Normal(0, 0.02)
            - Embeddings: Normal(0, 0.02)
            - Biases: Zero

        Called by:
            - self.apply() in __init__
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the model.

        This is the main computation path that processes token indices through
        the entire transformer architecture to produce logits (unnormalized
        probabilities) over the vocabulary.

        Args:
            idx (torch.Tensor): Token indices of shape (batch_size, sequence_length)
                               Each value should be in range [0, vocab_size)
            targets (Optional[torch.Tensor]): Target token indices for computing loss
                                            Shape: (batch_size, sequence_length)
                                            If None, only logits are computed (inference mode)

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - logits: Predicted token logits
                    * Training mode: Shape (batch_size, seq_len, vocab_size)
                    * Inference mode: Shape (batch_size, 1, vocab_size) - only last position
                - loss: Cross-entropy loss if targets provided, else None

        Processing flow:
            1. Get token embeddings from input indices
            2. Add positional embeddings
            3. Apply dropout
            4. Process through transformer blocks
            5. Apply final layer norm
            6. Project to vocabulary size
            7. Compute loss if training

        Called by:
            - Trainer (during training)
            - generate() method (during text generation)
            - InferencePipeline (during inference)
        """
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.block_size
        ), f"Sequence length {t} exceeds block size {self.config.block_size}"

        # Generate position indices
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # Forward pass through transformer
        # Token embeddings: (b, t) -> (b, t, n_embd)
        tok_emb = self.transformer.wte(idx)
        # Position embeddings: (t,) -> (t, n_embd)
        pos_emb = self.transformer.wpe(pos)
        # Combine embeddings and apply dropout
        x = self.transformer.drop(tok_emb + pos_emb)

        # Process through transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # Final layer normalization
        x = self.transformer.ln_f(x)

        if targets is not None:
            # Training: compute loss on all positions
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # Inference: only compute logits for last position
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def crop_block_size(self, block_size: int) -> None:
        """
        Reduce the block size for fine-tuning on shorter sequences.

        This performs model surgery to support smaller context windows
        than the model was originally trained with. Useful when fine-tuning
        a pretrained model on tasks with shorter sequences.

        Args:
            block_size (int): New maximum sequence length (must be <= original)

        Effects:
            - Updates self.config.block_size
            - Truncates positional embeddings to new size
            - Updates causal masks in attention layers (if not using Flash)

        Example:
            >>> model = GPT.from_pretrained('gpt2')  # block_size = 1024
            >>> model.crop_block_size(512)  # Now accepts sequences up to 512

        Called by:
            - ModelLoader (when loading with custom block_size)
        """
        assert (
            block_size <= self.config.block_size
        ), "New block size must be smaller than original"

        self.config.block_size = block_size
        # Crop positional embeddings
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:block_size]
        )
        # Crop attention bias buffers if they exist
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(
        cls, model_type: str, override_args: Optional[Dict[str, Any]] = None
    ) -> "GPT":
        """
        Load pretrained GPT-2 weights from HuggingFace.

        This method downloads and converts HuggingFace GPT-2 checkpoints to
        our model format. It handles the differences in layer naming and
        weight matrix orientations (Conv1D vs Linear).

        Args:
            model_type (str): One of 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
                - 'gpt2': 124M parameters, 12 layers
                - 'gpt2-medium': 350M parameters, 24 layers
                - 'gpt2-large': 774M parameters, 36 layers
                - 'gpt2-xl': 1558M parameters, 48 layers
            override_args (Optional[Dict[str, Any]]): Config overrides
                Currently only 'dropout' is supported for fine-tuning

        Returns:
            GPT: Model instance with pretrained weights loaded

        Example:
            >>> model = GPT.from_pretrained('gpt2')
            >>> model = GPT.from_pretrained('gpt2-medium', {'dropout': 0.1})

        Called by:
            - ModelLoader.load_pretrained()
        """
        assert model_type in {
            "gpt2",
            "gpt2-medium",
            "gpt2-large",
            "gpt2-xl",
        }, f"Unknown model type: {model_type}"

        override_args = override_args or {}
        # Only dropout can be overridden
        assert all(
            k == "dropout" for k in override_args
        ), "Only 'dropout' can be overridden"

        from transformers import GPT2LMHeadModel

        print(f"Loading weights from pretrained GPT: {model_type}")

        # Model configurations for different GPT-2 sizes
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]

        # GPT-2 checkpoints use these specific values
        print("Forcing vocab_size=50257, block_size=1024, bias=True")
        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024
        config_args["bias"] = True

        # Apply overrides
        if "dropout" in override_args:
            print(f"Overriding dropout rate to {override_args['dropout']}")
            config_args["dropout"] = override_args["dropout"]

        # Initialize model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        # Remove attention bias buffer (not a parameter)
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")]

        # Load HuggingFace model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # Clean up HuggingFace state dict
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")]

        # These layers need transposition due to Conv1D vs Linear difference
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]

        # Copy weights with proper transposition
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"Mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # Transpose Conv1D weights to Linear format
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # Direct copy for other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        device: str = "cuda",
        compile: bool = False,
    ) -> "GPT":
        """
        Load model from a saved checkpoint.

        Handles loading checkpoints saved during training, including model
        weights and configuration. Supports both PyTorch 2.6's security
        features (weights_only=True) and older formats.

        Args:
            checkpoint_path (Union[str, Path]): Path to checkpoint file (.pt or .pth)
            device (str): Device to load model on ('cuda', 'cpu', 'mps'). Default: 'cuda'
            compile (bool): Whether to compile model with PyTorch 2.0 for faster
                          inference. Default: False

        Returns:
            GPT: Model instance loaded from checkpoint

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist

        Example:
            >>> model = GPT.load_from_checkpoint('out/ckpt.pt')
            >>> model = GPT.load_from_checkpoint('model.pt', device='cpu')

        Called by:
            - ModelLoader.load_checkpoint()
            - InferencePipeline (for custom models)
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"Loading model from checkpoint: {checkpoint_path}")

        # Add safe globals for custom classes
        import torch.serialization

        torch.serialization.add_safe_globals([GPTConfig])

        # Try to load with weights_only=True first (safer)
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=True
            )
        except Exception as e:
            # Fall back to weights_only=False if needed
            print(
                f"Loading with weights_only=True failed, falling back to weights_only=False"
            )
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )

        # Extract model configuration
        model_args = checkpoint["model_args"]
        config = GPTConfig(**model_args)

        # Initialize model
        model = cls(config)

        # Load state dict
        state_dict = checkpoint["model"]
        # Remove unwanted prefix if present
        unwanted_prefix = "_orig_mod."
        for k in list(state_dict.keys()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)

        # Compile if requested
        if compile and hasattr(torch, "compile"):
            print("Compiling model with PyTorch 2.0...")
            model = torch.compile(model)

        # Print training info if available
        if "iter_num" in checkpoint:
            print(f"Checkpoint from iteration {checkpoint['iter_num']}")
        if "best_val_loss" in checkpoint:
            print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")

        return model

    def save_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        iter_num: Optional[int] = None,
        best_val_loss: Optional[float] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Save model checkpoint.

        Saves model state and optionally training state for resuming later.
        The checkpoint includes model weights, configuration, and optional
        training metadata.

        Args:
            checkpoint_path (Union[str, Path]): Path to save checkpoint file
            optimizer (Optional[torch.optim.Optimizer]): Optimizer state to save
                                                       for training resumption
            iter_num (Optional[int]): Current iteration number
            best_val_loss (Optional[float]): Best validation loss achieved
            config (Optional[Dict[str, Any]]): Training configuration dict

        Checkpoint format:
            {
                'model': state_dict,          # Model weights
                'model_args': config_dict,    # Model configuration
                'optimizer': optimizer_state, # Optional
                'iter_num': int,             # Optional
                'best_val_loss': float,      # Optional
                'config': dict               # Optional training config
            }

        Called by:
            - Trainer.train() (during checkpointing)
        """
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model": self.state_dict(),
            "model_args": self.config.__dict__,
        }

        if optimizer is not None:
            checkpoint["optimizer"] = optimizer.state_dict()
        if iter_num is not None:
            checkpoint["iter_num"] = iter_num
        if best_val_loss is not None:
            checkpoint["best_val_loss"] = best_val_loss
        if config is not None:
            checkpoint["config"] = config

        print(f"Saving checkpoint to {checkpoint_path}")
        torch.save(checkpoint, checkpoint_path)

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float],
        device_type: str,
    ) -> torch.optim.Optimizer:
        """
        Configure AdamW optimizer with weight decay.

        Implements smart weight decay following GPT-2 training:
        - Apply weight decay only to 2D parameters (weight matrices)
        - No weight decay for 1D parameters (biases, LayerNorm params)

        This prevents regularizing parameters that should adapt freely.

        Args:
            weight_decay (float): L2 regularization coefficient (e.g., 0.1)
            learning_rate (float): Base learning rate (e.g., 6e-4)
            betas (Tuple[float, float]): Adam beta parameters (default: (0.9, 0.95))
            device_type (str): Device type ('cuda', 'cpu') for fused optimizer

        Returns:
            torch.optim.Optimizer: Configured AdamW optimizer with parameter groups

        Implementation details:
            - Creates two parameter groups: decay and no_decay
            - Uses fused AdamW on CUDA for better performance
            - Reports parameter counts for each group

        Called by:
            - Trainer.__init__() (during optimizer setup)
        """
        # Get all parameters that require gradients
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # Separate parameters by dimensionality
        # 2D parameters (matrices) get weight decay
        # 1D parameters (vectors) don't get weight decay
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        # Create parameter groups with different weight decay
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        # Report parameter counts
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"Num decayed parameter tensors: {len(decay_params)}, "
            f"with {num_decay_params:,} parameters"
        )
        print(
            f"Num non-decayed parameter tensors: {len(nodecay_params)}, "
            f"with {num_nodecay_params:,} parameters"
        )

        # Use fused AdamW if available (more efficient on CUDA)
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"Using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        """
        Estimate model FLOPs utilization (MFU) as ratio of A100 peak FLOPs.

        This helps understand training efficiency by comparing actual
        computational throughput to theoretical hardware peak.

        Args:
            fwdbwd_per_iter: Number of forward/backward passes per iteration
            dt: Time taken for iteration in seconds

        Returns:
            MFU ratio (0 to 1, where 1 = peak hardware utilization)

        Reference:
            PaLM paper Appendix B: https://arxiv.org/abs/2204.02311
        """
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size

        # Estimate FLOPs using PaLM paper formula
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter

        # Calculate achieved vs theoretical peak
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.

        Implements autoregressive generation where each new token is predicted
        based on all previous tokens. Uses temperature and top-k sampling for
        controlling randomness vs quality.

        Args:
            idx (torch.Tensor): Initial token indices (prompt)
                              Shape: (batch_size, sequence_length)
            max_new_tokens (int): Number of new tokens to generate
            temperature (float): Sampling temperature. Default: 1.0
                               - < 1.0: More focused/deterministic
                               - = 1.0: Normal sampling
                               - > 1.0: More random/creative
            top_k (Optional[int]): If set, only sample from top k most likely tokens
                                 This improves quality by filtering unlikely tokens

        Returns:
            torch.Tensor: Extended sequence including prompt and generated tokens
                         Shape: (batch_size, sequence_length + max_new_tokens)

        Algorithm:
            For each new token:
            1. Crop context to block_size if needed
            2. Forward pass to get logits for next token
            3. Apply temperature scaling
            4. Optionally filter to top-k tokens
            5. Sample from probability distribution
            6. Append to sequence

        Example:
            >>> prompt = torch.tensor([[1, 2, 3]])  # Shape: (1, 3)
            >>> output = model.generate(prompt, max_new_tokens=10)
            >>> output.shape  # (1, 13)

        Called by:
            - InferencePipeline.generate()
            - Interactive generation scripts
        """
        for _ in range(max_new_tokens):
            # Crop context if it exceeds block size
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )

            # Get predictions for next token
            logits, _ = self(idx_cond)

            # Focus on last position and apply temperature
            logits = logits[:, -1, :] / temperature

            # Optionally apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            # Convert to probabilities and sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
