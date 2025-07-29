"""
Centralized configuration management for the GPT project.

This module provides a comprehensive configuration system using Python dataclasses
for type safety and validation. It supports loading configurations from:
- YAML/JSON files
- Command-line arguments
- Python dictionaries
- Default values

## Configuration Classes:
1. **ModelConfig**: GPT model architecture parameters
2. **TrainingConfig**: Training hyperparameters and settings
3. **GenerationConfig**: Text generation parameters
4. **APIConfig**: Server and API settings
5. **ChatMessage**: Chat completion message format

## Key Features:
- Type hints for all parameters
- Automatic validation
- Nested configuration support
- Command-line override capability
- Serialization to/from files

## Usage Examples:
```python
# Load from file
config = ConfigLoader.load_from_file('config.yaml', TrainingConfig)

# Create from CLI args
config = ConfigLoader.create_config_from_args(args, TrainingConfig)

# Save to file
ConfigLoader.save_to_file(config, 'config.yaml')
```
"""

import json
import yaml
import argparse
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Union
from pathlib import Path


@dataclass
class ModelConfig:
    """
    Configuration for GPT model architecture.

    Defines the structural parameters of the GPT model. These parameters
    determine the model size, capacity, and computational requirements.

    Attributes:
        block_size (int): Maximum sequence length the model can process.
                         Also called context_length. Default: 1024
        vocab_size (int): Size of the token vocabulary. Default: 50304
                         (50257 for GPT-2, padded to nearest multiple of 64 for efficiency)
        n_layer (int): Number of transformer blocks (depth). Default: 12
        n_head (int): Number of attention heads per layer. Default: 12
        n_embd (int): Embedding dimension (hidden size). Default: 768
                     Must be divisible by n_head
        dropout (float): Dropout probability for regularization. Default: 0.0
                        Set > 0 for training, especially when fine-tuning
        bias (bool): Whether to use bias in Linear and LayerNorm layers.
                    Default: True (matches GPT-2). Set False for slightly
                    better performance

    Model Size Examples:
        - GPT-2 (124M): n_layer=12, n_head=12, n_embd=768
        - GPT-2 Medium (350M): n_layer=24, n_head=16, n_embd=1024
        - GPT-2 Large (774M): n_layer=36, n_head=20, n_embd=1280
        - GPT-2 XL (1.5B): n_layer=48, n_head=25, n_embd=1600
    """

    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2


@dataclass
class TrainingConfig:
    """
    Configuration for model training.

    Comprehensive settings for the training process, including model architecture,
    optimization hyperparameters, data settings, and logging options.

    The configuration is organized into logical groups:
    - Model: Architecture settings via ModelConfig
    - I/O: Output paths, checkpointing, initialization
    - Data: Dataset, batching, train/val split
    - Optimization: Learning rate, weight decay, gradient settings
    - System: Device, precision, compilation settings
    - Logging: Weights & Biases integration
    """

    # Model architecture (nested ModelConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    # I/O settings
    out_dir: str = "out"  # Directory for checkpoints and logs
    eval_interval: int = 2000  # How often to evaluate on val set (iterations)
    log_interval: int = 1  # How often to log training metrics
    eval_iters: int = 200  # Number of batches for evaluation
    eval_only: bool = False  # If True, only evaluate and exit
    always_save_checkpoint: bool = True  # Save checkpoint every eval (not just best)
    init_from: str = "scratch"  # 'scratch', 'resume', or 'gpt2*' model name

    # Data settings
    data_path: Optional[str] = (
        None  # Path to text file or URL (None = tiny shakespeare)
    )
    dataset: str = "custom"  # Dataset name for organization
    gradient_accumulation_steps: int = (
        40  # Accumulate gradients for effective larger batch
    )
    batch_size: int = 12  # Micro batch size per GPU
    train_val_split: float = 0.0005  # Fraction of data for validation

    # Weights & Biases logging
    wandb_log: bool = False
    wandb_project: str = "gpt-training"
    wandb_run_name: str = "gpt-run"

    # Optimizer settings (AdamW)
    learning_rate: float = 6e-4  # Peak learning rate
    max_iters: int = 600000  # Total training iterations
    weight_decay: float = 1e-1  # L2 penalty (applied to 2D params only)
    beta1: float = 0.9  # Adam beta1 (momentum)
    beta2: float = 0.95  # Adam beta2 (RMSprop term)
    grad_clip: float = 1.0  # Gradient clipping threshold (0 = no clip)

    # Learning rate schedule
    decay_lr: bool = True
    warmup_iters: int = 2000
    lr_decay_iters: int = 600000
    min_lr: float = 6e-5

    # System settings
    backend: str = "nccl"
    device: str = "cuda"
    dtype: str = "bfloat16"
    compile: bool = False
    seed: int = 1337


@dataclass
class GenerationConfig:
    """
    Configuration for text generation.

    Controls the text generation process including sampling strategies,
    length limits, and output formatting.

    Attributes:
        max_new_tokens (int): Maximum number of tokens to generate
        temperature (float): Sampling temperature (0.0 = greedy, 1.0 = normal, >1.0 = creative)
        top_k (int): Only sample from top k tokens (0 = no limit)
        top_p (float): Nucleus sampling threshold (1.0 = no limit)
        repetition_penalty (float): Penalty for repeating tokens (1.0 = no penalty)
        num_samples (int): Number of independent samples to generate
        stream (bool): Whether to stream tokens as they're generated
        stop_sequences (List[str]): Sequences that trigger generation stop
    """

    max_new_tokens: int = 100
    temperature: float = 0.8
    top_k: int = 200
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    num_samples: int = 1
    stream: bool = False
    stop_sequences: Optional[List[str]] = None


@dataclass
class APIConfig:
    """
    Configuration for API and web server.

    Settings for the Flask server that provides REST API endpoints
    and web interface for model interaction.

    Attributes:
        host (str): Server host address ('0.0.0.0' = all interfaces)
        port (int): Server port number
        debug (bool): Enable Flask debug mode
        model_type (str): Default model to load ('gpt2', 'gpt2-medium', etc.)
        checkpoint_path (str): Path to custom checkpoint (overrides model_type)
        device (str): Compute device ('cuda', 'cpu', 'mps')
        dtype (str): Model precision ('auto', 'float32', 'float16', 'bfloat16')
        compile (bool): Whether to compile model with PyTorch 2.0
        max_batch_size (int): Maximum batch size for parallel requests
        cors_enabled (bool): Enable CORS for cross-origin requests
    """

    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = False
    model_type: str = "gpt2"
    checkpoint_path: Optional[str] = None
    device: str = "cuda"
    dtype: str = "auto"
    compile: bool = False
    max_batch_size: int = 16
    cors_enabled: bool = True


@dataclass
class ChatMessage:
    """
    Represents a message in a chat conversation.

    Used for OpenAI-compatible chat completions API.

    Attributes:
        role (str): Message role - 'system', 'user', or 'assistant'
        content (str): The actual message content
        name (Optional[str]): Optional name of the message author
    """

    role: str  # 'system', 'user', or 'assistant'
    content: str
    name: Optional[str] = None


@dataclass
class ChatCompletionRequest:
    """OpenAI-compatible chat completion request."""

    model: str
    messages: List[ChatMessage]
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None


class ConfigLoader:
    """
    Utility class for loading and managing configurations.

    Provides static methods for:
    - Loading configs from YAML/JSON files
    - Saving configs to files
    - Converting between configs and command-line arguments
    - Merging configurations from multiple sources

    This class handles the complexity of nested configurations
    (e.g., ModelConfig within TrainingConfig) and type conversions.

    Usage:
        # Load from file
        config = ConfigLoader.load_from_file('config.yaml', TrainingConfig)

        # Save to file
        ConfigLoader.save_to_file(config, 'output.yaml')

        # Create from CLI args
        config = ConfigLoader.create_config_from_args(args, TrainingConfig)
    """

    @staticmethod
    def load_from_file(config_path: Union[str, Path], config_class: type) -> Any:
        """
        Load configuration from a YAML or JSON file.

        Supports nested configurations and automatically handles
        type conversions based on the dataclass annotations.

        Args:
            config_path (Union[str, Path]): Path to configuration file (.yaml, .yml, or .json)
            config_class (type): Dataclass type to instantiate (e.g., TrainingConfig)

        Returns:
            Instance of config_class with loaded values

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If file format is not supported

        Example:
            >>> config = ConfigLoader.load_from_file('training.yaml', TrainingConfig)
            >>> print(config.learning_rate)
            0.0006
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Load file content
        with open(config_path, "r") as f:
            if config_path.suffix in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            elif config_path.suffix == ".json":
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")

        # Handle nested configs
        if config_class == TrainingConfig and "model" in data:
            model_config = ModelConfig(**data.pop("model"))
            return config_class(model=model_config, **data)

        return config_class(**data)

    @staticmethod
    def save_to_file(config: Any, config_path: Union[str, Path]):
        """
        Save configuration to a YAML or JSON file.

        Automatically determines format based on file extension.
        Creates parent directories if they don't exist.

        Args:
            config: Configuration dataclass instance to save
            config_path (Union[str, Path]): Output path (.yaml, .yml, or .json)

        Raises:
            ValueError: If file format is not supported

        Example:
            >>> config = TrainingConfig(learning_rate=0.001)
            >>> ConfigLoader.save_to_file(config, 'my_config.yaml')
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dictionary
        data = asdict(config)

        # Save file
        with open(config_path, "w") as f:
            if config_path.suffix in [".yaml", ".yml"]:
                yaml.dump(data, f, default_flow_style=False)
            elif config_path.suffix == ".json":
                json.dump(data, f, indent=2)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")

    @staticmethod
    def add_config_arguments(parser: argparse.ArgumentParser, config_class: type):
        """
        Add configuration fields as command-line arguments.

        Automatically generates CLI arguments from dataclass fields,
        handling nested configs, type conversions, and boolean flags.

        Args:
            parser (argparse.ArgumentParser): Parser to add arguments to
            config_class (type): Dataclass type to extract fields from

        Features:
            - Converts field_name to --field-name format
            - Handles bool fields as flags
            - Supports nested configs (e.g., model.n_layer)
            - Preserves default values

        Called by: main() functions in train.py, generate.py, server.py
        """
        # Get default instance
        default_config = config_class()

        # Add arguments for each field
        for field_name, field_type in config_class.__annotations__.items():
            if field_name == "model" and config_class == TrainingConfig:
                # Handle nested model config
                model_group = parser.add_argument_group("model")
                ConfigLoader.add_config_arguments(model_group, ModelConfig)
                continue

            default_value = getattr(default_config, field_name)

            # Determine argument parameters
            if field_type == bool:
                parser.add_argument(
                    f'--{field_name.replace("_", "-")}',
                    type=lambda x: x.lower() in ["true", "1", "yes"],
                    default=default_value,
                    help=f"Default: {default_value}",
                )
            elif hasattr(field_type, "__origin__") and field_type.__origin__ is Union:
                # Handle Optional types
                parser.add_argument(
                    f'--{field_name.replace("_", "-")}',
                    default=default_value,
                    help=f"Default: {default_value}",
                )
            else:
                parser.add_argument(
                    f'--{field_name.replace("_", "-")}',
                    type=field_type,
                    default=default_value,
                    help=f"Default: {default_value}",
                )

    @staticmethod
    def create_config_from_args(args: argparse.Namespace, config_class: type) -> Any:
        """
        Create configuration instance from parsed command-line arguments.

        Args:
            args: Parsed command-line arguments
            config_class: Dataclass type to instantiate

        Returns:
            Instance of config_class with argument values
        """
        # Convert args to dictionary
        args_dict = vars(args)

        # Filter out None values and non-config arguments
        config_dict = {}

        # Get field names from config class
        field_names = {f.replace("_", "-") for f in config_class.__annotations__.keys()}

        for key, value in args_dict.items():
            key_normalized = key.replace("_", "-")
            if key_normalized in field_names and value is not None:
                config_dict[key.replace("-", "_")] = value

        # Handle nested model config for TrainingConfig
        if config_class == TrainingConfig:
            model_dict = {}
            model_fields = {
                f.replace("_", "-") for f in ModelConfig.__annotations__.keys()
            }

            for key, value in args_dict.items():
                key_normalized = key.replace("_", "-")
                if key_normalized in model_fields and value is not None:
                    model_dict[key.replace("-", "_")] = value

            if model_dict:
                config_dict["model"] = ModelConfig(**model_dict)

        return config_class(**config_dict)


def load_config(
    config_class: type,
    config_file: Optional[str] = None,
    args: Optional[argparse.Namespace] = None,
) -> Any:
    """
    Load configuration from file and/or command-line arguments.

    This is the main entry point for configuration loading, handling
    the merging of configurations from multiple sources with proper
    priority ordering.

    Priority (highest to lowest):
        1. Command-line arguments (args)
        2. Configuration file (config_file)
        3. Default values in dataclass

    Args:
        config_class (type): Configuration dataclass type (e.g., TrainingConfig)
        config_file (Optional[str]): Path to YAML/JSON configuration file
        args (Optional[argparse.Namespace]): Parsed command-line arguments

    Returns:
        Configuration instance with merged values

    Example:
        >>> # Load with both file and CLI overrides
        >>> args = parser.parse_args(['--learning-rate', '0.001'])
        >>> config = load_config(TrainingConfig, 'base.yaml', args)
        >>> # config has base.yaml values with learning_rate overridden to 0.001

    Called by:
        - train.main() for training configuration
        - generate.main() for generation settings
        - server.main() for API configuration
    """
    # Start with default config
    if config_file:
        config = ConfigLoader.load_from_file(config_file, config_class)
    else:
        config = config_class()

    # Override with command-line arguments
    if args:
        args_dict = vars(args)
        for key, value in args_dict.items():
            if value is not None and hasattr(config, key):
                setattr(config, key, value)

    return config
