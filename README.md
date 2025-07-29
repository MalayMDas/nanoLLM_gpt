# nanoLLM_gpt

A clean, modular implementation of GPT (Generative Pre-trained Transformer) with support for training, inference, and serving. This project provides a production-ready codebase with comprehensive documentation, following software engineering best practices.

**Note**: This project is built based on [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT/). The codebase has been refactored for modularity, extended with additional features, and enhanced with detailed documentation. So far, developing on the shoulders of giants, this particular repository was developed as a single person project to make it easy for me to create and test new LLM models. Parts of the code were created and reviewed using AI assistants (Claude, ChatGPT, Gemini), and tested manually and iteratively developed in cycles.

**Webserver to train and inference for a model and interactive terminals**

![webserver inference](assets/Webserver_inference.png)

![webserver training](assets/Webserver_training.png)

![command line interactive inference](assets/Interactive_terminal.png)

## Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules for configuration, data handling, training, and inference
- **Unified Model Loading**: Consistent interface for loading models from checkpoints or HuggingFace
- **Flexible Training**: Support for single GPU and multi-GPU distributed training with mixed precision
- **Multiple Interfaces**: REST API, web UI, and command-line tools
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI API clients
- **Configuration Management**: YAML/JSON configuration files with command-line overrides
- **Comprehensive Logging**: Integration with Weights & Biases and file-based logging
- **Extensive Documentation**: Detailed docstrings, function flow diagrams, and technical handbook
- **Production Ready**: Error handling, validation, and performance optimizations
- **Testing Suite**: Unit tests and integration tests with pytest

Refer to **[Technical Handbook](handbook.md)** for comprehensive technical reference.

## Project Structure

**nanoLLM_gpt** folder has the code to build model and train it and derrive inference from it.

```
nanoLLM_gpt
├── nanoLLM_gpt/
|   ├── __init__.py             # Package initialization and public API exports
|   │                           # Exports: GPT, GPTConfig, ModelConfig, TrainingConfig, 
|   │                           #         GenerationConfig, APIConfig
|   ├── model.py                # GPT model implementation (transformer architecture)
|   │                           # Classes: GPT, Block, CausalSelfAttention, MLP, LayerNorm
|   │                           # Key features: Flash Attention support, weight tying,
|   │                           #              HuggingFace compatibility
|   ├── config.py               # Configuration dataclasses and utilities
|   │                           # Classes: ModelConfig, TrainingConfig, GenerationConfig,
|   │                           #         APIConfig, ChatMessage, ConfigLoader
|   │                           # Features: YAML/JSON support, CLI argument parsing
|   ├── train.py                # Training script with distributed support
|   │                           # Class: Trainer
|   │                           # Features: DDP/FSDP support, mixed precision, 
|   │                           #          gradient accumulation, checkpointing
|   ├── generate.py             # Text generation CLI with interactive mode
|   │                           # Features: Interactive REPL, batch generation,
|   │                           #          streaming output, prompt files
|   ├── server.py               # Unified API and web server
|   │                           # Classes: TrainingManager
|   │                           # Endpoints: /v1/chat/completions, /v1/completions,
|   │                           #           /v1/models, web UI routes
|   │                           # Features: OpenAI-compatible API, training management
|   ├── utils/                  # Utility modules
|   │   ├── __init__.py         # Utils package initialization
|   │   ├── model_loader.py     # Centralized model loading and management
|   │   │                       # Class: ModelLoader
|   │   │                       # Features: Checkpoint/HuggingFace loading,
|   │   │                       #          device management, compilation support
|   │   ├── data_utils.py       # Data preparation and loading utilities
|   │   │                       # Classes: DataPreparer, DataLoader
|   │   │                       # Features: URL/file loading, tokenization,
|   │   │                       #          train/val split, memory-mapped files
|   │   ├── training_utils.py   # Training utilities (LR scheduler, logging)
|   │   │                       # Classes: LearningRateScheduler, TrainingLogger
|   │   │                       # Features: Cosine decay with warmup, W&B integration,
|   │   │                       #          gradient statistics, checkpoint tracking
|   │   └── inference.py        # Unified inference pipeline
|   │                           # Class: InferencePipeline
|   │                           # Features: Top-p/top-k sampling, streaming,
|   │                           #          batch generation, chat formatting
|   ├── templates/              # Web UI templates
|      └── index.html           # Web interface with training/generation UI
|                               # Features: Interactive text generation,
|                               #          training configuration and monitoring,
|                               #          API documentation and examples|  
|   
├── tests/                      # Test suite
|   ├── __init__.py             # Test package initialization
|   ├── test_comprehensive.py   # Comprehensive functionality tests
|   │                           # Coverage: All major components and workflows
|   ├── test_edge_cases.py      # Edge case and error handling tests
|   │                           # Coverage: Boundary conditions, invalid inputs
|   ├── test_model_loader.py    # Model loading specific tests
|   │                           # Coverage: Checkpoint/HF loading, device handling
|   └── integration_tests/      # Integration tests (require running server)
|       ├── __init__.py         # Integration test package
|       └── test_api.py         # API endpoint integration tests
|                               # Coverage: REST API, streaming, chat format
├── handbook.md                 # Technical handbook with detailed documentation
│                               # Contents: Architecture overview, function flows,
│                               #          configuration reference, usage guide,
│                               #          extension guide, troubleshooting
├── setup.py                    # Package setup and installation configuration
│                               # Entry points: gpt-train, gpt-generate, gpt-server
│                               # Optional deps: dev, datasets, wandb
├── requirements.txt            # Core dependencies
│                               # Core: torch, numpy, tiktoken, flask, pyyaml
├── pyproject.toml              # Modern Python packaging configuration
│                               # Build system: setuptools
├── Makefile                    # Development automation
│                               # Targets: install, test, format, lint, clean
├── .gitignore                  # Git ignore patterns
│                               # Ignores: __pycache__, *.pyc, out/, logs/, etc.
└── LICENSE                     # MIT License file

Additional Files (when created):
├── config/                     # Configuration examples directory
│   ├── train_config.yaml       # Example training configuration
│   ├── server_config.yaml      # Example server configuration
│   └── generation_config.yaml  # Example generation configuration
├── data/                       # Data directory (auto-created)
│   └── <dataset_name>/         # Tokenized data for each dataset
│       ├── train.bin           # Training data (memory-mapped)
│       └── val.bin             # Validation data (memory-mapped)
├── out/                        # Output directory (auto-created during training)
│   ├── ckpt.pt                 # Latest checkpoint
│   ├── best_ckpt.pt            # Best validation checkpoint
│   └── training.log            # Training logs
└── logs/                       # Server logs directory (auto-created)
    └── server.log              # API server logs
```

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0 or higher (for Flash Attention support)
- CUDA-capable GPU (recommended) or CPU

### From Source

```bash
# Clone the repository
git clone https://github.com/MalayMDas/nanoLLM_gpt.git
cd nanoLLM_gpt

# Install in development mode
pip install -e .

# Or install with optional dependencies
pip install -e ".[dev,datasets,wandb]"
```

### Development Setup

```bash
# Install all development dependencies
pip install -e ".[dev,datasets,wandb]"

# Run tests to verify installation
pytest tests/

# Run code quality checks
black nanoLLM_gpt/
flake8 nanoLLM_gpt/
mypy nanoLLM_gpt/
```

## Quick Start

### 1. Generate Text

Using the command-line interface:

```bash
# Generate from HuggingFace model
gpt-generate --model gpt2 --prompt "Once upon a time"

# Generate from checkpoint
gpt-generate --checkpoint out/ckpt.pt --prompt "The future of AI"

# Interactive mode
gpt-generate --model gpt2 --interactive
```

### 2. Train a Model

```bash
# Train from scratch
gpt-train --data-path input.txt --max-iters 5000

# Train with configuration file
gpt-train --config config/train_config.yaml

# Resume training
gpt-train --init-from resume --out-dir out

# Multi-GPU training
torchrun --nproc_per_node=4 gpt-train --config config/train_config.yaml
```

### 3. Start the Server

```bash
# Start with default configuration
gpt-server

# Start with custom configuration
gpt-server --config config/server_config.yaml

# Start with specific model
gpt-server --checkpoint out/ckpt.pt --port 8080
```

### 4. Running provided examples

```bash
# Runs provided example code
python examples/basic_usage.py
```
## Configuration

### Training Configuration

Create a `train_config.yaml` file:

```yaml
model:
  n_layer: 12
  n_head: 12
  n_embd: 768
  block_size: 1024

data_path: path/to/data.txt
max_iters: 100000
learning_rate: 6.0e-4
batch_size: 12
```

### Server Configuration

Create a `server_config.yaml` file:

```yaml
host: 0.0.0.0
port: 8080
model_type: gpt2
device: cuda
dtype: auto
```

## API Usage

The server provides an OpenAI-compatible API:

### Python Example

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed"
)

# Chat completion
response = client.chat.completions.create(
    model="gpt",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a haiku about programming."}
    ]
)

# Text completion
response = client.completions.create(
    model="gpt",
    prompt="Once upon a time",
    max_tokens=100,
    temperature=0.8
)
```

### Curl Example

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.8,
    "max_tokens": 100
  }'
```

## Web Interface

Access the web interface at `http://localhost:8080` after starting the server. Features include:

- **Text Generation**: Interactive text generation with adjustable parameters
- **Model Training**: Start and monitor training jobs
- **API Documentation**: Built-in API reference

## Advanced Usage

### Custom Data Preparation

```python
from nanoLLM_gpt.utils import DataPreparer

preparer = DataPreparer()
data_dir = preparer.prepare_data(
    data_path="https://example.com/data.txt",
    train_val_split=0.001
)
```

### Programmatic Model Loading

```python
from nanoLLM_gpt.utils import ModelLoader

# Load from checkpoint
model = ModelLoader.load_model(
    checkpoint_path="out/ckpt.pt",
    device="cuda",
    compile=True
)

# Load from HuggingFace
model = ModelLoader.load_model(
    model_type="gpt2-large",
    device="cuda"
)
```

### Custom Inference Pipeline

```python
from nanoLLM_gpt.utils import InferencePipeline
from nanoLLM_gpt.config import GenerationConfig

# Initialize pipeline
pipeline = InferencePipeline(device="cuda")
pipeline.load_model(model_type="gpt2")

# Generate with custom config
config = GenerationConfig(
    max_new_tokens=200,
    temperature=0.9,
    top_p=0.95
)

text = pipeline.generate("Once upon a time", config)
```

## Development

### Running Tests

```bash
pytest tests/
```

## Model Architecture

The implementation supports various GPT-2 model sizes:

| Model | Layers | Heads | Embedding Size | Parameters | Context Length |
|-------|--------|-------|----------------|------------|----------------|
| gpt2 | 12 | 12 | 768 | 124M | 1024 |
| gpt2-medium | 24 | 16 | 1024 | 350M | 1024 |
| gpt2-large | 36 | 20 | 1280 | 774M | 1024 |
| gpt2-xl | 48 | 25 | 1600 | 1558M | 1024 |

### Custom Model Sizes

You can create custom model configurations:

```yaml
# config/small_model.yaml
model:
  n_layer: 6          # Number of transformer blocks
  n_head: 8           # Number of attention heads
  n_embd: 512         # Embedding dimension
  block_size: 512     # Maximum sequence length
  vocab_size: 50304   # Vocabulary size (padded for efficiency)
  dropout: 0.1        # Dropout for training
  bias: false         # Disable bias for better performance
```

### Architecture Details

- **Attention**: Multi-head causal self-attention with Flash Attention support
- **FFN**: 4x expansion with GELU activation
- **Normalization**: Pre-normalization with LayerNorm
- **Position Encoding**: Learned positional embeddings
- **Weight Tying**: Input and output embeddings are tied

## Performance Tips

1. **Hardware Optimization**
   - Use PyTorch 2.0+ for Flash Attention support
   - Enable model compilation with `--compile` flag (~30% speedup)
   - Use Ampere or newer GPUs for best performance
   - Enable TF32 for matrix multiplications (automatic on supported hardware)

2. **Memory Optimization**
   - Use mixed precision training with `dtype: bfloat16` (recommended) or `float16`
   - Adjust batch size and gradient accumulation for your GPU memory
   - Use gradient checkpointing for very large models (if implemented)
   - Clear GPU cache periodically: `torch.cuda.empty_cache()`

3. **Training Optimization**
   - Use distributed training for large models: `torchrun --nproc_per_node=4`
   - Enable data loading optimizations with pinned memory
   - Adjust `num_workers` for data loading based on CPU cores
   - Use larger batch sizes with gradient accumulation for stability

4. **Inference Optimization**
   - Compile models for inference: `model = torch.compile(model)`
   - Use appropriate precision: `bfloat16` for quality, `int8` for speed
   - Implement key-value caching for longer sequences
   - Batch multiple requests together when possible

## Common Use Cases

### 1. Fine-tuning on Custom Data

```bash
# Prepare your data
echo "Your custom training text" > data.txt

# Fine-tune from GPT-2
gpt-train --init-from gpt2 --data-path data.txt \
  --max-iters 10000 --learning-rate 3e-5 \
  --eval-interval 500 --eval-iters 50
```

### 2. Running a Local ChatGPT-like Service

```bash
# Start server with GPT-2 Large
gpt-server --model-type gpt2-large --port 8080

# Use with OpenAI Python client
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="dummy")
response = client.chat.completions.create(
    model="gpt",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### 3. Batch Text Generation

```python
from nanoLLM_gpt.utils import InferencePipeline
from nanoLLM_gpt.config import GenerationConfig

pipeline = InferencePipeline()
pipeline.load_model(model_type="gpt2")

# Generate multiple samples
config = GenerationConfig(num_samples=5, temperature=0.9)
samples = pipeline.generate("The future of AI is", config)
for i, sample in enumerate(samples):
    print(f"Sample {i+1}: {sample}")
```

## Troubleshooting

### CUDA Out of Memory

- Reduce `batch_size` in configuration
- Increase `gradient_accumulation_steps`
- Use a smaller model variant
- Enable gradient checkpointing (if implemented)
- Use mixed precision: `--dtype float16` or `--dtype bfloat16`

### Slow Training

- Ensure you're using GPU: `--device cuda`
- Enable mixed precision: `--dtype bfloat16`
- Use model compilation: `--compile` (requires PyTorch 2.0+)
- Check data loading isn't bottleneck
- Use larger batch sizes with gradient accumulation

### API Connection Issues

- Verify server is running: `curl http://localhost:8080/health`
- Check firewall settings
- Ensure correct `base_url` in client configuration
- For CORS issues, check `cors_enabled` in server config

### Import Errors

- Ensure package is installed: `pip install -e .`
- Check Python path includes the project directory
- Verify all dependencies are installed: `pip install -r requirements.txt`

## Testing

The project includes comprehensive tests:

```bash
# Run all unit tests
pytest tests/

# Run specific test file
pytest tests/test_model_loader.py

# Run with coverage report
pytest --cov=nanoLLM_gpt tests/

# Run integration tests (requires running server)
python -m pytest tests/integration_tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Documentation

- **[Technical Handbook](handbook.md)**: Comprehensive technical reference with:
  - Architecture overview and design decisions
  - Detailed function flow diagrams for training and inference
  - Complete function reference table with inputs/outputs
  - Python concepts explained for beginners
  
- **API Documentation**: Available at `/api` endpoint when server is running
- **Code Documentation**: Extensive docstrings in all modules


## Acknowledgments

- Based on the GPT architecture from OpenAI
- Developed based on [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT/) - Andrej's code was the starting point of this repository
- The codebase has been significantly refactored for modularity and extended with additional features
- Parts of the code were created and reviewed using LLMs (Claude, ChatGPT, Gemini) and then further developed and tested manually in cycles.
- Uses HuggingFace Transformers for pretrained weights
- Uses tiktoken for GPT-2 compatible tokenization

