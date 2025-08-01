# nanoLLM_gpt Technical Handbook

This handbook provides a comprehensive technical reference for the nanoLLM_gpt project, including detailed function flows, architecture overview, and complete function reference tables.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Training Flow](#training-flow)
3. [Inference Flow](#inference-flow)
4. [Complete Function Reference](#complete-function-reference)
5. [Python Concepts for Beginners](#python-concepts-for-beginners)

## Architecture Overview

The nanoLLM_gpt project implements a GPT (Generative Pre-trained Transformer) model with a clean, modular architecture:

```
┌─────────────────────────────────────────────────────────┐
│                    Entry Points                         │
├─────────────┬─────────────┬─────────────┬──────────────┤
│   train.py  │ generate.py │  server.py  │   Tests      │
└──────┬──────┴──────┬──────┴──────┬──────┴──────┬───────┘
       │             │             │             │
┌──────┴─────────────┴─────────────┴─────────────┴───────┐
│                    Core Components                      │
├─────────────────────┬───────────────────────────────────┤
│      model.py       │           config.py               │
│  - GPT              │  - ModelConfig                    │
│  - Block            │  - TrainingConfig                 │
│  - CausalSelfAttn  │  - GenerationConfig               │
│  - MLP              │  - APIConfig                      │
│  - LayerNorm        │  - ConfigLoader                   │
└─────────────────────┴───────────────────────────────────┘
       │                                 │
┌──────┴─────────────────────────────────┴───────────────┐
│                  Utility Modules                        │
├─────────────┬─────────────┬──────────────┬────────────┤
│ data_utils  │model_loader │training_utils│ inference  │
│ DataPreparer│ ModelLoader │LRScheduler   │InferencePipeline│
│ DataLoader  │             │TrainingLogger│            │
└─────────────┴─────────────┴──────────────┴────────────┘
```

## Training Flow

The training process follows this sequence of function calls:

### 1. Training Initialization

```
main() [train.py]
├── ConfigLoader.add_config_arguments()
├── ConfigLoader.load_from_file() or create_config_from_args()
└── Trainer.__init__()
    ├── setup_distributed()
    │   └── init_process_group() [if DDP]
    ├── setup_device()
    │   └── torch.set_float32_matmul_precision()
    ├── setup_logger()
    │   └── TrainingLogger.__init__()
    ├── setup_data()
    │   ├── DataPreparer.prepare_data()
    │   │   ├── _load_text_data()
    │   │   └── _tokenize_and_save()
    │   └── DataLoader.__init__()
    ├── setup_model()
    │   ├── ModelLoader.create_model() [if from scratch]
    │   │   └── GPT.__init__()
    │   │       ├── ModuleDict(transformer.wte, transformer.wpe, ...)
    │   │       ├── ModuleList([Block() for _ in range(n_layer)])
    │   │       │   └── Block.__init__()
    │   │       │       ├── LayerNorm.__init__()
    │   │       │       ├── CausalSelfAttention.__init__()
    │   │       │       └── MLP.__init__()
    │   │       └── _init_weights()
    │   ├── GPT.load_from_checkpoint() [if resume]
    │   └── ModelLoader.load_model() [if pretrained]
    ├── setup_optimizer()
    │   └── GPT.configure_optimizers()
    └── setup_scheduler()
        └── LearningRateScheduler.__init__()
```

### 2. Training Loop

```
Trainer.train()
├── estimate_loss() [initial evaluation]
│   └── DataLoader.get_batch() × eval_iters
│       └── model.forward()
├── [Main Training Loop: while iter_num <= max_iters]
│   ├── train_step()
│   │   ├── LearningRateScheduler.get_lr()
│   │   ├── [Gradient Accumulation Loop]
│   │   │   ├── model.forward()
│   │   │   │   ├── transformer.wte(idx)
│   │   │   │   ├── transformer.wpe(pos)
│   │   │   │   ├── transformer.drop(tok_emb + pos_emb)
│   │   │   │   ├── [For each Block]
│   │   │   │   │   ├── Block.forward()
│   │   │   │   │   │   ├── ln_1(x)
│   │   │   │   │   │   ├── attn(ln_1_out)
│   │   │   │   │   │   │   ├── Linear(x) → q, k, v
│   │   │   │   │   │   │   ├── scaled_dot_product_attention()
│   │   │   │   │   │   │   └── c_proj(attention_out)
│   │   │   │   │   │   ├── ln_2(x + attn_out)
│   │   │   │   │   │   └── mlp(ln_2_out)
│   │   │   │   │   │       ├── c_fc(x)
│   │   │   │   │   │       ├── new_gelu(fc_out)
│   │   │   │   │   │       └── c_proj(gelu_out)
│   │   │   │   │   └── x + mlp_out
│   │   │   │   ├── ln_f(x)
│   │   │   │   └── lm_head(ln_f_out)
│   │   │   ├── F.cross_entropy(logits, targets)
│   │   │   ├── loss.backward()
│   │   │   └── DataLoader.get_batch() [prefetch next]
│   │   ├── clip_grad_norm_() [if grad_clip > 0]
│   │   └── optimizer.step()
│   ├── [Every eval_interval iterations]
│   │   ├── estimate_loss()
│   │   ├── TrainingLogger.log_evaluation()
│   │   └── save_checkpoint()
│   │       └── GPT.save_checkpoint()
│   └── TrainingLogger.log_iteration()
│       └── get_gradient_stats()
└── TrainingLogger.finish()
```

## Inference Flow

The inference/generation process follows this sequence:

### 1. Generation Initialization

```
main() [generate.py]
├── argparse.ArgumentParser()
├── InferencePipeline.__init__()
│   ├── tiktoken.get_encoding()
│   └── ModelLoader.get_model_context()
└── InferencePipeline.load_model()
    └── ModelLoader.load_model()
        ├── _setup_device()
        ├── _get_torch_dtype()
        ├── GPT.load_from_checkpoint() or
        ├── GPT.from_pretrained()
        └── torch.compile() [if enabled]
```

### 2. Text Generation

```
InferencePipeline.generate()
├── encode() [if prompt is string]
│   └── tokenizer.encode()
├── [Based on config]
│   ├── _generate_single() [if stream=False, num_samples=1]
│   │   ├── _generate_with_top_p() [if top_p < 1.0]
│   │   │   └── [Generation Loop]
│   │   │       ├── model.forward()
│   │   │       ├── _apply_top_p()
│   │   │       ├── F.softmax()
│   │   │       ├── torch.multinomial()
│   │   │       └── _should_stop()
│   │   └── model.generate() [else]
│   │       └── [Internal generation loop]
│   ├── _generate_batch() [if num_samples > 1]
│   │   └── [Multiple calls to generation logic]
│   └── _generate_stream() [if stream=True]
│       └── [Token-by-token generation with yield]
└── decode()
    └── tokenizer.decode()
```

### 3. Server API Flow

```
main() [server.py]
├── initialize_server()
│   ├── TrainingManager.__init__()
│   ├── InferencePipeline.__init__()
│   └── InferencePipeline.load_model() [if initial model]
└── app.run()
    ├── [API Endpoints]
    │   ├── /v1/chat/completions
    │   │   └── InferencePipeline.chat_completion()
    │   │       ├── _format_chat_prompt()
    │   │       └── generate() or _stream_chat_completion()
    │   ├── /v1/completions
    │   │   └── InferencePipeline.generate()
    │   └── /v1/models
    │       └── list_models()
    └── [Web UI Endpoints]
        ├── /api/model/load
        │   └── InferencePipeline.load_model()
        ├── /api/generate
        │   └── InferencePipeline.generate()
        └── /api/training/start
            └── TrainingManager.start_training()
                └── subprocess.Popen(['gpt-train', ...])
```

## Model Directory Organization

The project uses a **Model Directory** approach for better model management. Each model is stored in its own directory containing both the checkpoint and configuration files.

### Directory Structure

A Model Directory contains:
- `ckpt.pt`: The model checkpoint with weights and training state
- `config.yaml`: The configuration used for training (auto-saved)

### Benefits

1. **Organization**: Keep related files together
2. **Versioning**: Easy to manage multiple model versions
3. **Portability**: Move entire model by copying one directory
4. **Clarity**: Clear association between checkpoint and its config

### Usage in Code

#### Training with Model Directory
```python
from nanoLLM_gpt.config import TrainingConfig, ModelConfig

config = TrainingConfig(
    model=ModelConfig(n_layer=12, n_head=12, n_embd=768),
    out_dir="models/experiments/run_001",  # Model Directory
    data_path="data.txt",
    max_iters=10000
)

# After training, the directory will contain:
# models/experiments/run_001/
#   ├── ckpt.pt      # Model checkpoint
#   └── config.yaml  # Training configuration
```

#### Loading from Model Directory
```python
from nanoLLM_gpt.utils import ModelLoader

# Load model using directory paths
model = ModelLoader.load_model(
    checkpoint_path="models/production/v1.0/ckpt.pt",
    config_path="models/production/v1.0/config.yaml"
)
```

#### Resume Training
```bash
# The config.yaml is automatically loaded from the model directory
gpt-train --init-from resume --out-dir models/my_experiment
```

### Automatic Directory Naming

When fine-tuning HuggingFace models without specifying `--out-dir`:
- `gpt2` → `out_gpt2/`
- `gpt2-medium` → `out_gpt2-medium/`
- `gpt2-large` → `out_gpt2-large/`
- `gpt2-xl` → `out_gpt2-xl/`

### Web Interface Integration

The web interface has been updated with Model Directory support:

1. **Train Model Tab**:
   - "Model Directory" field specifies output location
   - Automatically updates based on training mode
   - Disabled for resume mode to prevent confusion

2. **Generate Text Tab**:
   - Single "Model Directory" field instead of separate paths
   - Automatically constructs paths to ckpt.pt and config.yaml

### Best Practices

1. **Naming Convention**: Use descriptive names including experiment details
   ```
   models/experiments/lr_3e4_batch_12_layer_6/
   models/production/v1.0_shakespeare/
   models/finetuned/gpt2_medical_v2/
   ```

2. **Directory Structure**: Organize by purpose
   ```
   models/
   ├── experiments/    # Development models
   ├── production/     # Deployed models
   └── finetuned/      # Fine-tuned models
   ```

3. **Documentation**: Include README in model directories
   ```
   models/production/v1.0/
   ├── ckpt.pt
   ├── config.yaml
   └── README.md      # Model description, training details
   ```

## Complete Function Reference

### Core Model Functions (model.py)

| Function Name | File | Description | Input Parameters | Output Parameters | Calling Functions | Called Functions |
|--------------|------|-------------|------------------|-------------------|-------------------|------------------|
| GPT.__init__ | model.py | Initialize GPT model with architecture config | config: ModelConfig | None | ModelLoader.create_model, GPT.from_pretrained | ModuleDict, ModuleList, Block.__init__, _init_weights |
| GPT.forward | model.py | Forward pass through model | idx: torch.Tensor (B,T), targets: Optional[torch.Tensor] | logits: (B,T,vocab_size), loss: Optional[scalar] | train_step, estimate_loss, generate | transformer modules, F.cross_entropy |
| GPT.crop_block_size | model.py | Reduce model's context length | block_size: int | None | setup_model | None |
| GPT._init_weights | model.py | Initialize model weights with specific strategy | module: nn.Module | None | apply() in __init__ | torch.nn.init.normal_, zeros_ |
| GPT.generate | model.py | Generate tokens autoregressively | idx: torch.Tensor, max_new_tokens: int, temperature: float, top_k: Optional[int] | torch.Tensor of generated tokens | InferencePipeline._generate_single | forward, F.softmax, torch.multinomial |
| GPT.save_checkpoint | model.py | Save model and training state | checkpoint_path, optimizer, iter_num, best_val_loss, config | None | Trainer.save_checkpoint | torch.save |
| GPT.load_from_checkpoint | model.py | Load model from checkpoint | checkpoint_path: str, device: str, compile: bool | GPT model instance | ModelLoader.load_model, setup_model | torch.load, torch.compile |
| GPT.from_pretrained | model.py | Load pretrained HuggingFace model | model_type: str, override_args: Dict | GPT model instance | ModelLoader.load_model | HF transformers, torch.nn.init |
| GPT.configure_optimizers | model.py | Create AdamW optimizer with weight decay groups | weight_decay, learning_rate, betas, device_type | torch.optim.AdamW | Trainer.setup_optimizer | torch.optim.AdamW |
| GPT.estimate_mfu | model.py | Calculate model FLOPs utilization | fwdbwd_per_iter: int, dt: float | float (MFU percentage) | train loop logging | None |
| GPT.get_num_params | model.py | Count non-embedding parameters | None | int (parameter count) | ModelLoader.get_model_info | parameters() |
| LayerNorm.__init__ | model.py | Initialize LayerNorm | ndim: int, bias: bool | None | Block.__init__ | nn.Parameter |
| LayerNorm.forward | model.py | Apply layer normalization | input: torch.Tensor | torch.Tensor (normalized) | Block.forward | F.layer_norm |
| CausalSelfAttention.__init__ | model.py | Initialize multi-head attention | config: ModelConfig | None | Block.__init__ | nn.Linear |
| CausalSelfAttention.forward | model.py | Compute self-attention | x: torch.Tensor (B,T,C) | torch.Tensor (B,T,C) | Block.forward | F.scaled_dot_product_attention |
| MLP.__init__ | model.py | Initialize feedforward network | config: ModelConfig | None | Block.__init__ | nn.Linear |
| MLP.forward | model.py | Apply feedforward transformation | x: torch.Tensor | torch.Tensor | Block.forward | new_gelu |
| Block.__init__ | model.py | Initialize transformer block | config: ModelConfig | None | GPT.__init__ | LayerNorm, CausalSelfAttention, MLP |
| Block.forward | model.py | Process through one transformer layer | x: torch.Tensor | torch.Tensor | GPT.forward | ln_1, attn, ln_2, mlp |
| new_gelu | model.py | GELU activation function | x: torch.Tensor | torch.Tensor | MLP.forward | torch operations |

### Configuration Functions (config.py)

| Function Name | File | Description | Input Parameters | Output Parameters | Calling Functions | Called Functions |
|--------------|------|-------------|------------------|-------------------|-------------------|------------------|
| ConfigLoader.load_from_file | config.py | Load config from YAML/JSON file | config_path: Path, config_class: type | Config instance | main functions, load_config | yaml.safe_load, json.load |
| ConfigLoader.save_to_file | config.py | Save config to file | config: Any, config_path: Path | None | save_checkpoint, TrainingManager | yaml.dump, json.dump |
| ConfigLoader.add_config_arguments | config.py | Add config fields as CLI arguments | parser: ArgumentParser, config_class: type | None | main functions | add_argument |
| ConfigLoader.create_config_from_args | config.py | Create config from parsed args | args: Namespace, config_class: type | Config instance | main functions | dataclass constructors |
| load_config | config.py | Load config from file and/or CLI | config_class, config_file, args | Config instance | main functions | ConfigLoader methods |

### Training Functions (train.py)

| Function Name | File | Description | Input Parameters | Output Parameters | Calling Functions | Called Functions |
|--------------|------|-------------|------------------|-------------------|-------------------|------------------|
| Trainer.__init__ | train.py | Initialize trainer | config: TrainingConfig | None | main | setup_* methods |
| Trainer.setup_distributed | train.py | Initialize DDP if needed | None | None | __init__ | init_process_group |
| Trainer.setup_device | train.py | Configure device and precision | None | None | __init__ | torch.cuda.set_device |
| Trainer.setup_logger | train.py | Initialize training logger | None | None | __init__ | TrainingLogger.__init__ |
| Trainer.setup_data | train.py | Prepare data loader | None | None | __init__ | DataPreparer.prepare_data, DataLoader.__init__ |
| Trainer.setup_model | train.py | Initialize or load model | None | None | __init__ | ModelLoader methods |
| Trainer.setup_optimizer | train.py | Configure AdamW optimizer | None | None | __init__ | GPT.configure_optimizers |
| Trainer.setup_scheduler | train.py | Setup LR scheduler | None | None | __init__ | LearningRateScheduler.__init__ |
| Trainer.estimate_loss | train.py | Evaluate on train/val sets | None | Dict[str, float] | train | DataLoader.get_batch, model.forward |
| Trainer.save_checkpoint | train.py | Save model checkpoint | losses: Dict[str, float] | None | train | GPT.save_checkpoint |
| Trainer.train_step | train.py | Execute single training step | X, Y: torch.Tensor | loss, lr, X, Y | train | model.forward, optimizer.step |
| Trainer.train | train.py | Main training loop | None | None | main | train_step, estimate_loss, save_checkpoint |
| main | train.py | Entry point for training | None | None | CLI/script | Trainer, train |

### Data Utility Functions (utils/data_utils.py)

| Function Name | File | Description | Input Parameters | Output Parameters | Calling Functions | Called Functions |
|--------------|------|-------------|------------------|-------------------|-------------------|------------------|
| DataPreparer.__init__ | data_utils.py | Initialize data preparer | tokenizer_name: str, num_proc: int | None | setup_data | tiktoken.get_encoding |
| DataPreparer.prepare_data | data_utils.py | Main data preparation | data_path, dataset_name, output_dir, train_val_split | str (data directory) | Trainer.setup_data | _load_text_data, _tokenize_and_save |
| DataPreparer._load_text_data | data_utils.py | Load text from file/URL | path: str | str (text content) | prepare_data | requests.get, open |
| DataPreparer._tokenize_and_save | data_utils.py | Tokenize and save binary | text, data_dir, train_val_split | None | prepare_data | tokenizer.encode_ordinary, tofile |
| DataPreparer._prepare_openwebtext | data_utils.py | Prepare OpenWebText dataset | output_dir: str | str (data directory) | prepare_data | datasets.load_dataset |
| DataLoader.__init__ | data_utils.py | Initialize data loader | data_dir, block_size, batch_size, device, device_type | None | Trainer.setup_data | Path operations |
| DataLoader.get_batch | data_utils.py | Load random batch | split: str | X, Y tensors | train_step, estimate_loss | np.memmap, torch operations |
| DataLoader.get_vocab_size | data_utils.py | Get vocabulary size | None | Optional[int] | setup_data | None |
| DataLoader.get_data_stats | data_utils.py | Get dataset statistics | None | Dict[str, Any] | logging/debugging | Path.stat |

### Model Loading Functions (utils/model_loader.py)

| Function Name | File | Description | Input Parameters | Output Parameters | Calling Functions | Called Functions |
|--------------|------|-------------|------------------|-------------------|-------------------|------------------|
| ModelLoader.load_model | model_loader.py | Load model from checkpoint/HF | checkpoint_path, model_type, device, dtype, compile, override_args | GPT model | InferencePipeline.load_model, server | GPT.load_from_checkpoint, GPT.from_pretrained |
| ModelLoader.create_model | model_loader.py | Create new model | config: ModelConfig, device, compile | GPT model | Trainer.setup_model | GPT.__init__, torch.compile |
| ModelLoader.get_model_context | model_loader.py | Get autocast context | device_type, dtype | Context manager | InferencePipeline.__init__ | torch.amp.autocast |
| ModelLoader._setup_device | model_loader.py | Setup and validate device | device: str | str (validated device) | load_model | torch.cuda.is_available |
| ModelLoader._get_torch_dtype | model_loader.py | Convert string to torch dtype | dtype: str, device: str | torch.dtype | load_model, get_model_context | None |
| ModelLoader.get_model_info | model_loader.py | Get model information | model: GPT | Dict[str, Any] | server endpoints | model.get_num_params |

### Training Utility Functions (utils/training_utils.py)

| Function Name | File | Description | Input Parameters | Output Parameters | Calling Functions | Called Functions |
|--------------|------|-------------|------------------|-------------------|-------------------|------------------|
| LearningRateScheduler.__init__ | training_utils.py | Initialize LR scheduler | learning_rate, min_lr, warmup_iters, lr_decay_iters, decay_lr | None | Trainer.setup_scheduler | None |
| LearningRateScheduler.get_lr | training_utils.py | Calculate current LR | it: int | float (learning rate) | Trainer.train_step | math.cos |
| LearningRateScheduler.get_schedule_info | training_utils.py | Get schedule parameters | None | Dict[str, Any] | logging | None |
| TrainingLogger.__init__ | training_utils.py | Initialize logger | log_dir, log_interval, use_wandb, wandb_project, wandb_run_name | None | Trainer.setup_logger | logging.basicConfig, wandb.init |
| TrainingLogger.log_iteration | training_utils.py | Log training metrics | iter_num, loss, learning_rate, dt, mfu, extra_metrics | None | Trainer.train | logging, wandb.log |
| TrainingLogger.log_evaluation | training_utils.py | Log eval metrics | iter_num, train_loss, val_loss, extra_metrics | None | Trainer.train | logging, wandb.log |
| TrainingLogger.log_checkpoint | training_utils.py | Log checkpoint save | iter_num, checkpoint_path, best | None | Trainer.save_checkpoint | logging, wandb.log |
| TrainingLogger.finish | training_utils.py | Cleanup and summary | None | None | Trainer.train | wandb.finish |
| get_gradient_stats | training_utils.py | Calculate gradient stats | model: nn.Module | Dict[str, float] | Trainer.train | tensor operations |
| count_parameters | training_utils.py | Count model parameters | model: nn.Module | Dict[str, int] | model info logging | parameters() |

### Inference Functions (utils/inference.py)

| Function Name | File | Description | Input Parameters | Output Parameters | Calling Functions | Called Functions |
|--------------|------|-------------|------------------|-------------------|-------------------|------------------|
| InferencePipeline.__init__ | inference.py | Initialize pipeline | model, tokenizer_name, device, dtype | None | generate.main, server | tiktoken.get_encoding, ModelLoader.get_model_context |
| InferencePipeline.load_model | inference.py | Load model into pipeline | checkpoint_path, model_type, compile, model_config | None | generate.main, server | ModelLoader.load_model |
| InferencePipeline.encode | inference.py | Encode text to tokens | text: str | List[int] | generate, chat_completion | tokenizer.encode |
| InferencePipeline.decode | inference.py | Decode tokens to text | tokens: List[int] | str | generate methods | tokenizer.decode |
| InferencePipeline.generate | inference.py | Main generation method | prompt, config, **kwargs | str/List[str]/Iterator[str] | API endpoints, CLI | _generate_single/batch/stream |
| InferencePipeline._generate_single | inference.py | Generate single sample | idx, prompt_length, config | str | generate | model.generate or _generate_with_top_p |
| InferencePipeline._generate_batch | inference.py | Generate multiple samples | idx, prompt_length, config | List[str] | generate | model.generate or _generate_with_top_p |
| InferencePipeline._generate_stream | inference.py | Stream generation | idx, prompt_length, config | Iterator[str] | generate | model.forward, _apply_top_p |
| InferencePipeline._generate_with_top_p | inference.py | Top-p sampling | idx, config | List[int] | _generate_single/batch | model.forward, _apply_top_p |
| InferencePipeline._apply_top_p | inference.py | Apply nucleus filtering | logits, top_p | torch.Tensor | generation methods | torch.sort, torch.cumsum |
| InferencePipeline._should_stop | inference.py | Check stop conditions | token_id, generated_tokens, config | bool | generation loops | decode |
| InferencePipeline.chat_completion | inference.py | Chat API generation | messages, config, **kwargs | Dict/Iterator[Dict] | server /v1/chat/completions | _format_chat_prompt, generate |
| InferencePipeline._format_chat_prompt | inference.py | Format chat messages | messages: List[Dict] | str | chat_completion | string operations |
| InferencePipeline._stream_chat_completion | inference.py | Stream chat response | prompt, messages, config, **kwargs | Iterator[Dict] | chat_completion | generate |

### Server Functions (server.py)

| Function Name | File | Description | Input Parameters | Output Parameters | Calling Functions | Called Functions |
|--------------|------|-------------|------------------|-------------------|-------------------|------------------|
| TrainingManager.__init__ | server.py | Initialize training manager | None | None | initialize_server | threading.Lock |
| TrainingManager.start_training | server.py | Start training subprocess | train_config: Dict, data_path: str | bool | /api/training/start | subprocess.Popen, threading.Thread |
| TrainingManager._read_logs | server.py | Read training logs | None | None | start_training (thread) | process.stdout.readline |
| TrainingManager.stop_training | server.py | Stop training process | None | bool | /api/training/stop | process.terminate |
| TrainingManager.get_status | server.py | Get training status | None | Dict[str, Any] | /api/training/status | None |
| chat_completions | server.py | Chat API endpoint | None (uses request) | Response | Flask route | InferencePipeline.chat_completion |
| completions | server.py | Completions API endpoint | None (uses request) | Response | Flask route | InferencePipeline.generate |
| list_models | server.py | List available models | None | Response | Flask route | None |
| load_model | server.py | Load model endpoint | None (uses request) | Response | Flask route | InferencePipeline.load_model |
| web_generate | server.py | Web UI generation | None (uses request) | Response | Flask route | InferencePipeline.generate |
| start_training | server.py | Start training endpoint | None (uses request) | Response | Flask route | TrainingManager.start_training |
| stop_training | server.py | Stop training endpoint | None | Response | Flask route | TrainingManager.stop_training |
| training_status | server.py | Get training status | None | Response | Flask route | TrainingManager.get_status |
| model_info | server.py | Get model info | None | Response | Flask route | ModelLoader.get_model_info |
| get_model_config | server.py | Get model config | None | Response | Flask route | model.config access |
| health | server.py | Health check | None | Response | Flask route | None |
| initialize_server | server.py | Initialize server | args: APIConfig | None | main | InferencePipeline.__init__, TrainingManager.__init__ |
| main | server.py | Server entry point | None | None | CLI/script | initialize_server, app.run |

### Generation Functions (generate.py)

| Function Name | File | Description | Input Parameters | Output Parameters | Calling Functions | Called Functions |
|--------------|------|-------------|------------------|-------------------|-------------------|------------------|
| interactive_mode | generate.py | Run interactive REPL | pipeline: InferencePipeline | None | main | pipeline.generate |
| main | generate.py | CLI entry point | None | None | CLI/script | InferencePipeline methods |

## Python Concepts used in this codebase

### 1. Decorators

Decorators modify function behavior. In our code:

```python
@torch.no_grad()  # Disables gradient computation for efficiency
def generate(self, ...):
    ...

@staticmethod  # Method doesn't need self parameter
def _apply_top_p(logits, top_p):
    ...

@dataclass  # Automatically creates __init__, __repr__, etc.
class ModelConfig:
    ...
```

### 2. Context Managers

Context managers handle setup/cleanup automatically:

```python
with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
    # Code here runs with automatic mixed precision
    logits = model(x)
# Precision automatically restored here
```

### 3. Type Hints

Type hints document expected types:

```python
def generate(
    self,
    prompt: Union[str, List[int]],  # Can be string OR list of ints
    config: Optional[GenerationConfig] = None,  # Can be GenerationConfig OR None
    **kwargs  # Accepts any additional keyword arguments
) -> Union[str, List[str], Iterator[str]]:  # Returns one of these types
```

### 4. Dataclasses

Dataclasses reduce boilerplate for data containers:

```python
@dataclass
class ModelConfig:
    n_layer: int = 12  # Default value is 12
    n_head: int = 12
    
# Automatically creates:
# - __init__(self, n_layer=12, n_head=12)
# - __repr__ for printing
# - __eq__ for comparison
```

### 5. Generator Functions

Generators produce values on-demand:

```python
def _generate_stream(self, ...):
    for token in range(max_tokens):
        # Calculate next token
        yield token  # Returns token and pauses until next() called
```

### 6. Module Structure

Python modules use `__init__.py` to define packages:

```
nanoLLM_gpt/
├── __init__.py  # Makes nanoLLM_gpt a package
├── model.py     # Can import as: from nanoLLM_gpt import GPT
└── utils/
    ├── __init__.py  # Makes utils a subpackage
    └── data_utils.py  # Import as: from nanoLLM_gpt.utils import DataLoader
```

### 7. Class Methods vs Static Methods vs Instance Methods

```python
class Example:
    def instance_method(self):  # Needs instance (self)
        return self.value
    
    @classmethod
    def class_method(cls):  # Gets class, not instance
        return cls()  # Can create instances
    
    @staticmethod
    def static_method(x):  # No self or cls
        return x * 2  # Just a regular function in class namespace
```

### 8. Special Methods (Dunder Methods)

```python
class GPT(nn.Module):
    def __init__(self, ...):  # Constructor
        super().__init__()  # Call parent constructor
    
    def __repr__(self):  # String representation
        return f"GPT(n_layer={self.config.n_layer})"
    
    def forward(self, x):  # Special in PyTorch - called by __call__
        # This allows: output = model(input)
        # Instead of: output = model.forward(input)
```

### 9. **kwargs and *args

```python
def function(*args, **kwargs):
    # *args: Captures positional arguments as tuple
    # **kwargs: Captures keyword arguments as dict
    
# Usage:
function(1, 2, 3, name="Alice", age=30)
# args = (1, 2, 3)
# kwargs = {'name': 'Alice', 'age': 30}
```

### 10. F-strings (Formatted String Literals)

```python
name = "GPT"
params = 124_439_808  # Underscores for readability
print(f"Model {name} has {params:,} parameters")  # Output: Model GPT has 124,439,808 parameters
print(f"That's {params/1e6:.1f}M parameters")  # Output: That's 124.4M parameters
```

## Configuration Parameters Reference

### Model Configuration (ModelConfig)

Configuration for GPT model architecture that defines the structural parameters of the model.

| Parameter | Data Type | Default Value | Description | Used in File/Function |
|-----------|-----------|---------------|-------------|----------------------|
| `block_size` | int | 1024 | Maximum sequence length the model can process (context length) | `model.py/GPT.__init__`, `data_utils.py/DataLoader.get_batch` |
| `vocab_size` | int | 50304 | Size of token vocabulary (50257 for GPT-2, padded to nearest multiple of 64 for efficiency) | `model.py/GPT.__init__` (embeddings, output layer) |
| `n_layer` | int | 12 | Number of transformer blocks (model depth) | `model.py/GPT.__init__` |
| `n_head` | int | 12 | Number of attention heads per layer | `model.py/CausalSelfAttention.__init__` |
| `n_embd` | int | 768 | Embedding dimension (hidden size), must be divisible by n_head | All layer sizes in `model.py` |
| `dropout` | float | 0.0 | Dropout probability for regularization (set > 0 for training) | `model.py` (all dropout layers) |
| `bias` | bool | True | Whether to use bias in Linear and LayerNorm layers | `model.py` (layer initialization) |

**Model Size Examples:**
- GPT-2 (124M): n_layer=12, n_head=12, n_embd=768
- GPT-2 Medium (350M): n_layer=24, n_head=16, n_embd=1024
- GPT-2 Large (774M): n_layer=36, n_head=20, n_embd=1280
- GPT-2 XL (1.5B): n_layer=48, n_head=25, n_embd=1600

### Training Configuration (TrainingConfig)

Comprehensive settings for the training process including model architecture, optimization hyperparameters, data settings, and logging options.

#### Distributed Training (DDP) Configuration

The training system supports distributed data parallel (DDP) training across multiple GPUs and nodes:

1. **DDP Parameters:**
   - `backend`: Backend for distributed training ("nccl" for GPU, "gloo" for CPU)
   - `ddp_enabled`: Whether to use distributed training (auto-detected from environment)
   - `nproc_per_node`: Number of processes per node (typically equals number of GPUs)
   - `nnodes`: Number of nodes in distributed training
   - `node_rank`: Rank of current node (0 for master)
   - `master_addr`: IP address of master node
   - `master_port`: Port for distributed communication

2. **Multi-GPU Training:**
   ```bash
   # Single node, 4 GPUs
   torchrun --nproc_per_node=4 -m nanoLLM_gpt.train \
     --data-path data.txt --out-dir out_ddp

   # Or use the web interface with DDP parameters enabled
   ```

3. **Multi-Node Training:**
   ```bash
   # On master node (node 0)
   torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
     --master_addr=192.168.1.100 --master_port=29500 \
     -m nanoLLM_gpt.train --config config.yaml

   # On worker node (node 1)
   torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 \
     --master_addr=192.168.1.100 --master_port=29500 \
     -m nanoLLM_gpt.train --config config.yaml
   ```

4. **Web Interface DDP Support:**
   The web interface includes DDP parameters in the "Distributed Training" section:
   - Backend selection (NCCL/Gloo)
   - Enable DDP checkbox
   - Number of processes per node
   - Number of nodes
   - Master address and port configuration

5. **Important DDP Notes:**
   - Effective batch size = batch_size × gradient_accumulation_steps × world_size
   - Learning rate scaling: Consider scaling LR with world size
   - Each process loads its own data shard automatically
   - Checkpoints are saved only by the master process (rank 0)
   - Use NCCL backend for GPUs (much faster than Gloo)
   - Ensure all nodes have the same environment and code version

### Resume Training Functionality

The training system supports three initialization modes, all utilizing the Model Directory concept:

1. **From Scratch** (`--init-from scratch`):
   - Creates a new model with random weights
   - Requires `--data-path` to be specified
   - Uses default or provided configuration
   - Saves to specified Model Directory (default: "out")

2. **Resume Training** (`--init-from resume`):
   - Loads from Model Directory: `{out_dir}/ckpt.pt` and `{out_dir}/config.yaml`
   - Automatically loads saved configuration from the Model Directory
   - If no config.yaml found, uses default configuration
   - Restores optimizer state and training progress (iter_num, best_val_loss)
   - Data path is optional - if not provided, uses the path from saved config
   - Continues saving to the same Model Directory

3. **Fine-tune Pretrained** (`--init-from gpt2*`):
   - Loads pretrained weights from HuggingFace
   - No config.yaml exists initially (uses model defaults)
   - Requires `--data-path` for training data
   - Auto-creates Model Directory: `out_<model_name>` (e.g., `out_gpt2-medium`)
   - Saves config.yaml during training for future resume
   - Supports models: gpt2, gpt2-medium, gpt2-large, gpt2-xl
   - Checkpoints saved to `out_<model_name>` directory

#### Web Interface HuggingFace Support:
The web interface provides an intuitive way to fine-tune HuggingFace models:
- Select "Fine-tune HuggingFace Model" in the Training Mode section
- Choose from available GPT-2 model sizes
- Upload or provide URL for domain-specific training data
- Model weights are automatically downloaded on first use
- Fine-tuned checkpoints are saved to model-specific directories

#### Configuration Precedence (highest to lowest):
1. Command-line arguments
2. Explicitly provided config file (`--config`)
3. Auto-loaded config.yaml (when resuming)
4. Default values in TrainingConfig

#### Example Resume Workflows with Model Directories:

```bash
# Simple resume - continues from model directory with saved config
gpt-train --init-from resume --out-dir models/my_experiment

# Resume with new data (keeps model architecture from config.yaml)
gpt-train --init-from resume --out-dir models/my_experiment \
  --data-path new_data.txt

# Resume with modified hyperparameters (overrides saved config)
gpt-train --init-from resume --out-dir models/my_experiment \
  --learning-rate 1e-4 --max-iters 10000

# Fine-tune pretrained model with automatic directory
gpt-train --init-from gpt2-medium --data-path domain_data.txt
# Creates: out_gpt2-medium/ckpt.pt and out_gpt2-medium/config.yaml

# Fine-tune with custom model directory
gpt-train --init-from gpt2-large --data-path medical_data.txt \
  --out-dir models/finetuned/gpt2_large_medical
```

| Parameter | Data Type | Default Value | Description | Used in File/Function |
|-----------|-----------|---------------|-------------|----------------------|
| **Model Settings** |
| `model` | ModelConfig | ModelConfig() | Nested model configuration | `train.py/Trainer.setup_model` |
| **I/O Settings** |
| `out_dir` | str | "out" | Directory for checkpoints and logs | `train.py/Trainer.save_checkpoint` |
| `eval_interval` | int | 2000 | How often to evaluate on validation set (iterations) | `train.py/Trainer.train` |
| `log_interval` | int | 1 | How often to log training metrics | `train.py/Trainer.train` |
| `eval_iters` | int | 200 | Number of batches for evaluation | `train.py/Trainer.estimate_loss` |
| `eval_only` | bool | False | If True, only evaluate and exit | `train.py/Trainer.train` |
| `always_save_checkpoint` | bool | True | Save checkpoint every eval (not just best) | `train.py/Trainer.train` |
| `init_from` | str | "scratch" | Initialization mode: 'scratch', 'resume', or 'gpt2*' | `train.py/Trainer.setup_model` |
| **Data Settings** |
| `data_path` | Optional[str] | None | Path to text file or URL (None = tiny shakespeare) | `train.py/Trainer.setup_data` |
| `dataset` | str | "custom" | Dataset name for organization | `data_utils.py/DataPreparer.prepare_data` |
| `batch_size` | int | 12 | Micro batch size per GPU | `data_utils.py/DataLoader.__init__` |
| `gradient_accumulation_steps` | int | 40 | Accumulate gradients for effective larger batch | `train.py/Trainer.train` |
| `train_val_split` | float | 0.0005 | Fraction of data for validation | `data_utils.py/DataPreparer._tokenize_and_save` |
| **Optimizer Settings** |
| `learning_rate` | float | 6e-4 | Peak learning rate | `train.py/Trainer.setup_optimizer` |
| `max_iters` | int | 600000 | Total training iterations | `train.py/Trainer.train` |
| `weight_decay` | float | 1e-1 | L2 penalty (applied to 2D params only) | `model.py/GPT.configure_optimizers` |
| `beta1` | float | 0.9 | Adam beta1 (momentum) | `model.py/GPT.configure_optimizers` |
| `beta2` | float | 0.95 | Adam beta2 (RMSprop term) | `model.py/GPT.configure_optimizers` |
| `grad_clip` | float | 1.0 | Gradient clipping threshold (0 = no clip) | `train.py/Trainer.train_step` |
| **Learning Rate Schedule** |
| `decay_lr` | bool | True | Use learning rate decay | `training_utils.py/LearningRateScheduler` |
| `warmup_iters` | int | 2000 | Linear warmup steps | `training_utils.py/LearningRateScheduler.get_lr` |
| `lr_decay_iters` | int | 600000 | Cosine decay duration | `training_utils.py/LearningRateScheduler.get_lr` |
| `min_lr` | float | 6e-5 | Minimum learning rate | `training_utils.py/LearningRateScheduler.get_lr` |
| **System Settings** |
| `backend` | str | "nccl" | Distributed training backend | `train.py/Trainer.setup_distributed` |
| `device` | str | "cuda" | Training device | `train.py/Trainer.setup_device` |
| `dtype` | str | "bfloat16" | Training precision | `train.py/Trainer.setup_device` |
| `compile` | bool | False | Use PyTorch 2.0 compilation | `model_loader.py/ModelLoader.create_model` |
| `seed` | int | 1337 | Random seed for reproducibility | `train.py/Trainer.__init__` |
| **Logging Settings** |
| `wandb_log` | bool | False | Use Weights & Biases logging | `training_utils.py/TrainingLogger.__init__` |
| `wandb_project` | str | "gpt-training" | W&B project name | `training_utils.py/TrainingLogger.__init__` |
| `wandb_run_name` | str | "gpt-run" | W&B run name | `training_utils.py/TrainingLogger.__init__` |

### Generation Configuration (GenerationConfig)

Controls the text generation process including sampling strategies, length limits, and output formatting.

| Parameter | Data Type | Default Value | Description | Used in File/Function |
|-----------|-----------|---------------|-------------|----------------------|
| `max_new_tokens` | int | 100 | Maximum number of tokens to generate | `model.py/GPT.generate`, `inference.py` |
| `temperature` | float | 0.8 | Sampling temperature (0.0=greedy, 1.0=normal, >1.0=creative) | `model.py/GPT.generate` |
| `top_k` | int | 200 | Only sample from top k tokens (0 = no limit) | `model.py/GPT.generate` |
| `top_p` | float | 1.0 | Nucleus sampling threshold (1.0 = no limit) | `inference.py/InferencePipeline._apply_top_p` |
| `repetition_penalty` | float | 1.0 | Penalty for repeating tokens (1.0 = no penalty) | `inference.py/InferencePipeline` |
| `num_samples` | int | 1 | Number of independent samples to generate | `inference.py/InferencePipeline.generate` |
| `stream` | bool | False | Whether to stream tokens as they're generated | `inference.py/InferencePipeline.generate` |
| `stop_sequences` | Optional[List[str]] | None | Sequences that trigger generation stop | `inference.py/InferencePipeline._should_stop` |

### API Configuration (APIConfig)

Settings for the Flask server that provides REST API endpoints and web interface.

| Parameter | Data Type | Default Value | Description | Used in File/Function |
|-----------|-----------|---------------|-------------|----------------------|
| `host` | str | "0.0.0.0" | Server host address ('0.0.0.0' = all interfaces) | `server.py/main` |
| `port` | int | 8080 | Server port number | `server.py/main` |
| `debug` | bool | False | Enable Flask debug mode | `server.py/main` |
| `model_type` | str | "gpt2" | Default model to load ('gpt2', 'gpt2-medium', etc.) | `server.py/initialize_server` |
| `checkpoint_path` | Optional[str] | None | Path to custom checkpoint (overrides model_type) | `server.py/initialize_server` |
| `device` | str | "cuda" | Compute device ('cuda', 'cpu', 'mps') | `server.py/initialize_server` |
| `dtype` | str | "auto" | Model precision ('auto', 'float32', 'float16', 'bfloat16') | `server.py/initialize_server` |
| `compile` | bool | False | Whether to compile model with PyTorch 2.0 | `server.py/initialize_server` |
| `max_batch_size` | int | 16 | Maximum batch size for parallel requests | `server.py` (request handling) |
| `cors_enabled` | bool | True | Enable CORS for cross-origin requests | `server.py/initialize_server` |

## Detailed Usage Guide

### Installation

#### Basic Installation
```bash
git clone <repository>
cd nanoLLM_gpt
pip install -e .
```

#### Development Installation
```bash
pip install -e ".[dev,datasets,wandb]"
```

#### Docker Installation

##### Building the Docker Image

1. **Build the image with multi-GPU support:**
```bash
# Build the Docker image
docker build -t nanollm-gpt .

# For faster rebuilds with build cache
docker build --cache-from nanollm-gpt -t nanollm-gpt .
```

2. **Build with custom CUDA version (if needed):**
```bash
# Modify the base image in Dockerfile first, then build
docker build --build-arg CUDA_VERSION=11.8 -t nanollm-gpt:cuda11.8 .
```

##### Running the Docker Container

1. **Basic usage with GPU support:**
```bash
# Check environment and available commands
docker run --gpus all -it nanollm-gpt

# Start the web server
docker run --gpus all -p 8080:8080 -it nanollm-gpt gpt-server

# Train a model with persistent storage
docker run --gpus all -v $(pwd)/models:/workspace/nanoLLM_gpt/models \
  -v $(pwd)/data:/workspace/nanoLLM_gpt/data \
  -v $(pwd)/out:/workspace/nanoLLM_gpt/out \
  -it nanollm-gpt gpt-train --data-path data/input.txt
```

2. **Multi-GPU training:**
```bash
# Use all available GPUs
docker run --gpus all -v $(pwd)/models:/workspace/nanoLLM_gpt/models \
  -it nanollm-gpt bash -c "torchrun --nproc_per_node=\$(nvidia-smi -L | wc -l) \
  -m nanoLLM_gpt.train --config config/train_config.yaml"

# Use specific GPUs (e.g., GPU 0 and 1)
docker run --gpus '"device=0,1"' -v $(pwd)/models:/workspace/nanoLLM_gpt/models \
  -it nanollm-gpt bash -c "torchrun --nproc_per_node=2 \
  -m nanoLLM_gpt.train --config config/train_config.yaml"
```

3. **Interactive development with Claude Code:**
```bash
# Run interactive shell with all tools available
docker run --gpus all -it \
  -v $(pwd):/workspace/nanoLLM_gpt \
  -v ~/.config/claude-code:/root/.config/claude-code \
  nanollm-gpt /bin/bash

# Inside container, set up Claude Code (first time only)
claude-code setup

# Use Claude Code for development
claude-code "Help me optimize the training loop"
```

4. **Docker Compose (create docker-compose.yml):**
```yaml
version: '3.8'
services:
  nanollm-gpt:
    image: nanollm-gpt
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    ports:
      - "8080:8080"
    volumes:
      - ./models:/workspace/nanoLLM_gpt/models
      - ./data:/workspace/nanoLLM_gpt/data
      - ./out:/workspace/nanoLLM_gpt/out
      - ./uploads:/workspace/nanoLLM_gpt/uploads
    command: gpt-server
```

Run with: `docker-compose up`

##### Saving and Loading Docker Images

1. **Save the image to a file:**
```bash
# Save to compressed tar file
docker save nanollm-gpt | gzip > nanollm-gpt.tar.gz

# Save with progress bar (requires pv)
docker save nanollm-gpt | pv | gzip > nanollm-gpt.tar.gz
```

2. **Load the image from a file:**
```bash
# Load from compressed tar file
docker load < nanollm-gpt.tar.gz

# Or with gzip
gunzip -c nanollm-gpt.tar.gz | docker load

# With progress bar
pv nanollm-gpt.tar.gz | gunzip | docker load
```

3. **Transfer image between machines:**
```bash
# On source machine
docker save nanollm-gpt | gzip | ssh user@target-machine "gunzip | docker load"

# Using a registry (recommended for production)
# Tag and push to registry
docker tag nanollm-gpt myregistry.com/nanollm-gpt:latest
docker push myregistry.com/nanollm-gpt:latest

# Pull on target machine
docker pull myregistry.com/nanollm-gpt:latest
```

##### Docker Best Practices

1. **Resource Management:**
```bash
# Limit memory and CPU
docker run --gpus all --memory="16g" --cpus="8" \
  -it nanollm-gpt gpt-train --config config/train.yaml

# Set shared memory size for DataLoader workers
docker run --gpus all --shm-size=8g \
  -it nanollm-gpt gpt-train --config config/train.yaml
```

2. **Debugging GPU Issues:**
```bash
# Check GPU visibility in container
docker run --gpus all nanollm-gpt nvidia-smi

# Test PyTorch GPU access
docker run --gpus all nanollm-gpt python -c \
  "import torch; print(f'GPUs available: {torch.cuda.device_count()}')"
```

3. **Development Workflow:**
```bash
# Mount source code for live development
docker run --gpus all -it \
  -v $(pwd)/nanoLLM_gpt:/workspace/nanoLLM_gpt/nanoLLM_gpt \
  -v $(pwd)/models:/workspace/nanoLLM_gpt/models \
  --name nanollm-dev \
  nanollm-gpt /bin/bash

# In another terminal, execute commands in running container
docker exec -it nanollm-dev gpt-train --help
```

### Training Guide

#### Basic Training

1. **Train from scratch with local data:**
```bash
gpt-train --data-path data.txt --max-iters 10000
```

2. **Train with configuration file:**
```yaml
# config/my_training.yaml
model:
  n_layer: 6
  n_head: 8
  n_embd: 512
data_path: data/my_dataset.txt
max_iters: 50000
learning_rate: 3e-4
```
```bash
gpt-train --config config/my_training.yaml
```

3. **Resume training from Model Directory:**
```bash
# Auto-loads config.yaml from the model directory
gpt-train --init-from resume --out-dir models/experiments/run1
```

4. **Fine-tune from pretrained with Model Directory:**
```bash
# Auto-creates out_gpt2/ directory
gpt-train --init-from gpt2 --data-path data.txt --learning-rate 1e-4

# Or specify custom directory
gpt-train --init-from gpt2 --data-path data.txt \
  --out-dir models/finetuned/gpt2_custom --learning-rate 1e-4
```

#### Distributed Training

1. **Single node, multiple GPUs:**
```bash
torchrun --nproc_per_node=4 -m nanoLLM_gpt.train --config config/train.yaml
```

2. **Multiple nodes:**
```bash
# On master node (node 0)
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
  --master_addr=192.168.1.1 --master_port=29500 \
  -m nanoLLM_gpt.train --config config/train.yaml

# On worker node (node 1)  
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 \
  --master_addr=192.168.1.1 --master_port=29500 \
  -m nanoLLM_gpt.train --config config/train.yaml
```

3. **SLURM cluster:**
```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8

srun torchrun \
  --nproc_per_node=$SLURM_GPUS_PER_NODE \
  --nnodes=$SLURM_NNODES \
  --node_rank=$SLURM_NODEID \
  --master_addr=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1) \
  --master_port=29500 \
  -m nanoLLM_gpt.train --config config/train.yaml
```

4. **Using the Web Interface for DDP:**
   - Navigate to the "Train Model" tab
   - Enable "Distributed Training"
   - Set number of processes (GPUs) per node
   - For multi-node: set number of nodes, master address, and port
   - The server will automatically use `torchrun` with your settings

#### Training Best Practices

1. **Gradient Accumulation** for large models on limited memory:
```yaml
batch_size: 4
gradient_accumulation_steps: 128  # Effective batch = 512
```

2. **Mixed Precision** for faster training:
```yaml
dtype: bfloat16  # or float16 for older GPUs
```

3. **Learning Rate Schedule:**
```yaml
warmup_iters: 2000      # Gradual warmup
lr_decay_iters: 100000  # Cosine decay
min_lr: 1e-5            # Don't decay to zero
```

### Text Generation

#### Command-Line Generation

1. **Basic generation:**
```bash
gpt-generate --model gpt2 --prompt "Once upon a time"
```

2. **From checkpoint:**
```bash
gpt-generate --checkpoint model.pt --prompt "The answer is" --max-tokens 50
```

3. **With custom parameters:**
```bash
gpt-generate --model gpt2-large \
  --prompt "In the year 2050" \
  --temperature 0.9 \
  --top-p 0.95 \
  --max-tokens 200
```

4. **Interactive mode:**
```bash
gpt-generate --model gpt2 --interactive
```

5. **From file:**
```bash
gpt-generate --checkpoint model.pt --prompt-file prompts.txt
```

#### Programmatic Generation

```python
from nanoLLM_gpt.utils import InferencePipeline
from nanoLLM_gpt.config import GenerationConfig

# Initialize pipeline
pipeline = InferencePipeline(device='cuda')
pipeline.load_model(checkpoint_path='model.pt')

# Simple generation
text = pipeline.generate("Hello world", max_new_tokens=50)

# Advanced generation
config = GenerationConfig(
    max_new_tokens=200,
    temperature=0.9,
    top_p=0.95,
    top_k=50,
    repetition_penalty=1.2
)
text = pipeline.generate("Once upon a time", config)

# Streaming
config.stream = True
for token in pipeline.generate("The future is", config):
    print(token, end='', flush=True)
```

### API Server

#### Starting the Server

1. **Basic start:**
```bash
gpt-server
```

2. **With configuration:**
```bash
gpt-server --config server_config.yaml
```

3. **With specific model:**
```bash
gpt-server --checkpoint model.pt --port 8080
```

4. **Debug mode:**
```bash
gpt-server --debug --port 5000
```

#### API Usage Examples

1. **Chat Completions** (OpenAI-compatible):
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="gpt",
    messages=[
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Explain quantum computing."}
    ],
    temperature=0.7,
    max_tokens=150
)
print(response.choices[0].message.content)
```

2. **Text Completions:**
```python
response = client.completions.create(
    model="gpt",
    prompt="The key to happiness is",
    max_tokens=100,
    temperature=0.8,
    top_k=50
)
print(response.choices[0].text)
```

3. **Streaming:**
```python
stream = client.chat.completions.create(
    model="gpt",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end='')
```

4. **cURL Examples:**
```bash
# Chat completion
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.8
  }'

# List models
curl http://localhost:8080/v1/models
```

### Web Interface

Access the web interface at `http://localhost:8080` after starting the server.

#### Features:

1. **Text Generation Tab:**
   - **Model Directory**: Single field for loading models
   - Load from HuggingFace or custom checkpoint directories
   - Interactive prompt input
   - Real-time parameter adjustment with sliders
   - Visual feedback during generation
   - Generated text display

2. **Training Tab:**
   - **Model Directory**: Unified output location field
   - Three training modes with automatic directory management:
     - New Training: Custom directory (default: "out")
     - Resume Training: Locked to existing directory
     - Fine-tune HuggingFace: Auto-generates (e.g., "out_gpt2-medium")
   - Upload training data or provide URL
   - Configure model architecture
   - Set training hyperparameters
   - Distributed training (DDP) support
   - Start/stop training
   - Real-time log streaming

3. **API Documentation Tab:**
   - Endpoint reference
   - Example code snippets
   - cURL commands
   - Python client examples

## Extending the Codebase

### Adding a New Model Architecture

1. **Create new model class** in `model.py`:
```python
class GPTWithMoE(GPT):
    """GPT with Mixture of Experts"""
    
    def __init__(self, config):
        super().__init__(config)
        # Add MoE layers
        self.moe_layers = nn.ModuleList([
            MoELayer(config) for _ in range(config.n_moe_layers)
        ])
    
    def forward(self, idx, targets=None):
        # Modified forward pass
        ...
```

2. **Update configuration** in `config.py`:
```python
@dataclass
class MoEConfig(ModelConfig):
    """Configuration for MoE model"""
    n_experts: int = 8
    n_moe_layers: int = 4
    expert_capacity: float = 1.25
```

3. **Update model loader** in `utils/model_loader.py`:
```python
@staticmethod
def create_model(config, device='cuda', compile=False):
    if isinstance(config, MoEConfig):
        model = GPTWithMoE(config)
    else:
        model = GPT(config)
    # ... rest of function
```

### Adding New Training Strategies

1. **Add to TrainingConfig** in `config.py`:
```python
@dataclass
class TrainingConfig:
    # ... existing fields
    
    # New optimizer settings
    use_adam_w3: bool = False
    momentum: float = 0.9
    
    # New scheduling
    use_cyclic_lr: bool = False
    cycle_length: int = 10000
```

2. **Update optimizer configuration** in `model.py`:
```python
def configure_optimizers(self, weight_decay, learning_rate, betas, 
                        device_type, use_adam_w3=False, momentum=0.9):
    if use_adam_w3:
        # Configure AdamW3 optimizer
        optimizer = AdamW3(optim_groups, lr=learning_rate, 
                          momentum=momentum, betas=betas)
    else:
        # Existing AdamW logic
        ...
```

3. **Update scheduler** in `utils/training_utils.py`:
```python
class CyclicLRScheduler(LearningRateScheduler):
    """Cyclic learning rate schedule"""
    
    def __init__(self, base_lr, max_lr, cycle_length):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.cycle_length = cycle_length
    
    def get_lr(self, it):
        cycle_pos = it % self.cycle_length
        return self.base_lr + (self.max_lr - self.base_lr) * \
               (1 + np.cos(np.pi * cycle_pos / self.cycle_length)) / 2
```

### Adding New Generation Methods

1. **Add to GenerationConfig** in `config.py`:
```python
@dataclass
class GenerationConfig:
    # ... existing fields
    
    # New sampling strategies
    use_beam_search: bool = False
    beam_width: int = 5
    length_penalty: float = 1.0
    
    # Constrained generation
    force_words: Optional[List[str]] = None
    banned_words: Optional[List[str]] = None
```

2. **Implement in InferencePipeline** in `utils/inference.py`:
```python
def _beam_search(self, idx, config):
    """Beam search implementation"""
    beam_scores = torch.zeros(config.beam_width, device=idx.device)
    beam_sequences = idx.repeat(config.beam_width, 1)
    
    for _ in range(config.max_new_tokens):
        # Get logits for all beams
        logits = self.model(beam_sequences)[0][:, -1, :]
        
        # Apply length penalty
        scores = logits / config.length_penalty
        
        # Select top beams
        # ... beam search logic
    
    return beam_sequences[0]  # Best sequence

def generate(self, prompt, config=None, **kwargs):
    # ... existing code
    
    if config.use_beam_search:
        return self._beam_search(idx, config)
    # ... rest of function
```

### Adding New API Endpoints

1. **Add to server.py**:
```python
@app.route('/v1/embeddings', methods=['POST'])
def embeddings():
    """Generate embeddings for input text"""
    data = request.json
    input_text = data.get('input', '')
    
    # Get embeddings from model
    with torch.no_grad():
        tokens = model_pipeline.encode(input_text)
        token_tensor = torch.tensor(tokens).unsqueeze(0)
        embeddings = model_pipeline.model.transformer.wte(token_tensor)
        
        # Average pooling
        embedding = embeddings.mean(dim=1).squeeze().tolist()
    
    return jsonify({
        "object": "embedding",
        "embedding": embedding,
        "model": config.model_type,
        "usage": {
            "prompt_tokens": len(tokens),
            "total_tokens": len(tokens)
        }
    })
```

2. **Update web interface** in `templates/index.html`:
```javascript
async function getEmbedding(text) {
    const response = await fetch('/v1/embeddings', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({input: text})
    });
    
    const result = await response.json();
    return result.embedding;
}
```

### Adding Data Processing Extensions

1. **Add new data format support** in `utils/data_utils.py`:
```python
def _load_jsonl_data(self, path):
    """Load data from JSONL format"""
    texts = []
    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line)
            texts.append(data.get('text', ''))
    return '\n\n'.join(texts)

def _load_parquet_data(self, path):
    """Load data from Parquet format"""
    import pandas as pd
    df = pd.read_parquet(path)
    return '\n\n'.join(df['text'].tolist())
```

2. **Add data augmentation**:
```python
class DataAugmenter:
    """Apply augmentations during training"""
    
    def __init__(self, strategies=['shuffle', 'mask']):
        self.strategies = strategies
    
    def augment_batch(self, batch):
        """Apply augmentations to batch"""
        if 'shuffle' in self.strategies:
            # Shuffle sentences within sequences
            batch = self._shuffle_sentences(batch)
        
        if 'mask' in self.strategies:
            # Random token masking
            batch = self._random_mask(batch)
        
        return batch
```

## Troubleshooting

### Common Issues and Solutions

1. **CUDA Out of Memory:**
   - Reduce batch size: `--batch-size 4`
   - Increase gradient accumulation: `--gradient-accumulation-steps 64`
   - Use mixed precision: `--dtype float16`
   - Enable gradient checkpointing (if implemented)

2. **Slow Training:**
   - Enable compilation: `--compile`
   - Use mixed precision: `--dtype bfloat16`
   - Check data loading isn't bottleneck
   - Use faster optimizer: disable bias correction

3. **Poor Generation Quality:**
   - Adjust temperature: lower for less random
   - Use top-p sampling: `--top-p 0.9`
   - Increase model size or training time
   - Check training loss convergence

4. **API Connection Errors:**
   - Verify server is running: `curl http://localhost:8080/health`
   - Check firewall/port settings
   - Ensure model is loaded (check logs)
   - Verify CORS settings for browser access

5. **Distributed Training Issues:**
   - Check network connectivity between nodes
   - Verify same PyTorch/CUDA versions
   - Use `NCCL_DEBUG=INFO` for debugging
   - Ensure shared filesystem for checkpoints

### Performance Optimization

1. **Training Speed:**
   - Use larger batch sizes with gradient accumulation
   - Enable torch.compile: `--compile`
   - Use Flash Attention (PyTorch 2.0+)
   - Profile with `torch.profiler`

2. **Inference Speed:**
   - Batch multiple requests
   - Use key-value caching
   - Compile model for inference
   - Use appropriate precision (fp16/int8)

3. **Memory Usage:**
   - Clear cache regularly: `torch.cuda.empty_cache()`
   - Use gradient checkpointing for large models
   - Offload optimizer states to CPU
   - Use DeepSpeed ZeRO optimizations

## Best Practices

### Code Organization

1. **Module Design:**
   - Single responsibility per module
   - Clear public APIs
   - Comprehensive docstrings
   - Type hints throughout

2. **Configuration Management:**
   - Use dataclasses for type safety
   - Provide sensible defaults
   - Validate at load time
   - Document all parameters

3. **Error Handling:**
   - Fail fast with clear messages
   - Provide recovery suggestions
   - Log errors appropriately
   - Handle edge cases gracefully

### Development Workflow

1. **Testing:**
   ```bash
   # Run all tests
   pytest tests/
   
   # Run with coverage
   pytest --cov=nanoLLM_gpt tests/
   
   # Run specific test
   pytest tests/test_model_loader.py::TestModelLoader::test_create_model
   ```

2. **Code Quality:**
   ```bash
   # Format code
   black nanoLLM_gpt/
   
   # Check style
   flake8 nanoLLM_gpt/
   
   # Type checking
   mypy nanoLLM_gpt/
   ```

3. **Documentation:**
   - Update docstrings when changing functions
   - Keep README.md current
   - Document breaking changes
   - Provide migration guides

### Production Deployment

1. **Model Serving:**
   - Use production WSGI server (gunicorn)
   - Enable request queuing
   - Implement rate limiting
   - Monitor resource usage

2. **Scaling:**
   - Horizontal scaling with load balancer
   - Model replication across GPUs
   - Caching for repeated requests
   - Async request handling

3. **Monitoring:**
   - Track inference latency
   - Monitor GPU utilization
   - Log error rates
   - Set up alerts for failures

### Security Considerations

1. **API Security:**
   - Implement authentication
   - Rate limit by API key
   - Validate input sizes
   - Sanitize file uploads

2. **Model Security:**
   - Verify checkpoint integrity
   - Limit generation length
   - Filter inappropriate content
   - Implement usage quotas

## Summary

This handbook provides a comprehensive reference for the nanoLLM_gpt project. The modular architecture ensures clean separation of concerns, making the codebase maintainable and extensible. The training and inference flows demonstrate how components interact to provide a complete GPT implementation.

For hands-on usage, refer to the README.md file for practical examples and quick start guides.