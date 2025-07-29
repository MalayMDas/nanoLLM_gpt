"""
Optimized GPT training script with support for single GPU and distributed training.

This is the main training script that orchestrates the entire training process
for GPT models. It handles everything from data loading to optimization, evaluation,
and checkpointing.

## Key Features:
- Single GPU and multi-GPU distributed training via DDP
- Mixed precision training with automatic mixed precision (AMP)
- Gradient accumulation for larger effective batch sizes
- Checkpoint saving and resuming for fault tolerance
- Weights & Biases integration for experiment tracking
- Learning rate scheduling with cosine decay
- Automatic model compilation with PyTorch 2.0

## Training Pipeline:
1. **Initialization**: Set up distributed training, device, and random seeds
2. **Data Loading**: Prepare tokenized data using DataPreparer and DataLoader
3. **Model Setup**: Initialize or load model, configure optimizer
4. **Training Loop**:
   - Forward pass and loss computation
   - Gradient accumulation
   - Optimizer step with gradient clipping
   - Learning rate scheduling
   - Periodic evaluation and checkpointing
5. **Cleanup**: Save final checkpoint and close distributed processes

## Usage Examples:
```bash
# Train with configuration file
python train.py --config config.yaml

# Train with command-line arguments
python train.py --data-path input.txt --max-iters 5000 --batch-size 12

# Resume training from checkpoint
python train.py --init-from resume --out-dir out

# Multi-GPU training (4 GPUs)
torchrun --nproc_per_node=4 train.py --config config.yaml

# Custom model architecture
python train.py --n-layer 24 --n-head 16 --n-embd 1024 --data-path data.txt
```

## Configuration Priority:
1. Command-line arguments (highest priority)
2. Configuration file (--config)
3. Default values in TrainingConfig

## Output Structure:
```
out_dir/
├── ckpt.pt          # Latest checkpoint
├── config.yaml      # Training configuration
├── train_log.txt    # Training metrics log
└── wandb/           # Weights & Biases logs (if enabled)
```
"""

import os
import time
import argparse
import warnings
from contextlib import nullcontext
from pathlib import Path

import torch

# Configure matmul precision for better performance on modern GPUs
# For PyTorch 2.0+, use set_float32_matmul_precision
# For development versions, we suppress the deprecation warning as the API is in flux
if torch.cuda.is_available():
    torch_version = torch.__version__
    if "dev" in torch_version or "+" in torch_version:
        # Development version - suppress the warning as APIs are changing
        warnings.filterwarnings("ignore", message=".*TF32.*", category=UserWarning)
        warnings.filterwarnings(
            "ignore", message=".*float32_matmul_precision.*", category=UserWarning
        )

    # Set precision for all versions
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        # If the API is not available or fails, continue without it
        pass

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from nanoLLM_gpt import GPT
from nanoLLM_gpt.config import TrainingConfig, ConfigLoader, load_config
from nanoLLM_gpt.utils import (
    ModelLoader,
    DataPreparer,
    DataLoader,
    LearningRateScheduler,
    TrainingLogger,
    get_gradient_stats,
)


class Trainer:
    """
    Main trainer class for GPT model.

    This class encapsulates the entire training process, handling:
    - Distributed training setup (DDP)
    - Device and precision configuration
    - Data loading and batching
    - Model initialization or loading
    - Optimizer and scheduler setup
    - Training loop with gradient accumulation
    - Evaluation and checkpointing
    - Logging and metrics tracking

    The trainer follows a modular design where each setup method can be
    overridden for custom behavior.

    Attributes:
        config (TrainingConfig): Complete training configuration
        model (GPT): The GPT model being trained
        optimizer (torch.optim.Optimizer): AdamW optimizer
        data_loader (DataLoader): Handles batch generation
        scheduler (LearningRateScheduler): LR scheduling
        logger (TrainingLogger): Metrics logging
        iter_num (int): Current training iteration
        best_val_loss (float): Best validation loss seen

    Flow:
        1. __init__: Initialize all components
        2. train(): Run the main training loop
        3. estimate_loss(): Evaluate on train/val sets
        4. save_checkpoint(): Save model state
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize trainer with configuration.

        Args:
            config (TrainingConfig): Complete training configuration including
                                   model config, data paths, hyperparameters

        Setup sequence:
            1. Distributed training (DDP) if multi-GPU
            2. Device and random seeds
            3. Logging (file and optional W&B)
            4. Data loading and tokenization
            5. Model initialization/loading
            6. Optimizer (AdamW with smart weight decay)
            7. Learning rate scheduler
        """
        self.config = config
        self.setup_distributed()
        self.setup_device()
        self.setup_logger()
        self.setup_data()
        self.setup_model()
        self.setup_optimizer()
        self.setup_scheduler()

        # Training state
        self.iter_num = 0
        self.best_val_loss = 1e9
        self.local_iter_num = 0
        self.running_mfu = -1.0

    def setup_distributed(self):
        """
        Initialize distributed training if running with DDP.

        Detects if running under torchrun/DDP by checking environment variables.
        Sets up process groups and calculates effective batch size.

        Environment variables (set by torchrun):
            - RANK: Global rank of this process
            - LOCAL_RANK: Local GPU index on this node
            - WORLD_SIZE: Total number of processes

        Effects:
            - Initializes process group for communication
            - Adjusts gradient accumulation for world size
            - Sets master_process flag for logging/saving

        Called by: __init__
        """
        self.ddp = int(os.environ.get("RANK", -1)) != -1

        if self.ddp:
            init_process_group(backend=self.config.backend)
            self.ddp_rank = int(os.environ["RANK"])
            self.ddp_local_rank = int(os.environ["LOCAL_RANK"])
            self.ddp_world_size = int(os.environ["WORLD_SIZE"])
            self.master_process = self.ddp_rank == 0
            self.seed_offset = self.ddp_rank

            # Scale gradient accumulation by world size
            assert self.config.gradient_accumulation_steps % self.ddp_world_size == 0
            self.config.gradient_accumulation_steps //= self.ddp_world_size
        else:
            self.master_process = True
            self.seed_offset = 0
            self.ddp_world_size = 1

        # Calculate tokens per iteration
        self.tokens_per_iter = (
            self.config.gradient_accumulation_steps
            * self.ddp_world_size
            * self.config.batch_size
            * self.config.model.block_size
        )

        if self.master_process:
            print(f"Tokens per iteration: {self.tokens_per_iter:,}")

    def setup_device(self):
        """
        Configure device and precision settings.

        Sets up:
            - CUDA device (specific GPU for DDP)
            - Random seeds for reproducibility
            - Precision settings (bfloat16 if available)
            - Autocast and GradScaler for mixed precision

        The random seed is offset by rank to ensure different
        data ordering across processes while maintaining reproducibility.

        Called by: __init__
        """
        if self.ddp:
            self.device = f"cuda:{self.ddp_local_rank}"
            torch.cuda.set_device(self.device)
        else:
            self.device = self.config.device

        # Set random seed with rank offset for data diversity
        torch.manual_seed(self.config.seed + self.seed_offset)

        # Enable TF32 for better performance
        if "cuda" in self.device:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Setup mixed precision
        self.device_type = "cuda" if "cuda" in self.device else "cpu"
        ptdtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[self.config.dtype]

        self.ctx = (
            nullcontext()
            if self.device_type == "cpu"
            else torch.amp.autocast(device_type=self.device_type, dtype=ptdtype)
        )

        # Setup gradient scaler for fp16
        self.scaler = torch.amp.GradScaler(
            self.device_type, enabled=(self.config.dtype == "float16")
        )

    def setup_logger(self):
        """
        Setup training logger for metrics tracking.

        Creates a TrainingLogger instance that handles:
            - Console output
            - File logging to out_dir/train_log.txt
            - Weights & Biases integration (if enabled)

        Only the master process logs to avoid duplicate outputs
        in distributed training.

        Called by: __init__
        Calls: TrainingLogger (from utils.training_utils)
        """
        if self.master_process:
            self.logger = TrainingLogger(
                log_dir=self.config.out_dir,
                log_interval=self.config.log_interval,
                use_wandb=self.config.wandb_log,
                wandb_project=self.config.wandb_project,
                wandb_run_name=self.config.wandb_run_name,
            )
        else:
            self.logger = None

    def setup_data(self):
        """
        Setup data loader for training.

        This method:
        1. Prepares data using DataPreparer:
           - Downloads/loads raw text
           - Tokenizes using tiktoken
           - Splits into train/val sets
           - Saves as memory-mapped binary files

        2. Creates DataLoader for efficient batching:
           - Memory-mapped data access
           - Random sampling for training
           - Sequential sampling for validation

        Data is only prepared once and cached for subsequent runs.

        Called by: __init__
        Calls:
            - DataPreparer.prepare_data() (tokenization)
            - DataLoader (batch generation)
        """
        # Prepare data
        preparer = DataPreparer()
        data_dir = preparer.prepare_data(
            data_path=self.config.data_path,
            dataset_name=self.config.dataset,
            train_val_split=self.config.train_val_split,
        )

        # Create data loader
        self.data_loader = DataLoader(
            data_dir=data_dir,
            block_size=self.config.model.block_size,
            batch_size=self.config.batch_size,
            device=self.device,
            device_type=self.device_type,
        )

        # Update vocab size if needed
        vocab_size = self.data_loader.get_vocab_size()
        if vocab_size and self.config.init_from == "scratch":
            self.config.model.vocab_size = vocab_size

    def setup_model(self):
        """
        Initialize or load the model.

        Three initialization modes:
        1. 'scratch': Create new model with random weights
        2. 'resume': Load checkpoint from out_dir/ckpt.pt
        3. 'gpt2*': Load pretrained GPT-2 model from HuggingFace

        Also handles:
            - Model compilation with PyTorch 2.0
            - Block size adjustment
            - DDP wrapping for distributed training
            - Loading training state (iter_num, best_val_loss)

        Sets:
            self.model: The model (possibly wrapped in DDP)
            self.raw_model: Unwrapped model for optimizer setup
            self.iter_num: Starting iteration (0 or from checkpoint)
            self.best_val_loss: Best validation loss seen

        Called by: __init__
        Calls:
            - ModelLoader.create_model() (for new models)
            - GPT.load_from_checkpoint() (for resuming)
            - ModelLoader.load_model() (for pretrained)
        """
        if self.config.init_from == "scratch":
            # Create new model
            print("Initializing new model from scratch")
            self.model = ModelLoader.create_model(
                self.config.model, self.device, self.config.compile
            )

        elif self.config.init_from == "resume":
            # Resume from checkpoint
            print(f"Resuming training from {self.config.out_dir}")
            checkpoint_path = Path(self.config.out_dir) / "ckpt.pt"
            self.model = GPT.load_from_checkpoint(
                checkpoint_path, self.device, self.config.compile
            )

            # Load training state
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )
            self.iter_num = checkpoint.get("iter_num", 0)
            self.best_val_loss = checkpoint.get("best_val_loss", 1e9)

        elif self.config.init_from.startswith("gpt2"):
            # Load pretrained model
            print(f"Initializing from pretrained: {self.config.init_from}")
            self.model = ModelLoader.load_model(
                model_type=self.config.init_from,
                device=self.device,
                compile=self.config.compile,
                override_args={"dropout": self.config.model.dropout},
            )

        # Crop block size if needed
        if self.config.model.block_size < self.model.config.block_size:
            self.model.crop_block_size(self.config.model.block_size)

        # Wrap in DDP if distributed
        if self.ddp:
            self.model = DDP(self.model, device_ids=[self.ddp_local_rank])

        # Store unwrapped model reference
        self.raw_model = self.model.module if self.ddp else self.model

    def setup_optimizer(self):
        """
        Configure AdamW optimizer.

        Uses the model's configure_optimizers method which:
            - Separates parameters into decay/no-decay groups
            - Applies weight decay only to 2D parameters
            - Uses fused AdamW on CUDA for efficiency

        Also loads optimizer state when resuming training.

        Called by: __init__
        Calls: GPT.configure_optimizers()
        """
        self.optimizer = self.raw_model.configure_optimizers(
            weight_decay=self.config.weight_decay,
            learning_rate=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            device_type=self.device_type,
        )

        # Load optimizer state if resuming
        if self.config.init_from == "resume":
            checkpoint_path = Path(self.config.out_dir) / "ckpt.pt"
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )
            if "optimizer" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer"])

    def setup_scheduler(self):
        """
        Setup learning rate scheduler.

        Creates a scheduler that implements:
            - Linear warmup from 0 to learning_rate
            - Cosine decay to min_lr after warmup
            - Optional constant LR (if decay_lr=False)

        Called by: __init__
        Calls: LearningRateScheduler (from utils)
        """
        self.lr_scheduler = LearningRateScheduler(
            learning_rate=self.config.learning_rate,
            min_lr=self.config.min_lr,
            warmup_iters=self.config.warmup_iters,
            lr_decay_iters=self.config.lr_decay_iters,
            decay_lr=self.config.decay_lr,
        )

    @torch.no_grad()
    def estimate_loss(self):
        """
        Estimate loss over multiple batches.

        Evaluates model on both train and validation sets by:
            - Switching model to eval mode (disables dropout)
            - Averaging loss over eval_iters batches
            - Using no_grad context for efficiency

        Returns:
            dict: {'train': avg_train_loss, 'val': avg_val_loss}

        Called by: train() (at eval_interval)
        Calls:
            - DataLoader.get_batch()
            - Model.forward()
        """
        out = {}
        self.model.eval()

        for split in ["train", "val"]:
            losses = torch.zeros(self.config.eval_iters)
            for k in range(self.config.eval_iters):
                X, Y = self.data_loader.get_batch(split)
                with self.ctx:
                    logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()

        self.model.train()
        return out

    def save_checkpoint(self, losses):
        """
        Save model checkpoint.

        Saves checkpoint when:
            - Validation loss improves (new best)
            - always_save_checkpoint is True

        Checkpoint includes:
            - Model state dict
            - Optimizer state (for resuming)
            - Training metadata (iter_num, best_val_loss)
            - Full configuration

        Also saves config.yaml for easy inspection.

        Args:
            losses (dict): {'train': loss, 'val': loss} from estimate_loss

        Called by: train() (after evaluation)
        Calls:
            - GPT.save_checkpoint()
            - ConfigLoader.save_to_file()
        """
        if self.iter_num == 0 or not self.master_process:
            return

        if losses["val"] < self.best_val_loss or self.config.always_save_checkpoint:
            self.best_val_loss = losses["val"]

            checkpoint_path = Path(self.config.out_dir) / "ckpt.pt"
            self.raw_model.save_checkpoint(
                checkpoint_path=checkpoint_path,
                optimizer=self.optimizer,
                iter_num=self.iter_num,
                best_val_loss=self.best_val_loss,
                config=self.config.__dict__,
            )

            # Save config file alongside checkpoint
            config_path = Path(self.config.out_dir) / "config.yaml"
            ConfigLoader.save_to_file(self.config, config_path)

            if self.logger:
                self.logger.log_checkpoint(
                    self.iter_num,
                    str(checkpoint_path),
                    best=(losses["val"] == self.best_val_loss),
                )

    def train_step(self, X, Y):
        """
        Execute single training step with gradient accumulation.

        Implements the core training logic:
        1. Update learning rate based on schedule
        2. Gradient accumulation loop:
           - Forward pass with autocast
           - Scale loss by accumulation steps
           - Backward pass with gradient scaling
           - Prefetch next batch during backward
        3. Gradient clipping (if enabled)
        4. Optimizer step and zero gradients

        For DDP, gradient sync is disabled until the last
        accumulation step for efficiency.

        Args:
            X (torch.Tensor): Input batch (batch_size, block_size)
            Y (torch.Tensor): Target batch (batch_size, block_size)

        Returns:
            tuple: (loss_value, gradient_norm)

        Called by: train() (main loop)
        """
        # Update learning rate
        lr = self.lr_scheduler.get_lr(self.iter_num)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        # Gradient accumulation
        for micro_step in range(self.config.gradient_accumulation_steps):
            if self.ddp:
                # Sync gradients only on last micro-step
                self.model.require_backward_grad_sync = (
                    micro_step == self.config.gradient_accumulation_steps - 1
                )

            with self.ctx:
                logits, loss = self.model(X, Y)
                loss = loss / self.config.gradient_accumulation_steps

            # Get next batch while computing gradients
            X, Y = self.data_loader.get_batch("train")

            # Backward pass
            self.scaler.scale(loss).backward()

        # Gradient clipping
        if self.config.grad_clip != 0.0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip
            )

        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        return loss, lr, X, Y

    def train(self):
        """
        Main training loop.

        This is the core training method that:
        1. Performs initial evaluation
        2. Runs training iterations:
           - Calls train_step for gradient updates
           - Evaluates periodically on train/val sets
           - Saves checkpoints when validation improves
           - Logs metrics (loss, LR, MFU, gradients)
        3. Handles cleanup (saves final checkpoint, closes logger)

        The loop continues until max_iters is reached. Early stopping
        can be implemented by monitoring validation loss.

        Training efficiency is measured via MFU (Model FLOPs Utilization),
        which compares actual throughput to theoretical peak.

        Called by: main() (after Trainer initialization)
        Calls:
            - estimate_loss() (for evaluation)
            - train_step() (for parameter updates)
            - save_checkpoint() (when val loss improves)
            - Various logging methods
        """
        # Create output directory
        if self.master_process:
            os.makedirs(self.config.out_dir, exist_ok=True)

        # Initial evaluation
        if self.config.eval_interval > 0:
            losses = self.estimate_loss()
            if self.master_process:
                print(
                    f"Initial loss - train: {losses['train']:.4f}, val: {losses['val']:.4f}"
                )
                if self.logger:
                    self.logger.log_evaluation(0, losses["train"], losses["val"])

        # Exit if eval only
        if self.config.eval_only:
            return

        # Initial batch
        X, Y = self.data_loader.get_batch("train")
        t0 = time.time()

        # Training loop
        while self.iter_num <= self.config.max_iters:
            # Evaluation
            if self.iter_num % self.config.eval_interval == 0 and self.iter_num > 0:
                losses = self.estimate_loss()
                if self.master_process:
                    print(
                        f"step {self.iter_num}: "
                        f"train loss {losses['train']:.4f}, "
                        f"val loss {losses['val']:.4f}"
                    )

                    if self.logger:
                        self.logger.log_evaluation(
                            self.iter_num, losses["train"], losses["val"]
                        )

                    self.save_checkpoint(losses)

            # Training step
            loss, lr, X, Y = self.train_step(X, Y)

            # Timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            if self.iter_num % self.config.log_interval == 0 and self.master_process:
                # Calculate MFU
                if self.local_iter_num >= 5:
                    mfu = self.raw_model.estimate_mfu(
                        self.config.batch_size
                        * self.config.gradient_accumulation_steps,
                        dt,
                    )
                    self.running_mfu = (
                        mfu
                        if self.running_mfu == -1.0
                        else 0.9 * self.running_mfu + 0.1 * mfu
                    )
                else:
                    mfu = None

                # Get gradient stats
                grad_stats = get_gradient_stats(self.model)

                # Log metrics
                if self.logger:
                    self.logger.log_iteration(
                        iter_num=self.iter_num,
                        loss=loss.item() * self.config.gradient_accumulation_steps,
                        learning_rate=lr,
                        dt=dt,
                        mfu=self.running_mfu if self.running_mfu != -1.0 else None,
                        extra_metrics=grad_stats,
                    )

            self.iter_num += 1
            self.local_iter_num += 1

        # Cleanup
        if self.logger:
            self.logger.finish()

        if self.ddp:
            destroy_process_group()


def main():
    """
    Main entry point for the training script.

    Handles:
    1. Command-line argument parsing
    2. Configuration loading (file or CLI args)
    3. Trainer initialization
    4. Training execution

    Configuration priority:
        1. Command-line arguments (highest)
        2. Config file (--config)
        3. Default values

    This function is called when:
        - Running directly: python train.py
        - As module: python -m nanoLLM_gpt.train
        - Via entry point: gpt-train
    """
    parser = argparse.ArgumentParser(
        description="Train GPT models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Config file argument
    parser.add_argument(
        "--config", type=str, help="Path to configuration file (YAML or JSON)"
    )

    # Add all config fields as arguments
    ConfigLoader.add_config_arguments(parser, TrainingConfig)

    args = parser.parse_args()

    # Load configuration
    if args.config:
        config = ConfigLoader.load_from_file(args.config, TrainingConfig)
        # Override with command-line arguments
        config = load_config(TrainingConfig, args.config, args)
    else:
        config = ConfigLoader.create_config_from_args(args, TrainingConfig)

    # Create trainer and run
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
