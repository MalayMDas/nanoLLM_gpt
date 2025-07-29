"""
Training utilities including learning rate scheduling and logging.

This module provides essential utilities for the training process, including
learning rate scheduling, comprehensive logging, and gradient monitoring.

## Key Components:

1. **LearningRateScheduler**: Implements cosine decay with linear warmup
   - Linear warmup from 0 to max LR
   - Cosine decay to minimum LR
   - Constant LR option for experimentation

2. **TrainingLogger**: Unified logging interface
   - Console and file logging
   - Weights & Biases integration
   - Metric tracking and summarization

3. **Utility Functions**:
   - get_gradient_stats(): Monitor gradient health
   - count_parameters(): Parameter counting by category

## Learning Rate Schedule:
```
LR
│     ┌─── Max LR
│    ╱ ╲
│   ╱   ╲_____ Min LR
│  ╱
└──────────────► Iterations
  Warmup  Decay
```

## Usage Example:
```python
# Setup scheduler
scheduler = LearningRateScheduler(
    learning_rate=6e-4,
    min_lr=6e-5,
    warmup_iters=2000,
    lr_decay_iters=600000
)

# Setup logger
logger = TrainingLogger(
    log_dir='out',
    use_wandb=True,
    wandb_project='gpt-training'
)

# During training
lr = scheduler.get_lr(iter_num)
logger.log_iteration(iter_num, loss, lr, dt=step_time)
```
"""

import math
import time
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

import torch
import numpy as np


class LearningRateScheduler:
    """
    Cosine learning rate scheduler with linear warmup.

    Implements a learning rate schedule commonly used for transformer training:
    1. Linear warmup: LR increases linearly from 0 to max
    2. Cosine decay: LR decreases following cosine curve to min
    3. Constant: LR stays at min after decay period

    This schedule helps with:
    - Stable early training (warmup prevents instability)
    - Efficient learning (high LR after warmup)
    - Fine convergence (low LR at end)

    Attributes:
        learning_rate (float): Maximum learning rate (peak after warmup)
        min_lr (float): Minimum learning rate (after full decay)
        warmup_iters (int): Number of warmup steps
        lr_decay_iters (int): Total iterations for decay
        decay_lr (bool): Whether to use decay (False = constant LR)

    Methods:
        get_lr(): Calculate LR for given iteration
        get_schedule_info(): Return schedule parameters

    Mathematical Formula:
        - Warmup: lr = max_lr * (iter + 1) / (warmup + 1)
        - Decay: lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π * progress))

    Called by:
        - train.Trainer.train_step() every iteration
    """

    def __init__(
        self,
        learning_rate: float,
        min_lr: float,
        warmup_iters: int,
        lr_decay_iters: int,
        decay_lr: bool = True,
    ):
        """
        Initialize learning rate scheduler with cosine decay.

        Args:
            learning_rate (float): Peak learning rate after warmup
                - Typical range: 1e-4 to 6e-4 for GPT models
                - Higher for smaller models, lower for larger
            min_lr (float): Final learning rate after decay
                - Usually 10x smaller than max LR
                - E.g., 6e-5 if max is 6e-4
            warmup_iters (int): Linear warmup duration
                - Typical: 2000-4000 iterations
                - Longer warmup for larger models
            lr_decay_iters (int): Total iterations including warmup
                - Set based on total training budget
                - E.g., 600000 for full GPT-2 training
            decay_lr (bool): Enable cosine decay
                - True: Use warmup + decay schedule
                - False: Constant learning_rate (debugging)

        Example:
            >>> # Standard GPT-2 schedule
            >>> scheduler = LearningRateScheduler(
            ...     learning_rate=6e-4,
            ...     min_lr=6e-5,
            ...     warmup_iters=2000,
            ...     lr_decay_iters=600000
            ... )
        """
        self.learning_rate = learning_rate
        self.min_lr = min_lr
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters
        self.decay_lr = decay_lr

    def get_lr(self, it: int) -> float:
        """
        Calculate learning rate for given iteration.

        Implements three-phase schedule:
        1. **Linear warmup** (0 ≤ it < warmup_iters):
           - Gradually increase from 0 to max LR
           - Prevents training instability

        2. **Cosine decay** (warmup_iters ≤ it ≤ lr_decay_iters):
           - Smooth decay following cosine curve
           - Allows model to converge gradually

        3. **Constant minimum** (it > lr_decay_iters):
           - Maintain minimum LR
           - Fine-tuning phase

        Args:
            it (int): Current training iteration (0-indexed)

        Returns:
            float: Learning rate for current iteration

        Schedule Visualization:
            ```
            LR ^
               │    ╱╲
               │   ╱  ╲___
               │  ╱
               └────────────> iter
                 W    D    C
            ```
            W=Warmup, D=Decay, C=Constant

        Called by:
            - train.Trainer.train_step() to set optimizer LR

        Example:
            >>> scheduler = LearningRateScheduler(6e-4, 6e-5, 2000, 10000)
            >>> lr_0 = scheduler.get_lr(0)      # Near 0 (warmup start)
            >>> lr_2k = scheduler.get_lr(2000)  # 6e-4 (peak)
            >>> lr_10k = scheduler.get_lr(10000) # 6e-5 (min)
        """
        if not self.decay_lr:
            return self.learning_rate

        # Linear warmup
        if it < self.warmup_iters:
            return self.learning_rate * (it + 1) / (self.warmup_iters + 1)

        # Constant minimum after decay period
        if it > self.lr_decay_iters:
            return self.min_lr

        # Cosine decay
        decay_ratio = (it - self.warmup_iters) / (
            self.lr_decay_iters - self.warmup_iters
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.learning_rate - self.min_lr)

    def get_schedule_info(self) -> Dict[str, Any]:
        """
        Get information about the learning rate schedule.

        Returns configuration parameters for logging and debugging.

        Returns:
            Dict[str, Any]: Schedule configuration containing:
                - type: 'cosine_with_warmup'
                - max_lr: Peak learning rate
                - min_lr: Minimum learning rate
                - warmup_iters: Warmup duration
                - lr_decay_iters: Total decay duration
                - decay_enabled: Whether decay is active

        Used for:
            - Logging hyperparameters
            - Reproducing training runs
            - Debugging schedule issues
        """
        return {
            "type": "cosine_with_warmup",
            "max_lr": self.learning_rate,
            "min_lr": self.min_lr,
            "warmup_iters": self.warmup_iters,
            "lr_decay_iters": self.lr_decay_iters,
            "decay_enabled": self.decay_lr,
        }


class TrainingLogger:
    """
    Comprehensive logging system for training runs.

    Provides unified interface for logging training metrics to multiple
    destinations: console, files, and Weights & Biases. Tracks key
    metrics and provides summaries for analysis.

    Features:
        - Multi-destination logging (console, file, W&B)
        - Automatic metric tracking and history
        - Performance monitoring (step times, MFU)
        - Gradient health tracking
        - Checkpoint logging

    Attributes:
        log_interval (int): Iteration frequency for logging
        use_wandb (bool): Whether W&B logging is active
        logger: Python logger instance
        wandb: W&B module reference (if available)
        step_times (List[float]): History of iteration times
        losses (List[float]): History of loss values

    Log Format:
        ```
        2024-01-15 10:30:45 - INFO - iter 1000: loss 2.4532, time 342.15ms, mfu 45.23%, lr 6.00e-04
        ```

    Called by:
        - train.Trainer throughout training loop
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        log_interval: int = 1,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
    ):
        """
        Initialize training logger with multiple backends.

        Sets up logging infrastructure including file output,
        console display, and optional W&B integration.

        Args:
            log_dir (Optional[str]): Directory for log files
                - Creates 'training.log' in this directory
                - None: Console logging only
            log_interval (int): Log every N iterations
                - 1: Log every iteration (verbose)
                - 10: Log every 10th iteration (moderate)
                - 100: Log every 100th iteration (minimal)
            use_wandb (bool): Enable Weights & Biases logging
                - Requires 'wandb' package installed
                - Provides web dashboard for metrics
            wandb_project (Optional[str]): W&B project name
                - Groups related experiments
                - Default: 'gpt-training'
            wandb_run_name (Optional[str]): Unique run identifier
                - For distinguishing experiments
                - Auto-generated if not provided

        File Structure:
            ```
            log_dir/
            └── training.log    # Detailed training logs
            ```

        W&B Features:
            - Real-time metric visualization
            - Hyperparameter tracking
            - Model comparison
            - Loss curves and gradients
        """
        self.log_interval = log_interval
        self.use_wandb = use_wandb

        # Setup file logging
        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

            # Configure logging
            log_file = log_dir / "training.log"
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(message)s",
                handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
            )
        else:
            logging.basicConfig(
                level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
            )

        self.logger = logging.getLogger(__name__)

        # Initialize W&B if requested
        if use_wandb:
            try:
                import wandb

                wandb.init(
                    project=wandb_project or "gpt-training",
                    name=wandb_run_name,
                    resume="allow",
                )
                self.wandb = wandb
            except ImportError:
                self.logger.warning(
                    "wandb requested but not installed. Install with: pip install wandb"
                )
                self.use_wandb = False
                self.wandb = None
        else:
            self.wandb = None

        # Tracking variables
        self.step_times = []
        self.losses = []

    def log_iteration(
        self,
        iter_num: int,
        loss: float,
        learning_rate: float,
        dt: Optional[float] = None,
        mfu: Optional[float] = None,
        extra_metrics: Optional[Dict[str, Any]] = None,
    ):
        """
        Log metrics for a single training iteration.

        Records training progress with optional performance metrics.
        Only logs if iter_num is divisible by log_interval.

        Args:
            iter_num (int): Current iteration number
            loss (float): Training loss value
                - Cross-entropy loss for language modeling
                - Lower is better
            learning_rate (float): Current optimizer learning rate
                - From LearningRateScheduler
            dt (Optional[float]): Iteration time in seconds
                - Used to calculate throughput
                - None skips timing metrics
            mfu (Optional[float]): Model FLOPs Utilization (0-1)
                - Fraction of theoretical peak FLOPS
                - Higher means better hardware usage
            extra_metrics (Optional[Dict[str, Any]]): Additional metrics
                - E.g., {'grad_norm': 1.5, 'tokens_per_sec': 50000}
                - Automatically prefixed with 'train/'

        Metrics Logged:
            - iter: Iteration number
            - train/loss: Training loss
            - train/learning_rate: Current LR
            - train/step_time: Time per iteration (if dt provided)
            - train/tokens_per_second: Throughput (if dt provided)
            - train/mfu: Hardware utilization % (if mfu provided)
            - train/{key}: Any extra_metrics provided

        Called by:
            - train.Trainer.train() in main training loop

        Example:
            >>> logger.log_iteration(
            ...     iter_num=1000,
            ...     loss=2.456,
            ...     learning_rate=6e-4,
            ...     dt=0.342,
            ...     mfu=0.452,
            ...     extra_metrics={'grad_norm': 1.23}
            ... )
        """
        if iter_num % self.log_interval != 0:
            return

        # Build log message
        log_parts = [f"iter {iter_num}: loss {loss:.4f}"]

        if dt is not None:
            log_parts.append(f"time {dt*1000:.2f}ms")
            self.step_times.append(dt)

        if mfu is not None:
            log_parts.append(f"mfu {mfu*100:.2f}%")

        log_parts.append(f"lr {learning_rate:.2e}")

        if extra_metrics:
            for key, value in extra_metrics.items():
                if isinstance(value, float):
                    log_parts.append(f"{key} {value:.4f}")
                else:
                    log_parts.append(f"{key} {value}")

        # Log to console/file
        self.logger.info(", ".join(log_parts))

        # Log to W&B
        if self.use_wandb and self.wandb:
            wandb_dict = {
                "iter": iter_num,
                "train/loss": loss,
                "train/learning_rate": learning_rate,
            }

            if dt is not None:
                wandb_dict["train/step_time"] = dt
                wandb_dict["train/tokens_per_second"] = 1.0 / dt if dt > 0 else 0

            if mfu is not None:
                wandb_dict["train/mfu"] = mfu * 100

            if extra_metrics:
                for key, value in extra_metrics.items():
                    wandb_dict[f"train/{key}"] = value

            self.wandb.log(wandb_dict)

        # Track loss history
        self.losses.append(loss)

    def log_evaluation(
        self,
        iter_num: int,
        train_loss: float,
        val_loss: float,
        extra_metrics: Optional[Dict[str, Any]] = None,
    ):
        """
        Log evaluation metrics on train and validation sets.

        Records model performance on both splits for tracking
        generalization and potential overfitting.

        Args:
            iter_num (int): Current iteration number
            train_loss (float): Average loss on training set
                - From estimate_loss() on random train batches
            val_loss (float): Average loss on validation set
                - Key metric for model selection
                - Rising val_loss indicates overfitting
            extra_metrics (Optional[Dict[str, Any]]): Additional metrics
                - E.g., perplexity, accuracy, BLEU score

        Metrics Logged:
            - eval/train_loss: Training set performance
            - eval/val_loss: Validation set performance
            - eval/{key}: Any extra_metrics provided

        Overfitting Detection:
            - val_loss > train_loss: Normal (some gap expected)
            - val_loss >> train_loss: Overfitting
            - val_loss increasing: Stop training

        Called by:
            - train.Trainer.train() at eval_interval

        Example:
            >>> logger.log_evaluation(
            ...     iter_num=5000,
            ...     train_loss=2.34,
            ...     val_loss=2.45,
            ...     extra_metrics={'perplexity': math.exp(2.45)}
            ... )
        """
        self.logger.info(
            f"step {iter_num}: train loss {train_loss:.4f}, val loss {val_loss:.4f}"
        )

        if self.use_wandb and self.wandb:
            wandb_dict = {
                "iter": iter_num,
                "eval/train_loss": train_loss,
                "eval/val_loss": val_loss,
            }

            if extra_metrics:
                for key, value in extra_metrics.items():
                    wandb_dict[f"eval/{key}"] = value

            self.wandb.log(wandb_dict)

    def log_checkpoint(self, iter_num: int, checkpoint_path: str, best: bool = False):
        """
        Log checkpoint saving event.

        Records when model checkpoints are saved, with special
        indication for best validation performance.

        Args:
            iter_num (int): Iteration when checkpoint was saved
            checkpoint_path (str): Path where checkpoint was saved
            best (bool): Whether this is best validation loss so far

        Logged Information:
            - Checkpoint path
            - Whether it's the best model
            - W&B: checkpoint/saved and checkpoint/best flags

        Called by:
            - train.Trainer.save_checkpoint() after saving
        """
        message = f"Saved checkpoint to {checkpoint_path}"
        if best:
            message += " (best validation loss)"
        self.logger.info(message)

        if self.use_wandb and self.wandb:
            self.wandb.log(
                {
                    "checkpoint/iter": iter_num,
                    "checkpoint/saved": 1,
                    "checkpoint/best": int(best),
                }
            )

    def finish(self):
        """
        Finish logging session and print summary statistics.

        Cleans up logging resources and prints final training
        statistics including average step time and loss progression.

        Summary Includes:
            - Average step time (for performance analysis)
            - Final loss value
            - Best loss achieved
            - W&B run completion

        Called by:
            - train.Trainer.train() at end of training
        """
        if self.use_wandb and self.wandb:
            self.wandb.finish()

        # Log summary statistics
        if self.step_times:
            avg_time = np.mean(self.step_times)
            self.logger.info(f"Average step time: {avg_time*1000:.2f}ms")

        if self.losses:
            final_loss = self.losses[-1]
            min_loss = min(self.losses)
            self.logger.info(f"Final loss: {final_loss:.4f}, Best loss: {min_loss:.4f}")


def get_gradient_stats(model: torch.nn.Module) -> Dict[str, float]:
    """
    Calculate gradient statistics for monitoring training health.

    Computes key gradient metrics to detect training issues like
    vanishing/exploding gradients or dead neurons.

    Args:
        model (torch.nn.Module): Model with computed gradients
            - Must be called after backward() but before optimizer.step()

    Returns:
        Dict[str, float]: Gradient statistics
            - grad_norm: L2 norm of all gradients
            - grad_min: Minimum absolute gradient value
            - grad_max: Maximum absolute gradient value
            - num_params_with_grad: Count of parameters with gradients

    Gradient Health Indicators:
        - grad_norm < 0.001: Possible vanishing gradients
        - grad_norm > 100: Possible exploding gradients
        - grad_min ≈ 0: Some parameters not learning
        - grad_max > 1: May need gradient clipping

    Called by:
        - train.Trainer.train() for gradient monitoring

    Calls:
        - torch.norm() for L2 norm calculation

    Example:
        >>> loss.backward()
        >>> grad_stats = get_gradient_stats(model)
        >>> if grad_stats['grad_norm'] > 100:
        ...     print("Warning: Large gradients detected!")
    """
    total_norm = 0.0
    param_count = 0
    min_grad = float("inf")
    max_grad = float("-inf")

    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2).item()
            total_norm += param_norm**2
            param_count += 1

            grad_abs = p.grad.data.abs()
            min_grad = min(min_grad, grad_abs.min().item())
            max_grad = max(max_grad, grad_abs.max().item())

    total_norm = total_norm**0.5

    return {
        "grad_norm": total_norm,
        "grad_min": min_grad,
        "grad_max": max_grad,
        "num_params_with_grad": param_count,
    }


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters by category.

    Provides detailed parameter counting for model size analysis
    and comparison. Distinguishes between total and trainable parameters.

    Args:
        model (torch.nn.Module): Model to analyze

    Returns:
        Dict[str, int]: Parameter counts
            - total: All parameters in model
            - trainable: Parameters with requires_grad=True
            - non_trainable: Frozen parameters
            - total_millions: Total count in millions
            - trainable_millions: Trainable count in millions

    Use Cases:
        - Model size verification
        - Memory requirement estimation
        - Comparing model variants
        - Checking parameter freezing

    Memory Estimation:
        - FP32: params * 4 bytes
        - FP16/BF16: params * 2 bytes
        - + Optimizer states (2x-3x for Adam)

    Called by:
        - Model initialization logging
        - Model comparison scripts

    Example:
        >>> params = count_parameters(model)
        >>> print(f"Model has {params['total_millions']:.1f}M parameters")
        >>> print(f"{params['trainable_millions']:.1f}M are trainable")
    """
    total_params = 0
    trainable_params = 0

    for p in model.parameters():
        total_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()

    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": total_params - trainable_params,
        "total_millions": total_params / 1e6,
        "trainable_millions": trainable_params / 1e6,
    }
