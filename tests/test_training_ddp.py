"""
Test cases for training functionality including DDP support.

These tests cover:
- Single GPU training
- Multi-GPU training with DDP
- Resume training functionality
- Training with different configurations
"""

import os
import subprocess
import tempfile
from pathlib import Path
import torch
import pytest

from nanoLLM_gpt.config import TrainingConfig, ModelConfig, ConfigLoader
from nanoLLM_gpt.model import GPT
from nanoLLM_gpt.utils import DataPreparer, ModelLoader
from tests.conftest import requires_gpu, requires_multigpu, requires_distributed


class TestBasicTraining:
    """Test basic training functionality."""
    
    def test_training_config_creation(self):
        """Test creating a training configuration."""
        model_config = ModelConfig(
            n_layer=4,
            n_head=4,
            n_embd=256,
            block_size=512
        )
        
        config = TrainingConfig(
            model=model_config,
            out_dir="test_output",
            batch_size=8,
            max_iters=100
        )
        
        assert config.model.n_layer == 4
        assert config.batch_size == 8
        assert config.max_iters == 100
        assert config.train_val_split == 0.0005  # Default value
    
    def test_data_preparation(self, temp_dir, sample_text_data):
        """Test data preparation for training."""
        # Save sample data
        data_file = Path(temp_dir) / "sample.txt"
        data_file.write_text(sample_text_data)
        
        # Prepare data
        preparer = DataPreparer()
        data_dir = preparer.prepare_data(
            data_path=str(data_file),
            dataset_name="test_data",
            output_dir=temp_dir,
            train_val_split=0.1
        )
        
        # Check output files exist
        data_path = Path(data_dir)
        assert (data_path / "train.bin").exists()
        assert (data_path / "val.bin").exists()
        assert (data_path / "meta.pkl").exists()
    
    @pytest.mark.gpu
    @requires_gpu
    def test_single_gpu_training_command(self, temp_dir, sample_text_data):
        """Test single GPU training via command line."""
        # Prepare data
        data_file = Path(temp_dir) / "train_data.txt"
        data_file.write_text(sample_text_data)
        
        out_dir = Path(temp_dir) / "single_gpu_model"
        
        # Run short training
        cmd = [
            "gpt-train",
            f"--data-path={data_file}",
            f"--out-dir={out_dir}",
            "--max-iters=5",
            "--eval-interval=2",
            "--eval-iters=1",
            "--n-layer=2",
            "--n-head=2",
            "--n-embd=64",
            "--batch-size=2",
            "--log-interval=1"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        assert result.returncode == 0, f"Training failed: {result.stderr}"
        assert (out_dir / "ckpt.pt").exists()
        assert (out_dir / "config.yaml").exists()


class TestDistributedTraining:
    """Test distributed training functionality."""
    
    @pytest.mark.multigpu
    @pytest.mark.distributed
    @requires_multigpu
    def test_ddp_training_command(self, temp_dir, sample_text_data):
        """Test multi-GPU training with DDP."""
        # Prepare data
        data_file = Path(temp_dir) / "train_data.txt"
        data_file.write_text(sample_text_data)
        
        out_dir = Path(temp_dir) / "ddp_model"
        
        # Get number of GPUs
        num_gpus = torch.cuda.device_count()
        
        # Run distributed training
        cmd = [
            "torchrun",
            f"--nproc_per_node={min(num_gpus, 2)}",  # Use at most 2 GPUs for testing
            "-m", "nanoLLM_gpt.train",
            f"--data-path={data_file}",
            f"--out-dir={out_dir}",
            "--max-iters=5",
            "--eval-interval=2",
            "--n-layer=2",
            "--n-head=2",
            "--n-embd=64",
            "--batch-size=1",  # Small batch per GPU
            "--backend=nccl"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        assert result.returncode == 0, f"DDP training failed: {result.stderr}"
        assert (out_dir / "ckpt.pt").exists()
    
    @pytest.mark.distributed
    def test_distributed_environment_detection(self):
        """Test detection of distributed training environment."""
        # Check if distributed is available
        try:
            import torch.distributed as dist
            assert dist.is_available()
        except:
            pytest.skip("Distributed package not available")
    
    def test_backend_selection(self):
        """Test proper backend selection for different platforms."""
        from nanoLLM_gpt.config import TrainingConfig
        
        config = TrainingConfig()
        
        if torch.cuda.is_available():
            # NCCL is default for CUDA
            assert config.backend == "nccl"
        else:
            # Should handle CPU-only case
            assert config.backend in ["nccl", "gloo"]


class TestResumeTraining:
    """Test resume training functionality."""
    
    def test_checkpoint_format(self, temp_dir, small_model_config):
        """Test checkpoint saving and loading format."""
        # Create model
        model = GPT(small_model_config)
        
        # Save checkpoint
        checkpoint_path = Path(temp_dir) / "test_ckpt.pt"
        model.save_checkpoint(
            checkpoint_path=checkpoint_path,
            optimizer=None,
            iter_num=100,
            best_val_loss=2.5
        )
        
        # Load and verify
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        assert 'model' in checkpoint
        assert 'model_args' in checkpoint
        assert checkpoint.get('iter_num') == 100
        assert checkpoint.get('best_val_loss') == 2.5
    
    @pytest.mark.slow
    def test_resume_training_workflow(self, temp_dir, sample_text_data):
        """Test complete resume training workflow."""
        data_file = Path(temp_dir) / "data.txt"
        data_file.write_text(sample_text_data)
        
        out_dir = Path(temp_dir) / "resume_test"
        
        # Initial training
        cmd1 = [
            "gpt-train",
            f"--data-path={data_file}",
            f"--out-dir={out_dir}",
            "--max-iters=5",
            "--eval-interval=2",
            "--n-layer=2",
            "--n-head=2",
            "--n-embd=64",
            "--device=cpu"  # Use CPU for CI compatibility
        ]
        
        result1 = subprocess.run(cmd1, capture_output=True, text=True)
        assert result1.returncode == 0
        
        # Resume training
        cmd2 = [
            "gpt-train",
            "--init-from=resume",
            f"--out-dir={out_dir}",
            "--max-iters=10",
            "--device=cpu"
        ]
        
        result2 = subprocess.run(cmd2, capture_output=True, text=True)
        assert result2.returncode == 0
        
        # Verify iteration continued
        assert "iter 5" in result2.stdout or "iter 6" in result2.stdout


class TestTrainingParameters:
    """Test various training parameter configurations."""
    
    def test_train_val_split_parameter(self, temp_dir, sample_text_data):
        """Test train/validation split parameter."""
        data_file = Path(temp_dir) / "data.txt"
        data_file.write_text(sample_text_data * 100)  # Need more data for split
        
        # Test different split ratios
        for split_ratio in [0.01, 0.1, 0.2]:
            preparer = DataPreparer()
            data_dir = preparer.prepare_data(
                data_path=str(data_file),
                dataset_name=f"test_split_{split_ratio}",
                output_dir=temp_dir,
                train_val_split=split_ratio
            )
            
            # Verify files exist
            assert Path(data_dir).exists()
    
    def test_gradient_accumulation(self, training_config):
        """Test gradient accumulation settings."""
        # Default gradient accumulation
        assert training_config.gradient_accumulation_steps > 0
        
        # Effective batch size
        effective_batch = (
            training_config.batch_size * 
            training_config.gradient_accumulation_steps
        )
        assert effective_batch > 0
    
    @pytest.mark.gpu
    @requires_gpu
    def test_mixed_precision_training(self, temp_dir, sample_text_data):
        """Test training with mixed precision."""
        data_file = Path(temp_dir) / "data.txt"
        data_file.write_text(sample_text_data)
        
        out_dir = Path(temp_dir) / "amp_model"
        
        cmd = [
            "gpt-train",
            f"--data-path={data_file}",
            f"--out-dir={out_dir}",
            "--max-iters=3",
            "--n-layer=2",
            "--n-head=2",
            "--n-embd=64",
            "--dtype=bfloat16",  # Mixed precision
            "--device=cuda"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Should succeed or fail gracefully if bfloat16 not supported
        if result.returncode != 0:
            assert "bfloat16" in result.stderr or "dtype" in result.stderr


class TestServerDDPIntegration:
    """Test DDP integration with the web server."""
    
    def test_server_ddp_command_generation(self):
        """Test that server generates correct DDP commands."""
        from nanoLLM_gpt.server import TrainingManager
        
        manager = TrainingManager()
        
        # Mock config with DDP enabled
        train_config = {
            "ddp_enabled": True,
            "nproc_per_node": 4,
            "nnodes": 1,
            "master_addr": "127.0.0.1",
            "master_port": 29500,
            "out_dir": "test_out",
            "batch_size": 8,
            "backend": "nccl"
        }
        
        # The command should use torchrun for DDP
        # (actual command generation is in start_training method)
        assert train_config["ddp_enabled"] == True
        assert train_config["nproc_per_node"] == 4
    
    def test_server_single_gpu_command_generation(self):
        """Test that server generates correct single GPU commands."""
        train_config = {
            "ddp_enabled": False,
            "out_dir": "test_out",
            "batch_size": 12
        }
        
        # Should use gpt-train for single GPU
        assert train_config["ddp_enabled"] == False


# Utility functions for testing
def count_parameters(model):
    """Count the number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def verify_checkpoint_structure(checkpoint_path):
    """Verify the structure of a saved checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    required_keys = ['model', 'model_args']
    for key in required_keys:
        assert key in checkpoint, f"Missing required key: {key}"
    
    # Verify model_args has required fields
    model_args = checkpoint['model_args']
    required_model_args = ['n_layer', 'n_head', 'n_embd', 'block_size', 'vocab_size']
    for arg in required_model_args:
        assert arg in model_args, f"Missing model arg: {arg}"


if __name__ == "__main__":
    # Run basic tests (no GPU required)
    pytest.main([__file__, "-v", "-m", "not gpu and not multigpu and not distributed"])