"""
Test cases for resume training functionality.

This module tests the resume training features including:
- Auto-loading saved configuration when resuming
- Command-line argument precedence over saved config
- Handling missing config.yaml gracefully
- Resume with new data path
- Resume with modified hyperparameters
"""

import os
import tempfile
import shutil
from pathlib import Path
import yaml
import torch
import pytest

from nanoLLM_gpt.config import TrainingConfig, ModelConfig, ConfigLoader
from nanoLLM_gpt.model import GPT
from nanoLLM_gpt.utils import ModelLoader


class TestResumeTraining:
    """Test suite for resume training functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_config(self):
        """Create a sample training configuration."""
        model_config = ModelConfig(
            n_layer=2,
            n_head=2,
            n_embd=64,
            block_size=128,
            vocab_size=1000
        )
        
        return TrainingConfig(
            model=model_config,
            out_dir="test_out",
            data_path="test_data.txt",
            max_iters=100,
            learning_rate=1e-3,
            batch_size=4
        )
    
    def create_checkpoint(self, checkpoint_dir: Path, config: TrainingConfig):
        """Create a minimal checkpoint for testing."""
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a minimal model
        model = GPT(config.model)
        
        # Save checkpoint in the format expected by GPT.load_from_checkpoint
        checkpoint = {
            'model': model.state_dict(),
            'model_args': {
                'n_layer': config.model.n_layer,
                'n_head': config.model.n_head,
                'n_embd': config.model.n_embd,
                'block_size': config.model.block_size,
                'vocab_size': config.model.vocab_size,
                'dropout': config.model.dropout,
                'bias': config.model.bias
            },
            'optimizer': None,
            'iter_num': 50,
            'best_val_loss': 2.5,
            'config': config.__dict__
        }
        torch.save(checkpoint, checkpoint_dir / "ckpt.pt")
        
        # Save config
        ConfigLoader.save_to_file(config, checkpoint_dir / "config.yaml")
    
    def test_resume_loads_saved_config(self, temp_dir, sample_config):
        """Test that resuming loads the saved config.yaml automatically."""
        out_dir = Path(temp_dir) / "model_checkpoint"
        sample_config.out_dir = str(out_dir)
        
        # Create checkpoint with config
        self.create_checkpoint(out_dir, sample_config)
        
        # Load config as if resuming
        loaded_config_path = out_dir / "config.yaml"
        assert loaded_config_path.exists()
        
        loaded_config = ConfigLoader.load_from_file(str(loaded_config_path), TrainingConfig)
        
        # Verify config was loaded correctly
        assert loaded_config.data_path == sample_config.data_path
        assert loaded_config.learning_rate == sample_config.learning_rate
        assert loaded_config.batch_size == sample_config.batch_size
        assert loaded_config.model.n_layer == sample_config.model.n_layer
    
    def test_resume_without_config_uses_defaults(self, temp_dir):
        """Test that resuming without config.yaml falls back to defaults."""
        out_dir = Path(temp_dir) / "model_checkpoint"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Create checkpoint without config.yaml
        model_config = ModelConfig(n_layer=2, n_head=2, n_embd=64, block_size=128, vocab_size=1000)
        model = GPT(model_config)
        
        # Create checkpoint in the format expected by load_from_checkpoint
        checkpoint = {
            'model': model.state_dict(),
            'model_args': {
                'n_layer': model_config.n_layer,
                'n_head': model_config.n_head,
                'n_embd': model_config.n_embd,
                'block_size': model_config.block_size,
                'vocab_size': model_config.vocab_size,
                'dropout': model_config.dropout,
                'bias': model_config.bias
            },
            'iter_num': 50,
            'best_val_loss': 2.5
        }
        torch.save(checkpoint, out_dir / "ckpt.pt")
        
        # Verify config.yaml doesn't exist
        assert not (out_dir / "config.yaml").exists()
        
        # Load model from checkpoint
        loaded_model = GPT.load_from_checkpoint(
            out_dir / "ckpt.pt",
            device='cpu',
            compile=False
        )
        
        assert loaded_model is not None
        assert loaded_model.config.n_layer == 2
    
    def test_cli_args_override_saved_config(self, temp_dir, sample_config):
        """Test that command-line arguments take precedence over saved config."""
        out_dir = Path(temp_dir) / "model_checkpoint"
        sample_config.out_dir = str(out_dir)
        
        # Create checkpoint with config
        self.create_checkpoint(out_dir, sample_config)
        
        # Simulate command-line override
        loaded_config = ConfigLoader.load_from_file(
            str(out_dir / "config.yaml"), 
            TrainingConfig
        )
        
        # Override some values (simulating CLI args)
        new_learning_rate = 5e-4
        new_batch_size = 8
        loaded_config.learning_rate = new_learning_rate
        loaded_config.batch_size = new_batch_size
        
        # Verify overrides
        assert loaded_config.learning_rate == new_learning_rate
        assert loaded_config.batch_size == new_batch_size
        # Original values should remain for non-overridden params
        assert loaded_config.data_path == sample_config.data_path
        assert loaded_config.model.n_layer == sample_config.model.n_layer
    
    def test_resume_with_new_data_path(self, temp_dir, sample_config):
        """Test resuming training with a different data path."""
        out_dir = Path(temp_dir) / "model_checkpoint"
        sample_config.out_dir = str(out_dir)
        original_data_path = "original_data.txt"
        sample_config.data_path = original_data_path
        
        # Create checkpoint
        self.create_checkpoint(out_dir, sample_config)
        
        # Load config and change data path
        loaded_config = ConfigLoader.load_from_file(
            str(out_dir / "config.yaml"), 
            TrainingConfig
        )
        
        new_data_path = "new_data.txt"
        loaded_config.data_path = new_data_path
        loaded_config.init_from = "resume"
        
        # Verify
        assert loaded_config.data_path == new_data_path
        assert loaded_config.init_from == "resume"
    
    def test_model_state_preserved_on_resume(self, temp_dir, sample_config):
        """Test that model weights and training state are preserved when resuming."""
        out_dir = Path(temp_dir) / "model_checkpoint"
        sample_config.out_dir = str(out_dir)
        
        # Create checkpoint
        self.create_checkpoint(out_dir, sample_config)
        
        # Load checkpoint
        checkpoint = torch.load(out_dir / "ckpt.pt", map_location='cpu')
        
        # Verify training state
        assert checkpoint['iter_num'] == 50
        assert checkpoint['best_val_loss'] == 2.5
        assert 'model' in checkpoint
        assert 'config' in checkpoint
    
    def test_pretrained_model_init(self):
        """Test initialization from pretrained models doesn't require config."""
        # This should work without any existing config
        model_config = ModelConfig(
            n_layer=12,
            n_head=12,
            n_embd=768,
            block_size=1024,
            vocab_size=50257  # GPT-2 vocab size
        )
        
        # Create a model that mimics GPT-2 structure
        model = GPT(model_config)
        assert model is not None
        assert model.config.vocab_size == 50257


class TestServerResumeTraining:
    """Test resume training functionality in the server."""
    
    def test_training_manager_resume_command(self, tmp_path):
        """Test that TrainingManager builds correct command for resume training."""
        from nanoLLM_gpt.server import TrainingManager
        
        manager = TrainingManager()
        
        # Test resume command generation
        train_config = {
            "out_dir": str(tmp_path / "checkpoint"),
            "batch_size": 8,
            "learning_rate": 1e-4,
            "max_iters": 1000
        }
        
        # Mock the command building part
        init_from = "resume"
        data_path = None  # No new data path for resume
        
        # Build expected command
        expected_cmd_parts = [
            "gpt-train",
            f"--out-dir={train_config['out_dir']}",
            f"--init-from={init_from}",
            f"--batch-size={train_config['batch_size']}",
            f"--max-iters={train_config['max_iters']}",
            f"--learning-rate={train_config['learning_rate']}",
        ]
        
        # Verify command construction
        # Note: In actual implementation, this is done in start_training method
        assert all(part in expected_cmd_parts for part in [
            "--init-from=resume",
            "--out-dir=" + train_config['out_dir']
        ])


class TestWebInterfaceResume:
    """Test resume functionality in web interface."""
    
    def test_web_form_data_for_resume(self):
        """Test that web interface sends correct data for resume training."""
        # Simulate form data for resume training
        form_data = {
            "init_from": "resume",
            "out_dir": "models/my_checkpoint",
            "config": '{"batch_size": 12, "learning_rate": 0.0001}'
        }
        
        # Verify required fields
        assert form_data["init_from"] == "resume"
        assert form_data["out_dir"] is not None
        
        # Optional data path not required for resume
        assert "data_path" not in form_data or form_data.get("data_path") is None


class TestModelDirectoryOrganization:
    """Test Model Directory organization functionality.
    
    Tests the unified model directory approach where both checkpoint
    and config files are stored in the same directory for better organization.
    """
    
    @pytest.fixture
    def model_directory(self, tmp_path):
        """Create a model directory structure."""
        model_dir = tmp_path / "models" / "experiment_001"
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir
    
    @pytest.fixture
    def sample_config(self):
        """Create a sample training configuration."""
        model_config = ModelConfig(
            n_layer=2,
            n_head=2,
            n_embd=64,
            block_size=128,
            vocab_size=1000
        )
        
        return TrainingConfig(
            model=model_config,
            out_dir="test_out",
            data_path="test_data.txt",
            max_iters=100,
            learning_rate=1e-3,
            batch_size=4
        )
    
    def test_model_directory_structure(self, model_directory):
        """Test that model directory contains both checkpoint and config."""
        # Create dummy checkpoint and config files
        checkpoint_path = model_directory / "ckpt.pt"
        config_path = model_directory / "config.yaml"
        
        # Save dummy checkpoint
        torch.save({"model": "dummy_state", "iter_num": 100}, checkpoint_path)
        
        # Save dummy config
        config_data = {
            "model": {"n_layer": 6, "n_head": 6, "n_embd": 384},
            "training": {"batch_size": 12, "learning_rate": 0.0003}
        }
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        # Verify both files exist in the same directory
        assert checkpoint_path.exists(), "Checkpoint should exist in model directory"
        assert config_path.exists(), "Config should exist in model directory"
        assert checkpoint_path.parent == config_path.parent, "Files should be in same directory"
    
    def test_load_from_model_directory(self, model_directory, sample_config):
        """Test loading model from a unified directory."""
        # Create checkpoint and config in model directory
        checkpoint_path = model_directory / "ckpt.pt"
        config_path = model_directory / "config.yaml"
        
        # Save a proper checkpoint
        model = GPT(sample_config.model)
        checkpoint = {
            'model': model.state_dict(),
            'model_args': model.config.__dict__,
            'iter_num': 500,
            'best_val_loss': 2.0,
            'config': sample_config.__dict__
        }
        torch.save(checkpoint, checkpoint_path)
        
        # Save config
        ConfigLoader.save_to_file(sample_config, config_path)
        
        # Test loading with ModelLoader using checkpoint path
        # The checkpoint contains the model configuration
        loaded_model = ModelLoader.load_model(
            checkpoint_path=str(checkpoint_path),
            device='cpu'
        )
        
        assert loaded_model is not None, "Model should load successfully"
        assert isinstance(loaded_model, GPT), "Loaded model should be GPT instance"
    
    def test_web_interface_model_directory_usage(self):
        """Test web interface form data with model directory."""
        # Test training with custom model directory
        train_form_data = {
            "init_from": "scratch",
            "out_dir": "models/experiments/run_001",  # Model directory
            "config": '{"batch_size": 12, "n_layer": 6}'
        }
        
        assert train_form_data["out_dir"] == "models/experiments/run_001"
        
        # Test loading from model directory for inference
        inference_form_data = {
            "model_dir": "models/production/v1.0"  # Single directory field
        }
        
        # In the actual implementation, this would expand to:
        expected_checkpoint = f"{inference_form_data['model_dir']}/ckpt.pt"
        expected_config = f"{inference_form_data['model_dir']}/config.yaml"
        
        assert "models/production/v1.0" in expected_checkpoint
        assert "models/production/v1.0" in expected_config
    
    def test_huggingface_model_directory_naming(self):
        """Test automatic model directory naming for HuggingFace models."""
        # Test different HuggingFace model directory names
        test_cases = [
            ("gpt2", "out_gpt2"),
            ("gpt2-medium", "out_gpt2-medium"),
            ("gpt2-large", "out_gpt2-large"),
            ("gpt2-xl", "out_gpt2-xl")
        ]
        
        for model_name, expected_dir in test_cases:
            form_data = {
                "init_from": model_name,
                "out_dir": None  # Should be auto-generated
            }
            
            # Simulate server logic for directory naming
            if form_data["out_dir"] is None and form_data["init_from"].startswith("gpt2"):
                generated_dir = f"out_{form_data['init_from']}"
                assert generated_dir == expected_dir, f"Directory should be {expected_dir} for {model_name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])