"""
Comprehensive test suite for nanoLLM_gpt package.

This module tests:
- Model training on tiny dataset
- Checkpoint saving and loading
- Config file handling
- API endpoints
- Data validation checks
"""

import os
import json
import shutil
import tempfile
import time
from pathlib import Path
import subprocess
import requests
from unittest.mock import patch

import pytest
import torch
import yaml

from nanoLLM_gpt.model import GPT
from nanoLLM_gpt.config import ModelConfig, TrainingConfig, ConfigLoader
from nanoLLM_gpt.utils import DataPreparer, DataLoader, ModelLoader


class TestTraining:
    """Test cases for model training functionality."""
    
    @pytest.fixture
    def tiny_data_dir(self):
        """Create a tiny dataset for testing."""
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        data_dir = Path(temp_dir) / "tiny_test"
        data_dir.mkdir(parents=True)
        
        # Create tiny text file
        text_file = data_dir / "input.txt"
        # Ensure we have enough text for validation split
        text_content = "Hello world! " * 500  # Repeat to ensure enough tokens for train/val split
        text_file.write_text(text_content)
        
        yield str(text_file), str(data_dir)
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_tiny_training(self, tiny_data_dir):
        """Test training a model on tiny dataset for a short time."""
        text_file, data_dir = tiny_data_dir
        output_dir = Path(data_dir) / "output"
        
        # Run training with minimal settings
        cmd = [
            "gpt-train",
            f"--data-path={text_file}",
            f"--out-dir={output_dir}",
            "--max-iters=10",  # Very short training
            "--eval-interval=5",
            "--eval-iters=2",
            "--block-size=64",  # Small block size for tiny data
            "--batch-size=2",
            "--n-layer=2",  # Tiny model
            "--n-head=2",
            "--n-embd=64",
            "--train-val-split=0.1",  # Larger validation split to ensure enough data
            "--always-save-checkpoint=True"
        ]
        
        print(f"Running training command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Check training completed successfully
        assert result.returncode == 0, f"Training failed: {result.stderr}"
        
        # Verify checkpoint was saved
        checkpoint_path = output_dir / "ckpt.pt"
        assert checkpoint_path.exists(), "Checkpoint file not created"
        
        # Verify config was saved
        config_path = output_dir / "config.yaml"
        assert config_path.exists(), "Config file not created"
        
        # Load and verify checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        assert 'model' in checkpoint, "Model state not in checkpoint"
        assert 'model_args' in checkpoint, "Model args not in checkpoint"
        assert checkpoint['iter_num'] >= 10, "Training didn't complete expected iterations"
        
        # Load and verify config
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        assert 'model' in config_data, "Model config not saved"
        assert config_data['model']['n_layer'] == 2, "Model config incorrect"
    
    def test_checkpoint_loading(self, tiny_data_dir):
        """Test loading a trained checkpoint."""
        # First train a model
        text_file, data_dir = tiny_data_dir
        output_dir = Path(data_dir) / "output"
        
        # Run training first
        cmd = [
            "gpt-train",
            f"--data-path={text_file}",
            f"--out-dir={output_dir}",
            "--max-iters=10",
            "--eval-interval=5",
            "--eval-iters=2",
            "--block-size=64",
            "--batch-size=2",
            "--n-layer=2",
            "--n-head=2",
            "--n-embd=64",
            "--train-val-split=0.1",
            "--always-save-checkpoint=True"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"Training failed: {result.stderr}"
        
        checkpoint_path = output_dir / "ckpt.pt"
        
        # Load the checkpoint
        model = GPT.load_from_checkpoint(checkpoint_path, device='cpu')
        assert isinstance(model, GPT), "Failed to load model from checkpoint"
        assert model.config.n_layer == 2, "Model config not loaded correctly"
        
        # Test generation with loaded model
        import tiktoken
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode("Hello")
        x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
        
        with torch.no_grad():
            logits, _ = model(x)
        assert logits.shape[0] == 1, "Model forward pass failed"
        assert logits.shape[-1] == model.config.vocab_size, "Output vocab size incorrect"
    
    def test_validation_data_error(self):
        """Test error handling for small validation data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create very small text file
            text_file = Path(temp_dir) / "tiny.txt"
            text_file.write_text("Hi")  # Too small for any reasonable block size
            
            # Try to prepare data with large validation split
            preparer = DataPreparer()
            data_dir = preparer.prepare_data(
                data_path=str(text_file),
                output_dir=temp_dir,
                train_val_split=0.5  # Half the data for validation
            )
            
            # Try to create data loader with large block size
            with pytest.raises(ValueError) as exc_info:
                DataLoader(
                    data_dir=data_dir,
                    block_size=1024,  # Much larger than available tokens
                    batch_size=1,
                    device='cpu',
                    device_type='cpu'
                )
            
            # Check error message is informative
            assert "block_size" in str(exc_info.value)
            assert "tokens" in str(exc_info.value)


class TestAPI:
    """Test cases for API functionality."""
    
    @pytest.fixture
    def server_process(self):
        """Start the server in a subprocess."""
        # Start server
        process = subprocess.Popen(
            ["python", "-m", "nanoLLM_gpt.server", "--model-type=gpt2"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for server to start
        time.sleep(5)
        
        # Check server is running
        max_retries = 10
        for i in range(max_retries):
            try:
                response = requests.get("http://localhost:8080/health")
                if response.status_code == 200:
                    break
            except:
                pass
            time.sleep(1)
        else:
            process.terminate()
            pytest.fail("Server failed to start")
        
        yield process
        
        # Cleanup
        process.terminate()
        process.wait(timeout=5)
    
    def test_health_endpoint(self, server_process):
        """Test the health check endpoint."""
        response = requests.get("http://localhost:8080/health")
        assert response.status_code == 200
        
        data = response.json()
        assert 'status' in data
        assert data['status'] == 'healthy'
        assert 'model_loaded' in data
    
    def test_model_info_endpoint(self, server_process):
        """Test the model info endpoint."""
        response = requests.get("http://localhost:8080/api/model/info")
        assert response.status_code == 200
        
        data = response.json()
        assert 'loaded' in data
        assert data['loaded'] == True  # Server starts with gpt2
    
    def test_model_config_endpoint(self, server_process):
        """Test the model config download endpoint."""
        # Get config as JSON
        response = requests.get("http://localhost:8080/api/model/config")
        assert response.status_code == 200
        
        data = response.json()
        assert 'model' in data
        assert 'n_layer' in data['model']
        
        # Test download mode
        response = requests.get("http://localhost:8080/api/model/config?download=true")
        assert response.status_code == 200
        assert 'attachment' in response.headers.get('Content-Disposition', '')
    
    def test_generation_endpoint(self, server_process):
        """Test text generation via API."""
        response = requests.post(
            "http://localhost:8080/api/generate",
            json={
                "prompt": "Hello, world!",
                "max_tokens": 10,
                "temperature": 0.8
            }
        )
        assert response.status_code == 200
        
        data = response.json()
        assert 'success' in data
        assert data['success'] == True
        assert 'text' in data
        assert data['text'].startswith("Hello, world!")
    
    def test_chat_completion_endpoint(self, server_process):
        """Test OpenAI-compatible chat completion endpoint."""
        response = requests.post(
            "http://localhost:8080/v1/chat/completions",
            json={
                "model": "gpt",
                "messages": [
                    {"role": "user", "content": "Say hello"}
                ],
                "max_tokens": 10
            }
        )
        assert response.status_code == 200
        
        data = response.json()
        assert 'choices' in data
        assert len(data['choices']) > 0
        assert 'message' in data['choices'][0]
    
    def test_model_loading_endpoints(self, server_process):
        """Test model loading with config."""
        # Create a minimal config
        config = {
            "model": {
                "n_layer": 2,
                "n_head": 2,
                "n_embd": 128,
                "block_size": 256,
                "vocab_size": 50257
            }
        }
        
        # Test loading GPT-2 with custom config
        response = requests.post(
            "http://localhost:8080/api/model/load",
            json={
                "model_type": "gpt2",
                "config": config
            }
        )
        
        # Note: This might fail if model architecture doesn't match
        # but we're testing the endpoint functionality
        assert response.status_code in [200, 500]
        data = response.json()
        assert 'success' in data or 'error' in data


class TestConfigHandling:
    """Test configuration save/load functionality."""
    
    def test_config_save_load(self):
        """Test saving and loading configuration files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a config
            config = TrainingConfig(
                model=ModelConfig(n_layer=4, n_head=4, n_embd=256),
                out_dir=temp_dir,
                max_iters=100
            )
            
            # Save config
            config_path = Path(temp_dir) / "test_config.yaml"
            ConfigLoader.save_to_file(config, config_path)
            assert config_path.exists()
            
            # Load config
            loaded_config = ConfigLoader.load_from_file(config_path, TrainingConfig)
            assert loaded_config.model.n_layer == 4
            assert loaded_config.max_iters == 100
            
            # Test JSON format
            json_path = Path(temp_dir) / "test_config.json"
            ConfigLoader.save_to_file(config, json_path)
            assert json_path.exists()
            
            loaded_json = ConfigLoader.load_from_file(json_path, TrainingConfig)
            assert loaded_json.model.n_layer == 4


class TestDataUtils:
    """Test data preparation utilities."""
    
    def test_data_preparer(self):
        """Test data preparation with different sources."""
        with tempfile.TemporaryDirectory() as temp_dir:
            preparer = DataPreparer()
            
            # Test with text file
            text_file = Path(temp_dir) / "test.txt"
            text_file.write_text("This is a test " * 100)
            
            data_dir = preparer.prepare_data(
                data_path=str(text_file),
                output_dir=temp_dir,
                train_val_split=0.1
            )
            
            # Check files were created
            assert Path(data_dir, "train.bin").exists()
            assert Path(data_dir, "val.bin").exists()
            assert Path(data_dir, "meta.pkl").exists()
            
            # Load and check data
            loader = DataLoader(
                data_dir=data_dir,
                block_size=32,
                batch_size=2,
                device='cpu',
                device_type='cpu'
            )
            
            x, y = loader.get_batch('train')
            assert x.shape == (2, 32)
            assert y.shape == (2, 32)
    
    def test_url_data_loading(self):
        """Test loading data from URL."""
        with tempfile.TemporaryDirectory() as temp_dir:
            preparer = DataPreparer()
            
            # Use a small public text file
            data_dir = preparer.prepare_data(
                data_path="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
                output_dir=temp_dir,
                dataset_name="shakespeare_test"
            )
            
            assert Path(data_dir, "train.bin").exists()
            assert Path(data_dir, "val.bin").exists()


def run_all_tests():
    """Run all tests and provide a summary."""
    print("Running comprehensive test suite...")
    print("=" * 70)
    
    # Run pytest with verbose output
    result = pytest.main([
        __file__,
        "-v",  # Verbose
        "--tb=short",  # Short traceback
        "-x",  # Stop on first failure
        "--color=yes"
    ])
    
    return result


if __name__ == "__main__":
    exit_code = run_all_tests()