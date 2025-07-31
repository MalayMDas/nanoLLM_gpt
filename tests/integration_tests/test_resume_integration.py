"""
Integration tests for resume training functionality.

These tests require more setup and test the full resume workflow
including actual training steps.
"""

import os
import tempfile
import shutil
import subprocess
import time
from pathlib import Path
import yaml
import requests
import pytest

from nanoLLM_gpt.config import TrainingConfig, ModelConfig, ConfigLoader
from tests.conftest import requires_server


class TestResumeTrainingIntegration:
    """Integration tests for resume training workflow."""
    
    @pytest.fixture
    def temp_training_dir(self):
        """Create a temporary directory for training outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_data_file(self, temp_training_dir):
        """Create a small sample data file for testing."""
        data_path = Path(temp_training_dir) / "sample_data.txt"
        with open(data_path, 'w') as f:
            f.write("This is a test. " * 100)  # Small dataset
        return str(data_path)
    
    @pytest.mark.slow
    def test_full_resume_workflow(self, temp_training_dir, sample_data_file):
        """Test complete training -> interrupt -> resume workflow."""
        out_dir = Path(temp_training_dir) / "model_output"
        
        # Step 1: Start initial training
        initial_cmd = [
            "gpt-train",
            f"--data-path={sample_data_file}",
            f"--out-dir={out_dir}",
            "--max-iters=20",  # Very short for testing
            "--eval-interval=5",
            "--eval-iters=2",
            "--n-layer=2",
            "--n-head=2",
            "--n-embd=64",
            "--batch-size=2"
        ]
        
        # Run initial training
        process = subprocess.run(initial_cmd, capture_output=True, text=True)
        assert process.returncode == 0, f"Initial training failed: {process.stderr}"
        
        # Verify checkpoint and config were created
        assert (out_dir / "ckpt.pt").exists()
        assert (out_dir / "config.yaml").exists()
        
        # Step 2: Resume training
        resume_cmd = [
            "gpt-train",
            "--init-from=resume",
            f"--out-dir={out_dir}",
            "--max-iters=40"  # Continue for more iterations
        ]
        
        # Run resume training
        process = subprocess.run(resume_cmd, capture_output=True, text=True)
        assert process.returncode == 0, f"Resume training failed: {process.stderr}"
        
        # Verify training continued (check logs for iteration numbers > 20)
        assert "iter 20" in process.stdout or "iter 21" in process.stdout
    
    @pytest.mark.slow
    def test_resume_with_different_data(self, temp_training_dir, sample_data_file):
        """Test resuming training with a different dataset."""
        out_dir = Path(temp_training_dir) / "model_output"
        
        # Create second data file
        new_data_path = Path(temp_training_dir) / "new_data.txt"
        with open(new_data_path, 'w') as f:
            f.write("Different training data. " * 100)
        
        # Initial training
        initial_cmd = [
            "gpt-train",
            f"--data-path={sample_data_file}",
            f"--out-dir={out_dir}",
            "--max-iters=10",
            "--eval-interval=5",
            "--n-layer=2",
            "--n-head=2",
            "--n-embd=64"
        ]
        
        subprocess.run(initial_cmd, capture_output=True, text=True)
        
        # Resume with new data
        resume_cmd = [
            "gpt-train",
            "--init-from=resume",
            f"--out-dir={out_dir}",
            f"--data-path={new_data_path}",
            "--max-iters=20"
        ]
        
        process = subprocess.run(resume_cmd, capture_output=True, text=True)
        assert process.returncode == 0
        
        # Load config to verify new data path was used
        config = ConfigLoader.load_from_file(str(out_dir / "config.yaml"), TrainingConfig)
        assert config.data_path == str(new_data_path)


class TestServerResumeIntegration:
    """Integration tests for resume training via server API."""
    
    @pytest.fixture
    def server_url(self):
        """Base URL for test server (must be running)."""
        return "http://localhost:8080"
    
    @pytest.mark.requires_server
    @requires_server
    def test_resume_training_via_api(self, server_url, tmp_path):
        """Test resume training through the web API."""
        # Create a small training data file
        data_content = "Test training data. " * 50
        
        # Step 1: Start initial training
        files = {'file': ('train.txt', data_content, 'text/plain')}
        data = {
            'config': '{"max_iters": 10, "eval_interval": 5, "n_layer": 2, "n_head": 2, "n_embd": 64}',
            'init_from': 'scratch'
        }
        
        response = requests.post(f"{server_url}/api/training/start", files=files, data=data)
        assert response.status_code == 200
        assert response.json()['success']
        
        # Wait for training to complete
        time.sleep(10)
        
        # Stop training
        response = requests.post(f"{server_url}/api/training/stop")
        assert response.status_code == 200
        
        # Step 2: Resume training
        resume_data = {
            'init_from': 'resume',
            'out_dir': 'out',
            'config': '{"max_iters": 20}'
        }
        
        response = requests.post(f"{server_url}/api/training/start", data=resume_data)
        assert response.status_code == 200
        assert response.json()['success']
        
        # Check training status
        response = requests.get(f"{server_url}/api/training/status")
        assert response.status_code == 200
        status = response.json()
        assert status['is_training']
    
    @pytest.mark.requires_server
    @requires_server
    def test_load_model_from_checkpoint_dir(self, server_url):
        """Test loading a model from a training directory via API."""
        # Assuming a checkpoint exists in 'out' directory
        data = {
            'checkpoint_path': 'out/ckpt.pt',
            'config_path': 'out/config.yaml'
        }
        
        response = requests.post(f"{server_url}/api/model/load", data=data)
        
        if response.status_code == 200:
            result = response.json()
            assert result.get('success', False)
            
            # Verify model is loaded
            response = requests.get(f"{server_url}/api/model/info")
            assert response.status_code == 200
            info = response.json()
            assert info['loaded']


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not requires_server"])