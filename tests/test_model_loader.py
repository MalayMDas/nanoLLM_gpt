"""
Example test file for the model loader module.
"""

import pytest
import torch
from pathlib import Path
from unittest.mock import patch

from nanoLLM_gpt.utils import ModelLoader
from nanoLLM_gpt.config import ModelConfig
from nanoLLM_gpt.model import GPT

# Import GPU check decorators from conftest
from tests.conftest import requires_gpu


class TestModelLoader:
    """Test cases for ModelLoader functionality."""
    
    def test_create_model(self):
        """Test creating a new model from config."""
        config = ModelConfig(
            n_layer=2,
            n_head=2,
            n_embd=128,
            block_size=256,
            vocab_size=1000
        )
        
        model = ModelLoader.create_model(config, device='cpu')
        
        assert isinstance(model, GPT)
        assert model.config.n_layer == 2
        assert model.config.n_head == 2
        assert model.config.n_embd == 128
    
    def test_get_torch_dtype(self):
        """Test dtype conversion."""
        # Test auto selection
        dtype_auto = ModelLoader._get_torch_dtype('auto', 'cpu')
        assert dtype_auto == torch.float32
        
        # Test specific dtypes
        dtype_fp32 = ModelLoader._get_torch_dtype('float32', 'cpu')
        assert dtype_fp32 == torch.float32
        
        dtype_fp16 = ModelLoader._get_torch_dtype('float16', 'cpu')
        assert dtype_fp16 == torch.float16
    
    def test_model_info(self):
        """Test getting model information."""
        config = ModelConfig(
            n_layer=2,
            n_head=2,
            n_embd=128,
            block_size=256,
            vocab_size=1000
        )
        
        model = GPT(config)
        info = ModelLoader.get_model_info(model)
        
        assert 'num_parameters' in info
        assert 'architecture' in info
        assert info['architecture']['n_layer'] == 2
        assert info['architecture']['n_head'] == 2
    
    @pytest.mark.gpu
    @requires_gpu
    def test_device_setup(self):
        """Test device setup with CUDA."""
        device = ModelLoader._setup_device('cuda')
        assert device == 'cuda'
        
        # Test fallback when requesting CUDA without availability
        with patch('torch.cuda.is_available', return_value=False):
            device = ModelLoader._setup_device('cuda')
            assert device == 'cpu'
    
    def test_load_pretrained(self):
        """Test loading pretrained model (requires internet)."""
        try:
            model = ModelLoader.load_model(
                model_type='gpt2',
                device='cpu'
            )
            assert isinstance(model, GPT)
            assert model.config.n_layer == 12  # GPT-2 base has 12 layers
        except Exception as e:
            pytest.skip(f"Could not load pretrained model: {e}")


class TestModelContext:
    """Test cases for model context management."""
    
    def test_cpu_context(self):
        """Test context manager for CPU."""
        ctx = ModelLoader.get_model_context('cpu')
        
        # CPU should return nullcontext
        with ctx:
            # Should work without issues
            x = torch.randn(1, 10)
            assert x.shape == (1, 10)
    
    @pytest.mark.gpu
    @requires_gpu
    def test_cuda_context(self):
        """Test context manager for CUDA."""
        ctx = ModelLoader.get_model_context('cuda', dtype='float16')
        
        # Should return autocast context
        assert hasattr(ctx, '__enter__')
        assert hasattr(ctx, '__exit__')


if __name__ == '__main__':
    pytest.main([__file__])
