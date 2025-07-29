"""
Edge case tests for nanoLLM_gpt package.

Tests various error conditions and edge cases.
"""

import pytest
import torch
import tempfile
from pathlib import Path

from nanoLLM_gpt.model import GPT
from nanoLLM_gpt.config import ModelConfig
from nanoLLM_gpt.utils import DataPreparer, DataLoader


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_data_file(self):
        """Test handling of empty data files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create empty file
            empty_file = Path(temp_dir) / "empty.txt"
            empty_file.write_text("")
            
            preparer = DataPreparer()
            
            # Should handle empty file gracefully
            data_dir = preparer.prepare_data(
                data_path=str(empty_file),
                output_dir=temp_dir
            )
            
            # Check files exist but are minimal
            assert Path(data_dir, "train.bin").exists()
            assert Path(data_dir, "val.bin").exists()
    
    def test_block_size_larger_than_data(self):
        """Test when block_size exceeds available data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create small data file
            small_file = Path(temp_dir) / "small.txt"
            small_file.write_text("Hello")
            
            preparer = DataPreparer()
            data_dir = preparer.prepare_data(
                data_path=str(small_file),
                output_dir=temp_dir,
                train_val_split=0.1
            )
            
            # Try to create loader with large block size
            with pytest.raises(ValueError) as exc_info:
                DataLoader(
                    data_dir=data_dir,
                    block_size=1000,  # Much larger than data
                    batch_size=1,
                    device='cpu',
                    device_type='cpu'
                )
            
            # Verify error message
            assert "tokens" in str(exc_info.value).lower()
            assert "block_size" in str(exc_info.value).lower()
    
    def test_invalid_model_config(self):
        """Test model creation with invalid configurations."""
        # Test with invalid vocab size
        with pytest.raises(AssertionError):
            config = ModelConfig(vocab_size=None)
            GPT(config)
        
        # Test with invalid block size
        with pytest.raises(AssertionError):
            config = ModelConfig(block_size=None)
            GPT(config)
    
    def test_checkpoint_not_found(self):
        """Test loading non-existent checkpoint."""
        with pytest.raises(FileNotFoundError) as exc_info:
            GPT.load_from_checkpoint("non_existent_checkpoint.pt")
        
        assert "not found" in str(exc_info.value).lower()
    
    def test_mismatched_block_size(self):
        """Test handling of block size mismatch."""
        # Create a model with large block size
        config = ModelConfig(
            n_layer=2,
            n_head=2,
            n_embd=128,
            block_size=1024,
            vocab_size=1000
        )
        model = GPT(config)
        
        # Crop to smaller block size
        model.crop_block_size(512)
        assert model.config.block_size == 512
        
        # Verify position embeddings were cropped
        assert model.transformer.wpe.weight.shape[0] == 512
        
        # Test invalid crop (larger than original)
        with pytest.raises(AssertionError):
            model.crop_block_size(2048)
    
    def test_data_type_conversions(self):
        """Test data type handling in data loader."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test data
            text_file = Path(temp_dir) / "test.txt"
            text_file.write_text("Test data " * 500)  # More data for proper split
            
            preparer = DataPreparer()
            data_dir = preparer.prepare_data(
                data_path=str(text_file),
                output_dir=temp_dir,
                train_val_split=0.1  # Larger validation split
            )
            
            loader = DataLoader(
                data_dir=data_dir,
                block_size=16,
                batch_size=2,
                device='cpu',
                device_type='cpu'
            )
            
            x, y = loader.get_batch('train')
            
            # Check data types
            assert x.dtype == torch.long
            assert y.dtype == torch.long
            assert x.device.type == 'cpu'
    
    def test_special_characters_in_data(self):
        """Test handling of special characters and unicode."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create file with special characters
            special_file = Path(temp_dir) / "special.txt"
            special_file.write_text(
                "Hello 世界! 🌍 Special chars: @#$%^&*() " * 100,  # More data
                encoding='utf-8'
            )
            
            preparer = DataPreparer()
            
            # Should handle special characters
            data_dir = preparer.prepare_data(
                data_path=str(special_file),
                output_dir=temp_dir,
                train_val_split=0.1  # Larger validation split
            )
            
            # Verify data was tokenized
            assert Path(data_dir, "train.bin").exists()
            
            # Load and check
            loader = DataLoader(
                data_dir=data_dir,
                block_size=32,
                batch_size=1,
                device='cpu',
                device_type='cpu'
            )
            
            x, y = loader.get_batch('train')
            assert x.shape[0] == 1
            assert x.shape[1] == 32


class TestModelOperations:
    """Test various model operations."""
    
    def test_model_device_movement(self):
        """Test moving model between devices."""
        config = ModelConfig(
            n_layer=2,
            n_head=2,
            n_embd=64,
            block_size=128,
            vocab_size=1000
        )
        
        # Create model on CPU
        model = GPT(config)
        assert next(model.parameters()).device.type == 'cpu'
        
        # Move to CUDA if available
        if torch.cuda.is_available():
            model = model.cuda()
            assert next(model.parameters()).device.type == 'cuda'
            
            # Move back to CPU
            model = model.cpu()
            assert next(model.parameters()).device.type == 'cpu'
    
    def test_model_inference_modes(self):
        """Test model in different inference modes."""
        config = ModelConfig(
            n_layer=2,
            n_head=2,
            n_embd=64,
            block_size=128,
            vocab_size=1000
        )
        
        model = GPT(config)
        
        # Test eval mode
        model.eval()
        assert not model.training
        
        # Test train mode
        model.train()
        assert model.training
        
        # Test no_grad context
        x = torch.randint(0, 1000, (1, 10))
        with torch.no_grad():
            logits, loss = model(x)
            assert logits.requires_grad == False
    
    def test_batch_size_variations(self):
        """Test model with different batch sizes."""
        config = ModelConfig(
            n_layer=2,
            n_head=2,
            n_embd=64,
            block_size=128,
            vocab_size=1000
        )
        
        model = GPT(config)
        model.eval()
        
        # Test different batch sizes
        for batch_size in [1, 4, 16]:
            x = torch.randint(0, 1000, (batch_size, 32))
            with torch.no_grad():
                # During inference, model only returns logits for last position
                logits, _ = model(x)
            
            assert logits.shape[0] == batch_size
            assert logits.shape[1] == 1  # Only last position during inference
            assert logits.shape[2] == 1000
            
            # Test with targets (training mode)
            targets = torch.randint(0, 1000, (batch_size, 32))
            logits, loss = model(x, targets)
            assert logits.shape[0] == batch_size
            assert logits.shape[1] == 32  # All positions during training
            assert logits.shape[2] == 1000
            assert loss is not None


if __name__ == "__main__":
    # Run edge case tests
    pytest.main([__file__, "-v", "--tb=short"])