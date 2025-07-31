"""
Pytest configuration and shared fixtures for all tests.

This file provides:
- Environment detection for GPU/multi-GPU setups
- Conditional test skipping based on available resources
- Shared fixtures for common test scenarios
"""

import os
import pytest
import torch
import tempfile
import shutil
from pathlib import Path
import requests


def get_gpu_count():
    """Get the number of available GPUs."""
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


def is_distributed_available():
    """Check if distributed training is available."""
    try:
        import torch.distributed as dist
        # Check if we can import required backends
        return hasattr(dist, 'is_available') and dist.is_available()
    except ImportError:
        return False


# Skip conditions
requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU not available"
)

requires_multigpu = pytest.mark.skipif(
    get_gpu_count() < 2,
    reason=f"Multiple GPUs not available (found {get_gpu_count()})"
)

requires_distributed = pytest.mark.skipif(
    not is_distributed_available() or get_gpu_count() < 2,
    reason="Distributed training not available or insufficient GPUs"
)


def is_server_running(url="http://localhost:8080"):
    """Check if the test server is running."""
    try:
        response = requests.get(f"{url}/health", timeout=1)
        return response.status_code == 200
    except:
        return False


requires_server = pytest.mark.skipif(
    not is_server_running(),
    reason="Test server not running at http://localhost:8080"
)


# Fixtures
@pytest.fixture
def temp_dir():
    """Create a temporary directory that's automatically cleaned up."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def device():
    """Get the appropriate device for testing."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture
def sample_text_data():
    """Create sample text data for testing."""
    return """
    The quick brown fox jumps over the lazy dog.
    Machine learning is transforming the world.
    Natural language processing enables amazing applications.
    Deep learning models are becoming increasingly powerful.
    The future of AI is bright and full of possibilities.
    """ * 10  # Repeat to have more data


@pytest.fixture
def small_model_config():
    """Create a small model configuration for faster testing."""
    from nanoLLM_gpt.config import ModelConfig
    
    return ModelConfig(
        n_layer=2,
        n_head=2,
        n_embd=64,
        block_size=128,
        vocab_size=1000,
        dropout=0.0
    )


@pytest.fixture
def training_config(small_model_config, temp_dir):
    """Create a minimal training configuration for testing."""
    from nanoLLM_gpt.config import TrainingConfig
    
    return TrainingConfig(
        model=small_model_config,
        out_dir=str(Path(temp_dir) / "test_output"),
        batch_size=2,
        max_iters=10,
        learning_rate=1e-3,
        eval_interval=5,
        eval_iters=2,
        train_val_split=0.1,
        log_interval=1
    )


# Markers for different test categories
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "multigpu: mark test as requiring multiple GPUs")
    config.addinivalue_line("markers", "distributed: mark test as requiring distributed setup")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "requires_server: mark test as requiring running server")
    config.addinivalue_line("markers", "integration: mark test as integration test")


# Command line options
def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-gpu",
        action="store_true",
        default=False,
        help="Run GPU tests"
    )
    parser.addoption(
        "--run-multigpu",
        action="store_true",
        default=False,
        help="Run multi-GPU tests"
    )
    parser.addoption(
        "--run-distributed",
        action="store_true",
        default=False,
        help="Run distributed tests"
    )
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers and command line options."""
    # Skip GPU tests unless explicitly requested
    if not config.getoption("--run-gpu"):
        skip_gpu = pytest.mark.skip(reason="need --run-gpu option to run")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
    
    # Skip multi-GPU tests unless explicitly requested
    if not config.getoption("--run-multigpu"):
        skip_multigpu = pytest.mark.skip(reason="need --run-multigpu option to run")
        for item in items:
            if "multigpu" in item.keywords:
                item.add_marker(skip_multigpu)
    
    # Skip distributed tests unless explicitly requested
    if not config.getoption("--run-distributed"):
        skip_distributed = pytest.mark.skip(reason="need --run-distributed option to run")
        for item in items:
            if "distributed" in item.keywords:
                item.add_marker(skip_distributed)
    
    # Skip slow tests unless explicitly requested
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)