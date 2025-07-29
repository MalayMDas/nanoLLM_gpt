# Integration Tests

These tests require a running server and are meant to be run as standalone scripts, not through pytest.

## Running Integration Tests

1. First, start the server:
   ```bash
   gpt-server
   ```

2. In another terminal, run the desired test:
   ```bash
   python tests/integration_tests/test_checkpoint_loading.py
   python tests/integration_tests/test_huggingface_loading.py
   python tests/integration_tests/test_config_functionality.py
   ```

## Test Descriptions

- **test_checkpoint_loading.py**: Tests loading custom model checkpoints through the web server
- **test_huggingface_loading.py**: Tests loading HuggingFace models through the API
- **test_config_functionality.py**: Tests configuration saving/loading and server endpoints

These tests are separated from the main test suite because they require external dependencies (running server) and are not suitable for automated CI/CD pipelines.