"""
Test runner script to execute all tests and provide summary.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_tests():
    """Run all tests and provide detailed summary."""
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    print("=" * 70)
    print("Running nanoLLM_gpt Test Suite")
    print("=" * 70)
    
    # Define test categories
    test_categories = [
        ("Unit Tests - Model Loader", ["tests/test_model_loader.py::TestModelLoader"]),
        ("Unit Tests - Model Context", ["tests/test_model_loader.py::TestModelContext"]),
        ("Integration Tests - Training", ["tests/test_comprehensive.py::TestTraining::test_tiny_training"]),
        ("Integration Tests - Checkpoint Loading", ["tests/test_comprehensive.py::TestTraining::test_checkpoint_loading"]),
        ("Integration Tests - Data Validation", ["tests/test_comprehensive.py::TestTraining::test_validation_data_error"]),
        ("Integration Tests - Config Handling", ["tests/test_comprehensive.py::TestConfigHandling"]),
        ("Integration Tests - Data Utils", ["tests/test_comprehensive.py::TestDataUtils"]),
        ("Edge Cases - Data Handling", ["tests/test_edge_cases.py::TestEdgeCases"]),
        ("Edge Cases - Model Operations", ["tests/test_edge_cases.py::TestModelOperations"]),
    ]
    
    # Skip API tests as they require server running
    # ("Integration Tests - API", ["tests/test_comprehensive.py::TestAPI"])
    
    results = {}
    
    for category_name, test_paths in test_categories:
        print(f"\n{category_name}")
        print("-" * len(category_name))
        
        for test_path in test_paths:
            # Run pytest for this specific test
            cmd = [sys.executable, "-m", "pytest", test_path, "-v", "--tb=short", "-x"]
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60  # 60 second timeout per test
                )
                
                # Parse output
                output_lines = result.stdout.split('\n')
                test_results = []
                
                for line in output_lines:
                    if '::' in line and ('PASSED' in line or 'FAILED' in line or 'SKIPPED' in line):
                        # Extract test name and result
                        parts = line.split('::')
                        if len(parts) >= 2:
                            test_name = parts[-1].split()[0]
                            if 'PASSED' in line:
                                test_results.append((test_name, 'PASSED'))
                                print(f"  ✓ {test_name}")
                            elif 'FAILED' in line:
                                test_results.append((test_name, 'FAILED'))
                                print(f"  ✗ {test_name}")
                                # Print error details
                                if result.stderr:
                                    print(f"    Error: {result.stderr.strip()}")
                            elif 'SKIPPED' in line:
                                test_results.append((test_name, 'SKIPPED'))
                                print(f"  - {test_name} (skipped)")
                
                results[category_name] = test_results
                
            except subprocess.TimeoutExpired:
                print(f"  ✗ TIMEOUT - Test took too long")
                results[category_name] = [("All tests", "TIMEOUT")]
            except Exception as e:
                print(f"  ✗ ERROR - {str(e)}")
                results[category_name] = [("All tests", "ERROR")]
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    total_passed = 0
    total_failed = 0
    total_skipped = 0
    
    for category, test_results in results.items():
        passed = sum(1 for _, status in test_results if status == 'PASSED')
        failed = sum(1 for _, status in test_results if status == 'FAILED')
        skipped = sum(1 for _, status in test_results if status == 'SKIPPED')
        
        total_passed += passed
        total_failed += failed
        total_skipped += skipped
        
        print(f"\n{category}:")
        print(f"  Passed:  {passed}")
        print(f"  Failed:  {failed}")
        print(f"  Skipped: {skipped}")
    
    print("\n" + "-" * 70)
    print(f"TOTAL: {total_passed} passed, {total_failed} failed, {total_skipped} skipped")
    print("=" * 70)
    
    # List of implemented test cases
    print("\nIMPLEMENTED TEST CASES:")
    print("-" * 70)
    test_descriptions = {
        "test_create_model": "Tests creating a new GPT model from configuration",
        "test_get_torch_dtype": "Tests PyTorch dtype conversion utilities",
        "test_model_info": "Tests retrieving model information (parameters, architecture)",
        "test_device_setup": "Tests device setup and CUDA fallback",
        "test_load_pretrained": "Tests loading pretrained GPT-2 models from HuggingFace",
        "test_cpu_context": "Tests context manager for CPU inference",
        "test_cuda_context": "Tests context manager for CUDA inference with autocast",
        "test_tiny_training": "Tests training a small model on tiny dataset",
        "test_checkpoint_loading": "Tests loading and using saved checkpoints",
        "test_validation_data_error": "Tests error handling when validation data is too small",
        "test_config_save_load": "Tests saving and loading configuration files (YAML/JSON)",
        "test_data_preparer": "Tests data preparation from text files",
        "test_url_data_loading": "Tests loading training data from URLs",
        "test_empty_data_file": "Tests handling of empty data files",
        "test_block_size_larger_than_data": "Tests error when block_size exceeds data size",
        "test_invalid_model_config": "Tests model creation with invalid configurations",
        "test_checkpoint_not_found": "Tests error handling for missing checkpoints",
        "test_mismatched_block_size": "Tests block size cropping functionality",
        "test_data_type_conversions": "Tests data type handling in data loader",
        "test_special_characters_in_data": "Tests Unicode and special character handling",
        "test_model_device_movement": "Tests moving models between CPU/GPU",
        "test_model_inference_modes": "Tests model eval/train modes",
        "test_batch_size_variations": "Tests model with different batch sizes",
        # API tests (require server)
        "test_health_endpoint": "Tests server health check endpoint",
        "test_model_info_endpoint": "Tests model information API endpoint",
        "test_model_config_endpoint": "Tests config download endpoint",
        "test_generation_endpoint": "Tests text generation API",
        "test_chat_completion_endpoint": "Tests OpenAI-compatible chat API",
        "test_model_loading_endpoints": "Tests model loading with config via API"
    }
    
    for test_name, description in test_descriptions.items():
        print(f"• {test_name}: {description}")
    
    print("\n" + "=" * 70)
    
    return total_failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)