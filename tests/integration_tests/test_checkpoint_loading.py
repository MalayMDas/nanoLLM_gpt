"""
Test script to verify custom checkpoint loading works correctly.
"""

import requests
import json
import os
from pathlib import Path

def test_checkpoint_loading():
    """Test loading custom checkpoints through the web server."""
    base_url = "http://localhost:8080"
    
    print("Testing Custom Checkpoint Loading")
    print("=" * 50)
    
    # Test different checkpoint scenarios
    test_cases = [
        {
            "name": "Local checkpoint with config",
            "checkpoint_path": "out/ckpt.pt",
            "config_path": "out/config.yaml"
        },
        {
            "name": "Local checkpoint without config",
            "checkpoint_path": "out/ckpt.pt",
            "config_path": None
        }
    ]
    
    for test_case in test_cases:
        print(f"\nTesting: {test_case['name']}")
        
        checkpoint_path = test_case['checkpoint_path']
        config_path = test_case['config_path']
        
        # Check if checkpoint exists
        if not Path(checkpoint_path).exists():
            print(f"✗ Checkpoint not found at {checkpoint_path}")
            print("  Please train a model first or provide a valid checkpoint path")
            continue
        
        # Prepare request data
        request_data = {
            "checkpoint_path": checkpoint_path
        }
        
        if config_path:
            if Path(config_path).exists():
                request_data["config_path"] = config_path
            else:
                print(f"  Warning: Config file not found at {config_path}")
        
        # Send request
        try:
            print(f"  Loading checkpoint from: {checkpoint_path}")
            if config_path:
                print(f"  Using config from: {config_path}")
            
            response = requests.post(
                f"{base_url}/api/model/load",
                json=request_data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    print("  ✓ Successfully loaded checkpoint")
                    
                    # Display model info
                    model_info = result.get('model_info', {})
                    print(f"    Parameters: {model_info.get('num_parameters_millions', 0):.1f}M")
                    print(f"    Device: {model_info.get('device', 'unknown')}")
                    print(f"    Dtype: {model_info.get('dtype', 'unknown')}")
                    
                    # Display architecture if available
                    if 'architecture' in model_info:
                        arch = model_info['architecture']
                        print(f"    Architecture: n_layer={arch.get('n_layer')}, "
                              f"n_head={arch.get('n_head')}, n_embd={arch.get('n_embd')}")
                    
                    # Test generation
                    test_generation(base_url)
                else:
                    print(f"  ✗ Failed to load checkpoint: {result.get('error')}")
            else:
                print(f"  ✗ HTTP error {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"    Error: {error_data.get('error', response.text)}")
                except:
                    print(f"    Response: {response.text}")
        
        except requests.exceptions.Timeout:
            print("  ✗ Request timeout")
        except Exception as e:
            print(f"  ✗ Error: {e}")


def test_generation(base_url):
    """Test text generation with the loaded model."""
    try:
        response = requests.post(
            f"{base_url}/api/generate",
            json={
                "prompt": "The quick brown fox",
                "max_tokens": 20,
                "temperature": 0.8
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print("    ✓ Generation test passed")
                generated_text = result.get('text', '')
                print(f"      Generated: '{generated_text[:50]}...'")
            else:
                print(f"    ✗ Generation failed: {result.get('error')}")
        else:
            print(f"    ✗ Generation HTTP error {response.status_code}")
    except Exception as e:
        print(f"    ✗ Generation error: {e}")


def main():
    """Run all tests."""
    base_url = "http://localhost:8080"
    
    # Check if server is running
    print("Checking server connection...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✓ Server is running")
        else:
            print("✗ Server returned unexpected status")
            return
    except Exception as e:
        print(f"✗ Cannot connect to server at {base_url}")
        print("  Please start the server with: python -m nanoLLM_gpt.server")
        return
    
    # Run checkpoint loading tests
    test_checkpoint_loading()
    
    print("\n" + "=" * 50)
    print("Checkpoint loading test completed!")
    print("\nNote: The fix handles PyTorch 2.6's new security requirements by:")
    print("1. Adding ModelConfig to safe globals")
    print("2. Attempting to load with weights_only=True first")
    print("3. Falling back to weights_only=False if needed")


if __name__ == "__main__":
    main()