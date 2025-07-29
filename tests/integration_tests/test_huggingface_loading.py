"""
Test script to verify HuggingFace model loading works correctly.
"""

import requests
import json
import time

def test_huggingface_model_loading():
    """Test loading HuggingFace models through the web server."""
    base_url = "http://localhost:8080"
    
    print("Testing HuggingFace Model Loading")
    print("=" * 50)
    
    # Test different HuggingFace models
    models_to_test = [
        ("gpt2", "GPT-2 (124M)"),
        ("gpt2-medium", "GPT-2 Medium (350M)"),
    ]
    
    for model_type, model_name in models_to_test:
        print(f"\nTesting {model_name}...")
        
        try:
            # Load the model
            response = requests.post(
                f"{base_url}/api/model/load",
                json={"model_type": model_type},
                timeout=60  # Give it time to download
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    print(f"✓ Successfully loaded {model_name}")
                    
                    # Display model info
                    model_info = result.get('model_info', {})
                    print(f"  Parameters: {model_info.get('num_parameters_millions', 0):.1f}M")
                    print(f"  Device: {model_info.get('device', 'unknown')}")
                    print(f"  Dtype: {model_info.get('dtype', 'unknown')}")
                    
                    # Test generation with the loaded model
                    test_generation(base_url, model_name)
                else:
                    print(f"✗ Failed to load {model_name}: {result.get('error')}")
            else:
                print(f"✗ HTTP error {response.status_code}")
                print(f"  Response: {response.text}")
        
        except requests.exceptions.Timeout:
            print(f"✗ Timeout loading {model_name} (model download may take time)")
        except Exception as e:
            print(f"✗ Error loading {model_name}: {e}")


def test_generation(base_url, model_name):
    """Test text generation with the loaded model."""
    try:
        response = requests.post(
            f"{base_url}/api/generate",
            json={
                "prompt": "Hello, world! This is",
                "max_tokens": 20,
                "temperature": 0.8
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print(f"  ✓ Generation test passed")
                print(f"    Generated: {result.get('text', '')[:50]}...")
            else:
                print(f"  ✗ Generation failed: {result.get('error')}")
        else:
            print(f"  ✗ Generation HTTP error {response.status_code}")
    except Exception as e:
        print(f"  ✗ Generation error: {e}")


def test_model_info_endpoint(base_url):
    """Test the model info endpoint."""
    print("\nTesting model info endpoint...")
    
    try:
        response = requests.get(f"{base_url}/api/model/info")
        
        if response.status_code == 200:
            info = response.json()
            print("✓ Model info endpoint working")
            print(f"  Response: {json.dumps(info, indent=2)}")
        else:
            print(f"✗ Model info endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Model info error: {e}")


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
    
    # Run tests
    test_huggingface_model_loading()
    test_model_info_endpoint(base_url)
    
    print("\n" + "=" * 50)
    print("Test completed!")


if __name__ == "__main__":
    main()