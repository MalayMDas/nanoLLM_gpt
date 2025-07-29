"""
Basic usage examples for the GPT project.

This script demonstrates how to use the refactored GPT project
for common tasks like text generation and model training.
"""

import torch
from pathlib import Path

from nanoLLM_gpt import GPT, ModelConfig, TrainingConfig, GenerationConfig
from nanoLLM_gpt.utils import ModelLoader, InferencePipeline, DataPreparer


def example_1_simple_generation():
    """Example 1: Simple text generation with a pretrained model."""
    print("Example 1: Simple Text Generation")
    print("-" * 50)
    
    # Initialize inference pipeline
    pipeline = InferencePipeline(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load pretrained GPT-2
    print("Loading GPT-2 model...")
    pipeline.load_model(model_type='gpt2')
    
    # Generate text
    prompt = "The future of artificial intelligence is"
    print(f"Prompt: {prompt}")
    print("Generating...")
    
    text = pipeline.generate(
        prompt,
        max_new_tokens=50,
        temperature=0.8,
        top_k=50
    )
    
    print(f"Generated: {text}")
    print()


def example_2_custom_generation_config():
    """Example 2: Generation with custom configuration."""
    print("Example 2: Custom Generation Configuration")
    print("-" * 50)
    
    # Create custom generation config
    config = GenerationConfig(
        max_new_tokens=100,
        temperature=0.9,
        top_p=0.95,
        top_k=100,
        repetition_penalty=1.2
    )
    
    # Initialize pipeline
    pipeline = InferencePipeline()
    pipeline.load_model(model_type='gpt2')
    
    # Generate multiple samples
    prompt = "Once upon a time, in a land far away,"
    print(f"Prompt: {prompt}")
    print("Generating 3 samples...")
    
    config.num_samples = 3
    samples = pipeline.generate(prompt, config)
    
    for i, sample in enumerate(samples, 1):
        print(f"\nSample {i}:")
        print(prompt + sample)
    print()


def example_3_chat_completion():
    """Example 3: Chat completion API."""
    print("Example 3: Chat Completion")
    print("-" * 50)
    
    # Initialize pipeline
    pipeline = InferencePipeline()
    pipeline.load_model(model_type='gpt2')
    
    # Create chat messages
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Write a haiku about programming."}
    ]
    
    print("Messages:")
    for msg in messages:
        print(f"  {msg['role']}: {msg['content']}")
    print("\nGenerating response...")
    
    # Generate chat completion
    response = pipeline.chat_completion(messages)
    
    print(f"Assistant: {response['choices'][0]['message']['content']}")
    print()


def example_4_model_from_scratch():
    """Example 4: Create and use a model from scratch."""
    print("Example 4: Model from Scratch")
    print("-" * 50)
    
    # Create a small model configuration
    config = ModelConfig(
        n_layer=4,
        n_head=4,
        n_embd=256,
        block_size=512,
        vocab_size=50257,  # GPT-2 vocabulary size
        dropout=0.1
    )
    
    print("Creating small GPT model:")
    print(f"  Layers: {config.n_layer}")
    print(f"  Heads: {config.n_head}")
    print(f"  Embedding size: {config.n_embd}")
    
    # Create model
    model = GPT(config)
    num_params = model.get_num_params()
    print(f"  Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    
    # Note: This model is untrained, so generation will be random
    print("\nNote: This is an untrained model, output will be random.")
    
    # Create pipeline with the model
    pipeline = InferencePipeline(model=model)
    
    # Generate random text
    try:
        text = pipeline.generate("Hello", max_new_tokens=20)
        # Handle potential encoding issues on Windows
        print(f"Random output: {text.encode('ascii', 'replace').decode('ascii')}")
    except Exception as e:
        print(f"Generation completed but display failed: {str(e)}")
    print()


def example_5_data_preparation():
    """Example 5: Prepare data for training."""
    print("Example 5: Data Preparation")
    print("-" * 50)
    
    # Create sample text file
    sample_text = """
    Artificial intelligence is transforming the world.
    Machine learning models are becoming more powerful.
    Deep learning has revolutionized computer vision.
    Natural language processing enables amazing applications.
    The future of AI is bright and full of possibilities.
    """
    
    # Save to temporary file
    temp_file = Path("sample_data.txt")
    temp_file.write_text(sample_text.strip())
    print(f"Created sample data file: {temp_file}")
    
    # Prepare data
    preparer = DataPreparer()
    data_dir = preparer.prepare_data(
        data_path=str(temp_file),
        dataset_name='example',
        train_val_split=0.2  # 20% for validation
    )
    
    print(f"Data prepared in: {data_dir}")
    
    # Load and check data statistics
    from nanoLLM_gpt.utils import DataLoader
    
    loader = DataLoader(
        data_dir=data_dir,
        block_size=128,
        batch_size=4,
        device='cpu',
        device_type='cpu'
    )
    
    stats = loader.get_data_stats()
    print("\nData statistics:")
    for split in ['train', 'val']:
        if split in stats:
            print(f"  {split}: {stats[split]['num_tokens']} tokens")
    
    # Clean up
    temp_file.unlink()
    print()


def example_6_streaming_generation():
    """Example 6: Streaming text generation."""
    print("Example 6: Streaming Generation")
    print("-" * 50)
    
    # Initialize pipeline
    pipeline = InferencePipeline()
    pipeline.load_model(model_type='gpt2')
    
    # Configure for streaming
    config = GenerationConfig(
        max_new_tokens=50,
        temperature=0.8,
        stream=True
    )
    
    prompt = "The key to happiness is"
    print(f"Prompt: {prompt}")
    print("Streaming response: ", end='')
    print(prompt, end='', flush=True)
    
    # Stream tokens as they're generated
    for token in pipeline.generate(prompt, config):
        print(token, end='', flush=True)
    
    print("\n")


def main():
    """Run all examples."""
    examples = [
        example_1_simple_generation,
        example_2_custom_generation_config,
        example_3_chat_completion,
        example_4_model_from_scratch,
        example_5_data_preparation,
        example_6_streaming_generation
    ]
    
    print("GPT Project - Basic Usage Examples")
    print("=" * 50)
    print()
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"Error in {example.__name__}: {e}")
            print()
    
    print("All examples completed!")


if __name__ == '__main__':
    main()
