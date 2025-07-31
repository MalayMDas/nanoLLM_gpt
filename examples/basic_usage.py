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


def example_7_training_from_scratch():
    """Example 7: Training a model from scratch with Model Directory.
    
    Demonstrates how to train models and organize outputs using the
    model directory feature. The model directory contains both the
    checkpoint (ckpt.pt) and configuration (config.yaml) files.
    """
    print("Example 7: Training from Scratch")
    print("-" * 50)
    print("This example shows how to train a model programmatically.")
    print("For actual training, use the command-line interface:\n")
    
    # Single GPU training with custom model directory
    print("Single GPU training with custom model directory:")
    print("  gpt-train --data-path data.txt --out-dir custom_models/my_experiment --max-iters 1000")
    print()
    
    # Multi-GPU training with DDP
    print("Multi-GPU training (4 GPUs) with model directory:")
    print("  torchrun --nproc_per_node=4 -m nanoLLM_gpt.train \\")
    print("    --data-path data.txt --out-dir experiments/large_model --max-iters 1000")
    print()
    
    # Training configuration example
    print("Training configuration with model directory organization:")
    config = TrainingConfig(
        model=ModelConfig(
            n_layer=6,
            n_head=6,
            n_embd=384,
            block_size=512
        ),
        out_dir="models/my_custom_gpt",  # Model directory for checkpoints and config
        batch_size=12,
        max_iters=1000,
        learning_rate=3e-4,
        train_val_split=0.1
    )
    
    print(f"  Model: {config.model.n_layer} layers, {config.model.n_embd} dim")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Train/val split: {config.train_val_split}")
    print(f"  Model directory: {config.out_dir}")
    print(f"    - Checkpoint will be saved to: {config.out_dir}/ckpt.pt")
    print(f"    - Config will be saved to: {config.out_dir}/config.yaml")
    print()


def example_8_resume_training():
    """Example 8: Resume training from checkpoint using Model Directory.
    
    Shows how to resume training from a model directory that contains
    both the checkpoint and configuration. The config.yaml is automatically
    loaded when resuming, maintaining all original settings unless overridden.
    """
    print("Example 8: Resume Training with Model Directory")
    print("-" * 50)
    print("Examples of resuming training from model directories:\n")
    
    # Resume with same configuration
    print("1. Resume with original configuration from model directory:")
    print("   # Automatically loads models/my_custom_gpt/config.yaml")
    print("   gpt-train --init-from resume --out-dir models/my_custom_gpt")
    print()
    
    # Resume with new data
    print("2. Resume with new training data:")
    print("   # Keeps model architecture from config.yaml but uses new data")
    print("   gpt-train --init-from resume --out-dir models/my_custom_gpt \\")
    print("     --data-path new_data.txt")
    print()
    
    # Resume with modified hyperparameters
    print("3. Resume with different learning rate:")
    print("   # Overrides learning rate from saved config")
    print("   gpt-train --init-from resume --out-dir models/my_custom_gpt \\")
    print("     --learning-rate 1e-4 --max-iters 2000")
    print()
    
    # Multi-GPU resume
    print("4. Resume on multiple GPUs from model directory:")
    print("   torchrun --nproc_per_node=4 -m nanoLLM_gpt.train \\")
    print("     --init-from resume --out-dir experiments/large_model")
    print()
    
    # Loading for inference
    print("5. Load model from directory for inference:")
    print("   ```python")
    print("   # Option 1: Using model directory")
    print("   pipeline = InferencePipeline()")
    print("   pipeline.load_model(")
    print("       checkpoint_path='models/my_custom_gpt/ckpt.pt',")
    print("       config_path='models/my_custom_gpt/config.yaml'")
    print("   )")
    print("   ```")
    print()


def example_9_finetune_pretrained():
    """Example 9: Fine-tune a pretrained model with Model Directory.
    
    Demonstrates fine-tuning pretrained models with organized output
    directories. When fine-tuning HuggingFace models, the default model
    directory follows the pattern 'out_<model_name>' for easy identification.
    """
    print("Example 9: Fine-tune Pretrained Model with Model Directory")
    print("-" * 50)
    print("Examples of fine-tuning pretrained models:\n")
    
    # Fine-tune GPT-2 with automatic directory
    print("1. Fine-tune GPT-2 with automatic model directory:")
    print("   # Creates model directory: out_gpt2/")
    print("   gpt-train --init-from gpt2 --data-path domain_data.txt \\")
    print("     --learning-rate 5e-5")
    print()
    
    # Fine-tune with custom directory
    print("2. Fine-tune GPT-2 Medium with custom model directory:")
    print("   gpt-train --init-from gpt2-medium --data-path data.txt \\")
    print("     --out-dir finetuned_models/gpt2_medium_medical \\")
    print("     --learning-rate 3e-5 --batch-size 8 --gradient-accumulation-steps 4")
    print()
    
    # Multi-GPU fine-tuning
    print("3. Fine-tune on multiple GPUs with organized directories:")
    print("   torchrun --nproc_per_node=4 -m nanoLLM_gpt.train \\")
    print("     --init-from gpt2-large --data-path data.txt \\")
    print("     --out-dir experiments/gpt2_large_domain_specific --batch-size 4")
    print()
    
    # Loading fine-tuned model from directory
    print("4. Load and use fine-tuned model from model directory:")
    print("   ```python")
    print("   pipeline = InferencePipeline()")
    print("   # Load from the complete model directory")
    print("   pipeline.load_model(")
    print("       checkpoint_path='out_gpt2/ckpt.pt',")
    print("       config_path='out_gpt2/config.yaml'  # Optional, uses defaults if missing")
    print("   )")
    print("   text = pipeline.generate('Domain specific prompt')")
    print("   ```")
    print()
    
    # Web interface usage
    print("5. Using the web interface:")
    print("   - Select 'Fine-tune HuggingFace Model' in Training Mode")
    print("   - Choose model (e.g., gpt2-medium)")
    print("   - Model Directory auto-fills to 'out_gpt2-medium'")
    print("   - Upload your domain-specific training data")
    print("   - Start training")
    print()


def example_10_distributed_training():
    """Example 10: Distributed training configurations."""
    print("Example 10: Distributed Training Configurations")
    print("-" * 50)
    print("Various distributed training scenarios:\n")
    
    # Single node, multiple GPUs
    print("1. Single machine with 8 GPUs:")
    print("   torchrun --nproc_per_node=8 -m nanoLLM_gpt.train \\")
    print("     --data-path large_dataset.txt --out-dir big_model \\")
    print("     --n-layer 24 --n-head 16 --n-embd 1024 \\")
    print("     --batch-size 4 --gradient-accumulation-steps 8")
    print()
    
    # Multi-node training
    print("2. Multi-node training (2 nodes, 4 GPUs each):")
    print("   # On node 1 (master):")
    print("   torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \\")
    print("     --master_addr=192.168.1.100 --master_port=29500 \\")
    print("     -m nanoLLM_gpt.train --data-path data.txt \\")
    print("     --out-dir distributed_model")
    print()
    print("   # On node 2:")
    print("   torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \\")
    print("     --master_addr=192.168.1.100 --master_port=29500 \\")
    print("     -m nanoLLM_gpt.train --data-path data.txt \\")
    print("     --out-dir distributed_model")
    print()
    
    # Mixed precision training
    print("3. Mixed precision training for memory efficiency:")
    print("   torchrun --nproc_per_node=4 -m nanoLLM_gpt.train \\")
    print("     --data-path data.txt --out-dir efficient_model \\")
    print("     --dtype bfloat16 --compile")
    print()


def example_11_model_directory_usage():
    """Example 11: Model Directory Organization and Usage.
    
    Comprehensive example showing how model directories work for
    organizing checkpoints, configs, and managing multiple experiments.
    """
    print("Example 11: Model Directory Organization")
    print("-" * 50)
    print("Best practices for organizing models with directories:\n")
    
    # Directory structure
    print("1. Recommended directory structure:")
    print("   models/")
    print("   ├── experiments/")
    print("   │   ├── baseline_model/")
    print("   │   │   ├── ckpt.pt")
    print("   │   │   └── config.yaml")
    print("   │   └── improved_model/")
    print("   │       ├── ckpt.pt")
    print("   │       └── config.yaml")
    print("   ├── production/")
    print("   │   └── v1.0/")
    print("   │       ├── ckpt.pt")
    print("   │       └── config.yaml")
    print("   └── finetuned/")
    print("       ├── out_gpt2/")
    print("       │   ├── ckpt.pt")
    print("       │   └── config.yaml")
    print("       └── out_gpt2-medium/")
    print("           ├── ckpt.pt")
    print("           └── config.yaml")
    print()
    
    # Training with organized directories
    print("2. Training with organized model directories:")
    print("   # Experiment tracking")
    print("   gpt-train --data-path data.txt \\")
    print("     --out-dir models/experiments/run_001_lr_3e4")
    print()
    
    # Loading from directory
    print("3. Loading models from directories programmatically:")
    print("   ```python")
    print("   from nanoLLM_gpt.utils import ModelLoader")
    print("   ")
    print("   # Load model from a specific directory")
    print("   model = ModelLoader.load_model(")
    print("       checkpoint_path='models/production/v1.0/ckpt.pt',")
    print("       config_path='models/production/v1.0/config.yaml'")
    print("   )")
    print("   ")
    print("   # Or use with inference pipeline")
    print("   pipeline = InferencePipeline()")
    print("   pipeline.load_model(")
    print("       checkpoint_path='models/experiments/best_model/ckpt.pt'")
    print("   )")
    print("   ```")
    print()
    
    # Web interface model directory
    print("4. Web interface model directory usage:")
    print("   - Train Model tab: Set 'Model Directory' to organize outputs")
    print("   - Generate Text tab: Enter model directory path to load both files")
    print("   - Directory path auto-completes based on training mode")
    print()
    
    # Best practices
    print("5. Best practices:")
    print("   - Use descriptive directory names (include hyperparameters)")
    print("   - Keep checkpoint and config together in same directory")
    print("   - Version production models (v1.0, v1.1, etc.)")
    print("   - Separate experiments from production models")
    print("   - Document each model's purpose in a README within directory")
    print()


def main():
    """Run all examples."""
    examples = [
        example_1_simple_generation,
        example_2_custom_generation_config,
        example_3_chat_completion,
        example_4_model_from_scratch,
        example_5_data_preparation,
        example_6_streaming_generation,
        example_7_training_from_scratch,
        example_8_resume_training,
        example_9_finetune_pretrained,
        example_10_distributed_training,
        example_11_model_directory_usage
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
