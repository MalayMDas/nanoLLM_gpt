"""
Text generation script for GPT models.

This script provides a command-line interface for generating text using
trained GPT models or pretrained models from HuggingFace. It supports
various generation strategies and interactive mode for experimentation.

## Key Features:

1. **Multiple Model Sources**:
   - Local checkpoints from training
   - Pretrained HuggingFace models (gpt2, gpt2-medium, etc.)

2. **Generation Modes**:
   - Single sample generation
   - Multiple samples with different random seeds
   - Interactive mode for experimentation
   - Streaming output for real-time display

3. **Sampling Strategies**:
   - Temperature control (creativity vs coherence)
   - Top-k filtering (quality threshold)
   - Top-p (nucleus) sampling (dynamic vocabulary)

4. **Input Options**:
   - Direct prompt via command line
   - Prompt from file
   - Interactive prompt entry

## Usage Examples:

```bash
# Generate from checkpoint
python generate.py --checkpoint out/ckpt.pt --prompt "Once upon a time"

# Generate from HuggingFace model
python generate.py --model gpt2-large --prompt "The future of AI"

# Generate with custom parameters
python generate.py --model gpt2 --temperature 0.9 --top-k 40 --max-tokens 200

# Interactive mode
python generate.py --model gpt2 --interactive

# Generate from file
python generate.py --checkpoint out/ckpt.pt --prompt-file prompt.txt

# Multiple samples
python generate.py --model gpt2 --prompt "Write a poem" --num-samples 3

# With PyTorch 2.0 compilation
python generate.py --checkpoint out/ckpt.pt --compile --prompt "Hello world"
```

## Interactive Mode Commands:
- 'quit' - Exit the program
- 'help' - Show current settings
- 'set <param> <value>' - Modify generation parameters
  - set temperature 0.9
  - set max_tokens 200
  - set top_k 50
  - set top_p 0.95

## Generation Parameters:

- **temperature**: Controls randomness (0.0 = deterministic, 1.0 = normal, >1.0 = creative)
- **top_k**: Only consider the k most likely tokens at each step
- **top_p**: Nucleus sampling - consider tokens with cumulative probability < p
- **max_tokens**: Maximum number of new tokens to generate

## Performance Tips:

1. Use --compile flag with PyTorch 2.0 for ~30% speedup
2. Use bfloat16 dtype on modern GPUs for faster inference
3. Adjust temperature based on task (0.7-0.8 for general text, 0.9-1.0 for creative)
"""

import argparse
import warnings
from pathlib import Path

import torch

# Configure matmul precision for better performance on modern GPUs
# For PyTorch 2.0+, use set_float32_matmul_precision
# For development versions, we suppress the deprecation warning as the API is in flux
if torch.cuda.is_available():
    torch_version = torch.__version__
    if "dev" in torch_version or "+" in torch_version:
        # Development version - suppress the warning as APIs are changing
        warnings.filterwarnings("ignore", message=".*TF32.*", category=UserWarning)
        warnings.filterwarnings(
            "ignore", message=".*float32_matmul_precision.*", category=UserWarning
        )

    # Set precision for all versions
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        # If the API is not available or fails, continue without it
        pass

from nanoLLM_gpt.config import GenerationConfig
from nanoLLM_gpt.utils import InferencePipeline


def interactive_mode(pipeline: InferencePipeline):
    """
    Run interactive text generation mode.

    Provides an interactive REPL (Read-Eval-Print Loop) for experimenting
    with text generation. Users can enter prompts, adjust parameters,
    and see results in real-time.

    Features:
        - Live parameter adjustment without restarting
        - Streaming output for immediate feedback
        - Command system for control
        - Persistent parameter settings across prompts

    Args:
        pipeline (InferencePipeline): Initialized inference pipeline with loaded model

    Commands:
        - quit: Exit interactive mode
        - help: Display current parameters and commands
        - set <param> <value>: Modify generation parameters
            - max_tokens: Maximum new tokens (int)
            - temperature: Sampling temperature (float)
            - top_k: Top-k filtering (int)
            - top_p: Nucleus sampling threshold (float)

    Called by:
        - main() when --interactive flag is provided

    Example Session:
        ```
        Prompt: Tell me a story
        [Generated text appears here...]

        Prompt: set temperature 0.9
        Set temperature to 0.9

        Prompt: Continue the story
        [More creative text with higher temperature...]
        ```
    """
    print("\nInteractive text generation mode")
    print("Commands:")
    print("  'quit' - Exit")
    print("  'help' - Show options")
    print("  'set <param> <value>' - Change generation parameters")
    print("-" * 50)

    # Default config
    config = GenerationConfig()

    while True:
        try:
            prompt = input("\nPrompt: ").strip()

            if prompt.lower() == "quit":
                break

            elif prompt.lower() == "help":
                print("\nGeneration parameters:")
                print(f"  max_tokens: {config.max_new_tokens}")
                print(f"  temperature: {config.temperature}")
                print(f"  top_k: {config.top_k}")
                print(f"  top_p: {config.top_p}")
                print("\nSet parameters with: set <param> <value>")
                print("Example: set temperature 0.8")
                continue

            elif prompt.startswith("set "):
                parts = prompt.split()
                if len(parts) == 3:
                    param, value = parts[1], parts[2]
                    try:
                        if param == "max_tokens":
                            config.max_new_tokens = int(value)
                        elif param == "temperature":
                            config.temperature = float(value)
                        elif param == "top_k":
                            config.top_k = int(value)
                        elif param == "top_p":
                            config.top_p = float(value)
                        else:
                            print(f"Unknown parameter: {param}")
                            continue
                        print(f"Set {param} to {value}")
                    except ValueError:
                        print("Invalid value")
                continue

            if not prompt:
                continue

            # Generate text
            print("\nGenerating...\n")
            print(prompt, end="")

            # Stream generation
            config.stream = True
            for token in pipeline.generate(prompt, config):
                print(token, end="", flush=True)

            print("\n" + "-" * 50)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """
    Main entry point for text generation script.

    Handles command-line argument parsing, model loading, and orchestrates
    the generation process based on user inputs.

    Process Flow:
        1. Parse command-line arguments
        2. Set random seeds for reproducibility
        3. Initialize inference pipeline
        4. Load model (checkpoint or HuggingFace)
        5. Handle generation mode:
            - Interactive: Enter REPL loop
            - Batch: Generate multiple samples
            - Single: Generate one sample with streaming

    Exit Codes:
        0: Success
        1: Error (model loading, generation failure)

    Environment Variables:
        CUDA_VISIBLE_DEVICES: Control GPU selection
        TOKENIZERS_PARALLELISM: Disable tokenizer warnings

    Called by:
        - Command line: python generate.py [args]
        - Entry point: gpt-generate [args]
    """
    parser = argparse.ArgumentParser(
        description="Generate text from GPT models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ArgumentParser configuration for comprehensive CLI interface.
    # Groups arguments logically: model source, prompt source, generation params, system settings

    # Model source (mutually exclusive: either checkpoint OR HuggingFace model)
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--checkpoint",
        type=str,
        help="Path to model checkpoint file (.pt or .pth) from training",
    )
    model_group.add_argument(
        "--model",
        type=str,
        default="gpt2",
        choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
        help="HuggingFace model type - gpt2 (124M), gpt2-medium (350M), "
        "gpt2-large (774M), gpt2-xl (1.5B) (default: gpt2)",
    )

    # Prompt source (mutually exclusive: direct prompt, file, or interactive)
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument(
        "--prompt",
        type=str,
        help="Text prompt for generation (enclose in quotes for multi-word prompts)",
    )
    prompt_group.add_argument(
        "--prompt-file",
        type=str,
        help="Path to file containing prompt text (UTF-8 encoded)",
    )
    prompt_group.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive generation mode - enter prompts in a REPL loop",
    )

    # Generation parameters - control text quality and diversity
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum new tokens to generate. Higher values produce longer text "
        "but may lose coherence (default: 100)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature - controls randomness. 0.0=deterministic, "
        "0.8=balanced, 1.0+=creative (default: 0.8)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=200,
        help="Top-k filtering - only consider k most likely tokens. "
        "Lower=higher quality, Higher=more diversity (default: 200)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p (nucleus) sampling - cumulative probability threshold. "
        "0.9=diverse but coherent, 1.0=disabled (default: 1.0)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of independent samples to generate from same prompt (default: 1)",
    )

    # System parameters - hardware and optimization settings
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on - cuda, cuda:0, cpu, mps "
        "(default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Model precision - auto selects based on hardware, "
        "bfloat16 preferred for Ampere+ GPUs (default: auto)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile model with PyTorch 2.0 torch.compile() for ~30%% speedup "
        "(requires PyTorch 2.0+)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for reproducible generation (default: 1337)",
    )

    args = parser.parse_args()

    # Set random seed for reproducible generation
    # This ensures the same prompt produces the same output with same parameters
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Initialize inference pipeline with device and precision settings
    print("Loading model...")
    pipeline = InferencePipeline(device=args.device, dtype=args.dtype)

    # Load model from checkpoint or HuggingFace
    # Checkpoint takes precedence if both are somehow specified
    pipeline.load_model(
        checkpoint_path=args.checkpoint, model_type=args.model, compile=args.compile
    )

    # Model is now loaded and ready for generation
    model_source = args.checkpoint if args.checkpoint else f"HuggingFace {args.model}"
    print(f"Model loaded from {model_source}")

    # Handle interactive mode - enters REPL loop
    if args.interactive:
        interactive_mode(pipeline)
        return  # Exit after interactive session

    # Get prompt from specified source
    if args.prompt_file:
        # Read prompt from file
        try:
            with open(args.prompt_file, "r", encoding="utf-8") as f:
                prompt = f.read()
        except FileNotFoundError:
            print(f"Error: Prompt file '{args.prompt_file}' not found")
            return
        except Exception as e:
            print(f"Error reading prompt file: {e}")
            return
    elif args.prompt:
        # Use command-line prompt
        prompt = args.prompt
    else:
        # Default prompt if none provided
        prompt = "Hello, I'm a language model,"

    # Create generation config from command-line arguments
    # This config controls all aspects of the generation process
    config = GenerationConfig(
        max_new_tokens=args.max_tokens,  # Length limit
        temperature=args.temperature,  # Randomness control
        top_k=args.top_k,  # Quality filtering
        top_p=args.top_p,  # Nucleus sampling
        num_samples=args.num_samples,  # Number of variations
    )

    # Generate text based on configuration
    print(f"\nGenerating {args.num_samples} sample(s)...")
    print(f"Prompt: {repr(prompt)}")
    print("-" * 80)

    if args.num_samples > 1:
        # Batch generation mode - generate multiple independent samples
        # Each sample uses different random sampling for variety
        samples = pipeline.generate(prompt, config)
        for i, sample in enumerate(samples):
            print(f"\n--- Sample {i+1} ---")
            print(prompt + sample)
            if i < len(samples) - 1:
                print("-" * 80)
    else:
        # Single sample generation with streaming output
        # Streaming provides real-time token-by-token display
        print(prompt, end="")  # Print prompt without newline
        config.stream = True  # Enable streaming mode

        # Generate and print tokens as they're produced
        for token in pipeline.generate(prompt, config):
            print(token, end="", flush=True)  # flush ensures immediate display
        print()  # Final newline


if __name__ == "__main__":
    main()
