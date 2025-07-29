"""
Centralized inference pipeline for text generation.

This module provides a unified interface for text generation across all
entry points (API, web UI, CLI), ensuring consistent behavior and optimal
performance for inference tasks.

## Key Features:

1. **Unified Generation Interface**:
   - Single pipeline for all text generation needs
   - Consistent tokenization and decoding
   - Support for streaming and batch generation

2. **Advanced Sampling Strategies**:
   - Temperature sampling for creativity control
   - Top-k filtering for quality
   - Top-p (nucleus) sampling for diversity
   - Stop sequences for controlled termination

3. **Chat Completion Support**:
   - OpenAI-compatible chat format
   - Message role handling (system/user/assistant)
   - Streaming chat responses

4. **Performance Optimizations**:
   - Automatic mixed precision (AMP)
   - KV-cache utilization
   - Efficient batch processing

## Usage Example:
```python
# Initialize pipeline
pipeline = InferencePipeline(
    device='cuda',
    dtype='auto'
)

# Load model
pipeline.load_model(model_type='gpt2-medium')

# Generate text
text = pipeline.generate(
    "Once upon a time",
    max_new_tokens=100,
    temperature=0.8,
    top_k=50
)

# Chat completion
response = pipeline.chat_completion([
    {"role": "user", "content": "Tell me a joke"}
])
```

## Sampling Methods:

1. **Temperature**: Controls randomness (0=deterministic, >1=creative)
2. **Top-k**: Sample from k most likely tokens
3. **Top-p**: Sample from tokens with cumulative probability < p
4. **Repetition penalty**: Reduce repetition (not implemented yet)
"""

import time
from typing import List, Optional, Union, Iterator, Dict, Any
from contextlib import nullcontext

import torch
import tiktoken

from ..model import GPT
from ..config import GenerationConfig
from .model_loader import ModelLoader


class InferencePipeline:
    """
    Unified text generation pipeline for all interfaces.

    This class provides a high-level interface for text generation,
    handling model loading, tokenization, sampling strategies, and
    output formatting. It's designed to be the single entry point
    for all inference needs across the project.

    The pipeline abstracts away:
    - Model and tokenizer management
    - Device and precision handling
    - Sampling algorithm implementation
    - Streaming and batch generation
    - Chat format conversions

    Attributes:
        model (GPT): Loaded language model
        tokenizer: Tiktoken encoder/decoder
        device (str): Compute device
        device_type (str): Device category ('cuda' or 'cpu')
        ctx: Autocast context for mixed precision

    Methods:
        load_model(): Load a model into the pipeline
        generate(): Generate text from prompt
        chat_completion(): OpenAI-compatible chat API
        encode/decode(): Text/token conversion

    Called by:
        - generate.main() for CLI generation
        - server.py API endpoints
        - Web UI backend handlers
    """

    def __init__(
        self,
        model: Optional[GPT] = None,
        tokenizer_name: str = "gpt2",
        device: str = "cuda",
        dtype: str = "auto",
    ):
        """
        Initialize inference pipeline with optional pre-loaded model.

        Sets up the inference environment including tokenizer, device
        configuration, and autocast context for optimal performance.

        Args:
            model (Optional[GPT]): Pre-loaded model instance
                - If provided: Use this model directly
                - If None: Must call load_model() before generation
            tokenizer_name (str): Tiktoken tokenizer to use
                - 'gpt2': Standard GPT-2 tokenizer (50257 tokens)
                - Must match the model's training tokenizer
            device (str): Target device for inference
                - 'cuda' or 'cuda:N': GPU inference
                - 'cpu': CPU inference (slower)
                - 'mps': Apple Silicon GPU
            dtype (str): Precision for inference
                - 'auto': Automatically select best precision
                - 'float32': Full precision
                - 'float16': Half precision (GPU only)
                - 'bfloat16': Brain float (Ampere+ GPUs)

        Example:
            >>> # With pre-loaded model
            >>> model = GPT.from_pretrained('gpt2')
            >>> pipeline = InferencePipeline(model=model)

            >>> # Without pre-loaded model
            >>> pipeline = InferencePipeline(device='cuda')
            >>> pipeline.load_model(model_type='gpt2-medium')
        """
        self.model = model
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        self.device = device
        self.device_type = "cuda" if "cuda" in device else "cpu"
        self.ctx = ModelLoader.get_model_context(self.device_type, dtype)
        
        # Move model to the correct device if provided
        if self.model is not None:
            self.model = self.model.to(self.device)

    def load_model(
        self,
        checkpoint_path: Optional[str] = None,
        model_type: str = "gpt2",
        compile: bool = False,
        model_config: Optional[Any] = None,
    ):
        """
        Load a model into the pipeline for inference.

        Supports loading from local checkpoints or HuggingFace hub.
        Handles model compilation and configuration overrides.

        Args:
            checkpoint_path (Optional[str]): Path to saved checkpoint
                - Local file path: Load from checkpoint
                - None: Load from HuggingFace based on model_type
            model_type (str): HuggingFace model identifier
                - 'gpt2': 124M parameters
                - 'gpt2-medium': 350M parameters
                - 'gpt2-large': 774M parameters
                - 'gpt2-xl': 1.5B parameters
            compile (bool): Enable PyTorch 2.0 compilation
                - True: ~30% faster inference on compatible hardware
                - False: Standard eager mode execution
            model_config (Optional[Any]): Configuration overrides
                - Dict or ModelConfig instance
                - Used to modify architecture (e.g., dropout)

        Side Effects:
            - Sets self.model to loaded model
            - Model moved to configured device
            - Compilation applied if requested

        Calls:
            - ModelLoader.load_model() for actual loading

        Example:
            >>> # Load from HuggingFace
            >>> pipeline.load_model(model_type='gpt2-medium', compile=True)

            >>> # Load from checkpoint
            >>> pipeline.load_model(checkpoint_path='out/ckpt.pt')

            >>> # Load with custom config
            >>> config = {'dropout': 0.1}  # Add dropout for sampling
            >>> pipeline.load_model(model_type='gpt2', model_config=config)
        """
        if model_config is not None:
            # Use provided config when loading from checkpoint
            self.model = ModelLoader.load_model(
                checkpoint_path=checkpoint_path,
                model_type=model_type,
                device=self.device,
                compile=compile,
                override_args=(
                    model_config.__dict__
                    if hasattr(model_config, "__dict__")
                    else model_config
                ),
            )
        else:
            self.model = ModelLoader.load_model(
                checkpoint_path=checkpoint_path,
                model_type=model_type,
                device=self.device,
                compile=compile,
            )

    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs using tiktoken.

        Converts human-readable text into the token IDs that the model
        processes. Handles special tokens like <|endoftext|>.

        Args:
            text (str): Input text to tokenize

        Returns:
            List[int]: Token IDs representing the text

        Special Tokens:
            - <|endoftext|> (50256): End of text marker

        Example:
            >>> tokens = pipeline.encode("Hello world")
            >>> print(tokens)  # [15496, 995]
        """
        return self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, tokens: List[int]) -> str:
        """
        Decode token IDs back to text.

        Converts model output tokens into human-readable text.
        Handles byte-pair encoding artifacts automatically.

        Args:
            tokens (List[int]): Token IDs to decode

        Returns:
            str: Decoded text

        Example:
            >>> text = pipeline.decode([15496, 995])
            >>> print(text)  # "Hello world"
        """
        return self.tokenizer.decode(tokens)

    @torch.no_grad()
    def generate(
        self,
        prompt: Union[str, List[int]],
        config: Optional[GenerationConfig] = None,
        **kwargs,
    ) -> Union[str, List[str], Iterator[str]]:
        """
        Generate text from prompt using configured sampling strategy.

        Main generation method supporting various output formats and
        sampling strategies. Handles single/batch/streaming generation.

        Args:
            prompt (Union[str, List[int]]): Input prompt
                - str: Text prompt (will be tokenized)
                - List[int]: Pre-tokenized prompt
            config (Optional[GenerationConfig]): Generation settings
                - If None: Uses default GenerationConfig
                - Controls sampling, length, streaming, etc.
            **kwargs: Override specific config parameters
                - max_new_tokens: Maximum tokens to generate
                - temperature: Sampling temperature
                - top_k: Top-k filtering
                - top_p: Nucleus sampling threshold
                - stream: Enable streaming output
                - num_samples: Number of outputs
                - stop_sequences: Early stopping strings

        Returns:
            Union[str, List[str], Iterator[str]]:
                - str: Single generated text (default)
                - List[str]: Multiple samples (if num_samples > 1)
                - Iterator[str]: Token stream (if stream=True)

        Raises:
            RuntimeError: If no model is loaded

        Generation Modes:
            1. Single: Generate one complete response
            2. Batch: Generate multiple independent samples
            3. Stream: Yield tokens as they're generated

        Called by:
            - CLI generate.py main function
            - API /completions endpoint
            - Web UI generation handler

        Example:
            >>> # Simple generation
            >>> text = pipeline.generate("The meaning of life is")

            >>> # Custom parameters
            >>> text = pipeline.generate(
            ...     "Once upon a time",
            ...     max_new_tokens=200,
            ...     temperature=0.9,
            ...     top_p=0.95
            ... )

            >>> # Streaming
            >>> for token in pipeline.generate("Hello", stream=True):
            ...     print(token, end='', flush=True)
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        # Use default config if not provided
        if config is None:
            config = GenerationConfig()

        # Override config with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        # Encode prompt if string
        if isinstance(prompt, str):
            prompt_tokens = self.encode(prompt)
        else:
            prompt_tokens = prompt

        # Convert to tensor
        idx = torch.tensor(prompt_tokens, dtype=torch.long, device=self.device)[
            None, ...
        ]

        # Generate
        if config.stream:
            return self._generate_stream(idx, len(prompt_tokens), config)
        elif config.num_samples > 1:
            return self._generate_batch(idx, len(prompt_tokens), config)
        else:
            return self._generate_single(idx, len(prompt_tokens), config)

    def _generate_single(
        self, idx: torch.Tensor, prompt_length: int, config: GenerationConfig
    ) -> str:
        """
        Generate a single text sample (internal method).

        Handles non-streaming, single-sample generation using either
        the model's built-in generation or advanced top-p sampling.

        Args:
            idx (torch.Tensor): Input token tensor (1, seq_len)
            prompt_length (int): Number of prompt tokens
            config (GenerationConfig): Generation parameters

        Returns:
            str: Generated text (excluding prompt)

        Sampling Selection:
            - If top_p < 1.0: Use nucleus sampling
            - Otherwise: Use model's temperature + top_k

        Called by:
            - generate() when stream=False and num_samples=1
        """
        with self.ctx:
            # Use advanced generation if top_p is set
            if config.top_p < 1.0:
                tokens = self._generate_with_top_p(idx, config)
            else:
                # Use model's built-in generation
                output = self.model.generate(
                    idx,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_k=config.top_k,
                )
                tokens = output[0].tolist()

            # Decode tokens
            return self.decode(tokens[prompt_length:])

    def _generate_batch(
        self, idx: torch.Tensor, prompt_length: int, config: GenerationConfig
    ) -> List[str]:
        """
        Generate multiple text samples (internal method).

        Produces multiple independent samples from the same prompt,
        useful for exploring different completions.

        Args:
            idx (torch.Tensor): Input token tensor (1, seq_len)
            prompt_length (int): Number of prompt tokens
            config (GenerationConfig): Generation parameters

        Returns:
            List[str]: Generated texts (excluding prompt)

        Note:
            Each sample is generated independently with
            different random sampling.

        Called by:
            - generate() when num_samples > 1
        """
        samples = []

        with self.ctx:
            for _ in range(config.num_samples):
                if config.top_p < 1.0:
                    tokens = self._generate_with_top_p(idx, config)
                else:
                    output = self.model.generate(
                        idx,
                        max_new_tokens=config.max_new_tokens,
                        temperature=config.temperature,
                        top_k=config.top_k,
                    )
                    tokens = output[0].tolist()

                text = self.decode(tokens[prompt_length:])
                samples.append(text)

        return samples

    def _generate_stream(
        self, idx: torch.Tensor, prompt_length: int, config: GenerationConfig
    ) -> Iterator[str]:
        """
        Generate text in streaming mode (internal method).

        Yields tokens as they're generated for real-time output.
        Implements custom generation loop for fine control.

        Args:
            idx (torch.Tensor): Input token tensor (1, seq_len)
            prompt_length (int): Number of prompt tokens
            config (GenerationConfig): Generation parameters

        Yields:
            str: Individual tokens as generated

        Features:
            - Real-time token streaming
            - Respects block_size limits
            - Stop sequence detection
            - All sampling strategies supported

        Called by:
            - generate() when stream=True
            - chat_completion() for streaming responses
        """
        with self.ctx:
            generated_tokens = []

            for _ in range(config.max_new_tokens):
                # Get next token
                idx_cond = (
                    idx
                    if idx.size(1) <= self.model.config.block_size
                    else idx[:, -self.model.config.block_size :]
                )

                logits, _ = self.model(idx_cond)
                logits = logits[:, -1, :] / config.temperature

                # Apply sampling
                if config.top_p < 1.0:
                    logits = self._apply_top_p(logits, config.top_p)
                elif config.top_k is not None:
                    v, _ = torch.topk(logits, min(config.top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")

                # Sample token
                probs = torch.nn.functional.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

                # Check stop conditions
                token_id = idx_next.item()
                if self._should_stop(token_id, generated_tokens, config):
                    break

                # Append token
                idx = torch.cat((idx, idx_next), dim=1)
                generated_tokens.append(token_id)

                # Yield token
                yield self.decode([token_id])

    def _generate_with_top_p(
        self, idx: torch.Tensor, config: GenerationConfig
    ) -> List[int]:
        """
        Generate tokens using top-p (nucleus) sampling.

        Implements nucleus sampling where tokens are selected from
        the smallest set whose cumulative probability exceeds p.

        Args:
            idx (torch.Tensor): Input token tensor
            config (GenerationConfig): Must have top_p < 1.0

        Returns:
            List[int]: All tokens including prompt

        Algorithm:
            1. Sort tokens by probability
            2. Find cumulative probability threshold
            3. Zero out tokens beyond threshold
            4. Renormalize and sample

        Benefits:
            - More diverse than top-k
            - Adapts to confidence level
            - Reduces repetition

        Called by:
            - _generate_single() when top_p specified
            - _generate_batch() when top_p specified
        """
        tokens = idx[0].tolist()

        for _ in range(config.max_new_tokens):
            # Get logits
            idx_cond = (
                idx
                if idx.size(1) <= self.model.config.block_size
                else idx[:, -self.model.config.block_size :]
            )

            logits, _ = self.model(idx_cond)
            logits = logits[:, -1, :] / config.temperature

            # Apply top-p filtering
            logits = self._apply_top_p(logits, config.top_p)

            # Apply top-k if specified
            if config.top_k is not None:
                v, _ = torch.topk(logits, min(config.top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            # Sample
            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Check stop conditions
            token_id = idx_next.item()
            if self._should_stop(token_id, tokens, config):
                break

            # Append
            idx = torch.cat((idx, idx_next), dim=1)
            tokens.append(token_id)

        return tokens

    @staticmethod
    def _apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """
        Apply top-p (nucleus) filtering to logits.

        Filters logits to only include tokens from the smallest set
        with cumulative probability >= top_p. This creates a dynamic
        vocabulary size based on the model's confidence.

        Args:
            logits (torch.Tensor): Raw model logits (batch, vocab)
            top_p (float): Cumulative probability threshold (0-1)

        Returns:
            torch.Tensor: Filtered logits with -inf for excluded tokens

        Algorithm Details:
            1. Sort logits in descending order
            2. Calculate cumulative softmax probabilities
            3. Find cutoff where cumsum > top_p
            4. Mask tokens beyond cutoff with -inf

        Example:
            - top_p=0.9: Keep tokens until 90% probability mass
            - High confidence: Might only keep top 10 tokens
            - Low confidence: Might keep top 100 tokens

        Called by:
            - _generate_stream() during streaming
            - _generate_with_top_p() for full generation
        """
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1
        )

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = -float("Inf")

        return logits

    def _should_stop(
        self, token_id: int, generated_tokens: List[int], config: GenerationConfig
    ) -> bool:
        """
        Check if generation should stop.

        Evaluates multiple stopping conditions to determine when
        to end text generation.

        Args:
            token_id (int): Most recently generated token
            generated_tokens (List[int]): All generated tokens so far
            config (GenerationConfig): Config with stop_sequences

        Returns:
            bool: True if generation should stop

        Stop Conditions:
            1. EOS token encountered (token 50256)
            2. Stop sequence detected in generated text
            3. Maximum length reached (checked by caller)

        Called by:
            - _generate_stream() after each token
            - _generate_with_top_p() after each token
        """
        # Check for EOS token
        if token_id == self.tokenizer.eot_token:
            return True

        # Check stop sequences
        if config.stop_sequences:
            current_text = self.decode(generated_tokens)
            for stop_seq in config.stop_sequences:
                if current_text.endswith(stop_seq):
                    return True

        return False

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        config: Optional[GenerationConfig] = None,
        **kwargs,
    ) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """
        Generate OpenAI-compatible chat completion response.

        Converts chat messages into model prompt and generates response
        in OpenAI's chat completion format for API compatibility.

        Args:
            messages (List[Dict[str, str]]): Chat messages
                Each message must have:
                - 'role': 'system', 'user', or 'assistant'
                - 'content': Message text
                - 'name' (optional): Speaker name
            config (Optional[GenerationConfig]): Generation settings
            **kwargs: Override config parameters

        Returns:
            Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
                - Dict: Complete response (non-streaming)
                - Iterator: Chunked response (streaming)

        Response Format (non-streaming):
            ```json
            {
                "id": "chatcmpl-...",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "gpt",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "..."
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30
                }
            }
            ```

        Called by:
            - API /chat/completions endpoint
            - Chat UI backends

        Example:
            >>> response = pipeline.chat_completion([
            ...     {"role": "system", "content": "You are a helpful assistant."},
            ...     {"role": "user", "content": "What is the capital of France?"}
            ... ])
            >>> print(response['choices'][0]['message']['content'])
        """
        # Format messages into prompt
        prompt = self._format_chat_prompt(messages)

        # Generate response
        if config and config.stream:
            return self._stream_chat_completion(prompt, messages, config, **kwargs)
        else:
            response_text = self.generate(prompt, config, **kwargs)

            # Build response
            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "gpt",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": response_text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(self.encode(prompt)),
                    "completion_tokens": len(self.encode(response_text)),
                    "total_tokens": len(self.encode(prompt + response_text)),
                },
            }

    def _format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Format chat messages into a single prompt.

        Converts OpenAI-style message list into a formatted prompt
        string that the model can process.

        Args:
            messages (List[Dict[str, str]]): Chat messages with role/content

        Returns:
            str: Formatted prompt ending with "Assistant: "

        Format:
            ```
            System: {system message}

            User: {user message}

            Assistant: {assistant response}

            User: {next user message}

            Assistant:
            ```

        Note:
            The format is simple but effective. More sophisticated
            formatting (with special tokens) can improve quality.

        Called by:
            - chat_completion() to prepare prompt
        """
        prompt_parts = []

        for message in messages:
            role = message["role"]
            content = message["content"]

            if role == "system":
                prompt_parts.append(f"System: {content}\n\n")
            elif role == "user":
                prompt_parts.append(f"User: {content}\n\n")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}\n\n")

        # Add prefix for assistant's response
        prompt_parts.append("Assistant: ")

        return "".join(prompt_parts)

    def _stream_chat_completion(
        self,
        prompt: str,
        messages: List[Dict[str, str]],
        config: GenerationConfig,
        **kwargs,
    ) -> Iterator[Dict[str, Any]]:
        """
        Generate streaming chat completion response.

        Produces OpenAI-compatible streaming response format for
        real-time chat applications.

        Args:
            prompt (str): Formatted prompt from messages
            messages (List[Dict[str, str]]): Original messages (for context)
            config (GenerationConfig): Must have stream=True
            **kwargs: Additional generation parameters

        Yields:
            Dict[str, Any]: Streaming response chunks

        Chunk Format:
            ```json
            {
                "id": "chatcmpl-...",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": "gpt",
                "choices": [{
                    "index": 0,
                    "delta": {"content": "token"},
                    "finish_reason": null
                }]
            }
            ```

        Stream Sequence:
            1. Initial chunk with role
            2. Content chunks with tokens
            3. Final chunk with finish_reason

        Called by:
            - chat_completion() when stream=True
        """
        completion_id = f"chatcmpl-{int(time.time())}"
        created = int(time.time())

        # Initial response
        yield {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": "gpt",
            "choices": [
                {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
            ],
        }

        # Stream tokens
        for token in self.generate(prompt, config, stream=True, **kwargs):
            yield {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": "gpt",
                "choices": [
                    {"index": 0, "delta": {"content": token}, "finish_reason": None}
                ],
            }

        # Final response
        yield {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": "gpt",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
