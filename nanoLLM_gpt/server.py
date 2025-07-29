"""
Unified server for GPT model serving with both API and web interface.

This server provides a comprehensive platform for both using and training GPT models,
with a RESTful API for programmatic access and a web interface for interactive use.

## Key Features:

### 1. OpenAI-Compatible API
- `/v1/chat/completions`: Chat completion endpoint (streaming supported)
- `/v1/completions`: Text completion endpoint
- `/v1/models`: List available models
- Full compatibility with OpenAI client libraries

### 2. Web Interface
- Model loading from checkpoints or HuggingFace
- Interactive text generation with parameter control
- Training management with real-time logs
- Model configuration export/import

### 3. Training Management
- Start/stop training from web UI
- Upload training data or provide URLs
- Real-time training log streaming
- Automatic checkpoint saving

### 4. Dynamic Model Loading
- Load models without restarting server
- Support for local files and URLs
- HuggingFace model integration
- Custom configuration support

## Usage Examples:

```bash
# Start server with default GPT-2
python server.py

# Start with custom model
python server.py --checkpoint out/ckpt.pt

# Start with specific configuration
python server.py --config server_config.yaml

# Custom host and port
python server.py --host 0.0.0.0 --port 8080

# With compiled model for faster inference
python server.py --model-type gpt2-medium --compile
```

## API Usage:

```python
# Using OpenAI client
import openai
openai.api_base = "http://localhost:8080/v1"
openai.api_key = "dummy"  # Not used but required

response = openai.ChatCompletion.create(
    model="gpt2",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Web Interface Endpoints:

- `/`: Main web interface
- `/api/model/load`: Load new model
- `/api/generate`: Generate text
- `/api/training/start`: Start training
- `/api/training/status`: Get training status
- `/api/model/info`: Get model information

## Environment Variables:

- `CUDA_VISIBLE_DEVICES`: Control GPU selection
- `TOKENIZERS_PARALLELISM`: Disable tokenizer warnings

## Security Notes:

- CORS is enabled by default for API access
- No authentication by default (add for production)
- File uploads are sanitized but use caution
"""

import os
import json
import time
import threading
import subprocess
import argparse
import requests
import tempfile
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import uuid

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

from flask import (
    Flask,
    request,
    jsonify,
    Response,
    render_template,
    stream_with_context,
)
from flask_cors import CORS
from werkzeug.utils import secure_filename

from nanoLLM_gpt.config import APIConfig, GenerationConfig, ChatMessage, ConfigLoader
from nanoLLM_gpt.utils import InferencePipeline, ModelLoader


# Flask application with blueprints
app = Flask(__name__, template_folder="templates")
CORS(app)

# Global state
model_pipeline: Optional[InferencePipeline] = None
training_manager = None
config: Optional[APIConfig] = None
current_model_info: Dict[str, Any] = {}


class TrainingManager:
    """
    Manages model training processes with subprocess control.

    This class handles the lifecycle of training processes, including:
    - Starting training with configuration
    - Real-time log capture and streaming
    - Process monitoring and termination
    - Thread-safe state management

    The training runs as a subprocess to isolate it from the server
    process, preventing memory issues and allowing clean termination.

    Attributes:
        current_process: Active training subprocess
        training_logs: Circular buffer of recent log entries
        is_training: Current training state
        training_config: Active training configuration
        lock: Thread lock for state synchronization

    Methods:
        start_training(): Launch new training process
        stop_training(): Terminate active training
        get_status(): Get current training state
        _read_logs(): Internal log reader thread

    Thread Safety:
        All public methods are thread-safe using a lock to ensure
        consistent state across concurrent requests.

    Called by:
        - API endpoints for training control
        - Web interface training management
    """

    def __init__(self):
        self.current_process: Optional[subprocess.Popen] = None
        self.training_logs: List[Dict[str, Any]] = []
        self.is_training: bool = False
        self.training_config: Dict[str, Any] = {}
        self.lock = threading.Lock()

    def start_training(self, train_config: Dict[str, Any], data_path: str) -> bool:
        """
        Start a new training process with given configuration.

        Launches a subprocess running the training script with specified
        parameters. Only one training process can run at a time.

        Args:
            train_config (Dict[str, Any]): Training configuration
                - out_dir: Output directory for checkpoints
                - batch_size: Training batch size
                - max_iters: Maximum iterations
                - learning_rate: Initial learning rate
                - eval_interval: Evaluation frequency
                - n_layer, n_head, n_embd: Model architecture
                - block_size: Context length
            data_path (str): Path or URL to training data
                - Local file path: Direct file access
                - HTTP/HTTPS URL: Will be downloaded

        Returns:
            bool: True if training started, False if already training

        Process:
            1. Check if training already active
            2. Save configuration to output directory
            3. Build command-line arguments
            4. Launch subprocess with gpt-train
            5. Start log reader thread

        Side Effects:
            - Creates output directory if needed
            - Saves config.yaml in output directory
            - Starts background log reader thread

        Called by:
            - /api/training/start endpoint
        """
        with self.lock:
            if self.is_training:
                return False

            self.training_logs = []
            self.training_config = train_config
            self.is_training = True

            # Save training config to output directory
            out_dir = Path(train_config.get("out_dir", "out"))
            out_dir.mkdir(parents=True, exist_ok=True)

            # Create TrainingConfig object and save it
            from nanoLLM_gpt.config import TrainingConfig, ModelConfig, ConfigLoader

            model_config = ModelConfig(
                n_layer=train_config.get("n_layer", 12),
                n_head=train_config.get("n_head", 12),
                n_embd=train_config.get("n_embd", 768),
                block_size=train_config.get("block_size", 1024),
            )
            training_config = TrainingConfig(
                model=model_config,
                out_dir=train_config.get("out_dir", "out"),
                batch_size=train_config.get("batch_size", 12),
                max_iters=train_config.get("max_iters", 5000),
                learning_rate=train_config.get("learning_rate", 6e-4),
                eval_interval=train_config.get("eval_interval", 500),
                data_path=data_path,
            )
            ConfigLoader.save_to_file(training_config, out_dir / "config.yaml")

            # Prepare command
            cmd = [
                "gpt-train",
                f"--data-path={data_path}",
                f'--out-dir={train_config.get("out_dir", "out")}',
                f'--batch-size={train_config.get("batch_size", 12)}',
                f'--max-iters={train_config.get("max_iters", 5000)}',
                f'--learning-rate={train_config.get("learning_rate", 6e-4)}',
                f'--eval-interval={train_config.get("eval_interval", 500)}',
            ]

            # Add optional parameters
            for key in ["n_layer", "n_head", "n_embd", "block_size"]:
                if key in train_config:
                    cmd.append(f'--{key.replace("_", "-")}={train_config[key]}')

            # Start process
            self.current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
            )

            # Start log reader
            threading.Thread(target=self._read_logs, daemon=True).start()

            return True

    def _read_logs(self):
        """
        Read training logs from subprocess (internal thread method).

        Runs in a separate thread to continuously read output from the
        training subprocess. Maintains a circular buffer of recent logs.

        Log Format:
            Each log entry contains:
            - timestamp: ISO format timestamp
            - message: Log line from training script

        Buffer Management:
            - Keeps last 1000 log entries
            - Older entries are discarded
            - Thread-safe access via lock

        Lifecycle:
            - Starts when training begins
            - Runs until process terminates
            - Updates is_training flag on completion

        Called by:
            - start_training() via threading.Thread
        """
        try:
            for line in iter(self.current_process.stdout.readline, ""):
                if line:
                    log_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "message": line.strip(),
                    }
                    with self.lock:
                        self.training_logs.append(log_entry)
                        # Keep only last 1000 logs
                        if len(self.training_logs) > 1000:
                            self.training_logs.pop(0)

            self.current_process.wait()
        finally:
            with self.lock:
                self.is_training = False
                self.current_process = None

    def stop_training(self) -> bool:
        """
        Stop the current training process gracefully.

        Attempts graceful termination first, then force kills if needed.
        Ensures clean process shutdown to prevent zombie processes.

        Returns:
            bool: True if process was stopped, False if no active process

        Termination Sequence:
            1. Send SIGTERM for graceful shutdown
            2. Wait up to 5 seconds for termination
            3. Send SIGKILL if still running

        Called by:
            - /api/training/stop endpoint
            - Server shutdown handlers
        """
        with self.lock:
            if self.current_process:
                self.current_process.terminate()
                try:
                    self.current_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.current_process.kill()
                return True
            return False

    def get_status(self) -> Dict[str, Any]:
        """
        Get current training status and recent logs.

        Provides a snapshot of the training state for UI updates.
        Returns only the most recent logs to avoid large payloads.

        Returns:
            Dict[str, Any]: Status dictionary containing:
                - is_training: Whether training is active
                - config: Current training configuration
                - logs: Last 100 log entries

        Thread Safety:
            Uses lock to ensure consistent snapshot

        Called by:
            - /api/training/status endpoint
            - Web UI polling for updates
        """
        with self.lock:
            return {
                "is_training": self.is_training,
                "config": self.training_config,
                "logs": self.training_logs[-100:],
            }


# API Routes
@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    """
    OpenAI-compatible chat completions endpoint.

    Implements the OpenAI Chat Completions API for compatibility with
    existing tools and libraries. Supports both streaming and non-streaming
    responses.

    Request Format (JSON):
        {
            "model": "gpt2",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello!"}
            ],
            "temperature": 0.7,
            "max_tokens": 100,
            "stream": false
        }

    Response Format:
        - Non-streaming: Complete JSON response
        - Streaming: Server-sent events (SSE) with chunks

    Error Codes:
        - 503: Model not loaded
        - 500: Internal server error

    Called by:
        - OpenAI Python client
        - curl/HTTP clients
        - LangChain and other frameworks

    Calls:
        - InferencePipeline.chat_completion()
    """
    try:
        if not model_pipeline or not model_pipeline.model:
            return (
                jsonify(
                    {
                        "error": {
                            "message": "No model loaded. Please load a model first.",
                            "type": "model_not_loaded",
                            "code": 503,
                        }
                    }
                ),
                503,
            )

        data = request.json

        # Parse messages
        messages = []
        for msg in data.get("messages", []):
            messages.append({"role": msg["role"], "content": msg["content"]})

        # Create generation config
        gen_config = GenerationConfig(
            max_new_tokens=data.get("max_tokens", 100),
            temperature=data.get("temperature", 1.0),
            top_p=data.get("top_p", 1.0),
            stream=data.get("stream", False),
        )

        # Generate response
        if gen_config.stream:

            def generate():
                for chunk in model_pipeline.chat_completion(messages, gen_config):
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return Response(
                stream_with_context(generate()), mimetype="text/event-stream"
            )
        else:
            response = model_pipeline.chat_completion(messages, gen_config)
            return jsonify(response)

    except Exception as e:
        return (
            jsonify(
                {"error": {"message": str(e), "type": "internal_error", "code": 500}}
            ),
            500,
        )


@app.route("/v1/completions", methods=["POST"])
def completions():
    """
    OpenAI-compatible text completions endpoint.

    Implements the OpenAI Completions API for simple text generation
    without the chat format.

    Request Format (JSON):
        {
            "model": "gpt2",
            "prompt": "Once upon a time",
            "max_tokens": 100,
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 50
        }

    Response Format:
        {
            "id": "cmpl-...",
            "object": "text_completion",
            "created": 1234567890,
            "model": "gpt2",
            "choices": [{
                "text": " there was a...",
                "index": 0,
                "finish_reason": "stop"
            }]
        }

    Supported Parameters:
        - prompt: Input text
        - max_tokens: Maximum generation length
        - temperature: Sampling temperature
        - top_p: Nucleus sampling
        - top_k: Top-k filtering

    Called by:
        - OpenAI completion clients
        - Direct API integrations

    Calls:
        - InferencePipeline.generate()
    """
    try:
        if not model_pipeline or not model_pipeline.model:
            return (
                jsonify(
                    {
                        "error": {
                            "message": "No model loaded. Please load a model first.",
                            "type": "model_not_loaded",
                            "code": 503,
                        }
                    }
                ),
                503,
            )

        data = request.json
        prompt = data.get("prompt", "")

        # Create generation config
        gen_config = GenerationConfig(
            max_new_tokens=data.get("max_tokens", 100),
            temperature=data.get("temperature", 1.0),
            top_p=data.get("top_p", 1.0),
            top_k=data.get("top_k"),
            stream=data.get("stream", False),
        )

        # Generate
        text = model_pipeline.generate(prompt, gen_config)

        return jsonify(
            {
                "id": f"cmpl-{uuid.uuid4().hex[:8]}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": current_model_info.get("model_type", "custom"),
                "choices": [
                    {
                        "text": text,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": "stop",
                    }
                ],
            }
        )

    except Exception as e:
        return (
            jsonify(
                {"error": {"message": str(e), "type": "internal_error", "code": 500}}
            ),
            500,
        )


@app.route("/v1/models", methods=["GET"])
def list_models():
    """
    List available models (OpenAI-compatible).

    Returns a list of models that can be used with the API,
    including the currently loaded model and available HuggingFace models.

    Response Format:
        {
            "object": "list",
            "data": [
                {
                    "id": "gpt2",
                    "object": "model",
                    "created": 1234567890,
                    "owned_by": "huggingface"
                }
            ]
        }

    Model Sources:
        - Currently loaded model (if any)
        - HuggingFace models: gpt2, gpt2-medium, gpt2-large, gpt2-xl

    Called by:
        - Model selection UIs
        - API discovery tools
    """
    models = [
        {
            "id": current_model_info.get("model_type", "custom"),
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local",
        }
    ]

    # Add available HuggingFace models
    for model in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]:
        models.append(
            {
                "id": model,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "huggingface",
            }
        )

    return jsonify({"object": "list", "data": models})


# Web UI Routes
@app.route("/")
def index():
    """
    Serve the main web interface.

    Returns the HTML template for the web UI which provides:
    - Model loading interface
    - Text generation playground
    - Training management dashboard
    - Model information display

    Template: templates/index.html

    The web interface uses JavaScript to interact with the
    API endpoints for dynamic functionality.
    """
    return render_template("index.html")


@app.route("/api/model/load", methods=["POST"])
def load_model():
    """
    Load a new model dynamically without server restart.

    Supports multiple input methods for maximum flexibility:
    1. JSON with paths/URLs
    2. Form data with file uploads
    3. Mixed mode with config files

    Input Methods:
        - model_type: HuggingFace model name
        - checkpoint_path: Local path or URL to checkpoint
        - checkpoint file upload: Direct .pt file upload
        - config_path: Path or URL to config file
        - config file upload: Direct config upload
        - config data: Inline configuration JSON

    Request Formats:
        1. JSON:
            {
                "model_type": "gpt2-medium",
                "checkpoint_path": "https://example.com/model.pt",
                "config": {"n_layer": 24, ...}
            }

        2. Form data with files:
            - model_type: "custom"
            - checkpoint: <file upload>
            - config: <file upload>

    Response:
        {
            "success": true,
            "model_info": {
                "num_parameters": 124439808,
                "architecture": {...},
                ...
            }
        }

    Features:
        - URL downloading with progress
        - Temporary file cleanup
        - Config format auto-detection (JSON/YAML)
        - Error recovery

    Called by:
        - Web UI model loader
        - Programmatic model switching

    Calls:
        - InferencePipeline.load_model()
        - ModelLoader.get_model_info()
    """
    global model_pipeline, current_model_info

    try:
        # Handle both JSON and form data with files
        if request.is_json:
            data = request.json
            model_type = data.get("model_type")
            checkpoint_path = data.get("checkpoint_path")
            config_path = data.get("config_path")
            config_data = data.get("config")
        else:
            model_type = request.form.get("model_type")
            checkpoint_path = request.form.get("checkpoint_path")
            config_path = request.form.get("config_path")
            config_data = None

            # Check for uploaded files
            if "checkpoint" in request.files:
                checkpoint_file = request.files["checkpoint"]
                if checkpoint_file.filename:
                    filename = secure_filename(checkpoint_file.filename)
                    upload_dir = Path("uploads")
                    upload_dir.mkdir(exist_ok=True)
                    checkpoint_path = upload_dir / filename
                    checkpoint_file.save(checkpoint_path)
                    checkpoint_path = str(checkpoint_path)

            if "config" in request.files:
                config_file = request.files["config"]
                if config_file.filename:
                    # Read config file content
                    config_content = config_file.read().decode("utf-8")
                    if config_file.filename.endswith(".json"):
                        config_data = json.loads(config_content)
                    elif config_file.filename.endswith((".yaml", ".yml")):
                        import yaml

                        config_data = yaml.safe_load(config_content)

        # Load config from path or data
        model_config = None
        if config_path and config_path.startswith(("http://", "https://")):
            # Download config from URL
            response = requests.get(config_path)
            response.raise_for_status()
            if config_path.endswith(".json"):
                config_data = response.json()
            else:
                import yaml

                config_data = yaml.safe_load(response.text)
        elif config_path and Path(config_path).exists():
            # Load local config file
            from nanoLLM_gpt.config import ConfigLoader, TrainingConfig

            training_config = ConfigLoader.load_from_file(config_path, TrainingConfig)
            model_config = training_config.model

        if config_data and "model" in config_data:
            # Extract model config from training config
            from nanoLLM_gpt.config import ModelConfig

            model_config = ModelConfig(**config_data["model"])

        # Determine if checkpoint is a URL
        if checkpoint_path and checkpoint_path.startswith(("http://", "https://")):
            # Download checkpoint to temporary file
            print(f"Downloading checkpoint from {checkpoint_path}")
            response = requests.get(checkpoint_path, stream=True)
            response.raise_for_status()

            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    tmp_file.write(chunk)
                checkpoint_path = tmp_file.name

        # Initialize pipeline if not exists
        if not model_pipeline:
            model_pipeline = InferencePipeline(
                device=config.device if config else "cuda",
                dtype=config.dtype if config else "auto",
            )

        # Load model
        print(f"Loading model: {model_type or checkpoint_path}")
        model_pipeline.load_model(
            checkpoint_path=checkpoint_path,
            model_type=model_type or "gpt2",
            compile=config.compile if config else False,
            model_config=model_config,
        )

        # Update model info
        if model_pipeline.model:
            from nanoLLM_gpt.utils.model_loader import ModelLoader

            info = ModelLoader.get_model_info(model_pipeline.model)
            current_model_info = {
                "model_type": model_type,
                "checkpoint_path": (
                    checkpoint_path
                    if not checkpoint_path or not checkpoint_path.startswith("/tmp")
                    else None
                ),
                "config": model_config.__dict__ if model_config else None,
                **info,
            }

        # Clean up temporary file if it was downloaded
        if checkpoint_path and checkpoint_path.startswith("/tmp"):
            try:
                os.unlink(checkpoint_path)
            except:
                pass

        return jsonify({"success": True, "model_info": current_model_info})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/generate", methods=["POST"])
def web_generate():
    """
    Generate text for web interface.

    Simplified generation endpoint for the web UI with
    user-friendly parameter names and response format.

    Request Format (JSON):
        {
            "prompt": "Write a story about",
            "max_tokens": 100,
            "temperature": 0.8,
            "top_k": 200,
            "top_p": 0.95
        }

    Response Format:
        {
            "success": true,
            "text": "Write a story about a brave knight..."
        }

    Error Response:
        {
            "success": false,
            "error": "Error message"
        }

    Note:
        Response includes the original prompt concatenated
        with generated text for complete display.

    Called by:
        - Web UI generate button
        - Interactive playground

    Calls:
        - InferencePipeline.generate()
    """
    try:
        if not model_pipeline or not model_pipeline.model:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "No model loaded. Please load a model first.",
                    }
                ),
                503,
            )

        data = request.json
        prompt = data.get("prompt", "")

        # Create generation config
        gen_config = GenerationConfig(
            max_new_tokens=data.get("max_tokens", 100),
            temperature=data.get("temperature", 0.8),
            top_k=data.get("top_k", 200),
            top_p=data.get("top_p", 1.0),
        )

        # Generate text
        text = model_pipeline.generate(prompt, gen_config)

        return jsonify(
            {"success": True, "text": prompt + text}  # Include prompt in response
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/training/start", methods=["POST"])
def start_training():
    """
    Start model training from web interface.

    Accepts training data via file upload or URL, along with
    training configuration. Launches training subprocess.

    Request Format (multipart/form-data):
        - file: Training data file upload (optional)
        - url: URL to training data (optional)
        - config: JSON string with training parameters

    Config Parameters:
        {
            "out_dir": "out",
            "batch_size": 12,
            "max_iters": 5000,
            "learning_rate": 0.0006,
            "eval_interval": 500,
            "n_layer": 12,
            "n_head": 12,
            "n_embd": 768,
            "block_size": 1024
        }

    Data Sources (mutually exclusive):
        1. File upload: Text file with training data
        2. URL: HTTP/HTTPS URL to download data from

    Response:
        {"success": true}  # If training started
        {"success": false, "error": "message"}  # If failed

    Side Effects:
        - Creates output directory
        - Saves config.yaml
        - Starts training subprocess
        - Begins log streaming

    Called by:
        - Web UI training panel

    Calls:
        - TrainingManager.start_training()
    """
    try:
        # Determine data source
        data_path = None

        # Check for file upload
        if "file" in request.files:
            file = request.files["file"]
            if file.filename:
                filename = secure_filename(file.filename)
                upload_dir = Path("uploads")
                upload_dir.mkdir(exist_ok=True)
                data_path = upload_dir / filename
                file.save(data_path)
                data_path = str(data_path)

        # Check for URL
        elif "url" in request.form:
            url = request.form["url"]
            if url:
                # Training script will handle URL downloading
                data_path = url

        if not data_path:
            return jsonify({"success": False, "error": "No data source provided"}), 400

        # Get training config
        train_config = json.loads(request.form.get("config", "{}"))

        # Start training
        success = training_manager.start_training(train_config, data_path)

        return jsonify({"success": success})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/training/stop", methods=["POST"])
def stop_training():
    """
    Stop current training process.

    Terminates the active training subprocess if running.
    Graceful shutdown with forced termination fallback.

    Response:
        {"success": true}  # If training was stopped
        {"success": false}  # If no training was active

    Called by:
        - Web UI stop button
        - Emergency shutdown procedures

    Calls:
        - TrainingManager.stop_training()
    """
    success = training_manager.stop_training()
    return jsonify({"success": success})


@app.route("/api/training/status", methods=["GET"])
def training_status():
    """
    Get current training status and recent logs.

    Returns training state and last 100 log entries for
    UI updates. Designed for periodic polling.

    Response Format:
        {
            "is_training": true,
            "config": {
                "batch_size": 12,
                "learning_rate": 0.0006,
                ...
            },
            "logs": [
                {
                    "timestamp": "2024-01-15T10:30:45",
                    "message": "iter 100: loss 4.5234"
                },
                ...
            ]
        }

    Polling:
        Web UI typically polls this every 1-2 seconds
        during active training.

    Called by:
        - Web UI status updater
        - Training monitors

    Calls:
        - TrainingManager.get_status()
    """
    return jsonify(training_manager.get_status())


@app.route("/api/model/info", methods=["GET"])
def model_info():
    """
    Get detailed information about the loaded model.

    Returns comprehensive model statistics and configuration
    for display in the UI.

    Response Format (model loaded):
        {
            "loaded": true,
            "model_type": "gpt2",
            "checkpoint_path": "out/ckpt.pt",
            "num_parameters": 124439808,
            "num_parameters_millions": 124.44,
            "architecture": {
                "n_layer": 12,
                "n_head": 12,
                "n_embd": 768,
                "block_size": 1024,
                "vocab_size": 50257
            },
            "device": "cuda:0",
            "dtype": "torch.float16"
        }

    Response Format (no model):
        {"loaded": false}

    Called by:
        - Web UI model info panel
        - Status checks
    """
    if model_pipeline and model_pipeline.model:
        return jsonify({"loaded": True, **current_model_info})
    else:
        return jsonify({"loaded": False})


@app.route("/api/model/config", methods=["GET"])
def get_model_config():
    """
    Get the current model configuration.

    Returns model architecture configuration in a format
    suitable for saving or loading elsewhere.

    Query Parameters:
        - download=true: Return as downloadable file

    Response Format:
        {
            "model": {
                "block_size": 1024,
                "vocab_size": 50257,
                "n_layer": 12,
                "n_head": 12,
                "n_embd": 768,
                "dropout": 0.0,
                "bias": true
            }
        }

    Download Mode:
        Returns same JSON with Content-Disposition header
        for saving as model_config.json

    Called by:
        - Config export button
        - Model analysis tools
    """
    if model_pipeline and model_pipeline.model:
        from nanoLLM_gpt.config import ModelConfig

        model_config = model_pipeline.model.config

        # Convert to dict
        config_dict = {
            "block_size": model_config.block_size,
            "vocab_size": model_config.vocab_size,
            "n_layer": model_config.n_layer,
            "n_head": model_config.n_head,
            "n_embd": model_config.n_embd,
            "dropout": model_config.dropout,
            "bias": model_config.bias,
        }

        # Return as JSON with option to download
        if request.args.get("download") == "true":
            return Response(
                json.dumps({"model": config_dict}, indent=2),
                mimetype="application/json",
                headers={
                    "Content-Disposition": "attachment; filename=model_config.json"
                },
            )
        else:
            return jsonify({"model": config_dict})
    else:
        return jsonify({"error": "No model loaded"}), 404


@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint for monitoring.

    Provides quick status check for load balancers,
    monitoring systems, and health checks.

    Response Format:
        {
            "status": "healthy",
            "model_loaded": true,
            "training_active": false
        }

    Status Codes:
        - 200: Server is healthy
        - Never returns unhealthy (would not respond)

    Use Cases:
        - Kubernetes liveness/readiness probes
        - Load balancer health checks
        - Monitoring dashboards

    Called by:
        - Infrastructure monitoring
        - Container orchestration
    """
    return jsonify(
        {
            "status": "healthy",
            "model_loaded": model_pipeline is not None
            and model_pipeline.model is not None,
            "training_active": (
                training_manager.is_training if training_manager else False
            ),
        }
    )


def initialize_server(args: APIConfig):
    """
    Initialize server components with configuration.

    Sets up all server components including model pipeline,
    training manager, and initial model loading.

    Args:
        args (APIConfig): Server configuration including:
            - device: Compute device
            - dtype: Model precision
            - model_type: Initial model to load
            - checkpoint_path: Initial checkpoint
            - compile: Whether to compile model

    Global Effects:
        - Initializes model_pipeline
        - Initializes training_manager
        - Sets config global
        - Updates current_model_info

    Error Handling:
        - Warns if initial model fails to load
        - Server continues without model
        - Model can be loaded via web UI

    Called by:
        - main() during server startup

    Calls:
        - InferencePipeline initialization
        - TrainingManager initialization
        - ModelLoader.load_model()
        - ModelLoader.get_model_info()
    """
    global model_pipeline, training_manager, config, current_model_info

    config = args

    # Initialize training manager
    training_manager = TrainingManager()

    # Initialize model pipeline
    print("Initializing model pipeline...")
    model_pipeline = InferencePipeline(device=args.device, dtype=args.dtype)

    # Load initial model if specified
    if args.checkpoint_path or args.model_type:
        print("Loading initial model...")
        try:
            model_pipeline.load_model(
                checkpoint_path=args.checkpoint_path,
                model_type=args.model_type,
                compile=args.compile,
            )

            # Set initial model info
            if model_pipeline.model:
                from nanoLLM_gpt.utils.model_loader import ModelLoader

                info = ModelLoader.get_model_info(model_pipeline.model)
                current_model_info = {
                    "model_type": args.model_type,
                    "checkpoint_path": args.checkpoint_path,
                    **info,
                }

            print("Model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load initial model: {e}")
            print("Server started without a model. Load one from the web interface.")
    else:
        print("Server started without a model. Load one from the web interface.")

    print("Server initialized successfully")


def main():
    """
    Main entry point for the server application.

    Handles command-line parsing, configuration loading,
    server initialization, and Flask app startup.

    Command-line Options:
        - --config: Path to configuration file
        - --host: Server host (default: 0.0.0.0)
        - --port: Server port (default: 8080)
        - --model-type: Initial model (default: gpt2)
        - --checkpoint-path: Initial checkpoint
        - --device: Compute device
        - --dtype: Model precision
        - --compile: Enable compilation
        - --debug: Flask debug mode

    Configuration Priority:
        1. Command-line arguments (highest)
        2. Configuration file
        3. Default values

    Startup Sequence:
        1. Parse arguments
        2. Load configuration
        3. Initialize server components
        4. Create template directory
        5. Start Flask server

    Output:
        Prints server URLs for API and web access

    Called by:
        - Command line: python server.py
        - Entry point: gpt-server
    """
    parser = argparse.ArgumentParser(
        description="Run GPT model server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Config file
    parser.add_argument("--config", type=str, help="Path to server configuration file")

    # Add API config arguments
    ConfigLoader.add_config_arguments(parser, APIConfig)

    args = parser.parse_args()

    # Load configuration
    if args.config:
        config = ConfigLoader.load_from_file(args.config, APIConfig)
    else:
        config = ConfigLoader.create_config_from_args(args, APIConfig)

    # Initialize server
    initialize_server(config)

    # Create templates directory if needed
    template_dir = Path(__file__).parent / "templates"
    template_dir.mkdir(exist_ok=True)

    # Run server
    print(f"Starting server on http://{config.host}:{config.port}")
    print(f"API endpoint: http://{config.host}:{config.port}/v1/chat/completions")
    print(f"Web interface: http://{config.host}:{config.port}/")

    app.run(host=config.host, port=config.port, debug=config.debug, threaded=True)


if __name__ == "__main__":
    main()
