# Use NVIDIA CUDA base image with Ubuntu 24.04 and CUDA 12.6
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    git \
    wget \
    curl \
    vim \
    build-essential \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 20 LTS from NodeSource
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3.12 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Upgrade pip in virtual environment
RUN pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /workspace/nanoLLM_gpt

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA support (latest stable version)
RUN pip install --no-cache-dir torch torchvision torchaudio

# Copy the entire project
COPY . .

# Install the package in development mode
RUN pip install -e .

# Install additional development dependencies
RUN pip install -e ".[dev,datasets,wandb]"

# Install Claude Code via npm
RUN npm install -g @anthropic-ai/claude-code

# Create necessary directories
RUN mkdir -p /workspace/nanoLLM_gpt/data \
    /workspace/nanoLLM_gpt/out \
    /workspace/nanoLLM_gpt/logs \
    /workspace/nanoLLM_gpt/models \
    /workspace/nanoLLM_gpt/uploads

# Set up proper permissions
RUN chmod -R 755 /workspace/nanoLLM_gpt

# Create bashrc to ensure venv is activated
RUN echo 'export VIRTUAL_ENV=/opt/venv' >> ~/.bashrc && \
    echo 'export PATH="$VIRTUAL_ENV/bin:$PATH"' >> ~/.bashrc && \
    echo 'export PS1="(venv) \u@\h:\w\$ "' >> ~/.bashrc

# Expose ports for the web server
EXPOSE 8080

# Set up volume mounts for persistent data
VOLUME ["/workspace/nanoLLM_gpt/data", "/workspace/nanoLLM_gpt/out", "/workspace/nanoLLM_gpt/models", "/workspace/nanoLLM_gpt/uploads"]

# Add a script to check GPU availability
RUN echo '#!/bin/bash\n\
echo "=== Python Environment ==="\n\
echo "Virtual environment: $VIRTUAL_ENV"\n\
echo "Python path: $(which python)"\n\
echo "Python version: $(python --version)"\n\
echo "\n=== GPU Information ==="\n\
nvidia-smi\n\
echo "\n=== PyTorch GPU Check ==="\n\
python -c "import torch; print(\"PyTorch version: \", torch.__version__); print(\"CUDA available: \", torch.cuda.is_available()); print(\"CUDA device count: \", torch.cuda.device_count()); print(\"Current CUDA device: \", torch.cuda.current_device() if torch.cuda.is_available() else \"N/A\")"\n\
echo "\n=== Claude Code Version ==="\n\
claude --version || echo "Claude Code not configured. Run: claude setup"\n\
echo "\n=== nanoLLM_gpt Installation ==="\n\
python -c "import nanoLLM_gpt; print(\"nanoLLM_gpt successfully imported\")"' > /usr/local/bin/check-environment && \
    chmod +x /usr/local/bin/check-environment

# Default command - start bash with venv activated
CMD ["/bin/bash", "-c", "source ~/.bashrc && check-environment && echo -e '\n=== Available Commands ===\n\
gpt-train: Train a GPT model\n\
gpt-generate: Generate text using a trained model\n\
gpt-server: Start the web server\n\
claude: Use Claude Code CLI (requires setup)\n\
\nExample usage:\n\
  docker run --gpus all -it nanollm-gpt gpt-train --help\n\
  docker run --gpus all -p 8080:8080 nanollm-gpt gpt-server\n\n' && exec bash"]