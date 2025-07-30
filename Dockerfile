# Multi-stage Docker build for fusion kernel development
# Base image with CUDA support
FROM nvidia/cuda:12.1-devel-ubuntu22.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    vim \
    nano \
    htop \
    python3 \
    python3-pip \
    python3-dev \
    libblas-dev \
    liblapack-dev \
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install Python dependencies for deep learning and CUDA kernel development
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    numpy \
    scipy \
    matplotlib \
    seaborn \
    pandas \
    jupyter \
    jupyterlab \
    notebook \
    ipywidgets \
    tqdm \
    pytest \
    black \
    flake8 \
    mypy \
    pre-commit \
    tensorboard \
    wandb \
    pycuda \
    numba \
    cupy-cuda12x \
    cuda-python

# Install CUDA kernel development tools
RUN pip install --no-cache-dir \
    ninja \
    pybind11[global] \
    setuptools \
    wheel \
    packaging \
    cython \
    cffi

# Set working directory
WORKDIR /workspace

# Copy project files
COPY . /workspace/

# Create directories for development
RUN mkdir -p /workspace/kernels \
             /workspace/tests \
             /workspace/benchmarks \
             /workspace/examples \
             /workspace/notebooks

# Set up git configuration (will be overridden by user)
RUN git config --global user.name "Developer" && \
    git config --global user.email "developer@example.com"

# Expose ports for Jupyter and TensorBoard
EXPOSE 8888 6006

# Set up development environment
CMD ["/bin/bash"]