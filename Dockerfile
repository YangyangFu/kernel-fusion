# Multi-stage Docker build for fusion kernel development
# Base image with CUDA support
FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-devel as base


# Install additional Python packages
RUN pip install --no-cache-dir \
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
    tensorboard \
    wandb \
    pycuda
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

# Note: Project files will be mounted via docker-compose volumes
# This allows real-time editing and prevents copying files into the image

# Create directories for development (in case they don't exist)
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