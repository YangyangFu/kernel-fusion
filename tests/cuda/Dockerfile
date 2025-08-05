# Use NVIDIA CUDA development image
FROM nvidia/cuda:11.8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /workspace

# Copy the kernel fusion project
COPY . /workspace/kernel-fusion/

# Set working directory to tests
WORKDIR /workspace/kernel-fusion/tests/cuda

# Make build script executable
RUN chmod +x run_tests.sh

# Default command
CMD ["./run_tests.sh"]
