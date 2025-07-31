# Docker Environment Setup for Kernel Fusion Library

This directory contains Docker configurations for developing and deploying the Kernel Fusion library.

## Files Overview

- `Dockerfile.dev` - Development environment with full toolchain
- `Dockerfile.prod` - Production-ready minimal runtime image
- `docker-compose.yml` - Multi-service orchestration for different use cases
- `scripts/docker_build.sh` - Linux/Mac build script
- `scripts/docker_build.bat` - Windows build script

## Prerequisites

1. **Docker** - Install Docker Desktop or Docker Engine
2. **NVIDIA Container Toolkit** - Required for GPU support:
   ```bash
   # Ubuntu/Debian
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

## Quick Start

### Using Build Scripts (Recommended)

**Linux/Mac:**
```bash
# Make script executable
chmod +x scripts/docker_build.sh

# Build all images
./scripts/docker_build.sh build-all

# Start development environment
./scripts/docker_build.sh start-dev

# Run tests
./scripts/docker_build.sh test
```

**Windows:**
```cmd
# Build all images
scripts\docker_build.bat build-all

# Start development environment
scripts\docker_build.bat start-dev

# Run tests
scripts\docker_build.bat test
```

### Using Docker Compose Directly

```bash
# Build and start development environment
docker-compose up -d kernel-fusion-dev

# Access development container
docker exec -it kernel-fusion-dev bash

# Run tests
docker-compose run --rm kernel-fusion-test

# Start Jupyter notebook server
docker-compose up -d kernel-fusion-jupyter
# Access at http://localhost:8888

# Run benchmarks
docker-compose run --rm kernel-fusion-benchmark
```

## Available Services

### 1. Development Environment (`kernel-fusion-dev`)
- Full development toolchain with CUDA, CMake, and build tools
- **Live source code mounting** for immediate code changes
- Interactive shell access with helpful aliases
- Persistent caches for pip, CMake, and pytest
- Development environment variables (PYTHONPATH, CUDA_LAUNCH_BLOCKING)

### 2. Production Environment (`kernel-fusion-prod`)
- Minimal runtime image with only necessary dependencies
- Multi-stage build for optimized size
- Pre-built kernel-fusion package

### 3. Jupyter Server (`kernel-fusion-jupyter`)
- Interactive development with Jupyter notebooks
- Port 8888 exposed for browser access
- **No token/password required** for development convenience
- Includes visualization and analysis tools
- Auto-installs package in development mode

### 4. Testing Environment (`kernel-fusion-test`)
- Automated test execution
- Runs full test suite with pytest
- GPU and CPU test coverage

### 5. Benchmarking Environment (`kernel-fusion-benchmark`)
- Performance benchmarking and profiling
- Results saved to persistent volume
- Comparison with baseline implementations

## Environment Variables

Key environment variables used in containers:

- `CUDA_VISIBLE_DEVICES` - Controls GPU visibility
- `NVIDIA_VISIBLE_DEVICES` - NVIDIA runtime GPU control
- `NVIDIA_DRIVER_CAPABILITIES` - Driver capabilities (compute,utility)
- `CUDA_HOME` - CUDA installation path
- `PYTHONUNBUFFERED` - Unbuffered Python output

## Volumes

Persistent volumes for data and caching:

- `pip-cache` - Python package cache for faster installs
- `cmake-cache` - CMake build cache for faster compilation
- `pytest-cache` - Pytest cache for faster test runs
- `jupyter-data` - Jupyter configuration and notebooks
- `benchmark-results` - Benchmark output data

**Note:** Source code is mounted as a volume (`./:/workspace`) for live development.

## Build Scripts Commands

Both `docker_build.sh` (Linux/Mac) and `docker_build.bat` (Windows) support:

- `build-dev` - Build development image
- `build-prod` - Build production image
- `build-all` - Build both images
- `start-dev` - Start development environment
- `start-jupyter` - Start Jupyter server
- `test` - Run test suite
- `benchmark` - Run benchmarks
- `cleanup` - Clean up containers and images
- `help` - Show help message

## Development Workflow

1. **Setup:**
   ```bash
   # Clone repository and navigate to project
   git clone <repository>
   cd kernel-fusion
   
   # Build development image (only needed once or when Dockerfile changes)
   ./scripts/docker_build.sh build-dev
   ```

2. **Start Development Environment:**
   ```bash
   # Start development container with mounted source code
   ./scripts/docker_build.sh start-dev
   
   # Access container
   docker exec -it kernel-fusion-dev bash
   
   # Inside container - install in development mode
   setup-dev-env
   # Or manually: pip install -e .
   ```

3. **Live Development:**
   ```bash
   # Edit code on your host machine using any editor
   # Changes are immediately reflected in the container
   
   # Inside container - test your changes
   test-kernels
   
   # Rebuild only when needed (C++/CUDA changes)
   build-kernels
   
   # Run specific tests
   pytest tests/test_specific.py -v
   ```

4. **Testing and Benchmarking:**
   ```bash
   # Run full test suite
   ./scripts/docker_build.sh test
   
   # Run benchmarks
   ./scripts/docker_build.sh benchmark
   
   # Or manually in dev container
   python examples/benchmark_fusion.py
   ```

5. **Jupyter Development:**
   ```bash
   # Start Jupyter server
   ./scripts/docker_build.sh start-jupyter
   
   # Access at http://localhost:8888 (no password required)
   # Create notebooks in the mounted workspace
   ```

## Production Deployment

1. **Build production image:**
   ```bash
   ./scripts/docker_build.sh build-prod
   ```

2. **Run production container:**
   ```bash
   docker run --rm --gpus all kernel-fusion:latest python -c "import kernel_fusion; print('Ready!')"
   ```

3. **Push to registry:**
   ```bash
   docker tag kernel-fusion:latest your-registry/kernel-fusion:latest
   docker push your-registry/kernel-fusion:latest
   ```

## Troubleshooting

### GPU Not Available
- Ensure NVIDIA drivers are installed on host
- Verify NVIDIA Container Toolkit installation
- Check Docker runtime configuration: `docker info | grep nvidia`

### Build Failures
- Check CUDA compatibility (requires CUDA 12.6+)
- Verify sufficient disk space for build
- Check Docker daemon logs for errors

### Permission Issues
- On Linux, ensure user is in docker group: `sudo usermod -aG docker $USER`
- Restart shell session after group changes

### Memory Issues
- Increase Docker memory limit in Docker Desktop settings
- Use `--shm-size` for large tensor operations

## Key Development Features

### **🔄 Live Code Editing**
- Source code is mounted as a volume (`./:/workspace`)
- Edit files on your host machine with any editor/IDE
- Changes are **immediately reflected** in the container
- No need to rebuild Docker image for code changes
- Git operations work seamlessly on the host

### **⚡ Performance Optimizations**
- Persistent pip cache speeds up package installations
- CMake build cache reduces compilation time
- Pytest cache improves test performance
- Jupyter configuration preserved between runs

### **🛠️ Development Conveniences**
- Helpful bash aliases: `install-dev`, `test-kernels`, `benchmark`, `gpu-info`
- Auto-setup script: `setup-dev-env` installs package in development mode
- Jupyter server with no authentication for quick prototyping
- Development environment variables pre-configured

### **📁 Directory Structure in Container**
```
/workspace/               # Your mounted source code
├── kernel_fusion/        # Python package
├── src/                  # C++/CUDA source
├── tests/                # Test files
├── examples/             # Examples and benchmarks
├── setup.py              # Build configuration
└── ...
```

## Performance Notes

- **Development builds** include debug symbols and development tools
- **Production builds** are optimized for size and runtime performance
- **GPU memory** is shared between host and containers
- **Build cache** significantly speeds up subsequent builds
- **Volume mounting** has minimal performance overhead compared to copying

## Security Considerations

- Production images run as non-root user
- Development images may require root for package installation
- Limit GPU access with `CUDA_VISIBLE_DEVICES` if needed
- Use secrets for any authentication tokens or keys
- Jupyter server has no authentication in development mode (use only locally)
