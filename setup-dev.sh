#!/bin/bash

# Kernel Fusion Development Environment Setup Script

set -e

echo "🚀 Setting up Kernel Fusion Development Environment"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if NVIDIA Docker runtime is available
if ! docker info | grep -q nvidia; then
    echo "⚠️  NVIDIA Docker runtime not detected. GPU support may not be available."
    echo "   Please install nvidia-docker2 for GPU support."
fi

echo "📦 Building Docker image..."
docker-compose build

echo "✅ Setup complete!"
echo ""
echo "🎯 Usage:"
echo "  Development container:  docker-compose run --rm kernel-fusion-dev"
echo "  Jupyter Lab:           docker-compose up jupyter"
echo "  Interactive shell:     docker-compose run --rm kernel-fusion-dev bash"
echo ""
echo "📚 Jupyter will be available at: http://localhost:8888"
echo "📊 TensorBoard will be available at: http://localhost:6006"
