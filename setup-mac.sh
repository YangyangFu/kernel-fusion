#!/bin/bash

# MacBook Pro Development Setup Script

set -e

echo "🍎 Setting up Kernel Fusion Development Environment for MacBook Pro"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker Desktop for Mac first."
    echo "   Download from: https://www.docker.com/products/docker-desktop"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "📦 Building CPU-optimized Docker image..."
docker-compose -f docker-compose.cpu.yml build

echo "✅ Setup complete!"
echo ""
echo "🎯 Usage (CPU Development):"
echo "  Development container:  docker-compose -f docker-compose.cpu.yml run --rm kernel-fusion-cpu"
echo "  Jupyter Lab:           docker-compose -f docker-compose.cpu.yml up jupyter-cpu"
echo "  Interactive shell:     docker-compose -f docker-compose.cpu.yml run --rm kernel-fusion-cpu bash"
echo ""
echo "📚 Jupyter will be available at: http://localhost:8888"
echo "📊 TensorBoard will be available at: http://localhost:6006"
echo ""
echo "💡 Development Options for MacBook Pro:"
echo "   1. CPU-only development (current setup) - Great for:"
echo "      • Algorithm prototyping"
echo "      • Testing kernel logic"
echo "      • Learning Triton syntax"
echo "      • CPU-optimized implementations"
echo ""
echo "   2. Cloud GPU development - Consider:"
echo "      • Google Colab Pro/Pro+"
echo "      • AWS EC2 with GPU instances"
echo "      • Paperspace Gradient"
echo "      • Lambda Labs"
echo ""
echo "   3. Apple Silicon optimization:"
echo "      • PyTorch MPS backend"
echo "      • Metal Performance Shaders"
echo "      • Apple MLX framework"
