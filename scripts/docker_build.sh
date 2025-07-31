#!/bin/bash

# Docker Build and Management Scripts for Kernel Fusion Library

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if NVIDIA Docker runtime is available
check_nvidia_docker() {
    if ! docker info | grep -q nvidia; then
        print_warning "NVIDIA Docker runtime not detected. GPU support may not be available."
        return 1
    fi
    print_success "NVIDIA Docker runtime detected"
    return 0
}

# Build development image
build_dev() {
    print_status "Building development image..."
    docker build -f Dockerfile.dev -t kernel-fusion:dev .
    print_success "Development image built successfully"
}

# Build production image
build_prod() {
    print_status "Building production image..."
    docker build -f Dockerfile.prod -t kernel-fusion:latest .
    print_success "Production image built successfully"
}

# Build all images
build_all() {
    print_status "Building all Docker images..."
    build_dev
    build_prod
    print_success "All images built successfully"
}

# Start development environment
start_dev() {
    print_status "Starting development environment..."
    docker-compose up -d kernel-fusion-dev
    print_success "Development environment started"
    print_status "Access with: docker exec -it kernel-fusion-dev bash"
}

# Start Jupyter notebook server
start_jupyter() {
    print_status "Starting Jupyter notebook server..."
    docker-compose up -d kernel-fusion-jupyter
    print_success "Jupyter server started"
    print_status "Access at: http://localhost:8888"
    print_status "Check logs for token: docker-compose logs kernel-fusion-jupyter"
}

# Run tests
run_tests() {
    print_status "Running tests..."
    docker-compose run --rm kernel-fusion-test
}

# Run benchmarks
run_benchmarks() {
    print_status "Running benchmarks..."
    docker-compose run --rm kernel-fusion-benchmark
}

# Clean up containers and images
cleanup() {
    print_status "Cleaning up Docker containers and images..."
    docker-compose down -v
    docker image prune -f
    print_success "Cleanup completed"
}

# Show help
show_help() {
    echo "Kernel Fusion Docker Management Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build-dev      Build development image"
    echo "  build-prod     Build production image"
    echo "  build-all      Build all images"
    echo "  start-dev      Start development environment"
    echo "  start-jupyter  Start Jupyter notebook server"
    echo "  test           Run tests"
    echo "  benchmark      Run benchmarks"
    echo "  cleanup        Clean up containers and images"
    echo "  help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build-all      # Build both dev and prod images"
    echo "  $0 start-dev      # Start development container"
    echo "  $0 test           # Run test suite"
}

# Main script logic
case "$1" in
    "build-dev")
        check_nvidia_docker
        build_dev
        ;;
    "build-prod")
        check_nvidia_docker
        build_prod
        ;;
    "build-all")
        check_nvidia_docker
        build_all
        ;;
    "start-dev")
        check_nvidia_docker
        start_dev
        ;;
    "start-jupyter")
        check_nvidia_docker
        start_jupyter
        ;;
    "test")
        check_nvidia_docker
        run_tests
        ;;
    "benchmark")
        check_nvidia_docker
        run_benchmarks
        ;;
    "cleanup")
        cleanup
        ;;
    "help"|"--help"|"-h"|"")
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
