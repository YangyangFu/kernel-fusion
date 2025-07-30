# Makefile for Kernel Fusion Development

.PHONY: help build dev jupyter clean test lint format install build-mac dev-mac jupyter-mac

# Default target
help:
	@echo "Available targets:"
	@echo "  GPU Development (Linux/NVIDIA):"
	@echo "    build     - Build the Docker image"
	@echo "    dev       - Start development container"
	@echo "    jupyter   - Start Jupyter Lab server"
	@echo ""
	@echo "  CPU Development (MacBook Pro):"
	@echo "    build-mac - Build CPU-optimized Docker image"
	@echo "    dev-mac   - Start CPU development container"
	@echo "    jupyter-mac - Start CPU Jupyter Lab server"
	@echo ""
	@echo "  Common:"
	@echo "    test      - Run tests in container"
	@echo "    lint      - Run linting tools"
	@echo "    format    - Format code with black"
	@echo "    clean     - Clean up Docker resources"
	@echo "    install   - Install package in development mode"

# GPU Development (Linux/NVIDIA)
build:
	docker-compose build

dev:
	docker-compose run --rm kernel-fusion-dev

jupyter:
	docker-compose up jupyter

# CPU Development (MacBook Pro)
build-mac:
	docker-compose -f docker-compose.cpu.yml build

dev-mac:
	docker-compose -f docker-compose.cpu.yml run --rm kernel-fusion-cpu

jupyter-mac:
	docker-compose -f docker-compose.cpu.yml up jupyter-cpu

# Testing
test:
	@if [ -f docker-compose.cpu.yml ] && [ "$$(uname)" = "Darwin" ]; then \
		docker-compose -f docker-compose.cpu.yml run --rm kernel-fusion-cpu python -m pytest tests/ -v; \
	else \
		docker-compose run --rm kernel-fusion-dev python -m pytest tests/ -v; \
	fi

# Linting
lint:
	@if [ -f docker-compose.cpu.yml ] && [ "$$(uname)" = "Darwin" ]; then \
		docker-compose -f docker-compose.cpu.yml run --rm kernel-fusion-cpu flake8 kernel_fusion/ tests/; \
	else \
		docker-compose run --rm kernel-fusion-dev flake8 kernel_fusion/ tests/; \
	fi

# Formatting
format:
	@if [ -f docker-compose.cpu.yml ] && [ "$$(uname)" = "Darwin" ]; then \
		docker-compose -f docker-compose.cpu.yml run --rm kernel-fusion-cpu black kernel_fusion/ tests/ examples/; \
	else \
		docker-compose run --rm kernel-fusion-dev black kernel_fusion/ tests/ examples/; \
	fi

# Install package
install:
	@if [ -f docker-compose.cpu.yml ] && [ "$$(uname)" = "Darwin" ]; then \
		docker-compose -f docker-compose.cpu.yml run --rm kernel-fusion-cpu pip install -e .; \
	else \
		docker-compose run --rm kernel-fusion-dev pip install -e .; \
	fi

# Clean up
clean:
	docker-compose down --volumes --remove-orphans || true
	docker-compose -f docker-compose.cpu.yml down --volumes --remove-orphans || true
	docker system prune -f

# Setup development environment
setup:
	@if [ "$$(uname)" = "Darwin" ]; then \
		echo "Detected macOS - setting up CPU development environment"; \
		./setup-mac.sh; \
	else \
		echo "Setting up GPU development environment"; \
		./setup-dev.sh; \
	fi
