.PHONY: help clean build install test format lint type-check docs benchmark

# Default target
help:
	@echo "Available targets:"
	@echo "  clean      - Remove build artifacts"
	@echo "  build      - Build the extension in-place"
	@echo "  install    - Install in development mode"
	@echo "  test       - Run tests"
	@echo "  test-cuda  - Run CUDA-specific tests"
	@echo "  format     - Format code with black"
	@echo "  lint       - Run flake8 linting"
	@echo "  type-check - Run mypy type checking"
	@echo "  benchmark  - Run performance benchmarks"
	@echo "  docs       - Build documentation"

# Clean build artifacts
clean:
	rm -rf build/ dist/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name "*.so" -delete

# Build extension in-place
build:
	python setup.py build_ext --inplace

# Install in development mode
install:
	pip install -e ".[dev]"

# Run tests
test:
	pytest tests/ -v

# Run CUDA-specific tests
test-cuda:
	pytest tests/ -v -m cuda

# Run fast tests only
test-fast:
	pytest tests/ -v -m "not slow"

# Format code
format:
	black kernel_fusion/ tests/ examples/
	isort kernel_fusion/ tests/ examples/

# Lint code
lint:
	flake8 kernel_fusion/ tests/ examples/

# Type checking
type-check:
	mypy kernel_fusion/

# Run all quality checks
check: format lint type-check test-fast

# Run performance benchmarks
benchmark:
	python examples/benchmarks.py

# Build documentation
docs:
	cd docs && make html

# Package for distribution
package: clean
	python setup.py sdist bdist_wheel

# Upload to PyPI (test)
upload-test: package
	twine upload --repository testpypi dist/*

# Upload to PyPI (production)
upload: package
	twine upload dist/*
