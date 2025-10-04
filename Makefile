# Makefile for ExoHunter project
# Provides standardized commands for development and CI/CD

.PHONY: help install install-dev test test-verbose test-coverage lint format clean build docs

# Default target
help:
	@echo "ExoHunter Project Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  install      - Install production dependencies"
	@echo "  install-dev  - Install development dependencies"
	@echo "  test         - Run test suite with pytest"
	@echo "  test-verbose - Run tests with verbose output"
	@echo "  test-coverage- Run tests with coverage report"
	@echo "  lint         - Run code linting (if available)"
	@echo "  format       - Format code (if available)"
	@echo "  clean        - Clean temporary files and caches"
	@echo "  build        - Build the project"
	@echo "  docs         - Generate documentation"
	@echo ""

# Install production dependencies
install:
	@echo "Installing production dependencies..."
	pip install -r requirements.txt

# Install development dependencies (includes test dependencies)
install-dev: install
	@echo "Installing development dependencies..."
	pip install -e .
	@echo "Development setup complete!"

# Run test suite
test:
	@echo "Running test suite..."
	python -m pytest tests/ -v --tb=short

# Run tests with verbose output
test-verbose:
	@echo "Running tests with verbose output..."
	python -m pytest tests/ -v -s --tb=long

# Run tests with coverage report
test-coverage:
	@echo "Running tests with coverage..."
	python -m pytest tests/ --cov=. --cov-report=term-missing --cov-report=html

# Lint code (using flake8 if available)
lint:
	@echo "Running code linting..."
	@if command -v flake8 >/dev/null 2>&1; then \
		flake8 --max-line-length=100 --ignore=E501,W503 .; \
	else \
		echo "flake8 not installed, skipping lint check"; \
	fi

# Format code (using black if available)
format:
	@echo "Formatting code..."
	@if command -v black >/dev/null 2>&1; then \
		black --line-length=100 .; \
	else \
		echo "black not installed, skipping code formatting"; \
	fi

# Clean temporary files and caches
clean:
	@echo "Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type f -name ".coverage.*" -delete
	@echo "Clean complete!"

# Build the project
build: clean
	@echo "Building project..."
	python setup.py build
	@echo "Build complete!"

# Generate documentation
docs:
	@echo "Generating documentation..."
	@if command -v sphinx-build >/dev/null 2>&1; then \
		sphinx-build -b html docs/ docs/_build/html; \
	else \
		echo "Sphinx not installed, skipping documentation generation"; \
	fi

# Quick development setup
dev-setup: install-dev
	@echo "Development environment ready!"
	@echo "Run 'make test' to run the test suite"

# CI target - what CI/CD should run
ci: install test
	@echo "CI tasks completed successfully!"

# Run specific test modules
test-ingest:
	python -m pytest tests/test_ingest.py -v

test-labels:
	python -m pytest tests/test_labels.py -v

test-preprocessing:
	python -m pytest tests/test_preprocessing.py -v

test-train:
	python -m pytest tests/test_train.py -v

test-api:
	python -m pytest tests/test_api.py -v

# Test with specific markers (if using pytest markers)
test-unit:
	python -m pytest tests/ -m "not integration" -v

test-integration:
	python -m pytest tests/ -m "integration" -v

# Run tests in parallel (if pytest-xdist is available)
test-parallel:
	@if python -c "import pytest_xdist" 2>/dev/null; then \
		python -m pytest tests/ -n auto; \
	else \
		echo "pytest-xdist not installed, running tests sequentially"; \
		make test; \
	fi
