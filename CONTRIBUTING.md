# Contributing to ExoHunter

Thank you for your interest in contributing to ExoHunter! This document provides guidelines and best practices for contributing to the project.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Branching Model](#branching-model)
- [Commit Message Conventions](#commit-message-conventions)
- [Code Style and Formatting](#code-style-and-formatting)
- [Testing Requirements](#testing-requirements)
- [Pull Request Process](#pull-request-process)
- [Review Checklist](#review-checklist)
- [Release Process](#release-process)
- [Getting Help](#getting-help)

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Node.js 18+ and npm (for frontend development)
- Docker (for containerized development)
- Git

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/exohunter.git
   cd exohunter
   ```

2. **Set up Python Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

3. **Install Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

4. **Set up Frontend** (if contributing to UI)
   ```bash
   cd web/frontend
   npm install
   cd ../..
   ```

5. **Run Tests**
   ```bash
   pytest
   ```

## Branching Model

We use a simplified Git flow with the following branch structure:

### Main Branches

- **`main`** - Production-ready code, protected branch
- **`develop`** - Integration branch for features (optional for smaller teams)

### Feature Branches

- **`feature/feature-name`** - New features
- **`bugfix/issue-description`** - Bug fixes
- **`hotfix/critical-fix`** - Critical production fixes
- **`docs/documentation-update`** - Documentation improvements
- **`chore/maintenance-task`** - Maintenance and tooling

### Branch Naming Conventions

```bash
# Features
feature/stellar-classification
feature/data-preprocessing-pipeline

# Bug fixes
bugfix/transit-detection-accuracy
bugfix/api-validation-error

# Documentation
docs/api-documentation
docs/user-guide-update

# Maintenance
chore/update-dependencies
chore/ci-pipeline-optimization
```

### Workflow

1. **Create Branch**
   ```bash
   git checkout main
   git pull origin main
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes and Commit**
   ```bash
   git add .
   git commit -m "feat(scope): add stellar classification algorithm"
   ```

3. **Keep Updated**
   ```bash
   git fetch origin
   git rebase origin/main
   ```

4. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   # Create pull request via GitHub/GitLab interface
   ```

## Commit Message Conventions

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification for consistent and semantic commit messages.

### Format

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Types

- **`feat`** - New features
- **`fix`** - Bug fixes
- **`docs`** - Documentation changes
- **`style`** - Code style changes (formatting, missing semi-colons, etc.)
- **`refactor`** - Code refactoring without changing functionality
- **`perf`** - Performance improvements
- **`test`** - Adding or updating tests
- **`chore`** - Maintenance tasks, dependency updates
- **`ci`** - CI/CD configuration changes
- **`build`** - Build system or external dependency changes

### Scopes

Common scopes for ExoHunter:

- **`api`** - Backend API changes
- **`frontend`** - React frontend changes
- **`models`** - Machine learning models
- **`data`** - Data processing and pipelines
- **`docker`** - Container configuration
- **`docs`** - Documentation
- **`tests`** - Test-related changes
- **`release`** - Release process changes

### Examples

```bash
# Feature additions
feat(models): add random forest classifier for exoplanet detection
feat(api): implement transit data endpoint with pagination
feat(frontend): add interactive star chart visualization

# Bug fixes
fix(data): resolve CSV parsing error for large datasets
fix(api): handle edge case in orbital period calculation
fix(frontend): correct responsive layout on mobile devices

# Documentation
docs(api): update OpenAPI schema for new endpoints
docs: add machine learning model documentation
docs(contributing): update development setup instructions

# Refactoring
refactor(models): optimize feature extraction pipeline
refactor(api): simplify database query logic

# Testing
test(models): add unit tests for classification accuracy
test(api): increase coverage for edge cases

# Chores and maintenance
chore(deps): update scikit-learn to version 1.3.0
chore(release): prepare version 1.2.0
ci: optimize Docker build caching
```

### Breaking Changes

For breaking changes, add `BREAKING CHANGE:` in the footer:

```
feat(api): redesign authentication system

BREAKING CHANGE: API endpoints now require JWT tokens instead of API keys.
Migration guide available in docs/MIGRATION.md
```

## Code Style and Formatting

### Python Code Style

We use **Black** for code formatting and **isort** for import sorting.

#### Black Configuration

- Line length: 88 characters (Black default)
- Configuration in `pyproject.toml`:

```toml
[tool.black]
line-length = 88
target-version = ['py311']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.venv
  | venv
  | build
  | dist
)/
'''
```

#### isort Configuration

```toml
[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88
known_first_party = ["exohunter"]
known_third_party = ["fastapi", "sklearn", "pandas", "numpy"]
```

#### Pre-commit Setup

Install and use pre-commit hooks:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

#### Manual Formatting

```bash
# Format with Black
black .

# Sort imports with isort
isort .

# Check formatting
black --check .
isort --check-only .
```

### JavaScript/TypeScript (Frontend)

- **Prettier** for formatting
- **ESLint** for linting
- Configuration in `web/frontend/.prettierrc` and `.eslintrc.js`

```bash
# Format frontend code
cd web/frontend
npm run format
npm run lint
```

### Code Quality Tools

#### Linting

```bash
# Python linting with flake8
flake8 .

# Type checking with mypy
mypy .
```

#### Configuration

Add these tools to your development dependencies and configure in `pyproject.toml`:

```toml
[tool.flake8]
max-line-length = 88
extend-ignore = ["E203", "W503"]
exclude = [".git", "__pycache__", "venv", "build", "dist"]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
```

## Testing Requirements

### Coverage Target

- **Minimum coverage**: 85% overall
- **Critical components**: 95% coverage (models, API endpoints)
- **New code**: Must maintain or improve coverage

### Test Structure

```
tests/
├── unit/                    # Unit tests
│   ├── test_models/        # Model testing
│   ├── test_api/           # API testing
│   └── test_utils/         # Utility function tests
├── integration/            # Integration tests
├── performance/            # Performance benchmarks
└── test_model_performance.py # Model smoke tests
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=exohunter --cov-report=html --cov-report=term

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest -m "not slow"  # Skip slow tests

# Run performance tests
pytest tests/test_model_performance.py -v
```

### Test Requirements

1. **Unit Tests**: Required for all new functions and classes
2. **Integration Tests**: Required for API endpoints and data pipelines
3. **Performance Tests**: Required for model changes
4. **Documentation Tests**: Ensure examples in docs work

### Test Guidelines

- Use descriptive test names: `test_classify_exoplanet_with_valid_features()`
- Test edge cases and error conditions
- Mock external dependencies
- Use pytest fixtures for common setup
- Add performance benchmarks for critical paths

Example test structure:

```python
import pytest
from exohunter.models import ExoplanetClassifier

class TestExoplanetClassifier:
    @pytest.fixture
    def classifier(self):
        return ExoplanetClassifier()
    
    def test_classify_with_valid_features(self, classifier):
        """Test classification with valid input features."""
        features = [1.2, 0.8, 365.25, 0.05]  # Example features
        result = classifier.predict(features)
        assert result in [0, 1]  # Binary classification
    
    def test_classify_with_invalid_features_raises_error(self, classifier):
        """Test that invalid features raise appropriate error."""
        with pytest.raises(ValueError, match="Invalid feature dimensions"):
            classifier.predict([1, 2])  # Wrong number of features
```

## Pull Request Process

### Before Opening a PR

1. **Ensure your branch is up to date**
   ```bash
   git fetch origin
   git rebase origin/main
   ```

2. **Run full test suite**
   ```bash
   pytest --cov=exohunter
   ```

3. **Format code**
   ```bash
   black .
   isort .
   ```

4. **Update documentation if needed**

### PR Template

When creating a PR, include:

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature causing existing functionality to change)
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added tests for new functionality
- [ ] Coverage maintained/improved

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Example run passes
- [ ] README updated if needed
```

### PR Size Guidelines

- **Small PRs preferred**: < 400 lines changed
- **Large PRs**: > 800 lines should be split or have detailed justification
- **Focus**: One feature/fix per PR

## Review Checklist

### For Authors

Before requesting review, ensure:

- [ ] **Code Quality**
  - [ ] Code follows Black + isort formatting
  - [ ] No linting errors (flake8, mypy)
  - [ ] Functions and classes have docstrings
  - [ ] Variable names are descriptive

- [ ] **Testing**
  - [ ] Unit tests added for new functionality
  - [ ] Integration tests updated if needed
  - [ ] Performance tests pass
  - [ ] Coverage target met (85%+ overall)
  - [ ] All tests pass locally

- [ ] **Documentation**
  - [ ] README updated if user-facing changes
  - [ ] API documentation updated
  - [ ] Code comments added for complex logic
  - [ ] Docstrings follow Google/NumPy style

- [ ] **Functionality**
  - [ ] Example run passes (e.g., `python examples/basic_classification.py`)
  - [ ] Feature works as described
  - [ ] Edge cases handled
  - [ ] Error messages are helpful

- [ ] **Dependencies**
  - [ ] New dependencies justified and documented
  - [ ] `requirements.txt` updated if needed
  - [ ] Docker builds successfully

### For Reviewers

When reviewing, check:

- [ ] **Code Review**
  - [ ] Logic is correct and efficient
  - [ ] Security considerations addressed
  - [ ] Error handling is appropriate
  - [ ] Code is readable and maintainable

- [ ] **Design Review**
  - [ ] Architecture decisions are sound
  - [ ] Follows project patterns
  - [ ] API design is consistent
  - [ ] Database changes are backward compatible

- [ ] **Testing Review**
  - [ ] Tests cover the functionality
  - [ ] Tests are reliable and not flaky
  - [ ] Performance impact assessed
  - [ ] Integration points tested

### Review Process

1. **Automated Checks**: CI/CD pipeline runs tests and checks
2. **Code Review**: At least one approval required
3. **Manual Testing**: Reviewer tests functionality locally (for significant changes)
4. **Documentation Review**: Ensure documentation is clear and complete

## Release Process

### Version Management

We use semantic versioning (SemVer): `MAJOR.MINOR.PATCH`

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Steps

1. **Prepare Release**
   ```bash
   # Update version
   ./scripts/version.sh bump minor
   
   # Generate changelog
   ./scripts/generate_changelog.sh --append
   
   # Commit changes
   git add .
   git commit -m "chore(release): prepare version 1.2.0"
   ```

2. **Build Release**
   ```bash
   ./scripts/build_release.sh
   ```

3. **Tag and Push**
   ```bash
   VERSION=$(./scripts/version.sh current)
   git tag "v$VERSION"
   git push origin main
   git push origin "v$VERSION"
   ```

## Getting Help

### Communication Channels

- **Issues**: GitHub Issues for bugs and feature requests
- **Discussions**: GitHub Discussions for questions and ideas
- **Email**: [maintainer@exohunter.io](mailto:maintainer@exohunter.io) for private matters

### Documentation

- **User Guide**: `docs/USER_GUIDE.md`
- **Developer Guide**: `docs/DEVELOPER_GUIDE.md`
- **API Documentation**: Auto-generated from FastAPI
- **Scripts Documentation**: `scripts/README.md`

### Common Issues

**Setup Problems**
```bash
# Clean installation
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Test Failures**
```bash
# Check test environment
pytest --collect-only
python -m pytest tests/ -v

# Run specific failing test
pytest tests/test_specific.py::test_function -v
```

**Formatting Issues**
```bash
# Auto-fix formatting
black .
isort .
pre-commit run --all-files
```

## Code of Conduct

This project follows the [Contributor Covenant](https://www.contributor-covenant.org/) Code of Conduct. By participating, you are expected to uphold this code.

### Our Standards

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

---

Thank you for contributing to ExoHunter! Your efforts help advance exoplanet research and make space science more accessible. 🚀🪐
