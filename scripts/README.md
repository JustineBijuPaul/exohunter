# ExoHunter Release Scripts

This directory contains automation scripts for building and releasing ExoHunter.

## Scripts Overview

### 🚀 `build_release.sh`
**Main release packaging script**

Creates complete release packages including Docker images, model artifacts, frontend builds, and documentation.

```bash
# Full release build
./scripts/build_release.sh

# Quick build without tests
./scripts/build_release.sh --skip-tests --clean

# Build and publish to registry  
./scripts/build_release.sh --publish --registry gcr.io/your-project
```

**Features:**
- Runs comprehensive test suite
- Builds Docker images for API and frontend
- Compiles frontend for production
- Collects model artifacts and training scripts
- Packages documentation
- Creates release archives with checksums
- Generates release notes and metadata

**Output:**
- `release/` directory with complete release package
- Docker images (`.tar.gz` archives)
- Frontend build artifacts
- Model and preprocessing artifacts
- Documentation bundle
- Compressed release archives

### 📝 `version.sh`
**Version management across project files**

Updates version consistently across all project files using semantic versioning.

```bash
# Show current version
./scripts/version.sh current

# Bump version
./scripts/version.sh bump patch    # 1.0.0 -> 1.0.1
./scripts/version.sh bump minor    # 1.0.0 -> 1.1.0  
./scripts/version.sh bump major    # 1.0.0 -> 2.0.0

# Set specific version
./scripts/version.sh set 1.2.3
```

**Updates:**
- `pyproject.toml` - Project version
- `web/api/main.py` - API version constant
- `web/frontend/package.json` - Frontend package version

### 📖 `generate_changelog.sh`
**Automated changelog generation from git commits**

Generates changelog from conventional commit messages with proper categorization.

```bash
# Generate changelog from last tag to HEAD
./scripts/generate_changelog.sh

# Generate from specific tag
./scripts/generate_changelog.sh --from v1.0.0

# Append to existing changelog
./scripts/generate_changelog.sh --append

# Preview changes
./scripts/generate_changelog.sh --dry-run
```

**Features:**
- Parses conventional commit format (`type(scope): description`)
- Categories: Features, Bug Fixes, Documentation, Tests, Chores
- Identifies breaking changes
- Generates markdown with proper formatting
- Supports appending to existing changelog

### 🧪 `test_release_system.sh`
**Test script for release infrastructure**

Validates that all release components are properly configured.

```bash
./scripts/test_release_system.sh
```

## Release Workflow

### 1. Prepare Release
```bash
# Update version
./scripts/version.sh bump minor

# Update changelog
./scripts/generate_changelog.sh --append

# Commit changes
git add .
git commit -m "chore(release): prepare version 1.1.0"
```

### 2. Build Release
```bash
# Full release build
./scripts/build_release.sh

# Or quick build for testing
./scripts/build_release.sh --skip-tests --clean
```

### 3. Tag and Push
```bash
# Create git tag
VERSION=$(./scripts/version.sh current)
git tag "v$VERSION"

# Push to repository
git push origin main
git push origin "v$VERSION"
```

### 4. Deploy/Distribute
```bash
# Option 1: Publish Docker images
./scripts/build_release.sh --publish --registry your-registry.com

# Option 2: Upload release archives
# Upload files from release/archives/ to GitHub Releases, etc.

# Option 3: Deploy with Docker Compose
cd release/
docker load < docker/exohunter-api-1.0.0.tar.gz
docker load < docker/exohunter-frontend-1.0.0.tar.gz
docker-compose -f docs/deployment/docker-compose.yml up -d
```

## Release Package Contents

```
release/
├── artifacts/                          # Model and training artifacts
│   ├── *.pkl                          # Trained models
│   ├── *.joblib                       # Sklearn models
│   ├── example_toi.csv                 # Example dataset
│   ├── training_scripts/               # Model training code
│   ├── data_processing/                # Data pipeline code
│   ├── coverage_html/                  # Test coverage report
│   └── manifest.json                  # Artifacts metadata
├── docker/                            # Docker images
│   ├── exohunter-api-1.0.0.tar.gz    # API service image
│   ├── exohunter-frontend-1.0.0.tar.gz # Frontend image
│   └── images.json                    # Image metadata
├── frontend/                          # Frontend build artifacts
│   ├── build/                         # Production build
│   └── package.json                   # Package configuration
├── docs/                              # Documentation bundle
│   ├── README.md                      # Main documentation
│   ├── guides/                        # User/Developer guides
│   ├── deployment/                    # Deployment configs
│   ├── api_schema.json                # OpenAPI schema
│   └── *.toml, *.txt                  # Configuration files
├── archives/                          # Compressed release packages
│   ├── exohunter-1.0.0-full.tar.gz   # Complete package
│   ├── exohunter-1.0.0-deploy.tar.gz # Deployment only
│   ├── exohunter-1.0.0-artifacts.tar.gz # Models only
│   └── exohunter-1.0.0-checksums.sha256 # Integrity checksums
├── release-info.json                  # Build metadata
└── RELEASE_NOTES.md                   # Human-readable notes
```

## Environment Variables

- `EXOHUNTER_VERSION` - Override version detection
- `DOCKER_REGISTRY` - Default registry for publishing
- `SKIP_PERFORMANCE_TESTS` - Skip performance smoke tests

## Prerequisites

### Required Tools
- **git** - Version control
- **python** - Python runtime (3.11+)
- **docker** - Container builds (optional with `--skip-docker`)
- **node/npm** - Frontend builds (optional with `--skip-frontend`)

### Optional Tools
- **pytest** - Test execution
- **coverage** - Test coverage reports

## CI/CD Integration

### GitHub Actions Example
```yaml
name: Build Release
on:
  push:
    tags: ['v*']

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build Release
        run: ./scripts/build_release.sh --publish --registry ghcr.io
      - name: Upload Release Assets
        uses: actions/upload-artifact@v3
        with:
          path: release/archives/*
```

### GitLab CI Example
```yaml
release:
  stage: deploy
  script:
    - ./scripts/build_release.sh --publish --registry $CI_REGISTRY
  artifacts:
    paths:
      - release/
  only:
    - tags
```

## Troubleshooting

### Common Issues

**Build fails with "No models found"**
```bash
# Train a model first
python models/train_baseline.py

# Or skip model collection
./scripts/build_release.sh --skip-model-artifacts
```

**Docker build fails**
```bash
# Check Docker is running
docker info

# Or skip Docker builds
./scripts/build_release.sh --skip-docker
```

**Frontend build fails**
```bash
# Install Node.js dependencies
cd web/frontend && npm install

# Or skip frontend build
./scripts/build_release.sh --skip-frontend
```

**Version mismatch errors**
```bash
# Update all version files
./scripts/version.sh set 1.0.0
```

### Debug Mode
```bash
# Run with verbose output
bash -x ./scripts/build_release.sh

# Check intermediate files
ls -la release/
```

## Contributing

When adding new release functionality:

1. Update the main `build_release.sh` script
2. Add corresponding options and documentation
3. Test with `test_release_system.sh`
4. Update this README with new features

## Support

For issues with the release system:
- Check logs in release build output
- Verify prerequisites are installed
- Test with minimal options first
- Report bugs with full error output
