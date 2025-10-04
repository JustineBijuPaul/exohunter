#!/bin/bash
set -euo pipefail

# ExoHunter Release Build Script
# This script builds a complete release package including:
# - Docker images
# - Model artifacts
# - Frontend builds
# - Documentation
# - Release archives

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RELEASE_DIR="$PROJECT_ROOT/release"
BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
BUILD_COMMIT=$(git rev-parse --short HEAD)

# Default values
SKIP_TESTS=false
SKIP_DOCKER=false
SKIP_FRONTEND=false
CLEAN_BUILD=false
PUBLISH_IMAGES=false
REGISTRY=""
HELP=false

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${PURPLE}[STEP]${NC} $1"
}

show_help() {
    cat << EOF
ExoHunter Release Build Script

Usage: $0 [OPTIONS]

Options:
    --skip-tests        Skip running tests
    --skip-docker       Skip Docker image builds
    --skip-frontend     Skip frontend build
    --clean             Clean previous builds
    --publish           Publish Docker images to registry
    --registry URL      Docker registry URL (default: docker.io)
    --help              Show this help message

Examples:
    $0                              # Full build with tests
    $0 --skip-tests --clean         # Quick build without tests
    $0 --publish --registry gcr.io  # Build and publish to GCR

Environment Variables:
    EXOHUNTER_VERSION   Override version (default: from pyproject.toml)
    DOCKER_REGISTRY     Default registry for publishing
    
EOF
}

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --skip-docker)
                SKIP_DOCKER=true
                shift
                ;;
            --skip-frontend)
                SKIP_FRONTEND=true
                shift
                ;;
            --clean)
                CLEAN_BUILD=true
                shift
                ;;
            --publish)
                PUBLISH_IMAGES=true
                shift
                ;;
            --registry)
                REGISTRY="$2"
                shift 2
                ;;
            --help)
                HELP=true
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

get_version() {
    if [[ -n "${EXOHUNTER_VERSION:-}" ]]; then
        echo "$EXOHUNTER_VERSION"
        return
    fi
    
    # Extract version from pyproject.toml
    if [[ -f "$PROJECT_ROOT/pyproject.toml" ]]; then
        grep '^version = ' "$PROJECT_ROOT/pyproject.toml" | sed 's/version = "\(.*\)"/\1/'
    else
        # Fallback to API version
        grep 'API_VERSION = ' "$PROJECT_ROOT/web/api/main.py" | sed 's/API_VERSION = "\(.*\)"/\1/'
    fi
}

check_prerequisites() {
    log_step "Checking prerequisites..."
    
    local missing_tools=()
    
    # Check required tools
    command -v git >/dev/null 2>&1 || missing_tools+=("git")
    command -v python >/dev/null 2>&1 || missing_tools+=("python")
    
    if [[ "$SKIP_DOCKER" != "true" ]]; then
        command -v docker >/dev/null 2>&1 || missing_tools+=("docker")
    fi
    
    if [[ "$SKIP_FRONTEND" != "true" ]]; then
        command -v node >/dev/null 2>&1 || missing_tools+=("node")
        command -v npm >/dev/null 2>&1 || missing_tools+=("npm")
    fi
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        exit 1
    fi
    
    # Check if we're in a git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        log_error "Not in a git repository"
        exit 1
    fi
    
    # Check for uncommitted changes
    if [[ -n $(git status --porcelain) ]]; then
        log_warning "Working directory has uncommitted changes"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    log_success "Prerequisites check passed"
}

setup_release_directory() {
    log_step "Setting up release directory..."
    
    if [[ "$CLEAN_BUILD" == "true" ]] && [[ -d "$RELEASE_DIR" ]]; then
        log_info "Cleaning previous build..."
        rm -rf "$RELEASE_DIR"
    fi
    
    mkdir -p "$RELEASE_DIR"/{artifacts,docker,frontend,docs,archives}
    
    log_success "Release directory ready: $RELEASE_DIR"
}

run_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        log_warning "Skipping tests (--skip-tests)"
        return
    fi
    
    log_step "Running tests..."
    
    cd "$PROJECT_ROOT"
    
    # Activate virtual environment if it exists
    if [[ -f "venv/bin/activate" ]]; then
        source venv/bin/activate
    fi
    
    # Run unit tests
    log_info "Running unit tests..."
    python -m pytest tests/ -v --tb=short --maxfail=5
    
    # Run performance tests (soft assertions)
    log_info "Running performance smoke tests..."
    python -m pytest tests/test_model_performance.py --run-slow --run-performance -v || true
    
    # Generate test coverage report
    log_info "Generating coverage report..."
    python -m pytest tests/ --cov=exohunter --cov=web --cov=models --cov=data \
        --cov-report=html --cov-report=xml --cov-report=term
    
    # Copy coverage reports to release
    if [[ -d "htmlcov" ]]; then
        cp -r htmlcov "$RELEASE_DIR/artifacts/coverage_html"
    fi
    if [[ -f "coverage.xml" ]]; then
        cp coverage.xml "$RELEASE_DIR/artifacts/"
    fi
    
    log_success "Tests completed"
}

build_docker_images() {
    if [[ "$SKIP_DOCKER" == "true" ]]; then
        log_warning "Skipping Docker builds (--skip-docker)"
        return
    fi
    
    log_step "Building Docker images..."
    
    local version=$(get_version)
    local api_image="exohunter-api:${version}"
    local frontend_image="exohunter-frontend:${version}"
    
    cd "$PROJECT_ROOT"
    
    # Build API image
    log_info "Building API image: $api_image"
    docker build -t "$api_image" -t "exohunter-api:latest" \
        --build-arg VERSION="$version" \
        --build-arg BUILD_DATE="$BUILD_DATE" \
        --build-arg BUILD_COMMIT="$BUILD_COMMIT" \
        .
    
    # Build frontend image
    log_info "Building frontend image: $frontend_image"
    docker build -t "$frontend_image" -t "exohunter-frontend:latest" \
        --build-arg VERSION="$version" \
        --build-arg BUILD_DATE="$BUILD_DATE" \
        --build-arg BUILD_COMMIT="$BUILD_COMMIT" \
        web/frontend/
    
    # Save images to release directory
    log_info "Exporting Docker images..."
    docker save "$api_image" | gzip > "$RELEASE_DIR/docker/exohunter-api-${version}.tar.gz"
    docker save "$frontend_image" | gzip > "$RELEASE_DIR/docker/exohunter-frontend-${version}.tar.gz"
    
    # Create image manifest
    cat > "$RELEASE_DIR/docker/images.json" << EOF
{
  "build_date": "$BUILD_DATE",
  "build_commit": "$BUILD_COMMIT",
  "version": "$version",
  "images": [
    {
      "name": "exohunter-api",
      "tag": "$version",
      "file": "exohunter-api-${version}.tar.gz",
      "latest_tag": "exohunter-api:latest"
    },
    {
      "name": "exohunter-frontend", 
      "tag": "$version",
      "file": "exohunter-frontend-${version}.tar.gz",
      "latest_tag": "exohunter-frontend:latest"
    }
  ]
}
EOF
    
    # Publish images if requested
    if [[ "$PUBLISH_IMAGES" == "true" ]]; then
        publish_docker_images "$version"
    fi
    
    log_success "Docker images built and exported"
}

publish_docker_images() {
    local version="$1"
    local registry="${REGISTRY:-${DOCKER_REGISTRY:-docker.io}}"
    
    log_step "Publishing Docker images to $registry..."
    
    # Tag and push API image
    docker tag "exohunter-api:${version}" "${registry}/exohunter-api:${version}"
    docker tag "exohunter-api:${version}" "${registry}/exohunter-api:latest"
    docker push "${registry}/exohunter-api:${version}"
    docker push "${registry}/exohunter-api:latest"
    
    # Tag and push frontend image
    docker tag "exohunter-frontend:${version}" "${registry}/exohunter-frontend:${version}"
    docker tag "exohunter-frontend:${version}" "${registry}/exohunter-frontend:latest"
    docker push "${registry}/exohunter-frontend:${version}"
    docker push "${registry}/exohunter-frontend:latest"
    
    log_success "Images published to $registry"
}

build_frontend() {
    if [[ "$SKIP_FRONTEND" == "true" ]]; then
        log_warning "Skipping frontend build (--skip-frontend)"
        return
    fi
    
    log_step "Building frontend..."
    
    cd "$PROJECT_ROOT/web/frontend"
    
    # Install dependencies
    log_info "Installing frontend dependencies..."
    npm ci
    
    # Update version in package.json
    local version=$(get_version)
    npm version "$version" --no-git-tag-version
    
    # Build for production
    log_info "Building frontend for production..."
    npm run build
    
    # Copy build artifacts
    log_info "Copying frontend build artifacts..."
    cp -r dist "$RELEASE_DIR/frontend/build"
    cp package.json "$RELEASE_DIR/frontend/"
    cp package-lock.json "$RELEASE_DIR/frontend/" 2>/dev/null || true
    
    # Create frontend archive
    cd "$RELEASE_DIR/frontend"
    tar -czf "../archives/frontend-${version}.tar.gz" build/ package.json
    
    log_success "Frontend built successfully"
}

collect_model_artifacts() {
    log_step "Collecting model artifacts..."
    
    cd "$PROJECT_ROOT"
    
    # Copy trained models if they exist
    if [[ -d "models" ]]; then
        find models/ -name "*.pkl" -o -name "*.joblib" -o -name "*.json" | while read -r file; do
            cp "$file" "$RELEASE_DIR/artifacts/"
        done
    fi
    
    # Copy training scripts
    cp -r models/ "$RELEASE_DIR/artifacts/training_scripts/" 2>/dev/null || true
    
    # Copy data processing scripts
    cp -r data/ "$RELEASE_DIR/artifacts/data_processing/" 2>/dev/null || true
    
    # Copy example data
    if [[ -f "data/example_toi.csv" ]]; then
        cp "data/example_toi.csv" "$RELEASE_DIR/artifacts/"
    fi
    
    # Create artifacts manifest
    cat > "$RELEASE_DIR/artifacts/manifest.json" << EOF
{
  "build_date": "$BUILD_DATE",
  "build_commit": "$BUILD_COMMIT",
  "version": "$(get_version)",
  "artifacts": {
    "models": [
      $(find "$RELEASE_DIR/artifacts/" -name "*.pkl" -o -name "*.joblib" | sed 's/.*\/\(.*\)/"\1"/' | tr '\n' ',' | sed 's/,$//')
    ],
    "data_files": [
      $(find "$RELEASE_DIR/artifacts/" -name "*.csv" | sed 's/.*\/\(.*\)/"\1"/' | tr '\n' ',' | sed 's/,$//')
    ],
    "training_scripts": "training_scripts/",
    "data_processing": "data_processing/"
  }
}
EOF
    
    log_success "Model artifacts collected"
}

copy_documentation() {
    log_step "Copying documentation..."
    
    cd "$PROJECT_ROOT"
    
    # Copy main documentation
    cp README.md "$RELEASE_DIR/docs/"
    cp -r docs/ "$RELEASE_DIR/docs/guides/" 2>/dev/null || true
    
    # Copy deployment files
    cp -r deploy/ "$RELEASE_DIR/docs/deployment/" 2>/dev/null || true
    cp docker-compose.yml "$RELEASE_DIR/docs/deployment/" 2>/dev/null || true
    cp Dockerfile "$RELEASE_DIR/docs/deployment/" 2>/dev/null || true
    
    # Copy configuration files
    cp pyproject.toml "$RELEASE_DIR/docs/"
    cp requirements.txt "$RELEASE_DIR/docs/"
    
    # Generate API documentation
    if [[ -f "venv/bin/activate" ]]; then
        source venv/bin/activate
        python -c "
import json
from web.api.main import app
with open('$RELEASE_DIR/docs/api_schema.json', 'w') as f:
    json.dump(app.openapi(), f, indent=2)
" 2>/dev/null || log_warning "Could not generate API schema"
    fi
    
    log_success "Documentation copied"
}

create_release_archives() {
    log_step "Creating release archives..."
    
    local version=$(get_version)
    cd "$RELEASE_DIR"
    
    # Create full release archive
    log_info "Creating full release archive..."
    tar -czf "archives/exohunter-${version}-full.tar.gz" \
        artifacts/ docker/ frontend/ docs/ \
        --exclude="archives"
    
    # Create minimal deployment archive (Docker images + configs)
    log_info "Creating deployment archive..."
    tar -czf "archives/exohunter-${version}-deploy.tar.gz" \
        docker/ docs/deployment/
    
    # Create artifacts-only archive
    log_info "Creating artifacts archive..."
    tar -czf "archives/exohunter-${version}-artifacts.tar.gz" \
        artifacts/
    
    # Create checksums
    cd archives/
    sha256sum *.tar.gz > "exohunter-${version}-checksums.sha256"
    
    log_success "Release archives created"
}

generate_release_info() {
    log_step "Generating release information..."
    
    local version=$(get_version)
    local git_tag=$(git describe --tags --exact-match 2>/dev/null || echo "no-tag")
    local git_branch=$(git branch --show-current)
    
    # Create release info JSON
    cat > "$RELEASE_DIR/release-info.json" << EOF
{
  "version": "$version",
  "build_date": "$BUILD_DATE",
  "build_commit": "$BUILD_COMMIT",
  "git_tag": "$git_tag",
  "git_branch": "$git_branch",
  "build_system": {
    "os": "$(uname -s)",
    "arch": "$(uname -m)",
    "python_version": "$(python --version 2>&1 | cut -d' ' -f2)",
    "docker_version": "$(docker --version 2>/dev/null | cut -d' ' -f3 | tr -d ',' || echo 'not-available')",
    "node_version": "$(node --version 2>/dev/null || echo 'not-available')"
  },
  "components": {
    "api": {
      "built": $([ "$SKIP_DOCKER" != "true" ] && echo "true" || echo "false"),
      "image": "exohunter-api:$version"
    },
    "frontend": {
      "built": $([ "$SKIP_FRONTEND" != "true" ] && echo "true" || echo "false"),
      "image": "exohunter-frontend:$version"
    },
    "tests": {
      "run": $([ "$SKIP_TESTS" != "true" ] && echo "true" || echo "false")
    }
  },
  "files": {
    "archives": [
      "archives/exohunter-${version}-full.tar.gz",
      "archives/exohunter-${version}-deploy.tar.gz", 
      "archives/exohunter-${version}-artifacts.tar.gz"
    ],
    "docker_images": [
      "docker/exohunter-api-${version}.tar.gz",
      "docker/exohunter-frontend-${version}.tar.gz"
    ],
    "checksums": "archives/exohunter-${version}-checksums.sha256"
  }
}
EOF
    
    # Create human-readable release notes
    cat > "$RELEASE_DIR/RELEASE_NOTES.md" << EOF
# ExoHunter Release $version

## Build Information
- **Version**: $version
- **Build Date**: $BUILD_DATE
- **Commit**: $BUILD_COMMIT
- **Branch**: $git_branch
- **Tag**: $git_tag

## Contents

### Docker Images
- \`exohunter-api:$version\` - FastAPI backend service
- \`exohunter-frontend:$version\` - React frontend application

### Archives
- \`exohunter-${version}-full.tar.gz\` - Complete release package
- \`exohunter-${version}-deploy.tar.gz\` - Deployment-only package  
- \`exohunter-${version}-artifacts.tar.gz\` - Models and artifacts only

### Components
- **API Service**: Trained ML models for exoplanet classification
- **Web Frontend**: React-based user interface
- **Documentation**: User guides, API docs, deployment instructions
- **Model Artifacts**: Trained models, preprocessing scripts, example data

## Quick Start

### Using Docker Compose
\`\`\`bash
# Extract deployment package
tar -xzf exohunter-${version}-deploy.tar.gz

# Load Docker images
docker load < docker/exohunter-api-${version}.tar.gz
docker load < docker/exohunter-frontend-${version}.tar.gz

# Start services
docker-compose -f docs/deployment/docker-compose.yml up -d
\`\`\`

### Access Points
- **Web Interface**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## Verification
Verify archive integrity using:
\`\`\`bash
sha256sum -c exohunter-${version}-checksums.sha256
\`\`\`

For detailed documentation, see the \`docs/\` directory.
EOF
    
    log_success "Release information generated"
}

show_release_summary() {
    local version=$(get_version)
    
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}🚀 RELEASE BUILD COMPLETED SUCCESSFULLY${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo -e "${CYAN}Version:${NC} $version"
    echo -e "${CYAN}Build Date:${NC} $BUILD_DATE"
    echo -e "${CYAN}Commit:${NC} $BUILD_COMMIT"
    echo -e "${CYAN}Release Directory:${NC} $RELEASE_DIR"
    echo ""
    echo -e "${BLUE}📦 Release Contents:${NC}"
    
    if [[ -d "$RELEASE_DIR/archives" ]]; then
        echo -e "${YELLOW}Archives:${NC}"
        ls -lh "$RELEASE_DIR/archives/" | tail -n +2 | awk '{print "  " $9 " (" $5 ")"}'
    fi
    
    if [[ -d "$RELEASE_DIR/docker" ]]; then
        echo -e "${YELLOW}Docker Images:${NC}"
        ls -lh "$RELEASE_DIR/docker/"*.tar.gz 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}' || true
    fi
    
    echo ""
    echo -e "${BLUE}📋 Next Steps:${NC}"
    echo "  1. Review release notes: $RELEASE_DIR/RELEASE_NOTES.md"
    echo "  2. Test the release archives"
    echo "  3. Create git tag: git tag v$version"
    echo "  4. Push to repository: git push origin v$version"
    echo "  5. Upload release artifacts to distribution channels"
    echo ""
}

# Main execution
main() {
    parse_arguments "$@"
    
    if [[ "$HELP" == "true" ]]; then
        show_help
        exit 0
    fi
    
    cd "$PROJECT_ROOT"
    
    log_info "Starting ExoHunter release build..."
    log_info "Project root: $PROJECT_ROOT"
    log_info "Version: $(get_version)"
    
    check_prerequisites
    setup_release_directory
    run_tests
    build_docker_images
    build_frontend
    collect_model_artifacts
    copy_documentation
    create_release_archives
    generate_release_info
    
    show_release_summary
    
    log_success "Release build completed successfully! 🎉"
}

# Execute main function with all arguments
main "$@"
