#!/bin/bash
set -euo pipefail

# Version management script for ExoHunter
# Updates version across all project files

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

show_help() {
    cat << EOF
ExoHunter Version Management Script

Usage: $0 [COMMAND] [VERSION]

Commands:
    current     Show current version
    bump        Bump version (patch, minor, or major)
    set         Set specific version
    help        Show this help

Examples:
    $0 current                  # Show current version
    $0 bump patch              # Bump patch version (1.0.0 -> 1.0.1)
    $0 bump minor              # Bump minor version (1.0.0 -> 1.1.0)
    $0 bump major              # Bump major version (1.0.0 -> 2.0.0)
    $0 set 1.2.3               # Set version to 1.2.3

EOF
}

get_current_version() {
    if [[ -f "$PROJECT_ROOT/pyproject.toml" ]]; then
        grep '^version = ' "$PROJECT_ROOT/pyproject.toml" | sed 's/version = "\(.*\)"/\1/'
    else
        echo "1.0.0"
    fi
}

validate_version() {
    local version="$1"
    if [[ ! $version =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo -e "${RED}Error: Invalid version format. Use semantic versioning (e.g., 1.2.3)${NC}"
        exit 1
    fi
}

bump_version() {
    local bump_type="$1"
    local current_version=$(get_current_version)
    
    IFS='.' read -r major minor patch <<< "$current_version"
    
    case $bump_type in
        major)
            major=$((major + 1))
            minor=0
            patch=0
            ;;
        minor)
            minor=$((minor + 1))
            patch=0
            ;;
        patch)
            patch=$((patch + 1))
            ;;
        *)
            echo -e "${RED}Error: Invalid bump type. Use: major, minor, or patch${NC}"
            exit 1
            ;;
    esac
    
    echo "$major.$minor.$patch"
}

update_version_files() {
    local new_version="$1"
    
    echo -e "${YELLOW}Updating version to $new_version...${NC}"
    
    # Update pyproject.toml
    if [[ -f "$PROJECT_ROOT/pyproject.toml" ]]; then
        sed -i.bak "s/^version = .*/version = \"$new_version\"/" "$PROJECT_ROOT/pyproject.toml"
        rm "$PROJECT_ROOT/pyproject.toml.bak"
        echo "✓ Updated pyproject.toml"
    fi
    
    # Update API version
    if [[ -f "$PROJECT_ROOT/web/api/main.py" ]]; then
        sed -i.bak "s/API_VERSION = .*/API_VERSION = \"$new_version\"/" "$PROJECT_ROOT/web/api/main.py"
        rm "$PROJECT_ROOT/web/api/main.py.bak"
        echo "✓ Updated API version"
    fi
    
    # Update frontend package.json
    if [[ -f "$PROJECT_ROOT/web/frontend/package.json" ]]; then
        cd "$PROJECT_ROOT/web/frontend"
        npm version "$new_version" --no-git-tag-version
        echo "✓ Updated frontend package.json"
    fi
    
    # Update README if it contains version references
    if [[ -f "$PROJECT_ROOT/README.md" ]]; then
        # This is optional - only if README contains version badges or references
        echo "ℹ  Check README.md for version references to update manually"
    fi
    
    echo -e "${GREEN}Version updated successfully to $new_version${NC}"
}

case "${1:-}" in
    current)
        echo "Current version: $(get_current_version)"
        ;;
    bump)
        if [[ -z "${2:-}" ]]; then
            echo -e "${RED}Error: Bump type required (major, minor, patch)${NC}"
            exit 1
        fi
        new_version=$(bump_version "$2")
        validate_version "$new_version"
        update_version_files "$new_version"
        ;;
    set)
        if [[ -z "${2:-}" ]]; then
            echo -e "${RED}Error: Version required${NC}"
            exit 1
        fi
        validate_version "$2"
        update_version_files "$2"
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}Error: Unknown command${NC}"
        show_help
        exit 1
        ;;
esac
