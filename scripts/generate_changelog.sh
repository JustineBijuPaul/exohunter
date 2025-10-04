#!/bin/bash
set -euo pipefail

# Changelog generator for ExoHunter
# Generates changelog from git commits using conventional commit format

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CHANGELOG_FILE="$PROJECT_ROOT/CHANGELOG.md"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

show_help() {
    cat << EOF
ExoHunter Changelog Generator

Usage: $0 [OPTIONS]

Options:
    --from TAG      Generate changelog from this tag (default: last tag)
    --to TAG        Generate changelog to this tag (default: HEAD)
    --output FILE   Output file (default: CHANGELOG.md)
    --append        Append to existing changelog
    --dry-run       Show what would be generated without writing
    --help          Show this help

Examples:
    $0                          # Generate changelog from last tag to HEAD
    $0 --from v1.0.0           # Generate from specific tag
    $0 --append                # Append to existing changelog
    $0 --dry-run               # Preview changes

EOF
}

# Parse commit type and description
parse_commit() {
    local commit_hash="$1"
    local commit_message=$(git log --format="%s" -n 1 "$commit_hash")
    local commit_body=$(git log --format="%b" -n 1 "$commit_hash")
    local author=$(git log --format="%an" -n 1 "$commit_hash")
    local date=$(git log --format="%ad" --date=short -n 1 "$commit_hash")
    
    # Parse conventional commit format: type(scope): description
    if [[ $commit_message =~ ^([a-z]+)(\([^)]+\))?\:\ (.+)$ ]]; then
        local type="${BASH_REMATCH[1]}"
        local scope="${BASH_REMATCH[2]}"
        local description="${BASH_REMATCH[3]}"
        
        echo "$type|$scope|$description|$commit_hash|$author|$date|$commit_body"
    else
        # Non-conventional commit
        echo "other||$commit_message|$commit_hash|$author|$date|$commit_body"
    fi
}

# Get commits between two references
get_commits() {
    local from_ref="$1"
    local to_ref="$2"
    
    if [[ "$from_ref" == "NONE" ]]; then
        # Get all commits
        git rev-list --reverse "$to_ref"
    else
        # Get commits between tags
        git rev-list --reverse "${from_ref}..${to_ref}"
    fi
}

# Group commits by type
generate_changelog_section() {
    local version="$1"
    local date="$2"
    local from_ref="$3"
    local to_ref="$4"
    
    echo "## [$version] - $date"
    echo ""
    
    # Arrays to store different types of commits
    declare -a features=()
    declare -a fixes=()
    declare -a docs=()
    declare -a tests=()
    declare -a chores=()
    declare -a breaking=()
    declare -a others=()
    
    # Process each commit
    while IFS= read -r commit_hash; do
        local commit_info=$(parse_commit "$commit_hash")
        IFS='|' read -r type scope description hash author commit_date body <<< "$commit_info"
        
        # Check for breaking changes
        if echo "$description $body" | grep -qi "breaking change\|BREAKING CHANGE"; then
            breaking+=("- **BREAKING**: $description ($hash)")
            continue
        fi
        
        # Categorize by type
        case "$type" in
            feat|feature)
                features+=("- $description ($hash)")
                ;;
            fix|bugfix)
                fixes+=("- $description ($hash)")
                ;;
            docs|doc)
                docs+=("- $description ($hash)")
                ;;
            test|tests)
                tests+=("- $description ($hash)")
                ;;
            chore|build|ci|perf|refactor|style)
                chores+=("- $description ($hash)")
                ;;
            *)
                others+=("- $description ($hash)")
                ;;
        esac
    done <<< "$(get_commits "$from_ref" "$to_ref")"
    
    # Output sections in order of importance
    if [[ ${#breaking[@]} -gt 0 ]]; then
        echo "### 💥 Breaking Changes"
        echo ""
        printf '%s\n' "${breaking[@]}"
        echo ""
    fi
    
    if [[ ${#features[@]} -gt 0 ]]; then
        echo "### ✨ Features"
        echo ""
        printf '%s\n' "${features[@]}"
        echo ""
    fi
    
    if [[ ${#fixes[@]} -gt 0 ]]; then
        echo "### 🐛 Bug Fixes"
        echo ""
        printf '%s\n' "${fixes[@]}"
        echo ""
    fi
    
    if [[ ${#docs[@]} -gt 0 ]]; then
        echo "### 📚 Documentation"
        echo ""
        printf '%s\n' "${docs[@]}"
        echo ""
    fi
    
    if [[ ${#tests[@]} -gt 0 ]]; then
        echo "### 🧪 Tests"
        echo ""
        printf '%s\n' "${tests[@]}"
        echo ""
    fi
    
    if [[ ${#chores[@]} -gt 0 ]]; then
        echo "### 🔧 Chores & Maintenance"
        echo ""
        printf '%s\n' "${chores[@]}"
        echo ""
    fi
    
    if [[ ${#others[@]} -gt 0 ]]; then
        echo "### 🔀 Other Changes"
        echo ""
        printf '%s\n' "${others[@]}"
        echo ""
    fi
}

# Get the last tag
get_last_tag() {
    git describe --tags --abbrev=0 2>/dev/null || echo "NONE"
}

# Get current version
get_version() {
    if [[ -f "$PROJECT_ROOT/pyproject.toml" ]]; then
        grep '^version = ' "$PROJECT_ROOT/pyproject.toml" | sed 's/version = "\(.*\)"/\1/'
    else
        echo "1.0.0"
    fi
}

# Main changelog generation
generate_changelog() {
    local from_ref="$1"
    local to_ref="$2"
    local output_file="$3"
    local append_mode="$4"
    local dry_run="$5"
    
    local version=$(get_version)
    local date=$(date +"%Y-%m-%d")
    
    if [[ "$from_ref" == "auto" ]]; then
        from_ref=$(get_last_tag)
    fi
    
    local temp_file=$(mktemp)
    
    # Generate new changelog section
    generate_changelog_section "$version" "$date" "$from_ref" "$to_ref" > "$temp_file"
    
    if [[ "$dry_run" == "true" ]]; then
        echo -e "${BLUE}=== CHANGELOG PREVIEW ===${NC}"
        cat "$temp_file"
        echo -e "${BLUE}=========================${NC}"
        rm "$temp_file"
        return
    fi
    
    if [[ "$append_mode" == "true" ]] && [[ -f "$output_file" ]]; then
        # Insert new section after the header
        local new_changelog=$(mktemp)
        
        # Copy header
        head -n 1 "$output_file" > "$new_changelog" 2>/dev/null || echo "# Changelog" > "$new_changelog"
        echo "" >> "$new_changelog"
        
        # Add new section
        cat "$temp_file" >> "$new_changelog"
        echo "" >> "$new_changelog"
        
        # Add existing content (skip header)
        tail -n +2 "$output_file" >> "$new_changelog" 2>/dev/null || true
        
        mv "$new_changelog" "$output_file"
    else
        # Create new changelog
        {
            echo "# Changelog"
            echo ""
            echo "All notable changes to this project will be documented in this file."
            echo ""
            echo "The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),"
            echo "and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)."
            echo ""
            cat "$temp_file"
        } > "$output_file"
    fi
    
    rm "$temp_file"
    
    echo -e "${GREEN}Changelog generated: $output_file${NC}"
}

# Parse arguments
FROM_REF="auto"
TO_REF="HEAD"
OUTPUT_FILE="$CHANGELOG_FILE"
APPEND_MODE=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --from)
            FROM_REF="$2"
            shift 2
            ;;
        --to)
            TO_REF="$2"
            shift 2
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --append)
            APPEND_MODE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Change to project root
cd "$PROJECT_ROOT"

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "${RED}Error: Not in a git repository${NC}"
    exit 1
fi

# Generate changelog
generate_changelog "$FROM_REF" "$TO_REF" "$OUTPUT_FILE" "$APPEND_MODE" "$DRY_RUN"
