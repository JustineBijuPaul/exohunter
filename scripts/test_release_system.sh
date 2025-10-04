#!/bin/bash

# Test script for release build system
# This runs a minimal release build to test the infrastructure

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🧪 Testing ExoHunter Release Build System"
echo "========================================"

cd "$PROJECT_ROOT"

echo "📋 Current project status:"
echo "  - Version: $(./scripts/version.sh current)"
echo "  - Git status: $(git status --porcelain | wc -l) uncommitted changes"
echo "  - Git branch: $(git branch --show-current)"
echo ""

echo "🔧 Available scripts:"
ls -la scripts/*.sh | awk '{print "  - " $9}'
echo ""

echo "📦 Testing build script (dry run):"
echo "  Running: ./scripts/build_release.sh --skip-tests --skip-docker --skip-frontend --help"
./scripts/build_release.sh --help | head -20
echo ""

echo "📝 Testing version script:"
echo "  Current version: $(./scripts/version.sh current)"
echo ""

echo "📖 Testing changelog:"
if [[ -f "CHANGELOG.md" ]]; then
    echo "  ✅ CHANGELOG.md exists"
    echo "  📊 Lines: $(wc -l < CHANGELOG.md)"
else
    echo "  ❌ CHANGELOG.md missing"
fi
echo ""

echo "🏗️  Release build components:"
echo "  - pyproject.toml: $(test -f pyproject.toml && echo "✅" || echo "❌")"
echo "  - Dockerfile: $(test -f Dockerfile && echo "✅" || echo "❌")"
echo "  - docker-compose.yml: $(test -f docker-compose.yml && echo "✅" || echo "❌")"
echo "  - Frontend package.json: $(test -f web/frontend/package.json && echo "✅" || echo "❌")"
echo "  - Requirements: $(test -f requirements.txt && echo "✅" || echo "❌")"
echo ""

echo "✅ Release build system test completed!"
echo ""
echo "Next steps:"
echo "  1. Run full build: ./scripts/build_release.sh"
echo "  2. Version bump: ./scripts/version.sh bump patch"
echo "  3. Update changelog: ./scripts/generate_changelog.sh --append"
