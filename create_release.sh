#!/bin/bash
# Script to create GitHub release for v2.0.0

echo "Creating GitHub release for Obsidian Vault Tools v2.0.0..."

# Create the release using gh CLI
gh release create v2.0.0 \
  --title "🏰 Obsidian Vault Tools v2.0.0 - Unified Toolsuite Release" \
  --notes-file RELEASE_v2.0.0.md \
  --target main \
  --latest

echo "Release created! Don't forget to:"
echo "1. Push the tag: git push origin v2.0.0"
echo "2. Push commits: git push origin main"
echo "3. Check the release at: https://github.com/[your-username]/obsidian-vault-tools/releases"