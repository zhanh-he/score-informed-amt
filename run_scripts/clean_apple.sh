#!/bin/bash
set -euo pipefail

# Usage:
#   ./clean_cache.sh /path/to/root
#   # or rely on the default DATA_ROOT below

# DATA_ROOT_DEFAULT="/media/datadisk/home/22828187/zhanh"
DATA_ROOT_DEFAULT=".."

TARGET_DIR="${1:-$DATA_ROOT_DEFAULT}"

if [[ ! -d "$TARGET_DIR" ]]; then
  echo "Error: TARGET_DIR '$TARGET_DIR' does not exist." >&2
  exit 1
fi

echo "Cleaning cache artifacts under: $TARGET_DIR"

# Remove stray macOS metadata files.
echo "Removing .DS_Store files..."
find "$TARGET_DIR" -type f -name ".DS_Store" -print -delete
echo "Removing __MACOSX folders..."
find "$TARGET_DIR" -type d -name "__MACOSX" -print -exec rm -rf {} +

# Remove Python cache folders (helpful for shared workspaces).
echo "Removing __pycache__ folders..."
find "$TARGET_DIR" -type d -name "__pycache__" -print -exec rm -rf {} +

# Remove AppleDouble files (._filename) that macOS leaves behind.
echo "Removing AppleDouble files (._*)..."
find "$TARGET_DIR" -type f -name "._*" -print -delete

echo "Cleanup completed."
