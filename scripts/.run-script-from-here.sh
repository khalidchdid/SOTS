#!/bin/bash

# Exit on error, undefined variable, or pipeline failure
set -euo pipefail

# Name of the launcher script (used to avoid moving it)
LAUNCHER="$(basename "$0")"

# Ensure exactly one argument is provided
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <script_name>"
  exit 1
fi

# Script to execute
TARGET_SCRIPT="$1"

# Ensure the target script exists before doing anything
if [[ ! -f "$TARGET_SCRIPT" ]]; then
  echo "Error: $TARGET_SCRIPT not found"
  exit 1
fi

# Temporary directory used to store all files
TMP_DIR="scripts"

# Include hidden files in globbing '*' and avoid literal '*'
shopt -s dotglob nullglob

# Restore function: move everything back and remove temp directory
restore() {
  for item in "$TMP_DIR"/*; do
    mv "$item" . 2>/dev/null || true
  done
  rmdir "$TMP_DIR" 2>/dev/null || true
}

# Always restore, even on error or Ctrl+C
trap restore EXIT

# Create the temporary directory
mkdir "$TMP_DIR"

# Move everything except the launcher and temp directory
for item in *; do
  if [[ "$item" != "$LAUNCHER" && "$item" != "$TMP_DIR" ]]; then
    mv "$item" "$TMP_DIR/"
  fi
done

# Execute the target script from the original directory
# (the script is located in TMP_DIR, but runs from here)
bash "$TMP_DIR/$TARGET_SCRIPT"
