#!/bin/bash

# Directory to search
SEARCH_DIR="$1"

# Find all .wav files in the directory and its subdirectories
find "${SEARCH_DIR}" -type f -name "*.wav" | while read -r input_path; do
    # Skip if backup exists (means we already processed this file)
    if [ -f "${input_path}.backup" ]; then
        continue
    fi

    # Create backup and normalize
    cp "${input_path}" "${input_path}.backup"
    sox "${input_path}.backup" "${input_path}" norm -0.1
done
