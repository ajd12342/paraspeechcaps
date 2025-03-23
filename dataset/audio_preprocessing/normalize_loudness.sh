#!/bin/bash

# Directory to search
SEARCH_DIR="$1"

# Find all .wav files in the directory and its subdirectories
files=($(find "${SEARCH_DIR}" -type f -name "*.wav"))
total_files=${#files[@]}

if [ "$total_files" -eq 0 ]; then
    echo "No .wav files found in ${SEARCH_DIR}"
    exit 1
fi

# Function to display progress
show_progress() {
    local progress=$(( ($1 * 100) / total_files ))
    local bar_length=$((progress / 2))  # Scale to fit terminal width
    printf "\r[%-50s] %d%% (%d/%d)" "$(printf '#%.0s' $(seq 1 $bar_length))" "$progress" "$1" "$total_files"
}

# Process each file
processed=0
for input_path in "${files[@]}"; do
    # Skip if backup exists (means we already processed this file)
    if [ -f "${input_path}.backup" ]; then
        ((processed++))
        show_progress "$processed"
        continue
    fi

    # Create backup and normalize
    cp "${input_path}" "${input_path}.backup"
    sox "${input_path}.backup" "${input_path}" norm -0.1

    ((processed++))
    show_progress "$processed"
done

echo -e "\nProcessing complete!"
