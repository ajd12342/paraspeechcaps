#!/bin/bash

# Directory to search
SEARCH_DIR="$1"
INCLUDE_TOTAL=false

shift
# Parse remaining arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --show-total)
            INCLUDE_TOTAL=true
            ;;
    esac
    shift
done

# Check if there are any .wav files
if ! find "${SEARCH_DIR}" -type f -name "*.wav" -quit | grep -q .; then
    echo "No .wav files found in ${SEARCH_DIR}"
    exit 1
fi

total_files=0
if [ "$INCLUDE_TOTAL" = true ]; then
    total_files=$(find "${SEARCH_DIR}" -type f -name "*.wav" | wc -l)
fi

# Function to display progress
show_progress() {
    local processed=$1
    if [ "$INCLUDE_TOTAL" = true ]; then
        local progress=$(( ($processed * 100) / total_files ))
        local bar_length=$((progress / 2))  # Scale to fit terminal width
        printf "\r[%-50s] %d%% (%d/%d)" "$(printf '#%.0s' $(seq 1 $bar_length))" "$progress" "$processed" "$total_files"
    else
        local bar_length=$((processed % 50))  # Rotating progress bar
        printf "\r[%-50s] %d files processed" "$(printf '#%.0s' $(seq 1 $bar_length))" "$processed"
    fi
}

# Process files
processed=0
find "${SEARCH_DIR}" -type f -name "*.wav" | while read -r input_path; do
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

echo -e "Loudness normalization processing complete!"
