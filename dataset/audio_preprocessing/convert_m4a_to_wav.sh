#!/bin/bash

# Directory to search
SEARCH_DIR="$1"

# Find all .m4a files in the directory and its subdirectories
find "${SEARCH_DIR}" -type f -name "*.m4a" | while read -r m4a_file; do
    wav_file="${m4a_file%.m4a}.wav"
    ffmpeg -hide_banner -loglevel error -nostdin -y -i "${m4a_file}" "${wav_file}" # -nostdin to prevent ffmpeg from reading from while read
done