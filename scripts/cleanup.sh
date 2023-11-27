#!/bin/bash

cd "$(dirname "$0")"/..

DIRECTORY_ARRAY=(
    "data/raw/organized/imagens_ihq_er/"
    "data/processed/cropped_images/"
    "data/processed/folded_data/"
)

cleanup_directory() {
    if [ -d "$1" ]; then
        rm -rf "$1"/*
        echo "Contents of $1 directory cleaned up."
    else
        echo "$1 directory does not exist."
    fi
}

for dir in "${DIRECTORY_ARRAY[@]}"; do
    cleanup_directory "$dir"
done
