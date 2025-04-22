#!/bin/bash

# Script to run a mini test with just a few images to verify the fix

# Set paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REAL_DATA_DIR="${PROJECT_DIR}/DATA/DeepGuardDB_v1/SD_dataset/real"
FAKE_DATA_DIR="${PROJECT_DIR}/DATA/DeepGuardDB_v1/SD_dataset/fake"
OUTPUT_DIR="${PROJECT_DIR}/mini_test"
REAL_SUBSET_DIR="${OUTPUT_DIR}/real_subset"
FAKE_SUBSET_DIR="${OUTPUT_DIR}/fake_subset"

# Create output directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${REAL_SUBSET_DIR}"
mkdir -p "${FAKE_SUBSET_DIR}"

echo "===== AI Image Detection Mini Test ====="
echo "Using only 10 images per category to verify the fix"

# Clean any existing files
rm -f "${REAL_SUBSET_DIR}"/*
rm -f "${FAKE_SUBSET_DIR}"/*

# Copy a very small subset of images
echo "Copying 10 real images..."
find "${REAL_DATA_DIR}" -name "*.jpg" | head -n 10 | xargs -I{} cp {} "${REAL_SUBSET_DIR}/"

echo "Copying 10 fake images..."
find "${FAKE_DATA_DIR}" -name "*.jpg" | head -n 10 | xargs -I{} cp {} "${FAKE_SUBSET_DIR}/"

# Count files
REAL_COUNT=$(find "${REAL_SUBSET_DIR}" -type f | wc -l)
FAKE_COUNT=$(find "${FAKE_SUBSET_DIR}" -type f | wc -l)
echo "Created mini test with ${REAL_COUNT} real images and ${FAKE_COUNT} fake images"

# Process one image at a time with detailed output
echo -e "\n===== Testing Real Images ====="
for img in "${REAL_SUBSET_DIR}"/*.jpg; do
    echo "Processing: $(basename "$img")"
    python test_fix.py --image "$img"
    if [ $? -ne 0 ]; then
        echo "Failed to process $(basename "$img")"
    else
        echo "Successfully processed $(basename "$img")"
    fi
    echo "---------------------------------"
done

echo -e "\n===== Testing Fake Images ====="
for img in "${FAKE_SUBSET_DIR}"/*.jpg; do
    echo "Processing: $(basename "$img")"
    python test_fix.py --image "$img"
    if [ $? -ne 0 ]; then
        echo "Failed to process $(basename "$img")"
    else
        echo "Successfully processed $(basename "$img")"
    fi
    echo "---------------------------------"
done

echo -e "\n===== Mini Test Complete =====" 