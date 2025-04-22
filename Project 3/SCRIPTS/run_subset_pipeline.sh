#!/bin/bash

# Script to run a simplified pipeline for AI image detection
# Uses a small subset of images for faster processing

# Set paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REAL_DATA_DIR="${PROJECT_DIR}/DATA/DeepGuardDB_v1/SD_dataset/real"
FAKE_DATA_DIR="${PROJECT_DIR}/DATA/DeepGuardDB_v1/SD_dataset/fake"
OUTPUT_DIR="${PROJECT_DIR}/subset_results"
REAL_FEATURES="${OUTPUT_DIR}/real_subset_features.csv"
FAKE_FEATURES="${OUTPUT_DIR}/fake_subset_features.csv"
PREPARED_DATA_DIR="${OUTPUT_DIR}/prepared_data"
REAL_SUBSET_DIR="${OUTPUT_DIR}/real_subset"
FAKE_SUBSET_DIR="${OUTPUT_DIR}/fake_subset"

# Create output directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${PREPARED_DATA_DIR}"
mkdir -p "${REAL_SUBSET_DIR}"
mkdir -p "${FAKE_SUBSET_DIR}"

echo "===== AI Image Detection Subset Pipeline ====="
echo "Real data directory: ${REAL_DATA_DIR}"
echo "Fake data directory: ${FAKE_DATA_DIR}" 
echo "Output directory: ${OUTPUT_DIR}"
echo "Prepared data directory: ${PREPARED_DATA_DIR}"

# Create subset directories for faster processing
create_subset() {
    echo -e "\n===== Creating Image Subsets ====="
    
    # Clean any existing files
    rm -f "${REAL_SUBSET_DIR}"/*
    rm -f "${FAKE_SUBSET_DIR}"/*
    
    # Check if source directories exist
    if [ ! -d "${REAL_DATA_DIR}" ]; then
        echo "Error: Real data directory does not exist: ${REAL_DATA_DIR}"
        return 1
    fi
    
    if [ ! -d "${FAKE_DATA_DIR}" ]; then
        echo "Error: Fake data directory does not exist: ${FAKE_DATA_DIR}"
        return 1
    fi
    
    # Copy a small subset of images (150 images each)
    echo "Copying subset of real images..."
    find "${REAL_DATA_DIR}" -name "*.jpg" | head -n 150 | xargs -I{} cp {} "${REAL_SUBSET_DIR}/"
    
    echo "Copying subset of fake images..."
    find "${FAKE_DATA_DIR}" -name "*.jpg" | head -n 150 | xargs -I{} cp {} "${FAKE_SUBSET_DIR}/"
    
    # Count files
    REAL_COUNT=$(find "${REAL_SUBSET_DIR}" -type f | wc -l)
    FAKE_COUNT=$(find "${FAKE_SUBSET_DIR}" -type f | wc -l)
    
    echo "Created subset with ${REAL_COUNT} real images and ${FAKE_COUNT} fake images"
    
    if [ ${REAL_COUNT} -eq 0 ] || [ ${FAKE_COUNT} -eq 0 ]; then
        echo "Error: Failed to copy images to subset directories"
        return 1
    fi
}

# Extract features from the subsets
extract_features() {
    echo -e "\n===== Extracting Features from Subsets ====="
    
    echo "Extracting features from real images..."
    python feature_extraction.py --data_dir "${REAL_SUBSET_DIR}" --output "${REAL_FEATURES}" --workers 4
    
    echo "Extracting features from fake images..."
    python feature_extraction.py --data_dir "${FAKE_SUBSET_DIR}" --output "${FAKE_FEATURES}" --workers 4
    
    echo "Feature extraction completed!"
}

# Run the feature combination and processing
combine_features() {
    echo -e "\n===== Combining and Processing Features ====="
    python prepare_combined_features.py \
        --real_features "${REAL_FEATURES}" \
        --fake_features "${FAKE_FEATURES}" \
        --output_dir "${PREPARED_DATA_DIR}"
    
    if [ $? -ne 0 ]; then
        echo "Error: Feature combination failed!"
        return 1
    fi
    
    echo "Feature combination and processing completed successfully!"
    return 0
}

# Train different model types
train_models() {
    echo -e "\n===== Training Models ====="
    
    # Train Random Forest model (fastest)
    echo "Training Random Forest model..."
    python train_feature_model.py \
        --data_dir "${PREPARED_DATA_DIR}" \
        --output_dir "${OUTPUT_DIR}/random_forest" \
        --model_type "random_forest"
    
    # Train Gradient Boosting model
    echo "Training Gradient Boosting model..."
    python train_feature_model.py \
        --data_dir "${PREPARED_DATA_DIR}" \
        --output_dir "${OUTPUT_DIR}/gradient_boosting" \
        --model_type "gradient_boosting"
    
    echo "Model training completed!"
    return 0
}

# Display summary of results
display_summary() {
    echo -e "\n===== Results Summary ====="
    
    # Display accuracy for each model
    for model in "random_forest" "gradient_boosting"; do
        if [[ -f "${OUTPUT_DIR}/${model}/${model}_metrics.txt" ]]; then
            acc=$(grep "Accuracy:" "${OUTPUT_DIR}/${model}/${model}_metrics.txt" | cut -d' ' -f2)
            echo "${model} accuracy: ${acc}"
        else
            echo "${model} metrics not found"
        fi
    done
    
    echo -e "\nDetailed results are available in: ${OUTPUT_DIR}"
    echo "Pipeline execution completed!"
}

# Main execution flow
main() {
    # Step 1: Create image subsets
    create_subset
    if [ $? -ne 0 ]; then
        echo "Error: Creating subset failed."
        exit 1
    fi
    
    # Step 2: Extract features
    extract_features
    
    # Step 3: Combine and process features
    combine_features
    if [ $? -ne 0 ]; then
        echo "Error: Feature combination failed."
        exit 1
    fi
    
    # Step 4: Train models
    train_models
    if [ $? -ne 0 ]; then
        echo "Error: Model training failed."
        exit 1
    fi
    
    # Step 5: Display summary
    display_summary
    
    echo -e "\nSubset pipeline completed successfully!"
    exit 0
}

# Run the main function
main 