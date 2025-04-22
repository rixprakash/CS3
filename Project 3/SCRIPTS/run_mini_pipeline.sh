#!/bin/bash

# Script to run a mini pipeline for AI image detection with our already copied test images

# Set paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_DIR}/mini_test"
REAL_SUBSET_DIR="${OUTPUT_DIR}/real_subset"
FAKE_SUBSET_DIR="${OUTPUT_DIR}/fake_subset"
REAL_FEATURES="${OUTPUT_DIR}/real_features.csv"
FAKE_FEATURES="${OUTPUT_DIR}/fake_features.csv"
PREPARED_DATA_DIR="${OUTPUT_DIR}/prepared_data"

# Create output directories if they don't exist
mkdir -p "${PREPARED_DATA_DIR}"

echo "===== AI Image Detection Mini Pipeline ====="
echo "Using 10 real and 10 fake images"
echo "Real images directory: ${REAL_SUBSET_DIR}"
echo "Fake images directory: ${FAKE_SUBSET_DIR}"
echo "Output directory: ${OUTPUT_DIR}"

# Count files
REAL_COUNT=$(find "${REAL_SUBSET_DIR}" -type f | wc -l)
FAKE_COUNT=$(find "${FAKE_SUBSET_DIR}" -type f | wc -l)
echo "Working with ${REAL_COUNT} real images and ${FAKE_COUNT} fake images"

# Extract features from the subsets
extract_features() {
    echo -e "\n===== Extracting Features ====="
    
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
    # Step 1: Extract features
    extract_features
    
    # Step 2: Combine and process features
    combine_features
    if [ $? -ne 0 ]; then
        echo "Error: Feature combination failed."
        exit 1
    fi
    
    # Step 3: Train models
    train_models
    if [ $? -ne 0 ]; then
        echo "Error: Model training failed."
        exit 1
    fi
    
    # Step 4: Display summary
    display_summary
    
    echo -e "\nMini pipeline completed successfully!"
    exit 0
}

# Run the main function
main 