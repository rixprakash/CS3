#!/bin/bash

# Script to run the full pipeline for AI image detection
# This will wait for feature extraction to complete, then process and train models

# Set paths
REAL_FEATURES="../DATA/real_features.csv"
FAKE_FEATURES="../DATA/fake_features.csv"
OUTPUT_DIR="../model_results"
PREPARED_DATA_DIR="${OUTPUT_DIR}/prepared_data"

# Create output directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${PREPARED_DATA_DIR}"

echo "===== AI Image Detection Pipeline ====="
echo "Real features: ${REAL_FEATURES}"
echo "Fake features: ${FAKE_FEATURES}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Prepared data directory: ${PREPARED_DATA_DIR}"

# Check if feature extraction is still running
check_extraction_running() {
    pgrep -f "feature_extraction.py" > /dev/null
    return $?
}

# Wait for feature extraction to complete
wait_for_extraction() {
    echo -e "\nWaiting for feature extraction to complete..."
    while check_extraction_running; do
        echo "Feature extraction is still running. Waiting 60 seconds..."
        sleep 60
    done
    
    # Check if files exist after extraction completes
    if [[ -f "${REAL_FEATURES}" && -f "${FAKE_FEATURES}" ]]; then
        echo "Feature extraction completed successfully!"
        return 0
    else
        echo "Feature extraction finished but feature files not found!"
        echo "Please check logs for errors."
        return 1
    fi
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
    
    # Train Random Forest model
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
    
    # Train SVM model
    echo "Training SVM model..."
    python train_feature_model.py \
        --data_dir "${PREPARED_DATA_DIR}" \
        --output_dir "${OUTPUT_DIR}/svm" \
        --model_type "svm"
    
    echo "Model training completed!"
    return 0
}

# Display summary of results
display_summary() {
    echo -e "\n===== Results Summary ====="
    
    # Display accuracy for each model
    for model in "random_forest" "gradient_boosting" "svm"; do
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
    # Step 1: Wait for feature extraction to complete
    wait_for_extraction
    if [ $? -ne 0 ]; then
        echo "Error: Feature extraction failed or files not found."
        exit 1
    fi
    
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
    
    exit 0
}

# Run the main function
main 