#!/bin/bash
#SBATCH --job-name=ai_detection_subset
#SBATCH --output=ai_detection_subset_%j.out
#SBATCH --error=ai_detection_subset_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Load required modules
module purge
module load miniconda3/24.3.0-py3.11
module load cuda/12.4.1

echo "=== AI Image Detection Pipeline (SUBSET) ==="
echo "Started at $(date)"

# Set paths
HOME_DIR="$HOME/Project3"
SCRIPTS_DIR="${HOME_DIR}/SCRIPTS"
DATA_DIR="${HOME_DIR}/DATA/DeepGuardDB_v1/SD_dataset"
OUTPUT_DIR="${HOME_DIR}/subset_results"
REAL_SUBSET_DIR="${OUTPUT_DIR}/real_subset"
FAKE_SUBSET_DIR="${OUTPUT_DIR}/fake_subset"
REAL_FEATURES="${OUTPUT_DIR}/real_subset_features.csv"
FAKE_FEATURES="${OUTPUT_DIR}/fake_subset_features.csv"
PREPARED_DATA_DIR="${OUTPUT_DIR}/prepared_data"

# Create output directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${PREPARED_DATA_DIR}"
mkdir -p "${REAL_SUBSET_DIR}"
mkdir -p "${FAKE_SUBSET_DIR}"

# Create conda environment
echo "Creating conda environment..."
conda create -y -n ai_detector python=3.11
source activate ai_detector

# Install required packages
echo "Installing required packages..."
conda install -y numpy pandas matplotlib seaborn scikit-learn scikit-image tensorflow
conda install -y -c pytorch pytorch torchvision
conda install -y tqdm opencv joblib

# Function to handle errors
handle_error() {
    echo "Error: $1"
    conda deactivate
    exit 1
}

# Create subset directories for faster processing
create_subset() {
    echo -e "\n===== Creating Image Subsets ====="
    
    # Clean any existing files
    rm -f "${REAL_SUBSET_DIR}"/*
    rm -f "${FAKE_SUBSET_DIR}"/*
    
    # Copy a subset of images (150 images each)
    echo "Copying subset of real images..."
    find "${DATA_DIR}/real" -name "*.jpg" | head -n 150 | xargs -I{} cp {} "${REAL_SUBSET_DIR}/" || handle_error "Failed to copy real images"
    
    echo "Copying subset of fake images..."
    find "${DATA_DIR}/fake" -name "*.jpg" | head -n 150 | xargs -I{} cp {} "${FAKE_SUBSET_DIR}/" || handle_error "Failed to copy fake images"
    
    # Count files
    REAL_COUNT=$(find "${REAL_SUBSET_DIR}" -type f | wc -l)
    FAKE_COUNT=$(find "${FAKE_SUBSET_DIR}" -type f | wc -l)
    
    echo "Created subset with ${REAL_COUNT} real images and ${FAKE_COUNT} fake images"
    
    if [ ${REAL_COUNT} -eq 0 ] || [ ${FAKE_COUNT} -eq 0 ]; then
        handle_error "Failed to copy images to subset directories"
    fi
}

# Extract features from the subsets
extract_features() {
    echo -e "\n===== Extracting Features from Subsets ====="
    
    echo "Extracting features from real images..."
    python "${SCRIPTS_DIR}/feature_extraction.py" --data_dir "${REAL_SUBSET_DIR}" --output "${REAL_FEATURES}" --workers 8 || handle_error "Failed to extract real features"
    
    echo "Extracting features from fake images..."
    python "${SCRIPTS_DIR}/feature_extraction.py" --data_dir "${FAKE_SUBSET_DIR}" --output "${FAKE_FEATURES}" --workers 8 || handle_error "Failed to extract fake features"
    
    echo "Feature extraction completed!"
}

# Run the feature combination and processing
combine_features() {
    echo -e "\n===== Combining and Processing Features ====="
    python "${SCRIPTS_DIR}/prepare_combined_features.py" \
        --real_features "${REAL_FEATURES}" \
        --fake_features "${FAKE_FEATURES}" \
        --output_dir "${PREPARED_DATA_DIR}" || handle_error "Feature combination failed"
    
    echo "Feature combination and processing completed successfully!"
}

# Train different model types
train_models() {
    echo -e "\n===== Training Models ====="
    
    # Train Random Forest model (fastest)
    echo "Training Random Forest model..."
    python "${SCRIPTS_DIR}/train_feature_model.py" \
        --data_dir "${PREPARED_DATA_DIR}" \
        --output_dir "${OUTPUT_DIR}/random_forest" \
        --model_type "random_forest" || handle_error "Random Forest training failed"
    
    # Train Gradient Boosting model
    echo "Training Gradient Boosting model..."
    python "${SCRIPTS_DIR}/train_feature_model.py" \
        --data_dir "${PREPARED_DATA_DIR}" \
        --output_dir "${OUTPUT_DIR}/gradient_boosting" \
        --model_type "gradient_boosting" || handle_error "Gradient Boosting training failed"
    
    # Train SVM model
    echo "Training SVM model..."
    python "${SCRIPTS_DIR}/train_feature_model.py" \
        --data_dir "${PREPARED_DATA_DIR}" \
        --output_dir "${OUTPUT_DIR}/svm" \
        --model_type "svm" || handle_error "SVM training failed"
    
    echo "Model training completed!"
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
}

# Main execution flow
echo "Starting pipeline execution..."

# Step 1: Create image subsets
create_subset

# Step 2: Extract features
extract_features

# Step 3: Combine and process features
combine_features

# Step 4: Train models
train_models

# Step 5: Display summary
display_summary

echo -e "\nPipeline completed at $(date)"

# Clean up
conda deactivate

exit 0 