#!/bin/bash
#SBATCH --job-name=ai_detection_full
#SBATCH --output=ai_detection_full_%j.out
#SBATCH --error=ai_detection_full_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Load required modules
module purge
module load miniconda3/24.3.0-py3.11
module load cuda/12.4.1

echo "=== AI Image Detection Pipeline (FULL DATASET) ==="
echo "Started at $(date)"

# Set paths
HOME_DIR="$HOME/Project3"
SCRIPTS_DIR="${HOME_DIR}/SCRIPTS"
DATA_DIR="${HOME_DIR}/DATA/DeepGuardDB_v1/SD_dataset"
OUTPUT_DIR="${HOME_DIR}/full_results"
REAL_FEATURES="${OUTPUT_DIR}/real_features.csv"
FAKE_FEATURES="${OUTPUT_DIR}/fake_features.csv"
PREPARED_DATA_DIR="${OUTPUT_DIR}/prepared_data"

# Create output directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${PREPARED_DATA_DIR}"

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

# Extract features from the datasets
extract_features() {
    echo -e "\n===== Extracting Features from Full Dataset ====="
    
    echo "Extracting features from real images..."
    python "${SCRIPTS_DIR}/feature_extraction.py" --data_dir "${DATA_DIR}/real" --output "${REAL_FEATURES}" --workers 16 || handle_error "Failed to extract real features"
    
    echo "Extracting features from fake images..."
    python "${SCRIPTS_DIR}/feature_extraction.py" --data_dir "${DATA_DIR}/fake" --output "${FAKE_FEATURES}" --workers 16 || handle_error "Failed to extract fake features"
    
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
    
    # Train Random Forest model
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

# Step 1: Extract features
extract_features

# Step 2: Combine and process features
combine_features

# Step 3: Train models
train_models

# Step 4: Display summary
display_summary

echo -e "\nPipeline completed at $(date)"

# Clean up
conda deactivate

exit 0 