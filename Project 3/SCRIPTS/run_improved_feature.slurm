#!/bin/bash
#SBATCH --job-name=feature_model
#SBATCH --output=feature_model_%j.log
#SBATCH --error=feature_model_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=standard

# Load required modules
module purge
module load miniconda3/24.3.0-py3.11

# Create and activate conda environment
conda create -y -n ai_detector python=3.8
source activate ai_detector

# Install required packages
pip install --upgrade pip
pip install numpy pandas matplotlib seaborn opencv-python scikit-learn scikit-image tqdm joblib

# Set up output directory for feature models
OUTPUT_DIR="$HOME/Project3/model_results/feature_models"
mkdir -p "$OUTPUT_DIR"

# Check if features have been extracted
FEATURE_FILE="$HOME/Project3/features/image_features.csv"
if [ ! -f "$FEATURE_FILE" ]; then
    echo "Warning: Feature file not found at $FEATURE_FILE"
    echo "Will extract features during model training"
fi

# Run improved main script with feature model
echo "Starting feature model training at $(date)"
python $HOME/Project3/SCRIPTS/improved_main.py \
    --data_dir "$HOME/Project3/DATA" \
    --model_type feature \
    --output_dir "$OUTPUT_DIR" \
    --features_file "$FEATURE_FILE" \
    --epochs 20 \
    --batch_size 32

echo "Feature model training completed at $(date)"

# Deactivate conda environment
conda deactivate 