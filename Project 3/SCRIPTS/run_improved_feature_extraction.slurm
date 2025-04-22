#!/bin/bash
#SBATCH --job-name=feature_extract
#SBATCH --output=feature_extract_%j.log
#SBATCH --error=feature_extract_%j.err
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
pip install numpy pandas matplotlib seaborn opencv-python scikit-learn scikit-image tensorflow tqdm

# Set up output directory for features
OUTPUT_DIR="$HOME/Project3/features"
mkdir -p "$OUTPUT_DIR"

# Run improved main script with feature extraction only
echo "Starting feature extraction at $(date)"
python $HOME/Project3/SCRIPTS/improved_main.py \
    --data_dir "$HOME/Project3/DATA" \
    --extract_features \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 32

echo "Feature extraction completed at $(date)"

# Deactivate conda environment
conda deactivate 