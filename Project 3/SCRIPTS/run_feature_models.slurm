#!/bin/bash
#SBATCH --account=ds4002-lpa2a
#SBATCH --partition=standard
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --job-name=feature_models
#SBATCH --output=feature_models_%j.log
#SBATCH --error=feature_models_%j.err

# Load required modules
module purge
module load miniforge/24.3.0-py3.11

# Create and activate Python virtual environment
VENV_DIR="$HOME/Project3/venv"
if [ ! -d "$VENV_DIR" ]; then
    python -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# Install required packages
pip install --upgrade pip
pip install numpy pandas matplotlib seaborn opencv-python scikit-learn scikit-image scipy tqdm joblib

# Set the directories
DATA_DIR="$HOME/Project3/DATA/DeepGuardDB_v1"
FEATURES_FILE="$HOME/Project3/feature_extraction_results/full_features.csv"
OUTPUT_DIR="$HOME/Project3/model_results/feature_models"

# Create directories if they don't exist
mkdir -p "$OUTPUT_DIR"
mkdir -p "$(dirname "$FEATURES_FILE")"

# Check directory structure
echo "===== CHECKING DIRECTORY STRUCTURE ====="
echo "Current directory: $(pwd)"
echo "Data directory: $DATA_DIR"
echo "Data directory exists: $(test -d "$DATA_DIR" && echo "YES" || echo "NO")"
echo "Features file: $FEATURES_FILE"
echo "Features file exists: $(test -f "$FEATURES_FILE" && echo "YES" || echo "NO")"
echo "Output directory: $OUTPUT_DIR"
echo "Output directory exists: $(test -d "$OUTPUT_DIR" && echo "YES" || echo "NO")"

# Data directory contents
if [ -d "$DATA_DIR" ]; then
    echo "Contents of data directory:"
    ls -la "$DATA_DIR"
fi

# Print Python version and package information
echo "===== PYTHON ENVIRONMENT ====="
python -c "import sys; print('Python version:', sys.version)"
python -c "import numpy as np; print('NumPy version:', np.__version__)"
python -c "import pandas as pd; print('Pandas version:', pd.__version__)"
python -c "import sklearn; print('Scikit-learn version:', sklearn.__version__)"

# Change to the SCRIPTS directory
cd "$HOME/Project3/SCRIPTS"

echo "Starting feature-based models training at $(date)"
echo "Using $SLURM_CPUS_PER_TASK CPU cores"

# Run the training script with a small sample size for testing
python -u main.py \
    --data_dir "$DATA_DIR" \
    --features_file "$FEATURES_FILE" \
    --model_type feature \
    --output_dir "$OUTPUT_DIR" \
    --max_samples 100 \
    --verbose

# Check the output directory after completion
echo "===== OUTPUT FILES ====="
echo "Checking output directory contents:"
ls -la "$OUTPUT_DIR"

echo "Feature-based models training completed at $(date)"

# Deactivate virtual environment
deactivate 