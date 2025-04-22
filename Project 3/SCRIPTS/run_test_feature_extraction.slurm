#!/bin/bash
#SBATCH --account=ds4002-lpa2a
#SBATCH --partition=standard
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --job-name=test_features
#SBATCH --output=test_features_%j.log

# Load required modules
module load anaconda
module load cuda/11.6.2

# Create and activate conda environment if it doesn't exist
if [ ! -d "$HOME/anaconda/envs/ai_detector" ]; then
    conda create -y -n ai_detector python=3.8
fi
source activate ai_detector

# Install required packages if not already installed
pip install --no-cache-dir numpy pandas matplotlib seaborn opencv-python scikit-learn scikit-image tensorflow torch torchvision scipy tqdm

# Set the data directory
DATA_DIR="$SLURM_SUBMIT_DIR/../DATA/DeepGuardDB_v1"
OUTPUT_DIR="$SLURM_SUBMIT_DIR/../feature_test_results"
mkdir -p $OUTPUT_DIR

echo "Starting test feature extraction at $(date)"

# Run test feature extraction with a limited sample
python $SLURM_SUBMIT_DIR/test_feature_extraction.py --data_dir $DATA_DIR --samples 10 --output_dir $OUTPUT_DIR

echo "Test feature extraction completed at $(date)" 