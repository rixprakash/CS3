#!/bin/bash
#SBATCH --account=ds4002-lpa2a
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --job-name=train_models
#SBATCH --output=model_training_%j.log

# Load required modules
module load anaconda
module load cuda/11.6.2
module load cudnn

# Create and activate conda environment if it doesn't exist
if [ ! -d "$HOME/anaconda/envs/ai_detector" ]; then
    conda create -y -n ai_detector python=3.8
fi
source activate ai_detector

# Install required packages if not already installed
pip install --no-cache-dir numpy pandas matplotlib seaborn opencv-python scikit-learn scikit-image tensorflow torch torchvision scipy tqdm joblib

# Set the directories
DATA_DIR="$SLURM_SUBMIT_DIR/../DATA/DeepGuardDB_v1"
FEATURES_FILE="$SLURM_SUBMIT_DIR/../feature_extraction_results/full_features.csv"
OUTPUT_DIR="$SLURM_SUBMIT_DIR/../model_results"
mkdir -p $OUTPUT_DIR

# Check for GPU availability
nvidia-smi

echo "Starting model training at $(date)"

# Run the main script to train all model types
python $SLURM_SUBMIT_DIR/main.py --data_dir $DATA_DIR --features_file $FEATURES_FILE --model_type all --output_dir $OUTPUT_DIR

echo "Model training completed at $(date)" 