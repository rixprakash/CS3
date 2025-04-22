#!/bin/bash
#SBATCH --job-name=cnn_small
#SBATCH --output=cnn_small_%j.log
#SBATCH --error=cnn_small_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Load required modules
module purge
module load miniconda3/24.3.0-py3.11
module load cuda/12.4.1

# Set environment variables for better GPU performance
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_THREAD_MODE=gpu_private
export TF_GPU_THREAD_COUNT=1

# Create and activate conda environment
conda create -y -n ai_detector python=3.8
source activate ai_detector

# Install required packages
pip install --upgrade pip
pip install numpy pandas matplotlib seaborn opencv-python scikit-learn scikit-image tensorflow torch torchvision tqdm

# Set up output directory for CNN model
OUTPUT_DIR="$HOME/Project3/model_results/cnn_model_small"
CHECKPOINT_DIR="$HOME/Project3/checkpoints/cnn_small"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$CHECKPOINT_DIR"

# Print GPU information
echo "GPU Information:"
nvidia-smi

# Run improved main script with CNN model, limit to 100 samples per class for quick testing
echo "Starting CNN model training (small dataset) at $(date)"
python $HOME/Project3/SCRIPTS/improved_main.py \
    --data_dir "$HOME/Project3/DATA" \
    --model_type cnn \
    --output_dir "$OUTPUT_DIR" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --epochs 5 \
    --batch_size 32 \
    --max_samples 100

echo "CNN model training completed at $(date)"

# Check output directory after completion
echo "Output directory: $OUTPUT_DIR"
echo "Contents of output directory:"
ls -la "$OUTPUT_DIR"

# Deactivate conda environment
conda deactivate 