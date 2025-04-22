#!/bin/bash
#SBATCH --job-name=hybrid_model
#SBATCH --output=hybrid_model_%j.log
#SBATCH --error=hybrid_model_%j.err
#SBATCH --time=02:00:00
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

# Set up output directory for hybrid model
OUTPUT_DIR="$HOME/Project3/model_results/hybrid_model"
CHECKPOINT_DIR="$HOME/Project3/checkpoints/hybrid"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$CHECKPOINT_DIR"

# Print GPU information
echo "GPU Information:"
nvidia-smi

# Run improved main script with hybrid model
echo "Starting hybrid model training at $(date)"
python $HOME/Project3/SCRIPTS/improved_main.py \
    --data_dir "$HOME/Project3/DATA" \
    --model_type hybrid \
    --output_dir "$OUTPUT_DIR" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --epochs 20 \
    --batch_size 32

echo "Hybrid model training completed at $(date)"

# Deactivate conda environment
conda deactivate 