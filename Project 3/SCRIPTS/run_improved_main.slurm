#!/bin/bash
#SBATCH --job-name=ai_detector
#SBATCH --output=ai_detector_%j.log
#SBATCH --error=ai_detector_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Load required modules
module purge
module load miniforge/24.3.0-py3.11
module load cuda/12.4.1

# Set environment variables for better GPU performance
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_THREAD_MODE=gpu_private
export TF_GPU_THREAD_COUNT=1

# Create and activate Python virtual environment
VENV_DIR="$HOME/Project3/venv"
if [ ! -d "$VENV_DIR" ]; then
    python -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# Install required packages
pip install --upgrade pip
pip install numpy pandas matplotlib seaborn opencv-python scikit-learn scikit-image
pip install tensorflow torch torchvision tqdm

# Set up output directory
OUTPUT_DIR="$HOME/Project3/model_results"
CHECKPOINT_DIR="$HOME/Project3/checkpoints"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$CHECKPOINT_DIR"

# Print Python version and CUDA availability
python -c "import sys; print('Python version:', sys.version)"
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('CUDA available:', tf.test.is_gpu_available()); print('GPUs:', tf.config.list_physical_devices('GPU'))"
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA device count:', torch.cuda.device_count())"

# Run improved main script with GPU support
echo "Starting model training at $(date)"
python $HOME/Project3/SCRIPTS/improved_main.py \
    --data_dir "$HOME/Project3/DATA" \
    --model_type all \
    --output_dir "$OUTPUT_DIR" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --epochs 20 \
    --batch_size 32

echo "Model training completed at $(date)"

# Deactivate virtual environment
deactivate 