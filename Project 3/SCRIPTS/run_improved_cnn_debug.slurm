#!/bin/bash
#SBATCH --job-name=cnn_debug
#SBATCH --output=cnn_debug_%j.log
#SBATCH --error=cnn_debug_%j.err
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
export TF_CPP_MIN_LOG_LEVEL=0  # 0 = all logs, 1 = no INFO, 2 = no WARNING, 3 = no ERROR

# Debugging information
echo "====== DEBUGGING INFORMATION ======"
echo "Current directory: $(pwd)"
echo "User: $(whoami)"
echo "Python environment:"
which python
python --version

# Create and activate conda environment
conda create -y -n ai_detector python=3.8
source activate ai_detector

# Install required packages
pip install --upgrade pip
pip install numpy pandas matplotlib seaborn opencv-python scikit-learn scikit-image tensorflow torch torchvision tqdm

# Set up output directory for CNN model
OUTPUT_DIR="$HOME/Project3/model_results/cnn_model"
CHECKPOINT_DIR="$HOME/Project3/checkpoints/cnn"
DATA_DIR="$HOME/Project3/DATA"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$CHECKPOINT_DIR"

# Print GPU information
echo "GPU Information:"
nvidia-smi

# Debug data directory
echo "====== DATA DIRECTORY CHECK ======"
echo "Data directory: $DATA_DIR"
echo "Directory exists: $(if [ -d "$DATA_DIR" ]; then echo "YES"; else echo "NO"; fi)"
echo "Contents of data directory:"
ls -la "$DATA_DIR"

echo "Contents of DeepGuardDB_v1 directory:"
ls -la "$DATA_DIR/DeepGuardDB_v1"

# Debug script files
echo "====== SCRIPT FILES CHECK ======"
echo "Script directory: $HOME/Project3/SCRIPTS"
echo "Directory exists: $(if [ -d "$HOME/Project3/SCRIPTS" ]; then echo "YES"; else echo "NO"; fi)"
echo "Checking for improved_main.py:"
ls -la "$HOME/Project3/SCRIPTS/improved_main.py"
echo "File contents (first 20 lines):"
head -n 20 "$HOME/Project3/SCRIPTS/improved_main.py"

# Run improved main script with CNN model in verbose mode
echo "====== STARTING CNN MODEL TRAINING ======"
echo "Command: python $HOME/Project3/SCRIPTS/improved_main.py --data_dir "$DATA_DIR" --model_type cnn --output_dir "$OUTPUT_DIR" --checkpoint_dir "$CHECKPOINT_DIR" --epochs 20 --batch_size 32 --verbose"
echo "Starting CNN model training at $(date)"

# Run with Python's verbose mode
python -v "$HOME/Project3/SCRIPTS/improved_main.py" \
    --data_dir "$DATA_DIR" \
    --model_type cnn \
    --output_dir "$OUTPUT_DIR" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --epochs 20 \
    --batch_size 32

echo "CNN model training completed at $(date)"

# Check output directory after completion
echo "====== OUTPUT DIRECTORY CHECK ======"
echo "Output directory: $OUTPUT_DIR"
echo "Directory exists: $(if [ -d "$OUTPUT_DIR" ]; then echo "YES"; else echo "NO"; fi)"
echo "Contents of output directory:"
ls -la "$OUTPUT_DIR"

# Deactivate conda environment
conda deactivate 