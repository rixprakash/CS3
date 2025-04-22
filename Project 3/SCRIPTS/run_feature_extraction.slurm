#!/bin/bash
#SBATCH --job-name=feature_extract
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=bii_dsc_community

# Load modules
module purge
module load miniforge/24.3.0-py3.11
module load cuda/12.4.1

# Create and activate Python virtual environment
VENV_DIR="$HOME/Project3/venv"
if [ ! -d "$VENV_DIR" ]; then
    python -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# Install required packages
pip install --upgrade pip
pip install numpy pandas matplotlib seaborn opencv-python scikit-learn scikit-image tensorflow torch torchvision tqdm

# Set the data directory
DATA_DIR="$HOME/Project3/DATA/DeepGuardDB_v1"
OUTPUT_DIR="$HOME/Project3/feature_extraction_results"
FEATURES_FILE="$OUTPUT_DIR/image_features.csv"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Echo environment information
echo "WORKING IN DIRECTORY: $(pwd)"
echo "CHECKING DIRECTORY STRUCTURE:"
echo "Data directory: $DATA_DIR (exists: $([ -d "$DATA_DIR" ] && echo "YES" || echo "NO"))"
echo "Output directory: $OUTPUT_DIR (exists: $([ -d "$OUTPUT_DIR" ] && echo "YES" || echo "NO"))"
ls -la

# Check if we're in the correct directory
cd $HOME/Project3/SCRIPTS

# Check if main.py exists
if [ ! -f "main.py" ]; then
    echo "ERROR: main.py not found in $(pwd)"
    exit 1
fi

# Run the feature extraction
echo "STARTING FEATURE EXTRACTION at $(date)..."
python main.py --data_dir $DATA_DIR --extract_features --features_file $FEATURES_FILE --output_dir $OUTPUT_DIR

# Check for the output file
echo "FEATURE EXTRACTION COMPLETE at $(date). CHECKING OUTPUT:"
ls -la $OUTPUT_DIR 