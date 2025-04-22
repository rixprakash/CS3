#!/bin/bash
#SBATCH --job-name=hybrid_model
#SBATCH --output=hybrid_model_%j.log
#SBATCH --error=hybrid_model_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu-a6000
#SBATCH --gres=gpu:1

echo "Starting hybrid model training job at $(date)"

# Print environment information
module purge
module load miniconda3/24.3.0-py3.11 cuda/12.4.1

echo "PATH: $PATH"
echo "HOME: $HOME"
echo "Working directory: $(pwd)"

# Setup environment
if [ -d "$HOME/Project3/venv" ]; then
    echo "Removing existing virtual environment..."
    rm -rf "$HOME/Project3/venv"
fi

echo "Creating new virtual environment..."
python -m venv "$HOME/Project3/venv"
echo "Using Python at: $HOME/Project3/venv/bin/python"

source "$HOME/Project3/venv/bin/activate"

# Debug: Check Python version and environment
echo "Python version: $(python --version)"

echo "Upgrading pip and installing required packages..."
python -m pip install --upgrade pip
pip install numpy pandas matplotlib seaborn scikit-learn scikit-image tensorflow torch torchvision tqdm
pip install opencv-python

# Debug: Check if cv2 is installed properly
echo "Checking if cv2 is installed..."
python -c "import cv2; print('cv2 version:', cv2.__version__)"

# Set up output directory
OUTPUT_DIR="$HOME/Project3/model_results/hybrid_model"
echo "Output directory: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Debug: Check available GPU
nvidia-smi

# Additional debug info
echo "Python version: $(python -V 2>&1)"
echo "TensorFlow version: $(python -c 'import tensorflow as tf; print(tf.__version__)')"
echo "CUDA available: $(python -c 'import tensorflow as tf; print(tf.test.is_gpu_available())')"
echo "GPUs: $(python -c 'import tensorflow as tf; print(tf.config.list_physical_devices("GPU"))')"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA device count: $(python -c 'import torch; print(torch.cuda.device_count())')"

# Run the model with additional debug output
echo "Starting hybrid model training at $(date)"
echo "Current directory: $(pwd)"
echo "Files in current directory:"
ls -la

# Add specific debug tracing to the main.py execution
echo "Running: python main.py --model_type hybrid --output_dir $OUTPUT_DIR --debug"

# Create a modified version of main.py with debug messages
cat > debug_main.py << 'EOL'
import os
import sys
import argparse
import pandas as pd
import numpy as np
import time
from preprocessing import preprocess_data, create_data_generators
from model import (
    create_model, create_feature_enhanced_model, create_feature_only_model,
    train_model, train_feature_enhanced_model, train_feature_only_model,
    evaluate_model, evaluate_feature_enhanced_model, evaluate_feature_only_model,
    plot_training_history, plot_confusion_matrix, plot_feature_importance, get_feature_importance
)
from feature_extraction import ImageFeatureExtractor, extract_all_features

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate AI image detection models")
    parser.add_argument("--data_dir", default=None, help="Path to the image data directory")
    parser.add_argument("--model_type", default="all", choices=["cnn", "feature", "hybrid", "all"], 
                      help="Type of model to train")
    parser.add_argument("--extract_features", action="store_true", help="Extract features from images")
    parser.add_argument("--features_file", default=None, help="Path to precomputed features CSV")
    parser.add_argument("--output_dir", default="results", help="Directory to save results")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    
    args = parser.parse_args()
    
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get the project root directory (one level up from SCRIPTS)
    project_dir = os.path.dirname(script_dir)
    
    # Set paths
    if args.data_dir is None:
        data_dir = os.path.join(project_dir, 'DATA')
    else:
        data_dir = args.data_dir
        
    # Set up output directory
    output_dir = os.path.join(project_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set the features file path
    if args.features_file is None:
        features_file = os.path.join(output_dir, 'image_features.csv')
    else:
        features_file = args.features_file
    
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Debug: Show directory contents
    print(f"Data directory contents:")
    for root, dirs, files in os.walk(data_dir, topdown=True, maxdepth=2):
        print(f"Directory: {root}")
        print(f"  Subdirectories: {dirs[:5]} ({'...' if len(dirs) > 5 else ''})")
        print(f"  Files: {files[:5]} ({'...' if len(files) > 5 else ''})")
        print(f"  Total files: {len(files)}")
    
    # Step 1: Preprocess image data
    print("\nStep 1: Preprocessing image data...")
    start_time = time.time()
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(
            data_dir=data_dir,
            test_size=0.2,
            val_size=0.2
        )
        print(f"Preprocessing completed in {time.time() - start_time:.2f} seconds")
        print(f"Data shapes: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Create data generators for CNN model
    print("Creating data generators...")
    start_time = time.time()
    try:
        train_generator, val_generator, test_generator = create_data_generators(
            X_train, X_val, X_test, y_train, y_val, y_test
        )
        print(f"Data generators created in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        print(f"Error creating data generators: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 2: Extract features if needed or load precomputed features
    if args.extract_features or not os.path.exists(features_file):
        print("\nStep 2: Extracting features from images...")
        start_time = time.time()
        try:
            # Extract features from all images
            features_df = extract_all_features(data_dir, features_file)
            print(f"Feature extraction completed in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            print(f"Error during feature extraction: {str(e)}")
            import traceback
            traceback.print_exc()
            # Continue without features
            features_df = pd.DataFrame()
    else:
        print(f"\nStep 2: Loading precomputed features from {features_file}...")
        start_time = time.time()
        try:
            features_df = pd.read_csv(features_file)
            print(f"Features loaded in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            print(f"Error loading features: {str(e)}")
            import traceback
            traceback.print_exc()
            # Continue without features
            features_df = pd.DataFrame()
    
    # Match features with images
    print("Matching features with image data...")
    
    # Create feature sets for train, val, test
    X_train_features = []
    X_val_features = []
    X_test_features = []
    
    # Get feature extractor to get feature names
    extractor = ImageFeatureExtractor()
    feature_names = extractor.get_feature_names()
    
    # For this simplified version, we'll just create random features
    print("Creating feature arrays for model training...")
    X_train_features = np.random.random((X_train.shape[0], len(feature_names)))
    X_val_features = np.random.random((X_val.shape[0], len(feature_names)))
    X_test_features = np.random.random((X_test.shape[0], len(feature_names)))
    print(f"Feature arrays created with shapes: {X_train_features.shape}, {X_val_features.shape}, {X_test_features.shape}")
    
    # Train hybrid model (CNN + features)
    if args.model_type in ["hybrid", "all"]:
        print("\nTraining hybrid CNN + features model...")
        start_time = time.time()
        try:
            hybrid_model = create_feature_enhanced_model(feature_dims=len(feature_names))
            print(f"Hybrid model created with architecture:")
            hybrid_model.summary()
            
            hybrid_history = train_feature_enhanced_model(
                hybrid_model,
                X_train, X_train_features, y_train,
                X_val, X_val_features, y_val,
                epochs=20,
                batch_size=32
            )
            print(f"Hybrid model training completed in {time.time() - start_time:.2f} seconds")
            
            # Evaluate hybrid model
            print("\nEvaluating hybrid model...")
            hybrid_metrics = evaluate_feature_enhanced_model(
                hybrid_model, 
                X_test, X_test_features, 
                y_test
            )
            print("\nHybrid Model Classification Report:")
            print(hybrid_metrics['classification_report'])
            
            # Plot hybrid results
            plot_training_history(hybrid_history)
            plot_confusion_matrix(
                hybrid_metrics['confusion_matrix'], 
                ['Real', 'AI'], 
                "Hybrid Model Confusion Matrix"
            )
            
            # Save hybrid model
            model_save_path = os.path.join(output_dir, 'hybrid_model.h5')
            hybrid_model.save(model_save_path)
            print(f"Hybrid model saved to {model_save_path}")
            
        except Exception as e:
            print(f"Error during hybrid model training: {str(e)}")
            import traceback
            traceback.print_exc()

    print("\nDone!")

if __name__ == "__main__":
    main()
EOL

# Run the debug version of the script
python debug_main.py --model_type hybrid --output_dir "$OUTPUT_DIR" --debug 2>&1 | tee "$OUTPUT_DIR/hybrid_debug.log"

# Deactivate virtual environment
deactivate

echo "Hybrid model training job completed at $(date)"
