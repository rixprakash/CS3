#!/bin/bash
#SBATCH --job-name=hybrid_fix6
#SBATCH --output=hybrid_fix6_%j.out
#SBATCH --error=hybrid_fix6_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

echo "Starting Hybrid model training job with JSON-based data loading (ALL IMAGES) at $(date)"

# Clean module environment and load proper modules
module purge
module load apptainer/1.3.4
module load tensorflow/2.17.0

# Verify GPU availability
echo "Checking GPU availability:"
python -c "import tensorflow as tf; print('GPU Devices:', tf.config.list_physical_devices('GPU')); print('CUDA Visible Devices:', [tf.test.gpu_device_name()])"

# Set environment variables for GPU memory
export TF_FORCE_GPU_ALLOW_GROWTH=true
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/apps/software/standard/core/cuda/12.4.1"

# Set paths
DATA_DIR="$HOME/Project3/DATA/DeepGuardDB_v1"
JSON_FILE="$DATA_DIR/json_files/sd_json.json"
OUTPUT_DIR="$HOME/Project3/model_results/hybrid_fix6"
mkdir -p "$OUTPUT_DIR"

# Debug info
echo "DATA_DIR exists: $([ -d "$DATA_DIR" ] && echo "YES" || echo "NO")"
echo "JSON_FILE exists: $([ -f "$JSON_FILE" ] && echo "YES" || echo "NO")"
echo "Current directory: $(pwd)"
echo "Files in SCRIPTS directory:"
ls -la $HOME/Project3/SCRIPTS/

# Create a modified main.py that uses the JSON-based preprocessing for hybrid model
cat > $HOME/Project3/SCRIPTS/main_hybrid_fix6.py << 'EOL'
import os
import sys
import argparse
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import psutil
from preprocessing_json import preprocess_data, create_data_generators
import tensorflow as tf
from model import (
    create_model, create_feature_enhanced_model, create_feature_only_model,
    train_model, train_feature_enhanced_model, train_feature_only_model,
    evaluate_model, evaluate_feature_enhanced_model, evaluate_feature_only_model,
    plot_training_history, plot_confusion_matrix
)
from extract_features import extract_features_from_data

# Print TensorFlow details for debugging
print(f"TensorFlow version: {tf.__version__}")
print(f"Eager execution enabled: {tf.executing_eagerly()}")
print("GPU Devices:", tf.config.list_physical_devices('GPU'))
print("CUDA Visible Devices:", os.environ.get('CUDA_VISIBLE_DEVICES', 'None'))

# Set memory growth on GPU to avoid OOM errors
physical_devices = tf.config.list_physical_devices('GPU')
print(f"Found {len(physical_devices)} GPU devices")
try:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print("Memory growth enabled for GPUs")
except Exception as e:
    print(f"Failed to set memory growth: {e}")

def print_memory_usage():
    """Print current memory usage of the process"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"Current memory usage: {memory_info.rss / (1024 * 1024 * 1024):.2f} GB")

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate hybrid AI image detection model with JSON data")
    parser.add_argument("--data_dir", default=None, help="Path to the image data directory")
    parser.add_argument("--json_file", default=None, help="Path to the JSON file with image metadata")
    parser.add_argument("--output_dir", default="results", help="Directory to save results")
    parser.add_argument("--max_images", type=int, default=None, help="Maximum number of images to use")
    
    args = parser.parse_args()
    
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get the project root directory (one level up from SCRIPTS)
    project_dir = os.path.dirname(script_dir)
    
    # Set paths
    if args.data_dir is None:
        data_dir = os.path.join(project_dir, 'DATA/DeepGuardDB_v1')
    else:
        data_dir = args.data_dir
        
    # Set up output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Set the JSON file path
    json_path = args.json_file
    
    print(f"Data directory: {data_dir}")
    print(f"JSON file: {json_path}")
    print(f"Output directory: {output_dir}")
    
    # Debug: Show directory contents
    print(f"Data directory contents:")
    for root, dirs, files in os.walk(data_dir, topdown=True):
        print(f"Directory: {root}")
        print(f"  Subdirectories: {dirs}")
        # Only process the top level directory
        break
    
    # Step 1: Preprocess image data using the JSON-based approach
    print("\nStep 1: Preprocessing image data with JSON metadata...")
    start_time = time.time()
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(
            data_dir=data_dir,
            json_path=json_path,
            test_size=0.2,
            val_size=0.2,
            max_images=args.max_images
        )
        print(f"Preprocessing completed in {time.time() - start_time:.2f} seconds")
        print(f"Data shapes: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")
        
        # Print memory usage diagnostics
        print_memory_usage()
        
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Create data generators for CNN model
    print("Creating data generators...")
    start_time = time.time()
    try:
        batch_size = 16  # Reduced batch size to conserve memory
        train_generator, val_generator, test_generator = create_data_generators(
            X_train, X_val, X_test, y_train, y_val, y_test,
            batch_size=batch_size
        )
        print(f"Data generators created in {time.time() - start_time:.2f} seconds")
        print_memory_usage()
        
    except Exception as e:
        print(f"Error creating data generators: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Extract features for hybrid model
    print("\nExtracting features for hybrid model...")
    start_time = time.time()
    try:
        # Use smaller batches for feature extraction to avoid memory issues
        train_features = extract_features_from_data(X_train, batch_size=8)
        val_features = extract_features_from_data(X_val, batch_size=8)
        test_features = extract_features_from_data(X_test, batch_size=8)
        print(f"Feature extraction completed in {time.time() - start_time:.2f} seconds")
        print(f"Feature shapes: Train={train_features.shape}, Val={val_features.shape}, Test={test_features.shape}")
        print_memory_usage()
        
    except Exception as e:
        print(f"Error during feature extraction: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Train hybrid model
    print("\nTraining Hybrid model...")
    start_time = time.time()
    try:
        # Create the hybrid model with smaller batch size to conserve memory
        batch_size = 8  # Further reduced batch size for training
        print(f"Creating hybrid model with batch size {batch_size}...")
        
        # Create the model with smaller dense layers to reduce parameters
        hybrid_model = create_feature_enhanced_model(
            train_features.shape[1],
            dense_units=[64, 32]  # Reduced from default settings
        )
        print(f"Hybrid model created with architecture:")
        hybrid_model.summary()
        
        # Print memory after model creation
        print_memory_usage()
        
        # Train the model with early stopping and reduced epochs
        hybrid_history = train_feature_enhanced_model(
            hybrid_model,
            train_generator,
            val_generator,
            train_features,
            val_features,
            epochs=10,  # Reduced from 15
            batch_size=batch_size,
            patience=3  # Early stopping patience
        )
        print(f"Hybrid model training completed in {time.time() - start_time:.2f} seconds")
        
        # Evaluate hybrid model with smaller batch size
        print("\nEvaluating Hybrid model...")
        hybrid_metrics = evaluate_feature_enhanced_model(
            hybrid_model, 
            test_generator,
            test_features,
            batch_size=batch_size
        )
        print("\nHybrid Classification Report:")
        print(hybrid_metrics['classification_report'])
        
        # Save classification report to file
        with open(os.path.join(output_dir, 'hybrid_classification_report.txt'), 'w') as f:
            f.write(hybrid_metrics['classification_report'])
        
        # Plot Hybrid results
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(hybrid_history.history['accuracy'])
        plt.plot(hybrid_history.history['val_accuracy'])
        plt.title('Hybrid Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.subplot(1, 2, 2)
        plt.plot(hybrid_history.history['loss'])
        plt.plot(hybrid_history.history['val_loss'])
        plt.title('Hybrid Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'hybrid_training_history.png'))
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        plot_confusion_matrix(
            hybrid_metrics['confusion_matrix'], 
            ['Real', 'AI'], 
            "Hybrid Model Confusion Matrix"
        )
        plt.savefig(os.path.join(output_dir, 'hybrid_confusion_matrix.png'))
        
        # Save Hybrid model
        model_save_path = os.path.join(output_dir, 'hybrid_model.h5')
        hybrid_model.save(model_save_path)
        print(f"Hybrid model saved to {model_save_path}")
        
        # Final memory usage
        print_memory_usage()
        
    except Exception as e:
        print(f"Error during Hybrid model training: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\nDone!")

if __name__ == "__main__":
    main()
EOL

# Run the modified script
echo "Running Hybrid model training with JSON data (ALL IMAGES)..."
cd $HOME/Project3/SCRIPTS/
python main_hybrid_fix6.py --data_dir "$DATA_DIR" --json_file "$JSON_FILE" --output_dir "$OUTPUT_DIR"

echo "Hybrid model training job completed at $(date)" 