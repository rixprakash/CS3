#!/bin/bash
#SBATCH --job-name=cnn_all
#SBATCH --output=cnn_all_%j.out
#SBATCH --error=cnn_all_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

echo "Starting CNN model training job with JSON-based data loading (ALL IMAGES) at $(date)"

# Load modules
module purge
module load python/3.11.4 cuda/12.4.1

# Setup environment
VENV_DIR="$HOME/Project3/venv"
if [ -d "$VENV_DIR" ]; then
    echo "Removing existing venv..."
    rm -rf "$VENV_DIR"
fi
echo "Creating new virtual environment..."
python -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Install packages
pip install --upgrade pip
pip install numpy pandas matplotlib seaborn scikit-learn scikit-image tensorflow torch torchvision tqdm opencv-python psutil

# Set paths
DATA_DIR="$HOME/Project3/DATA/DeepGuardDB_v1"
JSON_FILE="$DATA_DIR/json_files/sd_json.json"
OUTPUT_DIR="$HOME/Project3/model_results/cnn_all"
mkdir -p "$OUTPUT_DIR"

# Debug info
echo "DATA_DIR exists: $([ -d "$DATA_DIR" ] && echo "YES" || echo "NO")"
echo "JSON_FILE exists: $([ -f "$JSON_FILE" ] && echo "YES" || echo "NO")"
echo "Current directory: $(pwd)"
echo "Files in SCRIPTS directory:"
ls -la $HOME/Project3/SCRIPTS/

# Create a modified main.py that uses the JSON-based preprocessing
cat > $HOME/Project3/SCRIPTS/main_json_all.py << 'EOL'
import os
import sys
import argparse
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import psutil
from preprocessing_json import preprocess_data, create_data_generators
from model import (
    create_model, create_feature_enhanced_model, create_feature_only_model,
    train_model, train_feature_enhanced_model, train_feature_only_model,
    evaluate_model, evaluate_feature_enhanced_model, evaluate_feature_only_model,
    plot_training_history, plot_confusion_matrix
)

def print_memory_usage():
    """Print current memory usage of the process"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"Current memory usage: {memory_info.rss / (1024 * 1024 * 1024):.2f} GB")

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate AI image detection models with JSON data")
    parser.add_argument("--data_dir", default=None, help="Path to the image data directory")
    parser.add_argument("--json_file", default=None, help="Path to the JSON file with image metadata")
    parser.add_argument("--model_type", default="cnn", choices=["cnn", "feature", "hybrid"], 
                      help="Type of model to train")
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
        print("\n==== DATASET MEMORY DIAGNOSTICS ====")
        print(f"Total images: {len(X_train) + len(X_val) + len(X_test)}")
        print(f"Image dimensions: {X_train[0].shape}")
        print(f"Data type: {X_train.dtype}")
        print(f"Memory usage per image: {X_train[0].nbytes / (1024 * 1024):.2f} MB")
        print(f"Total dataset memory:")
        print(f"  - Training set: {X_train.nbytes / (1024 * 1024 * 1024):.2f} GB")
        print(f"  - Validation set: {X_val.nbytes / (1024 * 1024 * 1024):.2f} GB")
        print(f"  - Test set: {X_test.nbytes / (1024 * 1024 * 1024):.2f} GB")
        print(f"  - Total: {(X_train.nbytes + X_val.nbytes + X_test.nbytes) / (1024 * 1024 * 1024):.2f} GB")
        print_memory_usage()
        print("====================================\n")
        
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Create data generators for CNN model
    print("Creating data generators...")
    start_time = time.time()
    try:
        batch_size = 32
        train_generator, val_generator, test_generator = create_data_generators(
            X_train, X_val, X_test, y_train, y_val, y_test,
            batch_size=batch_size
        )
        print(f"Data generators created in {time.time() - start_time:.2f} seconds")
        
        # Print batch memory usage
        print(f"Estimated batch memory: {X_train[0].nbytes * batch_size / (1024 * 1024):.2f} MB")
        print_memory_usage()
        
    except Exception as e:
        print(f"Error creating data generators: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Train CNN model
    if args.model_type == "cnn":
        print("\nTraining CNN model...")
        start_time = time.time()
        try:
            cnn_model = create_model()
            print(f"CNN model created with architecture:")
            cnn_model.summary()
            
            # Print memory after model creation
            print("\nMemory usage after model creation:")
            print_memory_usage()
            
            cnn_history = train_model(
                cnn_model,
                train_generator,
                val_generator,
                epochs=20,
                batch_size=batch_size
            )
            print(f"CNN model training completed in {time.time() - start_time:.2f} seconds")
            
            # Evaluate CNN model
            print("\nEvaluating CNN model...")
            cnn_metrics = evaluate_model(cnn_model, test_generator)
            print("\nCNN Classification Report:")
            print(cnn_metrics['classification_report'])
            
            # Save classification report to file
            with open(os.path.join(output_dir, 'cnn_classification_report.txt'), 'w') as f:
                f.write(cnn_metrics['classification_report'])
            
            # Plot CNN results
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(cnn_history.history['accuracy'])
            plt.plot(cnn_history.history['val_accuracy'])
            plt.title('CNN Model Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            
            plt.subplot(1, 2, 2)
            plt.plot(cnn_history.history['loss'])
            plt.plot(cnn_history.history['val_loss'])
            plt.title('CNN Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'cnn_training_history.png'))
            
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            plot_confusion_matrix(
                cnn_metrics['confusion_matrix'], 
                ['Real', 'AI'], 
                "CNN Model Confusion Matrix"
            )
            plt.savefig(os.path.join(output_dir, 'cnn_confusion_matrix.png'))
            
            # Save CNN model
            model_save_path = os.path.join(output_dir, 'cnn_model.h5')
            cnn_model.save(model_save_path)
            print(f"CNN model saved to {model_save_path}")
            
            # Final memory usage
            print("\nFinal memory usage:")
            print_memory_usage()
            
        except Exception as e:
            print(f"Error during CNN model training: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\nDone!")

if __name__ == "__main__":
    main()
EOL

# Run the modified script
echo "Running CNN model training with JSON data (ALL IMAGES)..."
cd $HOME/Project3/SCRIPTS/
python main_json_all.py --data_dir "$DATA_DIR" --json_file "$JSON_FILE" --model_type cnn --output_dir "$OUTPUT_DIR"

# Deactivate venv
deactivate

echo "CNN model training job completed at $(date)" 