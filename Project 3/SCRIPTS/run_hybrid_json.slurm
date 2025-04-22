#!/bin/bash
#SBATCH --job-name=hybrid_json
#SBATCH --output=hybrid_json_%j.out
#SBATCH --error=hybrid_json_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=bii_dsc_community

echo "Starting hybrid model training job with JSON-based data loading at $(date)"

# Load modules
module purge
module load miniconda3/24.3.0-py3.11 cuda/12.4.1

# Setup environment
VENV_DIR="$HOME/Project3/venv"
if [ -d "$VENV_DIR" ]; then
    rm -rf "$VENV_DIR"
fi
python -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Install packages
pip install --upgrade pip
pip install numpy pandas matplotlib seaborn scikit-learn scikit-image tensorflow torch torchvision tqdm opencv-python

# Set paths
DATA_DIR="$HOME/Project3/DATA/DeepGuardDB_v1"
JSON_FILE="$DATA_DIR/json_files/sd_json.json"
FEATURES_FILE="$HOME/Project3/feature_extraction_results/image_features.csv"
OUTPUT_DIR="$HOME/Project3/model_results/hybrid_json"
mkdir -p "$OUTPUT_DIR"

# Debug info
echo "DATA_DIR exists: $([ -d "$DATA_DIR" ] && echo "YES" || echo "NO")"
echo "JSON_FILE exists: $([ -f "$JSON_FILE" ] && echo "YES" || echo "NO")"
echo "FEATURES_FILE exists: $([ -f "$FEATURES_FILE" ] && echo "YES" || echo "NO")"
echo "Current directory: $(pwd)"
echo "Files in SCRIPTS directory:"
ls -la $HOME/Project3/SCRIPTS/

# Make sure we have the preprocessing_json.py file
if [ ! -f "$HOME/Project3/SCRIPTS/preprocessing_json.py" ]; then
    echo "preprocessing_json.py not found, exiting"
    exit 1
fi

# Create a modified main.py that uses the JSON-based preprocessing for hybrid model
cat > $HOME/Project3/SCRIPTS/main_hybrid_json.py << 'EOL'
import os
import sys
import argparse
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from preprocessing_json import preprocess_data, create_data_generators
from model import (
    create_model, create_feature_enhanced_model, create_feature_only_model,
    train_model, train_feature_enhanced_model, train_feature_only_model,
    evaluate_model, evaluate_feature_enhanced_model, evaluate_feature_only_model,
    plot_training_history, plot_confusion_matrix
)
from feature_extraction import ImageFeatureExtractor

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate hybrid AI image detection model with JSON data")
    parser.add_argument("--data_dir", default=None, help="Path to the image data directory")
    parser.add_argument("--json_file", default=None, help="Path to the JSON file with image metadata")
    parser.add_argument("--features_file", default=None, help="Path to the features CSV file")
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
    features_file = args.features_file
    
    print(f"Data directory: {data_dir}")
    print(f"JSON file: {json_path}")
    print(f"Features file: {features_file}")
    print(f"Output directory: {output_dir}")
    
    # Debug: Show directory contents
    print(f"Data directory contents:")
    for root, dirs, files in os.walk(data_dir, topdown=True, maxdepth=1):
        print(f"Directory: {root}")
        print(f"  Subdirectories: {dirs}")
    
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
    
    # Step 2: Generate or load features
    print("\nStep 2: Generating dummy features (since we're not using actual extracted features)...")
    # Get feature extractor to get feature names
    extractor = ImageFeatureExtractor()
    feature_names = extractor.get_feature_names()
    
    # For this simplified version, we'll just create random features
    print(f"Creating random features with {len(feature_names)} dimensions")
    X_train_features = np.random.random((X_train.shape[0], len(feature_names)))
    X_val_features = np.random.random((X_val.shape[0], len(feature_names)))
    X_test_features = np.random.random((X_test.shape[0], len(feature_names)))
    print(f"Feature arrays created with shapes: {X_train_features.shape}, {X_val_features.shape}, {X_test_features.shape}")
    
    # Train hybrid model (CNN + features)
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
        
        # Save classification report to file
        with open(os.path.join(output_dir, 'hybrid_classification_report.txt'), 'w') as f:
            f.write(hybrid_metrics['classification_report'])
        
        # Plot hybrid results
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

# Run the modified script
echo "Running hybrid model training with JSON data..."
cd $HOME/Project3/SCRIPTS/
python main_hybrid_json.py --data_dir "$DATA_DIR" --json_file "$JSON_FILE" --features_file "$FEATURES_FILE" --output_dir "$OUTPUT_DIR" --max_images 1000

# Deactivate venv
deactivate

echo "Hybrid model training job completed at $(date)" 