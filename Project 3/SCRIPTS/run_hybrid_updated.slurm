#!/bin/bash
#SBATCH --job-name=hybrid_updated
#SBATCH --output=hybrid_updated_%j.out
#SBATCH --error=hybrid_updated_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

echo "Starting Hybrid model training job with JSON-based data loading at $(date)"

# Load required modules
module purge
module load gcc/11.4.0
module load openmpi/4.1.4
module load python/3.11.4
module load cuda/12.2.2

echo "Loaded modules:"
module list

# Create virtual environment
VENV_DIR="$HOME/py3_11_hybrid_env"
python -m venv $VENV_DIR
source $VENV_DIR/bin/activate

# Install required packages
pip install --upgrade pip
pip install numpy pandas matplotlib seaborn scikit-learn scikit-image psutil
pip install tensorflow==2.15.0  # Newer version compatible with Python 3.11 and CUDA
pip install torch torchvision tqdm
pip install opencv-python

# Verify Python and library versions
python --version
pip list | grep -E "tensorflow|torch|numpy|pandas"

# Print TensorFlow and CUDA info
python -c "import tensorflow as tf; print('TensorFlow Version:', tf.__version__); print('GPU Available:', len(tf.config.list_physical_devices('GPU')) > 0); print('Built with CUDA:', tf.test.is_built_with_cuda())"

# Set paths
DATA_DIR="$HOME/Project3/DATA/DeepGuardDB_v1"
JSON_FILE="$DATA_DIR/json_files/sd_json.json"
OUTPUT_DIR="$HOME/Project3/model_results/hybrid_updated"
mkdir -p "$OUTPUT_DIR"

# Debug info
echo "DATA_DIR exists: $([ -d "$DATA_DIR" ] && echo "YES" || echo "NO")"
echo "JSON_FILE exists: $([ -f "$JSON_FILE" ] && echo "YES" || echo "NO")"
echo "Checking all available JSON files:"
find $DATA_DIR -name "*.json"
echo "Current directory: $(pwd)"
echo "Files in SCRIPTS directory:"
ls -la $HOME/Project3/SCRIPTS/

# Create a modified main.py that uses the JSON-based preprocessing for hybrid model
cat > $HOME/Project3/SCRIPTS/main_hybrid_updated.py << 'EOL'
import os
import sys
import argparse
import numpy as np
import time
import matplotlib.pyplot as plt
import psutil
import tensorflow as tf
from preprocessing_json import preprocess_data, create_data_generators
from feature_extraction import ImageFeatureExtractor
from model import (
    create_feature_enhanced_model, train_feature_enhanced_model,
    evaluate_model, plot_training_history, plot_confusion_matrix
)

# Configure GPU memory growth to prevent OOM errors
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Set TensorFlow to only use a portion of GPU memory initially
        # and grow as needed
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Memory growth enabled for {len(gpus)} GPUs")
        
        # Limit TensorFlow to only use the first GPU
        if len(gpus) > 1:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            print(f"Limited TensorFlow to first GPU only")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

# Print TensorFlow details for debugging
print(f"TensorFlow version: {tf.__version__}")
print(f"Eager execution enabled: {tf.executing_eagerly()}")
print("GPU Devices:", tf.config.list_physical_devices('GPU'))
print("CUDA Visible Devices:", os.environ.get('CUDA_VISIBLE_DEVICES', 'None'))

def print_memory_usage():
    """Print current memory usage of the process"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"Current memory usage: {memory_info.rss / (1024 * 1024 * 1024):.2f} GB")

# Function to extract features from preprocessed images using the feature extractor
def extract_features(images):
    """Extract features from a batch of preprocessed images"""
    print(f"Extracting features from {len(images)} images")
    # Initialize feature extractor
    extractor = ImageFeatureExtractor()
    
    # Extract features from each image
    features_list = []
    for i, img in enumerate(images):
        if i % 100 == 0:
            print(f"Processing image {i}/{len(images)}")
        
        # Convert normalized float image (0-1) back to uint8 (0-255) for feature extraction
        img_uint8 = (img * 255).astype(np.uint8)
        features = extractor.extract_features(img_uint8)
        features_list.append(features)
        
    # Convert to numpy array
    features_array = np.array(features_list)
    print(f"Feature shape: {features_array.shape}")
    return features_array

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate hybrid image detection model")
    parser.add_argument("--data_dir", default=None, help="Path to the image data directory")
    parser.add_argument("--json_file", default=None, help="Path to the JSON file with image metadata")
    parser.add_argument("--output_dir", default="results", help="Directory to save results")
    parser.add_argument("--max_images", type=int, default=None, help="Maximum number of images to use")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    
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
        # Use float32 instead of float64 to reduce memory usage
        X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(
            data_dir=data_dir,
            json_path=json_path,
            test_size=0.2,
            val_size=0.2,
            max_images=args.max_images,
            dtype='float32'  # Use float32 instead of default float64
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
    
    # Step 2: Extract features for the hybrid model
    print("\nStep 2: Extracting image features for hybrid model...")
    start_time = time.time()
    try:
        # Extract features for training, validation, and test sets
        print("Extracting features from training set...")
        X_train_features = extract_features(X_train)
        
        print("Extracting features from validation set...")
        X_val_features = extract_features(X_val)
        
        print("Extracting features from test set...")
        X_test_features = extract_features(X_test)
        
        print(f"Feature extraction completed in {time.time() - start_time:.2f} seconds")
        print(f"Feature shapes: X_train_features={X_train_features.shape}, X_val_features={X_val_features.shape}, X_test_features={X_test_features.shape}")
        
        # Print memory usage after feature extraction
        print("\nMemory usage after feature extraction:")
        print_memory_usage()
        
    except Exception as e:
        print(f"Error during feature extraction: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 3: Create data generators for CNN part
    print("\nStep 3: Creating data generators...")
    try:
        # Use a smaller batch size for hybrid model due to increased memory requirements
        batch_size = args.batch_size
        print(f"Using batch size: {batch_size}")
        
        # Create data generators for training and validation
        train_generator, val_generator = create_data_generators(
            X_train, X_val, X_test, y_train, y_val, y_test,
            batch_size=batch_size
        )
        
        print(f"Created data generators with batch size {batch_size}")
        
    except Exception as e:
        print(f"Error creating data generators: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 4: Create and train the hybrid model
    print("\nStep 4: Creating and training hybrid model...")
    try:
        start_time = time.time()
        
        # Create the hybrid model
        print("Creating hybrid model...")
        model = create_feature_enhanced_model(
            input_shape=X_train.shape[1:],
            feature_input_shape=X_train_features.shape[1:],
            num_classes=1
        )
        
        # Summary of the model architecture
        print("Hybrid Model Summary:")
        model.summary()
        
        # Train the hybrid model
        print("\nTraining hybrid model...")
        history = train_feature_enhanced_model(
            model=model,
            train_generator=train_generator,
            val_generator=val_generator,
            X_train_features=X_train_features,
            X_val_features=X_val_features,
            output_dir=output_dir,
            epochs=args.epochs
        )
        
        print(f"Model training completed in {time.time() - start_time:.2f} seconds")
        
        # Print memory usage after training
        print("\nMemory usage after training:")
        print_memory_usage()
        
    except Exception as e:
        print(f"Error during model creation or training: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 5: Evaluate the model on test set
    print("\nStep 5: Evaluating model on test set...")
    try:
        # Evaluate the model
        results = evaluate_model(
            model=model,
            test_data=(X_test, y_test),
            test_features=X_test_features,
            output_dir=output_dir
        )
        
        # Plot training history
        plot_training_history(history, output_dir)
        
        # Plot confusion matrix
        y_pred = model.predict([X_test, X_test_features])
        y_pred_classes = (y_pred > 0.5).astype(int)
        plot_confusion_matrix(y_test, y_pred_classes, output_dir)
        
        print("Evaluation results:")
        print(f"  Test loss: {results[0]:.4f}")
        print(f"  Test accuracy: {results[1]:.4f}")
        
    except Exception as e:
        print(f"Error during model evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\nDone! Results saved to:", output_dir)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Unhandled exception: {str(e)}")
        import traceback
        traceback.print_exc()
EOL

# Run the Python script
echo "Running hybrid model training..."
python $HOME/Project3/SCRIPTS/main_hybrid_updated.py \
  --data_dir $DATA_DIR \
  --json_file $JSON_FILE \
  --output_dir $OUTPUT_DIR \
  --batch_size 16 \
  --epochs 15

# Clean up
deactivate
echo "Hybrid model training completed at $(date)" 