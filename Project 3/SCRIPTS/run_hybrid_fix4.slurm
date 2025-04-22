#!/bin/bash
#SBATCH --job-name=hybrid_fix4
#SBATCH --output=hybrid_fix4_%j.out
#SBATCH --error=hybrid_fix4_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

echo "Starting Hybrid model training job with JSON-based data loading (ALL IMAGES) at $(date)"

# Load modules
module purge
module load gcc/11.4.0
module load cuda/11.8.0
module load bioconda/py3.10

echo "Loaded modules:"
module list

# Create a new conda environment
CONDA_ENV_NAME="tf_gpu_env_hybrid"
echo "Creating conda environment $CONDA_ENV_NAME..."
conda create -y -n $CONDA_ENV_NAME python=3.10
source activate $CONDA_ENV_NAME

# Verify Python and conda versions
python --version
conda --version

# Install packages with specific versions known to work with CUDA 11.8
echo "Installing required packages..."
conda install -y numpy pandas matplotlib seaborn scikit-learn scikit-image psutil
conda install -y -c conda-forge tensorflow=2.9.0 cudatoolkit=11.8 cudnn
conda install -y -c pytorch pytorch torchvision tqdm
conda install -y -c conda-forge opencv

# Print TensorFlow and CUDA info
python -c "import tensorflow as tf; print('TF Version:', tf.__version__); print('GPU Available:', len(tf.config.list_physical_devices('GPU')) > 0); print('Built with CUDA:', tf.test.is_built_with_cuda())"

# Set paths
DATA_DIR="$HOME/Project3/DATA/DeepGuardDB_v1"
JSON_FILE="$DATA_DIR/json_files/sd_json.json"
OUTPUT_DIR="$HOME/Project3/model_results/hybrid_fix4"
mkdir -p "$OUTPUT_DIR"

# Debug info
echo "DATA_DIR exists: $([ -d "$DATA_DIR" ] && echo "YES" || echo "NO")"
echo "JSON_FILE exists: $([ -f "$JSON_FILE" ] && echo "YES" || echo "NO")"
echo "Current directory: $(pwd)"
echo "Files in SCRIPTS directory:"
ls -la $HOME/Project3/SCRIPTS/

# Create a modified main.py that uses the JSON-based preprocessing for hybrid model
cat > $HOME/Project3/SCRIPTS/main_hybrid_fix4.py << 'EOL'
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
    
    # Create data generators for hybrid model
    print("\nCreating data generators for hybrid model...")
    start_time = time.time()
    try:
        batch_size = args.batch_size
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
    
    # Step 3: Train hybrid model
    print("\nStep 3: Training hybrid model...")
    start_time = time.time()
    try:
        hybrid_model = create_feature_enhanced_model(X_train_features.shape[1])
        print(f"Hybrid model created with architecture:")
        hybrid_model.summary()
        
        # Print memory after model creation
        print("\nMemory usage after model creation:")
        print_memory_usage()
        
        hybrid_history = train_feature_enhanced_model(
            hybrid_model,
            train_generator,
            val_generator,
            X_train_features,
            X_val_features,
            y_train,
            y_val,
            epochs=args.epochs,
            batch_size=batch_size
        )
        print(f"Hybrid model training completed in {time.time() - start_time:.2f} seconds")
        
        # Step 4: Evaluate hybrid model
        print("\nStep 4: Evaluating hybrid model...")
        # Prepare test data with features for evaluation
        test_img_batch = np.asarray([x for x, _ in test_generator])
        test_label_batch = np.asarray([y for _, y in test_generator])
        
        # Use a small batch size for prediction to avoid memory issues
        pred_batch_size = 16
        num_samples = len(test_img_batch)
        
        # Initialize predictions array
        preds = np.zeros((num_samples, 2))
        
        # Make predictions in batches
        for i in range(0, num_samples, pred_batch_size):
            end_idx = min(i + pred_batch_size, num_samples)
            batch_images = test_img_batch[i:end_idx]
            batch_features = X_test_features[i:end_idx]
            batch_preds = hybrid_model.predict([batch_images, batch_features])
            preds[i:end_idx] = batch_preds
        
        # Get predicted classes
        predicted_classes = np.argmax(preds, axis=1)
        true_classes = np.argmax(test_label_batch, axis=1)
        
        # Calculate metrics
        from sklearn.metrics import classification_report, confusion_matrix
        hybrid_report = classification_report(true_classes, predicted_classes, target_names=['Real', 'AI'])
        hybrid_cm = confusion_matrix(true_classes, predicted_classes)
        
        print("\nHybrid Model Classification Report:")
        print(hybrid_report)
        
        # Save classification report to file
        with open(os.path.join(output_dir, 'hybrid_classification_report.txt'), 'w') as f:
            f.write(hybrid_report)
        
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
            hybrid_cm, 
            ['Real', 'AI'], 
            "Hybrid Model Confusion Matrix"
        )
        plt.savefig(os.path.join(output_dir, 'hybrid_confusion_matrix.png'))
        
        # Save hybrid model
        model_save_path = os.path.join(output_dir, 'hybrid_model.h5')
        hybrid_model.save(model_save_path)
        print(f"Hybrid model saved to {model_save_path}")
        
        # Final memory usage
        print("\nFinal memory usage:")
        print_memory_usage()
        
    except Exception as e:
        print(f"Error during hybrid model training: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\nDone!")

if __name__ == "__main__":
    main()
EOL

# Run the modified script
echo "Running hybrid model training with fixed TensorFlow-CUDA compatibility..."
cd $HOME/Project3/SCRIPTS/
python main_hybrid_fix4.py --data_dir "$DATA_DIR" --json_file "$JSON_FILE" --output_dir "$OUTPUT_DIR" --batch_size 16 --epochs 15

# Deactivate conda environment
conda deactivate

echo "Hybrid model training job completed at $(date)" 