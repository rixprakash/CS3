#!/bin/bash
#SBATCH --job-name=cnn_updated
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Load required modules
module purge
module load gcc/11.4.0
module load openmpi/4.1.4
module load python/3.11.4
module load cuda/12.2.2

# Create virtual environment
VENV_DIR="$HOME/py3_11_env"
python -m venv $VENV_DIR
source $VENV_DIR/bin/activate

# Install required packages
pip install --upgrade pip
pip install numpy pandas matplotlib seaborn scikit-learn scikit-image psutil
pip install tensorflow==2.15.0  # Newer version compatible with Python 3.11 and CUDA
pip install torch torchvision
pip install opencv-python

# Define paths
DATA_DIR="$HOME/Project3/DATA/DeepGuardDB_v1"
JSON_PATH="$DATA_DIR/json_files/sd_json.json"  # Updated JSON path
OUTPUT_DIR="$HOME/Project3/model_results/cnn_updated"
mkdir -p $OUTPUT_DIR

# Debugging information
echo "Checking if data directory exists:"
ls -la $DATA_DIR
echo "Checking if JSON file exists:"
ls -la $JSON_PATH
echo "Checking all available JSON files:"
find $DATA_DIR -name "*.json"

# Create the Python script for the CNN model
cat > main_cnn_updated.py << 'EOF'
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import cv2
import time
import psutil
import warnings
import gc
warnings.filterwarnings('ignore')

print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Memory monitoring function
def print_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"Memory usage: {memory_info.rss / (1024 * 1024):.2f} MB")

def preprocess_data(data_dir, json_path, test_size=0.15, val_size=0.15, target_size=(224, 224), max_images=None):
    """
    Preprocess data and split into train, validation, and test sets.
    
    Args:
        data_dir: Directory containing image folders
        json_path: Path to the json file with image metadata
        test_size: Proportion of data to use for testing
        val_size: Proportion of data to use for validation
        target_size: Size to resize images to
        max_images: Maximum number of images to use per class (for debugging)
        
    Returns:
        train_images, train_labels, val_images, val_labels, test_images, test_labels
    """
    print(f"Loading data from {data_dir}")
    print(f"Using metadata from {json_path}")
    
    # Check if JSON file exists
    if os.path.exists(json_path):
        print(f"Loading JSON data from {json_path}")
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        
        # Get image paths and labels from JSON
        real_folder = os.path.join(data_dir, "SD_dataset", "real")
        fake_folder = os.path.join(data_dir, "SD_dataset", "fake")
        
        images = []
        labels = []
        
        # Process real images
        real_count = 0
        for img_file in os.listdir(real_folder):
            if img_file.endswith('.jpg') or img_file.endswith('.png'):
                img_path = os.path.join(real_folder, img_file)
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Warning: Could not load image {img_path}")
                        continue
                    img = cv2.resize(img, target_size)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = img / 255.0  # Normalize to [0,1]
                    images.append(img)
                    labels.append(0)  # 0 for real
                    real_count += 1
                    if max_images and real_count >= max_images:
                        break
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
        # Process fake images
        fake_count = 0
        for img_file in os.listdir(fake_folder):
            if img_file.endswith('.jpg') or img_file.endswith('.png'):
                img_path = os.path.join(fake_folder, img_file)
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Warning: Could not load image {img_path}")
                        continue
                    img = cv2.resize(img, target_size)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = img / 255.0  # Normalize to [0,1]
                    images.append(img)
                    labels.append(1)  # 1 for fake
                    fake_count += 1
                    if max_images and fake_count >= max_images:
                        break
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
        print(f"Loaded {real_count} real images and {fake_count} fake images")
        
        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)
        
        # Split into train, validation, and test sets
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, 
                                                       test_size=val_size/(1-test_size), 
                                                       stratify=y_train_val, 
                                                       random_state=42)
        
        print(f"Train set: {X_train.shape[0]} images")
        print(f"Validation set: {X_val.shape[0]} images")
        print(f"Test set: {X_test.shape[0]} images")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    else:
        print(f"JSON file {json_path} not found")
        return None, None, None, None, None, None

def create_data_generators(X_train, y_train, X_val, y_val, batch_size=32):
    """
    Create data generators for training and validation sets.
    
    Args:
        X_train: Training images
        y_train: Training labels
        X_val: Validation images
        y_val: Validation labels
        batch_size: Batch size
        
    Returns:
        train_generator, val_generator
    """
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # No augmentation for validation
    val_datagen = ImageDataGenerator()
    
    train_generator = train_datagen.flow(
        X_train,
        y_train,
        batch_size=batch_size
    )
    
    val_generator = val_datagen.flow(
        X_val,
        y_val,
        batch_size=batch_size
    )
    
    return train_generator, val_generator

def create_model(input_shape=(224, 224, 3)):
    """
    Create a CNN model based on EfficientNetB0.
    
    Args:
        input_shape: Input shape of the model
        
    Returns:
        model: Compiled model
    """
    # Base model (EfficientNetB0)
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    # Create model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def train_model(model, train_generator, val_generator, output_dir, epochs=30, steps_per_epoch=None, validation_steps=None):
    """
    Train the model.
    
    Args:
        model: Model to train
        train_generator: Training data generator
        val_generator: Validation data generator
        output_dir: Directory to save model and results
        epochs: Number of epochs to train
        steps_per_epoch: Number of steps per epoch (default: None, use all)
        validation_steps: Number of validation steps (default: None, use all)
        
    Returns:
        history: Training history
    """
    # Create callbacks
    checkpoint = ModelCheckpoint(
        os.path.join(output_dir, 'best_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        mode='max',
        verbose=1
    )
    
    # Train model
    start_time = time.time()
    
    history = model.fit(
        train_generator,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=[checkpoint, early_stopping],
        verbose=1
    )
    
    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds")
    
    return history

def evaluate_model(model, X_test, y_test, output_dir):
    """
    Evaluate the model on the test set.
    
    Args:
        model: Trained model
        X_test: Test images
        y_test: Test labels
        output_dir: Directory to save results
        
    Returns:
        results: Evaluation results
    """
    # Evaluate model
    print("Evaluating model on test set...")
    results = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test loss: {results[0]:.4f}")
    print(f"Test accuracy: {results[1]:.4f}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_classes = (y_pred > 0.5).astype(int)
    
    # Classification report
    print("Classification Report:")
    report = classification_report(y_test, y_pred_classes)
    print(report)
    
    # Save classification report
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    return results

def plot_training_history(history, output_dir):
    """
    Plot training history.
    
    Args:
        history: Training history
        output_dir: Directory to save plots
    """
    # Plot accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train a CNN model for image classification')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing image data')
    parser.add_argument('--json_path', type=str, required=True, help='Path to JSON file with metadata')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save model and results')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')
    parser.add_argument('--max_images', type=int, default=None, help='Maximum number of images to use per class (for debugging)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print GPU information
    print("GPU Information:")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            print(f"  {gpu}")
    else:
        print("  No GPUs found")
    
    # Set memory growth
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except:
            pass
    
    # Preprocess data
    print("Preprocessing data...")
    print_memory_usage()
    
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(
        args.data_dir, 
        args.json_path, 
        max_images=args.max_images
    )
    
    if X_train is None:
        print("Error: Could not preprocess data. Exiting.")
        return
    
    # Create data generators
    print("Creating data generators...")
    train_generator, val_generator = create_data_generators(
        X_train, 
        y_train, 
        X_val, 
        y_val, 
        batch_size=args.batch_size
    )
    
    # Create model
    print("Creating model...")
    print_memory_usage()
    
    model = create_model(input_shape=X_train.shape[1:])
    model.summary()
    
    # Train model
    print("Training model...")
    print_memory_usage()
    
    history = train_model(
        model, 
        train_generator, 
        val_generator, 
        args.output_dir, 
        epochs=args.epochs
    )
    
    # Evaluate model
    print("Evaluating model...")
    print_memory_usage()
    
    evaluate_model(model, X_test, y_test, args.output_dir)
    
    # Plot training history
    print("Plotting training history...")
    plot_training_history(history, args.output_dir)
    
    # Save model
    print("Saving model...")
    model.save(os.path.join(args.output_dir, 'final_model.h5'))
    
    print("Done!")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
EOF

# Run the script with the appropriate arguments
python main_cnn_updated.py --data_dir $DATA_DIR --json_path $JSON_PATH --output_dir $OUTPUT_DIR

# Clean up
deactivate
echo "CNN model training complete!" 