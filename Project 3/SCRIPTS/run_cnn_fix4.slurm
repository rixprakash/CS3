#!/bin/bash
#SBATCH --job-name=cnn_fix4
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
module load cuda/12.2.2
module load bioconda

# Create a conda environment for TensorFlow with GPU support
CONDA_ENV_NAME="tf_gpu_env_cnn"
conda create -y -n $CONDA_ENV_NAME python=3.10
source activate $CONDA_ENV_NAME

# Install required packages
conda install -y numpy pandas matplotlib seaborn scikit-learn scikit-image psutil
conda install -y -c conda-forge tensorflow=2.9.0 cudatoolkit=11.8 cudnn
conda install -y -c pytorch pytorch torchvision
conda install -y -c conda-forge opencv

# Define paths
DATA_DIR="$HOME/Project3/DATA/DeepGuardDB_v1"
JSON_PATH="$DATA_DIR/SD_dataset/SD_dataset.json"
OUTPUT_DIR="$HOME/Project3/model_results/cnn_fix4"
mkdir -p $OUTPUT_DIR

# Debugging information
echo "Checking if data directory exists:"
ls -la $DATA_DIR
echo "Checking if JSON file exists:"
ls -la $JSON_PATH

# Create the Python script for the CNN model
cat > main_cnn_fix4.py << 'EOF'
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

def create_cnn_model(input_shape=(224, 224, 3)):
    """
    Create a CNN model using EfficientNetB0 as a base model.
    
    Args:
        input_shape: Input shape of the images
        
    Returns:
        model
    """
    # Base model
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    # Combine base model and custom layers
    model = Model(inputs=base_model.input, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

def train_model(model, train_generator, val_generator, epochs=20, output_dir=None):
    """
    Train the model with early stopping and model checkpointing.
    
    Args:
        model: The model to train
        train_generator: Training data generator
        val_generator: Validation data generator
        epochs: Maximum number of epochs to train for
        output_dir: Directory to save model and plots
        
    Returns:
        history: Training history
        best_model: Best model from training
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, 'best_model.h5')
    else:
        model_path = 'best_model.h5'
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    model_checkpoint = ModelCheckpoint(
        model_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    # Train the model
    print("\nTraining the model...")
    print_memory_usage()
    
    steps_per_epoch = len(train_generator)
    validation_steps = len(val_generator)
    
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    
    # Load best model from checkpoint
    if os.path.exists(model_path):
        best_model = tf.keras.models.load_model(model_path)
    else:
        best_model = model
    
    return history, best_model

def evaluate_model(model, X_test, y_test, output_dir=None):
    """
    Evaluate the model on the test set.
    
    Args:
        model: Trained model
        X_test: Test images
        y_test: Test labels
        output_dir: Directory to save evaluation results
        
    Returns:
        test_loss, test_acc, test_auc, test_precision, test_recall
    """
    print("\nEvaluating the model on the test set...")
    test_results = model.evaluate(X_test, y_test, verbose=1)
    
    # Get predictions
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Print classification report
    print("\nClassification Report:")
    class_report = classification_report(y_test, y_pred)
    print(class_report)
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    
    plt.close()
    
    # Save classification report
    if output_dir:
        with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
            f.write(class_report)
    
    return test_results

def plot_training_history(history, output_dir=None):
    """
    Plot training and validation accuracy and loss.
    
    Args:
        history: Training history
        output_dir: Directory to save plots
    """
    # Plot training & validation accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'training_history.png'))
    
    plt.close()

def main(data_dir, json_path, output_dir):
    """
    Main function to preprocess data, create model, train, and evaluate.
    
    Args:
        data_dir: Directory containing image folders
        json_path: Path to the json file with image metadata
        output_dir: Directory to save model and results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Start timing
    start_time = time.time()
    
    # Preprocess data
    print("\nPreprocessing data...")
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(
        data_dir=data_dir, 
        json_path=json_path,
        test_size=0.15,
        val_size=0.15,
        target_size=(224, 224)
    )
    
    if X_train is None:
        print("Error preprocessing data. Exiting.")
        return
    
    # Create data generators
    print("\nCreating data generators...")
    batch_size = 16  # Reduced batch size to conserve memory
    train_generator, val_generator = create_data_generators(X_train, y_train, X_val, y_val, batch_size=batch_size)
    
    # Create model
    print("\nCreating CNN model...")
    model = create_cnn_model(input_shape=(224, 224, 3))
    model.summary()
    
    # Train model
    print("\nTraining model...")
    history, best_model = train_model(
        model=model,
        train_generator=train_generator,
        val_generator=val_generator,
        epochs=20,
        output_dir=output_dir
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    test_results = evaluate_model(best_model, X_test, y_test, output_dir=output_dir)
    
    # Plot training history
    print("\nPlotting training history...")
    plot_training_history(history, output_dir=output_dir)
    
    # Print results
    print("\nTest Results:")
    metric_names = ['loss', 'accuracy', 'auc', 'precision', 'recall']
    for i, metric in enumerate(metric_names):
        print(f"{metric}: {test_results[i]:.4f}")
    
    # Save model
    best_model.save(os.path.join(output_dir, 'cnn_model.h5'))
    
    # End timing
    end_time = time.time()
    print(f"\nTotal runtime: {(end_time - start_time) / 60:.2f} minutes")

# Run the main function
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train CNN model for AI image detection')
    parser.add_argument('--data_dir', type=str, help='Directory containing image folders')
    parser.add_argument('--json_path', type=str, help='Path to the json file with image metadata')
    parser.add_argument('--output_dir', type=str, help='Directory to save model and results')
    
    args = parser.parse_args()
    
    main(
        data_dir=args.data_dir,
        json_path=args.json_path,
        output_dir=args.output_dir
    )
EOF

# Run the Python script
echo "Running CNN model training..."
python main_cnn_fix4.py --data_dir "$DATA_DIR" --json_path "$JSON_PATH" --output_dir "$OUTPUT_DIR"

# Cleanup
conda deactivate
echo "CNN model training complete!" 