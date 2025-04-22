import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from pathlib import Path
import json

def load_image_data(data_dir, labels_file=None, resize_shape=(224, 224), grayscale=False):
    """
    Load images and labels from a directory
    
    Args:
        data_dir: Directory containing image data
        labels_file: Path to a CSV file with image_file,label format (optional)
        resize_shape: Tuple of (height, width) to resize images to
        grayscale: Whether to convert images to grayscale
        
    Returns:
        Tuple of (images, labels, filenames)
    """
    data_dir = Path(data_dir)
    
    # If labels file is provided, load labels from CSV
    if labels_file is not None:
        df = pd.read_csv(labels_file)
        filenames = df['image_file'].values
        labels = df['label'].values
        image_paths = [data_dir / f for f in filenames]
    else:
        # Otherwise, assume directory structure with class subdirectories
        image_paths = []
        labels = []
        filenames = []
        
        class_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
        
        for i, class_dir in enumerate(sorted(class_dirs)):
            class_files = [f for f in class_dir.glob('*') if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
            
            for f in class_files:
                image_paths.append(f)
                labels.append(i)
                filenames.append(f.name)
        
        labels = np.array(labels)
    
    # Load and preprocess images
    images = []
    valid_indices = []
    valid_filenames = []
    
    for i, (path, filename) in enumerate(zip(image_paths, filenames)):
        try:
            if grayscale:
                img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, resize_shape[::-1])  # cv2 uses (width, height)
                img = img[..., np.newaxis]  # Add channel dimension
            else:
                img = cv2.imread(str(path))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                img = cv2.resize(img, resize_shape[::-1])  # cv2 uses (width, height)
            
            images.append(img)
            valid_indices.append(i)
            valid_filenames.append(filename)
        except Exception as e:
            print(f"Error loading image {path}: {e}")
    
    if not images:
        raise ValueError("No valid images found in the specified directory")
    
    # Convert to numpy arrays
    X = np.array(images)
    y = labels[valid_indices] if len(labels) > 0 else np.array([])
    filenames = np.array(valid_filenames)
    
    return X, y, filenames

def create_train_val_test_split(X, y, test_size=0.2, val_size=0.2, stratify=True, random_state=42):
    """
    Split data into training, validation, and test sets
    
    Args:
        X: Input features or images
        y: Target labels
        test_size: Proportion of data to use for testing
        val_size: Proportion of non-test data to use for validation
        stratify: Whether to perform stratified splits
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # First split: separate test set
    if stratify and len(y) > 0:
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
    else:
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    
    # Second split: separate validation set from training set
    if stratify and len(y_trainval) > 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=val_size, stratify=y_trainval, random_state=random_state
        )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=val_size, random_state=random_state
        )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def preprocess_images(images, normalize=True):
    """
    Preprocess images for neural network input
    
    Args:
        images: Array of images
        normalize: Whether to normalize pixel values to [0, 1]
        
    Returns:
        Preprocessed images
    """
    # Convert to float32
    images = images.astype(np.float32)
    
    # Normalize to [0, 1]
    if normalize:
        images /= 255.0
    
    return images

def extract_image_features(images, feature_extractor=None, batch_size=32):
    """
    Extract features from images using a pre-trained model
    
    Args:
        images: Array of images
        feature_extractor: Pre-trained model for feature extraction
        batch_size: Batch size for processing
        
    Returns:
        Extracted features
    """
    if feature_extractor is None:
        # Use a pre-trained ResNet50 model by default
        feature_extractor = tf.keras.applications.ResNet50(
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
    
    # Process images in batches to avoid memory issues
    features = []
    
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        batch_features = feature_extractor.predict(batch)
        features.append(batch_features)
    
    # Combine all batches
    features = np.vstack(features)
    
    return features

def extract_custom_features(images):
    """
    Extract custom features from images (color histograms, textures, etc.)
    
    Args:
        images: Array of images
        
    Returns:
        Extracted features
    """
    features = []
    
    for img in images:
        # Extract features from each image
        img_features = []
        
        # 1. Color histograms (for RGB images)
        if img.shape[-1] == 3:
            for channel in range(3):
                hist = cv2.calcHist([img], [channel], None, [32], [0, 256])
                img_features.extend(hist.flatten())
        else:
            # For grayscale images
            hist = cv2.calcHist([img], [0], None, [32], [0, 256])
            img_features.extend(hist.flatten())
        
        # 2. Basic statistics: mean, std, min, max for each channel
        for channel in range(img.shape[-1]):
            channel_data = img[..., channel].flatten()
            img_features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.min(channel_data),
                np.max(channel_data)
            ])
        
        # 3. Haralick texture features (on grayscale version)
        if img.shape[-1] == 3:
            gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = img[..., 0].astype(np.uint8)
        
        # Simplify by using only a subset of Haralick features
        glcm = cv2.calcGLCM(gray, stepSizes=[1], distances=[1, 3, 5])
        texture_stats = cv2.calcGLCMStatistics(glcm, [cv2.GLCM_HOMOGENEITY, cv2.GLCM_CONTRAST, cv2.GLCM_ENERGY, cv2.GLCM_CORRELATION])
        img_features.extend(texture_stats.flatten())
        
        features.append(img_features)
    
    # Convert to numpy array
    features = np.array(features)
    
    return features

def standardize_features(features, scaler=None, fit=True):
    """
    Standardize features using sklearn's StandardScaler
    
    Args:
        features: Array of features
        scaler: Pre-fitted scaler (optional)
        fit: Whether to fit the scaler on this data
        
    Returns:
        Tuple of (standardized features, scaler)
    """
    if scaler is None:
        scaler = StandardScaler()
    
    if fit:
        scaler.fit(features)
    
    standardized = scaler.transform(features)
    
    return standardized, scaler

def create_balanced_batches(X, y, batch_size=32, shuffle=True):
    """
    Create balanced batches for training
    
    Args:
        X: Input features or images
        y: Target labels
        batch_size: Size of each batch
        shuffle: Whether to shuffle the data
        
    Returns:
        Generator yielding (batch_X, batch_y)
    """
    # Get unique classes and their indices
    classes = np.unique(y)
    class_indices = [np.where(y == c)[0] for c in classes]
    
    # Calculate samples per class per batch
    samples_per_class = batch_size // len(classes)
    
    # Shuffle indices if requested
    if shuffle:
        for i in range(len(class_indices)):
            np.random.shuffle(class_indices[i])
    
    # Create pointers for each class
    pointers = [0] * len(classes)
    
    while True:
        batch_X = []
        batch_y = []
        
        # Get samples_per_class from each class
        for i, c in enumerate(classes):
            indices = class_indices[i]
            
            # Reset pointer if necessary
            if pointers[i] + samples_per_class > len(indices):
                if shuffle:
                    np.random.shuffle(class_indices[i])
                pointers[i] = 0
            
            # Get batch indices for this class
            batch_indices = indices[pointers[i]:pointers[i] + samples_per_class]
            pointers[i] += samples_per_class
            
            # Add to batch
            batch_X.extend(X[batch_indices])
            batch_y.extend([c] * len(batch_indices))
        
        # Yield batch
        yield np.array(batch_X), np.array(batch_y)

def save_dataset_split(X_train, X_val, X_test, y_train, y_val, y_test, 
                      filenames_train=None, filenames_val=None, filenames_test=None,
                      output_dir=None):
    """
    Save dataset splits to a directory
    
    Args:
        X_train, X_val, X_test: Features for each split
        y_train, y_val, y_test: Labels for each split
        filenames_train, filenames_val, filenames_test: Original filenames for each split
        output_dir: Directory to save the dataset splits
        
    Returns:
        Path to the saved dataset information
    """
    if output_dir is None:
        return None
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Save labels and filenames
    splits_info = {
        'train': {
            'size': len(y_train),
            'class_distribution': {str(c): int((y_train == c).sum()) for c in np.unique(y_train)}
        },
        'val': {
            'size': len(y_val),
            'class_distribution': {str(c): int((y_val == c).sum()) for c in np.unique(y_val)}
        },
        'test': {
            'size': len(y_test),
            'class_distribution': {str(c): int((y_test == c).sum()) for c in np.unique(y_test)}
        }
    }
    
    # Save filenames if provided
    if filenames_train is not None:
        train_df = pd.DataFrame({'filename': filenames_train, 'label': y_train})
        train_df.to_csv(os.path.join(output_dir, 'train_files.csv'), index=False)
        
    if filenames_val is not None:
        val_df = pd.DataFrame({'filename': filenames_val, 'label': y_val})
        val_df.to_csv(os.path.join(output_dir, 'val_files.csv'), index=False)
        
    if filenames_test is not None:
        test_df = pd.DataFrame({'filename': filenames_test, 'label': y_test})
        test_df.to_csv(os.path.join(output_dir, 'test_files.csv'), index=False)
    
    # Save split information
    info_path = os.path.join(output_dir, 'splits_info.json')
    with open(info_path, 'w') as f:
        json.dump(splits_info, f, indent=4)
    
    return info_path

def create_cross_validation_splits(X, y, n_splits=5, stratify=True, random_state=42):
    """
    Create cross-validation splits
    
    Args:
        X: Input features or images
        y: Target labels
        n_splits: Number of splits
        stratify: Whether to perform stratified splits
        random_state: Random seed for reproducibility
        
    Returns:
        List of tuples (train_indices, val_indices)
    """
    if stratify and len(y) > 0:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Generate the indices for each split
    splits = []
    for train_idx, val_idx in cv.split(X, y):
        splits.append((train_idx, val_idx))
    
    return splits

def get_class_weights(y):
    """
    Calculate class weights for imbalanced datasets
    
    Args:
        y: Target labels
        
    Returns:
        Dictionary mapping class indices to weights
    """
    # Count samples per class
    class_counts = np.bincount(y.astype(int))
    
    # Calculate weights as inverse of frequency
    n_samples = len(y)
    n_classes = len(class_counts)
    
    weights = {}
    for i in range(n_classes):
        if class_counts[i] > 0:
            weights[i] = n_samples / (n_classes * class_counts[i])
    
    return weights

def augment_images(images, augmentation_factor=2, random_state=None):
    """
    Apply data augmentation to images
    
    Args:
        images: Array of images
        augmentation_factor: Number of augmented images to create per original image
        random_state: Random seed for reproducibility
        
    Returns:
        Augmented images
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Create augmentation pipeline using tf.keras.preprocessing
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Generate augmented images
    augmented_images = []
    
    for img in images:
        img = img.reshape((1,) + img.shape)
        batch = next(datagen.flow(img, batch_size=1))
        augmented_images.append(batch[0])
        
        # Generate additional augmented images if requested
        for _ in range(augmentation_factor - 1):
            batch = next(datagen.flow(img, batch_size=1))
            augmented_images.append(batch[0])
    
    # Combine original and augmented images
    augmented_images = np.array(augmented_images)
    
    return augmented_images 