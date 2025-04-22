import os
import json
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import pandas as pd
import glob

def load_json_data(json_path):
    """
    Load and parse the JSON dataset file.
    
    Args:
        json_path (str): Path to the JSON file
        
    Returns:
        pandas.DataFrame: DataFrame containing the dataset information
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def load_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess a single image.
    
    Args:
        image_path (str): Path to the image file
        target_size (tuple): Target size for resizing images
        
    Returns:
        numpy.ndarray: Preprocessed image or None if loading fails
    """
    try:
        img = cv2.imread(image_path)
        if img is not None:
            img = cv2.resize(img, target_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
    except Exception as e:
        print(f"Error loading image {image_path}: {str(e)}")
    return None

def load_images_from_json(json_path, data_dir, target_size=(224, 224), max_images=None):
    """
    Load images based on the JSON metadata, handling filename discrepancies.
    
    Args:
        json_path (str): Path to the JSON file
        data_dir (str): Base directory for the dataset
        target_size (tuple): Target size for resizing images
        max_images (int): Maximum number of image pairs to load
        
    Returns:
        tuple: (images, labels) containing all real and fake images with labels
    """
    print(f"Loading JSON data from {json_path}")
    
    # Load JSON data
    try:
        df = load_json_data(json_path)
        print(f"Loaded {len(df)} image pairs from JSON")
    except Exception as e:
        print(f"Error loading JSON data: {str(e)}")
        return np.array([]), np.array([])
    
    # Limit the number of images if specified
    if max_images is not None and max_images < len(df):
        df = df.sample(max_images, random_state=42)
        print(f"Sampled {len(df)} image pairs")
    
    # Find all available datasets
    datasets = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d.endswith('_dataset')]
    print(f"Found datasets: {datasets}")
    
    # Create lists to store images and labels
    all_images = []
    all_labels = []
    
    # Keep track of found and missing images
    found_real = 0
    found_fake = 0
    missing_real = 0
    missing_fake = 0
    
    # Process each image pair
    for _, row in df.iterrows():
        real_filename = row['real_image_file_name']
        fake_filename = row['fake_image_file_name']
        platform = row.get('platform', 'unknown')
        
        # Search for real image
        real_image_path = None
        # First try exact match in the real directory
        for dataset in datasets:
            real_dir = os.path.join(data_dir, dataset, 'real')
            if os.path.exists(real_dir):
                candidate_path = os.path.join(real_dir, real_filename)
                if os.path.exists(candidate_path):
                    real_image_path = candidate_path
                    break
        
        # Search for fake image (handle potential name changes)
        fake_image_path = None
        # First try exact match
        for dataset in datasets:
            fake_dir = os.path.join(data_dir, dataset, 'fake')
            if os.path.exists(fake_dir):
                candidate_path = os.path.join(fake_dir, fake_filename)
                if os.path.exists(candidate_path):
                    fake_image_path = candidate_path
                    break
                
                # If the platform is 'sd', try to match with image_X.jpg pattern
                if platform == 'sd':
                    # Look for files matching pattern image_*.jpg
                    potential_matches = glob.glob(os.path.join(fake_dir, "image_*.jpg"))
                    for match in potential_matches:
                        # We could add more sophisticated matching here if needed
                        fake_image_path = match
                        if fake_image_path is not None:
                            break
                
        # Load real image if found
        if real_image_path is not None:
            real_img = load_image(real_image_path, target_size)
            if real_img is not None:
                all_images.append(real_img)
                all_labels.append(0)  # Real image
                found_real += 1
            else:
                missing_real += 1
        else:
            missing_real += 1
            
        # Load fake image if found
        if fake_image_path is not None:
            fake_img = load_image(fake_image_path, target_size)
            if fake_img is not None:
                all_images.append(fake_img)
                all_labels.append(1)  # Fake image
                found_fake += 1
            else:
                missing_fake += 1
        else:
            missing_fake += 1
    
    print(f"Found {found_real} real images, {found_fake} fake images")
    print(f"Missing {missing_real} real images, {missing_fake} fake images")
    
    return np.array(all_images), np.array(all_labels)

def load_all_images_from_directories(data_dir, target_size=(224, 224), max_images=None):
    """
    Load all images from real and fake directories, ignoring JSON mapping.
    
    Args:
        data_dir (str): Base directory containing the dataset
        target_size (tuple): Target size for resizing images
        max_images (int): Maximum number of images to load per category
        
    Returns:
        tuple: (images, labels)
    """
    all_images = []
    all_labels = []
    
    # Find dataset directories
    datasets = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d.endswith('_dataset')]
    print(f"Found datasets: {datasets}")
    
    for dataset in datasets:
        dataset_path = os.path.join(data_dir, dataset)
        
        # Process real images
        real_dir = os.path.join(dataset_path, 'real')
        if os.path.exists(real_dir):
            real_files = []
            for ext in ['jpg', 'jpeg', 'png']:
                real_files.extend(glob.glob(os.path.join(real_dir, f"*.{ext}")))
            
            if max_images is not None and len(real_files) > max_images:
                real_files = real_files[:max_images]
            
            print(f"Loading {len(real_files)} real images from {real_dir}")
            
            for img_path in real_files:
                img = load_image(img_path, target_size)
                if img is not None:
                    all_images.append(img)
                    all_labels.append(0)  # Real image
        
        # Process fake images
        fake_dir = os.path.join(dataset_path, 'fake')
        if os.path.exists(fake_dir):
            fake_files = []
            for ext in ['jpg', 'jpeg', 'png']:
                fake_files.extend(glob.glob(os.path.join(fake_dir, f"*.{ext}")))
            
            if max_images is not None and len(fake_files) > max_images:
                fake_files = fake_files[:max_images]
            
            print(f"Loading {len(fake_files)} fake images from {fake_dir}")
            
            for img_path in fake_files:
                img = load_image(img_path, target_size)
                if img is not None:
                    all_images.append(img)
                    all_labels.append(1)  # Fake image
    
    return np.array(all_images), np.array(all_labels)

def preprocess_data(data_dir, json_path=None, test_size=0.2, val_size=0.2, target_size=(224, 224), max_images=None, dtype='float64'):
    """
    Preprocess image data and split into training, validation, and testing sets.
    
    Args:
        data_dir (str): Directory containing the dataset
        json_path (str): Path to the JSON file with image metadata (optional)
        test_size (float): Proportion of data to use for testing
        val_size (float): Proportion of training data to use for validation
        target_size (tuple): Target size for resizing images
        max_images (int): Maximum number of images to use
        dtype (str): Data type for the image arrays ('float32' or 'float64')
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Load images based on strategy
    if json_path is not None and os.path.exists(json_path):
        print(f"Using JSON-based image loading with file: {json_path}")
        images, labels = load_images_from_json(json_path, data_dir, target_size, max_images)
    else:
        print("Using directory-based image loading")
        images, labels = load_all_images_from_directories(data_dir, target_size, max_images)
    
    if len(images) == 0:
        raise ValueError("No images were found. Check the dataset directory and JSON file.")
    
    print(f"Loaded {len(images)} images in total")
    
    # Convert to arrays and normalize
    X = np.array(images, dtype=dtype)
    y = np.array(labels)
    
    # Normalize pixel values to [0, 1]
    X = X / 255.0
    
    # Split data into train and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Split training data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=42, stratify=y_train_val
    )
    
    print(f"Data split: Train={len(X_train)}, Validation={len(X_val)}, Test={len(X_test)}")
    
    class_counts = np.bincount(y)
    print(f"Class distribution: Real={class_counts[0]}, Fake={class_counts[1]}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def create_data_generators(X_train, X_val, X_test, y_train, y_val, y_test, batch_size=32):
    """
    Create data generators for training, validation, and testing.
    
    Args:
        X_train, X_val, X_test: Image data
        y_train, y_val, y_test: Labels
        batch_size (int): Batch size for generators
        
    Returns:
        tuple: (train_generator, val_generator, test_generator)
    """
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator()
    test_datagen = ImageDataGenerator()
    
    train_generator = train_datagen.flow(
        X_train, y_train,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_generator = val_datagen.flow(
        X_val, y_val,
        batch_size=batch_size,
        shuffle=False
    )
    
    test_generator = test_datagen.flow(
        X_test, y_test,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator

def generate_batches(X, y, batch_size=32, augment=True):
    """
    Generate batches of data with optional augmentation.
    
    Args:
        X: Image data
        y: Labels
        batch_size (int): Batch size
        augment (bool): Whether to apply data augmentation
        
    Yields:
        tuple: (batch_X, batch_y)
    """
    num_samples = len(X)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    ) if augment else ImageDataGenerator()
    
    for start_idx in range(0, num_samples, batch_size):
        batch_indices = indices[start_idx:start_idx + batch_size]
        batch_X = X[batch_indices]
        batch_y = y[batch_indices]
        
        if augment:
            for X_batch, y_batch in datagen.flow(batch_X, batch_y, batch_size=batch_size):
                yield X_batch, y_batch
                break
        else:
            yield batch_X, batch_y 