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

def find_image_paths(data_dir, categories=None):
    """
    Find all image paths in the dataset directory structure.
    
    Args:
        data_dir (str): Base directory containing the dataset
        categories (list): List of categories to include ('real', 'fake', or both)
        
    Returns:
        dict: Dictionary mapping categories to lists of image paths
    """
    if categories is None:
        categories = ['real', 'fake']
    
    image_paths = {cat: [] for cat in categories}
    
    print(f"Searching for images in: {data_dir}")
    print(f"Directory exists: {os.path.exists(data_dir)}")
    
    # Look for DeepGuardDB_v1 directory
    deepguard_dir = data_dir
    if os.path.basename(data_dir) != 'DeepGuardDB_v1' and os.path.exists(os.path.join(data_dir, 'DeepGuardDB_v1')):
        deepguard_dir = os.path.join(data_dir, 'DeepGuardDB_v1')
        print(f"Using DeepGuardDB_v1 directory: {deepguard_dir}")
    
    # List model directories (DALLE_dataset, GLIDE_dataset, etc.)
    model_dirs = [
        d for d in os.listdir(deepguard_dir) 
        if os.path.isdir(os.path.join(deepguard_dir, d)) and d.endswith('_dataset')
    ]
    
    print(f"Found model directories: {model_dirs}")
    
    # Traverse each model directory looking for real/fake subdirectories
    for model_dir in model_dirs:
        model_path = os.path.join(deepguard_dir, model_dir)
        print(f"Checking model directory: {model_path}")
        
        for category in categories:
            category_path = os.path.join(model_path, category)
            print(f"Checking category path: {category_path}")
            print(f"Category path exists: {os.path.exists(category_path)}")
            
            if os.path.exists(category_path):
                # Find all image files with various extensions
                for ext in ['jpg', 'jpeg', 'png']:
                    pattern = os.path.join(category_path, f"*.{ext}")
                    found_images = glob.glob(pattern)
                    print(f"Found {len(found_images)} images with extension {ext} in {category_path}")
                    image_paths[category].extend(found_images)
    
    # If no images found, try a simpler directory structure
    if all(len(paths) == 0 for paths in image_paths.values()):
        print("No images found in the expected structure, checking simpler structure...")
        for category in categories:
            # Try "images/real" and "images/fake" structure
            category_path = os.path.join(data_dir, 'images', category)
            if os.path.exists(category_path):
                for ext in ['jpg', 'jpeg', 'png']:
                    image_paths[category].extend(
                        glob.glob(os.path.join(category_path, f"*.{ext}"))
                    )
            
            # Try direct "real" and "fake" directories
            category_path = os.path.join(data_dir, category)
            if os.path.exists(category_path):
                for ext in ['jpg', 'jpeg', 'png']:
                    image_paths[category].extend(
                        glob.glob(os.path.join(category_path, f"*.{ext}"))
                    )
    
    print(f"Total images found: {sum(len(paths) for paths in image_paths.values())}")
    for category in categories:
        print(f"Sample {category} image paths: {image_paths[category][:5]}")
    
    return image_paths

def load_images_from_paths(image_paths, target_size=(224, 224), max_images=None):
    """
    Load images from file paths.
    
    Args:
        image_paths (dict): Dictionary mapping categories to lists of image paths
        target_size (tuple): Target size for resizing images
        max_images (int): Maximum number of images to load per category (None for all)
        
    Returns:
        tuple: (images, labels)
    """
    all_images = []
    all_labels = []
    
    for category, paths in image_paths.items():
        if max_images is not None:
            paths = paths[:max_images]
            
        print(f"Loading {len(paths)} {category} images...")
        
        for path in paths:
            img = load_image(path, target_size)
            if img is not None:
                all_images.append(img)
                all_labels.append(1 if category == 'fake' else 0)
    
    return np.array(all_images), np.array(all_labels)

def preprocess_data(data_dir, json_path=None, test_size=0.2, val_size=0.2, target_size=(224, 224)):
    """
    Preprocess and split the dataset into training, validation, and test sets.
    
    Args:
        data_dir (str): Directory containing the images
        json_path (str): Path to the JSON file containing dataset information (optional)
        test_size (float): Proportion of dataset to include in test split
        val_size (float): Proportion of training set to include in validation split
        target_size (tuple): Target size for resizing images
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # If JSON path is provided, use it to load dataset information
    if json_path is not None and os.path.exists(json_path):
        # Load dataset information from JSON
        df = load_json_data(json_path)
        
        # Load real and fake images based on JSON
        print("Loading images based on JSON metadata...")
        real_images, real_labels = load_images_from_dataframe(
            df, data_dir, 'real', target_size
        )
        fake_images, fake_labels = load_images_from_dataframe(
            df, data_dir, 'fake', target_size
        )
        
        # Combine data
        X = np.concatenate([real_images, fake_images])
        y = np.concatenate([real_labels, fake_labels])
    else:
        # Find images in the directory structure
        print("Finding images in directory structure...")
        image_paths = find_image_paths(data_dir)
        
        # Load images from paths
        X, y = load_images_from_paths(image_paths, target_size)
    
    # Check if we have loaded any images
    if len(X) == 0:
        raise ValueError("No images found in the specified directory structure. Please check the data path.")
    
    print(f"Loaded {len(X)} total images: {np.sum(y == 0)} real, {np.sum(y == 1)} fake")
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Split train into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=42, stratify=y_train
    )
    
    # Normalize pixel values
    X_train = X_train / 255.0
    X_val = X_val / 255.0
    X_test = X_test / 255.0
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def load_images_from_dataframe(df, data_dir, image_type='real', target_size=(224, 224)):
    """
    Load images from a DataFrame containing image information.
    
    Args:
        df (pandas.DataFrame): DataFrame containing image information
        data_dir (str): Directory containing the images
        image_type (str): Type of images to load ('real' or 'fake')
        target_size (tuple): Target size for resizing images
        
    Returns:
        tuple: (images, labels)
    """
    images = []
    labels = []
    
    for _, row in df.iterrows():
        if image_type == 'real':
            img_path = os.path.join(data_dir, row['real_image_file_name'])
            label = 0  # Real image
        else:
            img_path = os.path.join(data_dir, row['fake_image_file_name'])
            label = 1  # AI-generated image
            
        img = load_image(img_path, target_size)
        if img is not None:
            images.append(img)
            labels.append(label)
    
    return np.array(images), np.array(labels)

def create_data_generators(X_train, X_val, X_test, y_train, y_val, y_test, batch_size=32):
    """
    Create data generators with augmentation for training.
    
    Args:
        X_train (numpy.ndarray): Training images
        X_val (numpy.ndarray): Validation images
        X_test (numpy.ndarray): Test images
        y_train (numpy.ndarray): Training labels
        y_val (numpy.ndarray): Validation labels
        y_test (numpy.ndarray): Test labels
        batch_size (int): Batch size for training
        
    Returns:
        tuple: (train_generator, val_generator, test_generator)
    """
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
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