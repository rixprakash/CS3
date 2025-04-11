import os
import json
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import pandas as pd

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

def preprocess_data(data_dir, json_path, test_size=0.2, val_size=0.2, target_size=(224, 224)):
    """
    Preprocess and split the dataset into training, validation, and test sets.
    
    Args:
        data_dir (str): Directory containing the images
        json_path (str): Path to the JSON file containing dataset information
        test_size (float): Proportion of dataset to include in test split
        val_size (float): Proportion of training set to include in validation split
        target_size (tuple): Target size for resizing images
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Load dataset information
    df = load_json_data(json_path)
    
    # Load real and fake images
    real_images, real_labels = load_images_from_dataframe(
        df, data_dir, 'real', target_size
    )
    fake_images, fake_labels = load_images_from_dataframe(
        df, data_dir, 'fake', target_size
    )
    
    # Combine data
    X = np.concatenate([real_images, fake_images])
    y = np.concatenate([real_labels, fake_labels])
    
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
        fill_mode='nearest'
    )
    
    val_test_datagen = ImageDataGenerator()
    
    train_generator = train_datagen.flow(
        X_train, y_train, batch_size=batch_size, shuffle=True
    )
    
    val_generator = val_test_datagen.flow(
        X_val, y_val, batch_size=batch_size, shuffle=False
    )
    
    test_generator = val_test_datagen.flow(
        X_test, y_test, batch_size=batch_size, shuffle=False
    )
    
    return train_generator, val_generator, test_generator 