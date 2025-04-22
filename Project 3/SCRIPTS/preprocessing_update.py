import os
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2
import json
from keras.preprocessing.image import ImageDataGenerator

def preprocess_data(data_dir, json_path=None, test_size=0.2, val_size=0.2, target_size=(224, 224), max_images=None):
    """
    Preprocess and split the dataset into training, validation, and test sets.
    
    Args:
        data_dir (str): Directory containing the images
        json_path (str): Path to the JSON file containing dataset information (optional)
        test_size (float): Proportion of dataset to include in test split
        val_size (float): Proportion of training set to include in validation split
        target_size (tuple): Target size for resizing images
        max_images (int): Maximum number of images to load per category (default: None - load all images)
        
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
        
        # Load images from paths with maximum limit if specified
        X, y = load_images_from_paths(image_paths, target_size, max_images)
    
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

def load_images_from_paths(image_paths, target_size=(224, 224), max_images=None):
    """
    Load images from paths and convert labels.
    
    Args:
        image_paths (dict): Dictionary of image paths for real and fake images
        target_size (tuple): Target size for resizing images
        max_images (int): Maximum number of images to load per category
        
    Returns:
        tuple: (images, labels)
    """
    real_image_paths = image_paths.get('real', [])
    fake_image_paths = image_paths.get('fake', [])
    
    # Limit the number of images if max_images is specified
    if max_images is not None:
        print(f"Limiting to {max_images} images per class")
        real_image_paths = real_image_paths[:max_images] if len(real_image_paths) > max_images else real_image_paths
        fake_image_paths = fake_image_paths[:max_images] if len(fake_image_paths) > max_images else fake_image_paths
    
    print(f"Found {len(real_image_paths)} real images and {len(fake_image_paths)} fake images")
    
    # Load real images
    real_images = []
    for img_path in tqdm(real_image_paths, desc="Loading real images"):
        try:
            img = load_and_preprocess_image(img_path, target_size)
            real_images.append(img)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    
    # Load fake images
    fake_images = []
    for img_path in tqdm(fake_image_paths, desc="Loading fake images"):
        try:
            img = load_and_preprocess_image(img_path, target_size)
            fake_images.append(img)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    
    # Convert to numpy arrays
    real_images = np.array(real_images) if real_images else np.array([])
    fake_images = np.array(fake_images) if fake_images else np.array([])
    
    # Create labels (0 for real, 1 for fake)
    real_labels = np.zeros(len(real_images))
    fake_labels = np.ones(len(fake_images))
    
    # Combine data
    if len(real_images) > 0 and len(fake_images) > 0:
        images = np.concatenate([real_images, fake_images])
        labels = np.concatenate([real_labels, fake_labels])
    elif len(real_images) > 0:
        images = real_images
        labels = real_labels
    elif len(fake_images) > 0:
        images = fake_images
        labels = fake_labels
    else:
        images = np.array([])
        labels = np.array([])
    
    return images, labels 

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """
    Load an image from path and preprocess it.
    
    Args:
        image_path (str): Path to the image
        target_size (tuple): Target size for resizing
        
    Returns:
        numpy.ndarray: Preprocessed image
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize image
    img = cv2.resize(img, target_size)
    
    # Normalize pixel values to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    return img

def find_image_paths(data_dir):
    """
    Find all image paths in a data directory.
    
    Args:
        data_dir (str): Path to the directory containing images
        
    Returns:
        tuple: (real_image_paths, fake_image_paths)
    """
    # Define supported image extensions
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    # Initialize lists to store image paths
    real_image_paths = []
    fake_image_paths = []
    
    # Get paths for real images
    real_dir = os.path.join(data_dir, 'real')
    if os.path.exists(real_dir):
        for root, _, files in os.walk(real_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in supported_extensions):
                    real_image_paths.append(os.path.join(root, file))
    
    # Get paths for fake/AI-generated images
    fake_dir = os.path.join(data_dir, 'fake')
    if os.path.exists(fake_dir):
        for root, _, files in os.walk(fake_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in supported_extensions):
                    fake_image_paths.append(os.path.join(root, file))
    
    print(f"Found {len(real_image_paths)} real images and {len(fake_image_paths)} fake images")
    return real_image_paths, fake_image_paths

def preprocess_data(data_dir, test_size=0.15, val_size=0.15, random_state=42):
    """
    Preprocess data by splitting image paths into training, validation, and test sets.
    
    Args:
        data_dir (str): Directory containing the image data
        test_size (float): Proportion of data to use for testing
        val_size (float): Proportion of data to use for validation
        random_state (int): Random state for reproducibility
        
    Returns:
        dict: Dictionary containing train, val, and test data with labels
    """
    # Find image paths
    real_image_paths, fake_image_paths = find_image_paths(data_dir)
    
    # Create labels (0 for real, 1 for fake)
    real_labels = [0] * len(real_image_paths)
    fake_labels = [1] * len(fake_image_paths)
    
    # Combine data
    all_paths = real_image_paths + fake_image_paths
    all_labels = real_labels + fake_labels
    
    # First split: separate out test set
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        all_paths, all_labels, test_size=test_size, random_state=random_state, stratify=all_labels
    )
    
    # Second split: separate out validation set
    adjusted_val_size = val_size / (1 - test_size)  # Adjust val_size for the remaining data
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=adjusted_val_size, 
        random_state=random_state, stratify=train_val_labels
    )
    
    # Create data dictionary
    data = {
        'train': {'paths': train_paths, 'labels': train_labels},
        'val': {'paths': val_paths, 'labels': val_labels},
        'test': {'paths': test_paths, 'labels': test_labels}
    }
    
    # Print dataset statistics
    print(f"Dataset split:")
    print(f"  Train: {len(train_paths)} images ({len(train_paths)} images/{len(all_paths)} total = {len(train_paths)/len(all_paths):.2f})")
    print(f"  Validation: {len(val_paths)} images ({len(val_paths)} images/{len(all_paths)} total = {len(val_paths)/len(all_paths):.2f})")
    print(f"  Test: {len(test_paths)} images ({len(test_paths)} images/{len(all_paths)} total = {len(test_paths)/len(all_paths):.2f})")
    
    return data

def create_data_generators(data, batch_size=32, target_size=(224, 224), augment=True):
    """
    Create data generators for training models.
    
    Args:
        data (dict): Dictionary containing train, val, and test data with labels
        batch_size (int): Batch size for data generators
        target_size (tuple): Target size for image resizing
        augment (bool): Whether to apply data augmentation to training data
        
    Returns:
        tuple: (train_generator, val_generator, test_generator)
    """
    # Create ImageDataGenerator for training with augmentation if enabled
    if augment:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
    else:
        train_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create ImageDataGenerator for validation and test data (no augmentation)
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Define generator functions
    def generate_batches(paths, labels, datagen, batch_size, target_size, shuffle=True):
        num_samples = len(paths)
        indices = np.arange(num_samples)
        
        while True:
            if shuffle:
                np.random.shuffle(indices)
            
            for start_idx in range(0, num_samples, batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                batch_paths = [paths[i] for i in batch_indices]
                batch_labels = [labels[i] for i in batch_indices]
                
                # Load and preprocess images
                batch_images = []
                for path in batch_paths:
                    img = load_and_preprocess_image(path, target_size)
                    batch_images.append(img)
                
                batch_images = np.array(batch_images)
                batch_labels = np.array(batch_labels)
                
                # Apply augmentation if using a datagen with augmentation
                if datagen.featurewise_center or datagen.samplewise_center or \
                   datagen.featurewise_std_normalization or datagen.samplewise_std_normalization or \
                   datagen.zca_whitening or datagen.rotation_range or \
                   datagen.width_shift_range or datagen.height_shift_range or \
                   datagen.shear_range or datagen.zoom_range or \
                   datagen.channel_shift_range or datagen.fill_mode != 'nearest' or \
                   datagen.cval != 0. or datagen.horizontal_flip or \
                   datagen.vertical_flip or datagen.rescale != None:
                    batch_images = next(datagen.flow(batch_images, batch_size=len(batch_indices))).astype('float32')
                
                yield batch_images, batch_labels
    
    # Create generators
    train_generator = generate_batches(
        data['train']['paths'], data['train']['labels'], 
        train_datagen, batch_size, target_size
    )
    
    val_generator = generate_batches(
        data['val']['paths'], data['val']['labels'], 
        val_datagen, batch_size, target_size, shuffle=False
    )
    
    test_generator = generate_batches(
        data['test']['paths'], data['test']['labels'], 
        test_datagen, batch_size, target_size, shuffle=False
    )
    
    # Return generators along with steps information
    return {
        'train': {
            'generator': train_generator,
            'steps': len(data['train']['paths']) // batch_size + (1 if len(data['train']['paths']) % batch_size > 0 else 0)
        },
        'val': {
            'generator': val_generator,
            'steps': len(data['val']['paths']) // batch_size + (1 if len(data['val']['paths']) % batch_size > 0 else 0)
        },
        'test': {
            'generator': test_generator,
            'steps': len(data['test']['paths']) // batch_size + (1 if len(data['test']['paths']) % batch_size > 0 else 0)
        }
    } 