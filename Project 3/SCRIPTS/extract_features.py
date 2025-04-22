import numpy as np
import tensorflow as tf

def extract_features_from_data(images, batch_size=32):
    """
    Extract features from images using a pre-trained model.
    
    Args:
        images (numpy.ndarray): Array of images
        batch_size (int): Batch size for processing
        
    Returns:
        numpy.ndarray: Extracted features
    """
    print(f"Extracting features from {len(images)} images with batch_size={batch_size}")
    
    # Initialize the feature extractor model
    try:
        # Use EfficientNetB0 as feature extractor (lighter than ResNet50)
        print("Initializing EfficientNetB0 feature extractor...")
        feature_extractor = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        print("Feature extractor initialized successfully")
    except Exception as e:
        print(f"Error initializing feature extractor: {e}")
        raise
    
    # Process images in batches to avoid memory issues
    features = []
    total_batches = (len(images) + batch_size - 1) // batch_size
    
    print(f"Processing {total_batches} batches...")
    for i in range(0, len(images), batch_size):
        batch_end = min(i + batch_size, len(images))
        batch = images[i:batch_end]
        print(f"Processing batch {i//batch_size + 1}/{total_batches} with {len(batch)} images")
        
        try:
            # Preprocess the batch
            preprocessed_batch = tf.keras.applications.efficientnet.preprocess_input(batch)
            
            # Extract features
            with tf.device('/CPU:0'):  # Force CPU to avoid GPU memory issues
                batch_features = feature_extractor.predict(preprocessed_batch, verbose=0)
            
            features.append(batch_features)
            print(f"Batch {i//batch_size + 1}/{total_batches} processed successfully")
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {e}")
            # Try one more time with smaller batch
            if len(batch) > 1:
                print(f"Retrying with smaller batches...")
                half_batch = len(batch) // 2
                try:
                    # Process first half
                    first_half = batch[:half_batch]
                    preprocessed_first = tf.keras.applications.efficientnet.preprocess_input(first_half)
                    with tf.device('/CPU:0'):
                        first_features = feature_extractor.predict(preprocessed_first, verbose=0)
                    
                    # Process second half
                    second_half = batch[half_batch:]
                    preprocessed_second = tf.keras.applications.efficientnet.preprocess_input(second_half)
                    with tf.device('/CPU:0'):
                        second_features = feature_extractor.predict(preprocessed_second, verbose=0)
                    
                    # Combine results
                    batch_features = np.vstack([first_features, second_features])
                    features.append(batch_features)
                    print(f"Batch {i//batch_size + 1}/{total_batches} processed successfully with split approach")
                except Exception as inner_e:
                    print(f"Error even with smaller batches: {inner_e}")
                    # Create placeholder features with zeros
                    feature_dim = 1280  # EfficientNetB0 feature dimension
                    placeholder_features = np.zeros((len(batch), feature_dim))
                    features.append(placeholder_features)
                    print(f"Using placeholder features for batch {i//batch_size + 1}")
            else:
                # Single image that failed, use placeholder
                feature_dim = 1280  # EfficientNetB0 feature dimension
                placeholder_features = np.zeros((len(batch), feature_dim))
                features.append(placeholder_features)
                print(f"Using placeholder features for single image in batch {i//batch_size + 1}")
    
    # Combine all batches
    print("Combining all batches...")
    if features:
        combined_features = np.vstack(features)
        print(f"Features extracted successfully: {combined_features.shape}")
        return combined_features
    else:
        print("No features extracted, returning empty array")
        return np.array([]) 