import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse

def load_and_combine_features(real_features_path, fake_features_path, output_path=None):
    """
    Load features from real and fake images, combine them, and add labels.
    
    Args:
        real_features_path (str): Path to the CSV file with real image features
        fake_features_path (str): Path to the CSV file with fake image features
        output_path (str): Path to save the combined features CSV (optional)
        
    Returns:
        pd.DataFrame: Combined features with labels
    """
    print(f"Loading real features from {real_features_path}")
    real_features = pd.read_csv(real_features_path)
    real_features['label'] = 0  # 0 for real
    
    print(f"Loading fake features from {fake_features_path}")
    fake_features = pd.read_csv(fake_features_path)
    fake_features['label'] = 1  # 1 for fake/AI-generated
    
    # Combine the features
    print("Combining features...")
    combined_features = pd.concat([real_features, fake_features], ignore_index=True)
    
    # Save combined features if output path is provided
    if output_path:
        print(f"Saving combined features to {output_path}")
        combined_features.to_csv(output_path, index=False)
    
    print(f"Combined dataset has {len(combined_features)} samples")
    print(f"Real images: {len(real_features)}, Fake images: {len(fake_features)}")
    
    return combined_features

def prepare_train_test_split(combined_features, test_size=0.2, val_size=0.25, random_state=42, output_dir=None):
    """
    Split the combined features into train, validation, and test sets.
    
    Args:
        combined_features (pd.DataFrame): Combined features with labels
        test_size (float): Proportion of the dataset to include in the test split
        val_size (float): Proportion of the train set to include in the validation split
        random_state (int): Random seed for reproducibility
        output_dir (str): Directory to save the split datasets (optional)
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Separate features and labels
    X = combined_features.drop(['label', 'image_path'], axis=1)
    y = combined_features['label']
    
    # First split: separate test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: separate validation set from training set
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=random_state, stratify=y_train_val
    )
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Save the splits if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving splits to {output_dir}")
        
        # Save feature sets
        X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
        X_val.to_csv(os.path.join(output_dir, 'X_val.csv'), index=False)
        X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
        
        # Save labels
        pd.DataFrame(y_train, columns=['label']).to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
        pd.DataFrame(y_val, columns=['label']).to_csv(os.path.join(output_dir, 'y_val.csv'), index=False)
        pd.DataFrame(y_test, columns=['label']).to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def scale_features(X_train, X_val, X_test, output_dir=None):
    """
    Scale features using StandardScaler fit on the training set.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_val (pd.DataFrame): Validation features
        X_test (pd.DataFrame): Test features
        output_dir (str): Directory to save the scaled datasets and scaler (optional)
        
    Returns:
        tuple: (X_train_scaled, X_val_scaled, X_test_scaled, scaler)
    """
    # Initialize scaler
    scaler = StandardScaler()
    
    # Fit on training data and transform all sets
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrames with original column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # Save the scaled datasets and scaler if output directory is provided
    if output_dir:
        print(f"Saving scaled features to {output_dir}")
        X_train_scaled.to_csv(os.path.join(output_dir, 'X_train_scaled.csv'), index=False)
        X_val_scaled.to_csv(os.path.join(output_dir, 'X_val_scaled.csv'), index=False)
        X_test_scaled.to_csv(os.path.join(output_dir, 'X_test_scaled.csv'), index=False)
        
        # Save the scaler
        scaler_path = os.path.join(output_dir, 'feature_scaler.joblib')
        print(f"Saving scaler to {scaler_path}")
        import joblib
        joblib.dump(scaler, scaler_path)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare combined features from real and fake images")
    parser.add_argument("--real_features", required=True, help="Path to real image features CSV")
    parser.add_argument("--fake_features", required=True, help="Path to fake image features CSV")
    parser.add_argument("--output_dir", required=True, help="Directory to save outputs")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion for test set")
    parser.add_argument("--val_size", type=float, default=0.25, help="Proportion for validation set")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Combined features output path
    combined_path = os.path.join(args.output_dir, 'combined_features.csv')
    
    # Load and combine features
    combined_features = load_and_combine_features(
        args.real_features, 
        args.fake_features,
        output_path=combined_path
    )
    
    # Split the data
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_train_test_split(
        combined_features,
        test_size=args.test_size,
        val_size=args.val_size,
        output_dir=args.output_dir
    )
    
    # Scale the features
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_val, X_test, 
        output_dir=args.output_dir
    )
    
    print("Data preparation complete!") 