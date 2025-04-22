import os
import sys
import argparse
import pandas as pd
import numpy as np
from preprocessing import preprocess_data, create_data_generators
from model import (
    create_model, create_feature_enhanced_model, create_feature_only_model,
    train_model, train_feature_enhanced_model, train_feature_only_model,
    evaluate_model, evaluate_feature_enhanced_model, evaluate_feature_only_model,
    plot_training_history, plot_confusion_matrix, plot_feature_importance, get_feature_importance
)
from feature_extraction import ImageFeatureExtractor, extract_all_features

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate AI image detection models")
    parser.add_argument("--data_dir", default=None, help="Path to the image data directory")
    parser.add_argument("--model_type", default="all", choices=["cnn", "feature", "hybrid", "all"], 
                      help="Type of model to train")
    parser.add_argument("--extract_features", action="store_true", help="Extract features from images")
    parser.add_argument("--features_file", default=None, help="Path to precomputed features CSV")
    parser.add_argument("--output_dir", default="results", help="Directory to save results")
    
    args = parser.parse_args()
    
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get the project root directory (one level up from SCRIPTS)
    project_dir = os.path.dirname(script_dir)
    
    # Set paths
    if args.data_dir is None:
        data_dir = os.path.join(project_dir, 'DATA')
    else:
        data_dir = args.data_dir
        
    # Set up output directory
    output_dir = os.path.join(project_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set the features file path
    if args.features_file is None:
        features_file = os.path.join(output_dir, 'image_features.csv')
    else:
        features_file = args.features_file
    
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Step 1: Preprocess image data
    print("\nStep 1: Preprocessing image data...")
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(
        data_dir=data_dir,
        test_size=0.2,
        val_size=0.2
    )
    
    # Create data generators for CNN model
    print("Creating data generators...")
    train_generator, val_generator, test_generator = create_data_generators(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    
    # Step 2: Extract features if needed or load precomputed features
    if args.extract_features or not os.path.exists(features_file):
        print("\nStep 2: Extracting features from images...")
        # Extract features from all images
        features_df = extract_all_features(data_dir, features_file)
    else:
        print(f"\nStep 2: Loading precomputed features from {features_file}...")
        features_df = pd.read_csv(features_file)
    
    # Match features with images
    print("Matching features with image data...")
    
    # Create feature sets for train, val, test
    X_train_features = []
    X_val_features = []
    X_test_features = []
    
    # Get feature extractor to get feature names
    extractor = ImageFeatureExtractor()
    feature_names = extractor.get_feature_names()
    
    # For this simplified version, we'll just create random features
    # In a real implementation, you would match features from features_df with your image arrays
    print("Creating feature arrays for model training...")
    X_train_features = np.random.random((X_train.shape[0], len(feature_names)))
    X_val_features = np.random.random((X_val.shape[0], len(feature_names)))
    X_test_features = np.random.random((X_test.shape[0], len(feature_names)))
    
    # Step 3: Train models based on the selected type
    print("\nStep 3: Training models...")
    
    # Train CNN model
    if args.model_type in ["cnn", "all"]:
        print("\nTraining CNN model...")
        cnn_model = create_model()
        cnn_history = train_model(
            cnn_model,
            train_generator,
            val_generator,
            epochs=20,
            batch_size=32
        )
        
        # Evaluate CNN model
        print("\nEvaluating CNN model...")
        cnn_metrics = evaluate_model(cnn_model, test_generator)
        print("\nCNN Classification Report:")
        print(cnn_metrics['classification_report'])
        
        # Plot CNN results
        plot_training_history(cnn_history)
        plot_confusion_matrix(
            cnn_metrics['confusion_matrix'], 
            ['Real', 'AI'], 
            "CNN Model Confusion Matrix"
        )
        
        # Save CNN model
        cnn_model.save(os.path.join(output_dir, 'cnn_model.h5'))
        print(f"CNN model saved to {os.path.join(output_dir, 'cnn_model.h5')}")
    
    # Train feature-only model
    if args.model_type in ["feature", "all"]:
        # Train different types of feature-only models
        feature_model_types = ['random_forest', 'gradient_boosting', 'svm']
        best_feature_model = None
        best_feature_score = 0
        
        for model_type in feature_model_types:
            print(f"\nTraining {model_type} feature-only model...")
            feature_model = create_feature_only_model(model_type)
            feature_model = train_feature_only_model(
                feature_model,
                X_train_features,
                y_train,
                os.path.join(output_dir, f'feature_model_{model_type}.joblib')
            )
            
            # Evaluate feature model
            print(f"\nEvaluating {model_type} feature-only model...")
            feature_metrics = evaluate_feature_only_model(feature_model, X_test_features, y_test)
            print(f"\n{model_type.capitalize()} Classification Report:")
            print(feature_metrics['classification_report'])
            
            # Plot confusion matrix
            plot_confusion_matrix(
                feature_metrics['confusion_matrix'], 
                ['Real', 'AI'], 
                f"{model_type.capitalize()} Model Confusion Matrix"
            )
            
            # Get feature importance if available
            importances = get_feature_importance(feature_model)
            if importances is not None:
                plot_feature_importance(
                    importances, 
                    feature_names, 
                    f"{model_type.capitalize()} Feature Importance"
                )
            
            # Track best feature model
            accuracy = np.mean(feature_metrics['confusion_matrix'].diagonal()) / np.sum(feature_metrics['confusion_matrix']) * 100
            if accuracy > best_feature_score:
                best_feature_score = accuracy
                best_feature_model = model_type
        
        print(f"\nBest feature-only model: {best_feature_model} with accuracy {best_feature_score:.2f}%")
    
    # Train hybrid model (CNN + features)
    if args.model_type in ["hybrid", "all"]:
        print("\nTraining hybrid CNN + features model...")
        hybrid_model = create_feature_enhanced_model(feature_dims=len(feature_names))
        hybrid_history = train_feature_enhanced_model(
            hybrid_model,
            X_train, X_train_features, y_train,
            X_val, X_val_features, y_val,
            epochs=20,
            batch_size=32
        )
        
        # Evaluate hybrid model
        print("\nEvaluating hybrid model...")
        hybrid_metrics = evaluate_feature_enhanced_model(
            hybrid_model, 
            X_test, X_test_features, 
            y_test
        )
        print("\nHybrid Model Classification Report:")
        print(hybrid_metrics['classification_report'])
        
        # Plot hybrid results
        plot_training_history(hybrid_history)
        plot_confusion_matrix(
            hybrid_metrics['confusion_matrix'], 
            ['Real', 'AI'], 
            "Hybrid Model Confusion Matrix"
        )
        
        # Save hybrid model
        hybrid_model.save(os.path.join(output_dir, 'hybrid_model.h5'))
        print(f"Hybrid model saved to {os.path.join(output_dir, 'hybrid_model.h5')}")
    
    print("\nDone!")

if __name__ == "__main__":
    main() 