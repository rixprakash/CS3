import os
import pandas as pd
import numpy as np
import argparse
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(data_dir):
    """
    Load the prepared data splits from the given directory.
    
    Args:
        data_dir (str): Directory containing the data splits
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    print(f"Loading data from {data_dir}")
    
    # Load features
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train_scaled.csv'))
    X_val = pd.read_csv(os.path.join(data_dir, 'X_val_scaled.csv'))
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test_scaled.csv'))
    
    # Load labels
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))['label']
    y_val = pd.read_csv(os.path.join(data_dir, 'y_val.csv'))['label']
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv'))['label']
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def create_model(model_type='random_forest'):
    """
    Create a feature-based model of the specified type.
    
    Args:
        model_type (str): Type of model to create ('random_forest', 'gradient_boosting', or 'svm')
        
    Returns:
        object: The model
    """
    if model_type == 'random_forest':
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
    elif model_type == 'gradient_boosting':
        return GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
    elif model_type == 'svm':
        return SVC(
            kernel='rbf',
            C=10.0,
            gamma='scale',
            probability=True,
            class_weight='balanced',
            random_state=42
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def train_and_evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, output_dir, model_name):
    """
    Train and evaluate a model.
    
    Args:
        model: The model to train
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        X_test: Test features
        y_test: Test labels
        output_dir (str): Directory to save outputs
        model_name (str): Name of the model
        
    Returns:
        tuple: (trained_model, metrics)
    """
    print(f"Training {model_name} model...")
    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    val_preds = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_preds)
    print(f"Validation accuracy: {val_accuracy:.4f}")
    
    # Evaluate on test set
    test_preds = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_preds)
    test_report = classification_report(y_test, test_preds)
    test_cm = confusion_matrix(y_test, test_preds)
    
    print(f"Test accuracy: {test_accuracy:.4f}")
    print("Classification report:")
    print(test_report)
    
    # Save model
    model_path = os.path.join(output_dir, f'{model_name}_model.joblib')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Plot and save confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'AI-generated'],
                yticklabels=['Real', 'AI-generated'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    cm_path = os.path.join(output_dir, f'{model_name}_confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    
    # If model supports feature importance, plot it
    if hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_
        features = X_train.columns
        
        # Sort features by importance
        indices = np.argsort(feature_importances)[::-1]
        
        # Plot top 20 features
        plt.figure(figsize=(12, 8))
        plt.title(f'Feature Importances - {model_name}')
        plt.bar(range(min(20, len(indices))), 
                [feature_importances[i] for i in indices[:20]],
                align='center')
        plt.xticks(range(min(20, len(indices))), 
                  [features[i] for i in indices[:20]], 
                  rotation=90)
        plt.tight_layout()
        fi_path = os.path.join(output_dir, f'{model_name}_feature_importance.png')
        plt.savefig(fi_path)
        plt.close()
    
    metrics = {
        'accuracy': test_accuracy,
        'report': test_report,
        'confusion_matrix': test_cm
    }
    
    return model, metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a feature-based model for AI image detection")
    parser.add_argument("--data_dir", required=True, help="Directory containing the prepared data splits")
    parser.add_argument("--output_dir", required=True, help="Directory to save outputs")
    parser.add_argument("--model_type", default="random_forest", 
                      choices=["random_forest", "gradient_boosting", "svm"],
                      help="Type of model to train")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(args.data_dir)
    
    # Create and train model
    model = create_model(args.model_type)
    model, metrics = train_and_evaluate_model(
        model, X_train, y_train, X_val, y_val, X_test, y_test,
        args.output_dir, args.model_type
    )
    
    # Save metrics
    with open(os.path.join(args.output_dir, f'{args.model_type}_metrics.txt'), 'w') as f:
        f.write(f"Accuracy: {metrics['accuracy']}\n\n")
        f.write("Classification Report:\n")
        f.write(metrics['report'])
    
    print(f"Training complete! Results saved to {args.output_dir}") 