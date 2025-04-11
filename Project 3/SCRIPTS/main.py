import os
import sys
from preprocessing import preprocess_data, create_data_generators
from model import create_model, train_model, evaluate_model, plot_training_history

def main():
    # Set paths
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'DATA')
    json_path = os.path.join(data_dir, 'DeepGuardDB.json')
    
    # Preprocess data
    print("Preprocessing data...")
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(
        data_dir=data_dir,
        json_path=json_path,
        test_size=0.2,
        val_size=0.2
    )
    
    # Create data generators
    print("Creating data generators...")
    train_generator, val_generator, test_generator = create_data_generators(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    
    # Create and train model
    print("Creating model...")
    model = create_model()
    
    print("Training model...")
    history = train_model(
        model,
        train_generator,
        val_generator,
        epochs=50,
        batch_size=32
    )
    
    # Evaluate model
    print("Evaluating model...")
    metrics = evaluate_model(model, test_generator)
    print("\nClassification Report:")
    print(metrics['classification_report'])
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    
    # Plot training history
    print("Plotting training history...")
    plot_training_history(history)
    
    print("Done!")

if __name__ == "__main__":
    main() 