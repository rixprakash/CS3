import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def create_model(input_shape=(224, 224, 3), num_classes=1):
    """
    Create the EfficientNet-based model for AI image detection.
    
    Args:
        input_shape (tuple): Input shape of the images
        num_classes (int): Number of output classes
        
    Returns:
        tensorflow.keras.Model: Compiled model
    """
    # Load pre-trained EfficientNetB0
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='sigmoid')(x)
    
    # Create the model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, train_generator, val_generator, epochs=50, batch_size=32):
    """
    Train the model with callbacks for monitoring and early stopping.
    
    Args:
        model: Compiled model
        train_generator: Training data generator
        val_generator: Validation data generator
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        
    Returns:
        History object containing training history
    """
    # Define callbacks
    callbacks = [
        ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    # Train the model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks,
        batch_size=batch_size
    )
    
    return history

def evaluate_model(model, test_generator):
    """
    Evaluate the model on test data and generate performance metrics.
    
    Args:
        model: Trained model
        test_generator: Test data generator
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Get predictions
    y_pred = model.predict(test_generator)
    y_pred = (y_pred > 0.5).astype(int)
    
    # Get true labels
    y_true = test_generator.classes
    
    # Calculate metrics
    report = classification_report(y_true, y_pred, target_names=['Real', 'AI'])
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    return {
        'classification_report': report,
        'confusion_matrix': conf_matrix
    }

def plot_training_history(history):
    """
    Plot training and validation accuracy/loss curves.
    
    Args:
        history: History object from model training
    """
    # Plot accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close() 