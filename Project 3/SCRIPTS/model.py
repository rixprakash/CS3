import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os

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

def create_feature_enhanced_model(input_shape=(224, 224, 3), feature_dims=59, num_classes=1):
    """
    Create a model that combines CNN features with extracted image features.
    
    Args:
        input_shape (tuple): Input shape of the images
        feature_dims (int): Dimension of the extracted features
        num_classes (int): Number of output classes
        
    Returns:
        tensorflow.keras.Model: Compiled model
    """
    # Image input branch
    image_input = Input(shape=input_shape, name='image_input')
    
    # Feature input branch
    feature_input = Input(shape=(feature_dims,), name='feature_input')
    
    # CNN branch
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False
    
    # Process the image through CNN
    x = base_model(image_input)
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    # Process the features
    y = Dense(128, activation='relu')(feature_input)
    y = Dropout(0.3)(y)
    
    # Combine both branches
    combined = Concatenate()([x, y])
    
    # Final layers
    z = Dense(256, activation='relu')(combined)
    z = Dropout(0.4)(z)
    predictions = Dense(num_classes, activation='sigmoid')(z)
    
    # Create the model
    model = Model(inputs=[image_input, feature_input], outputs=predictions)
    
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

def train_feature_enhanced_model(model, X_train_images, X_train_features, y_train, 
                                X_val_images, X_val_features, y_val, 
                                epochs=50, batch_size=32):
    """
    Train the feature-enhanced model with callbacks for monitoring and early stopping.
    
    Args:
        model: Compiled model
        X_train_images: Training image data
        X_train_features: Training extracted features
        y_train: Training labels
        X_val_images: Validation image data
        X_val_features: Validation extracted features
        y_val: Validation labels
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        
    Returns:
        History object containing training history
    """
    # Define callbacks
    callbacks = [
        ModelCheckpoint(
            'best_feature_model.h5',
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
        {'image_input': X_train_images, 'feature_input': X_train_features},
        y_train,
        validation_data=({'image_input': X_val_images, 'feature_input': X_val_features}, y_val),
        epochs=epochs,
        callbacks=callbacks,
        batch_size=batch_size
    )
    
    return history

def create_feature_only_model(model_type='random_forest'):
    """
    Create a machine learning model that uses only extracted features.
    
    Args:
        model_type (str): Type of model to create ('random_forest', 'gradient_boosting', 'svm', 'mlp')
        
    Returns:
        sklearn model or pipeline
    """
    if model_type == 'random_forest':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                random_state=42
            ))
        ])
    
    elif model_type == 'gradient_boosting':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('model', GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            ))
        ])
    
    elif model_type == 'svm':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVC(
                C=1.0,
                kernel='rbf',
                probability=True,
                random_state=42
            ))
        ])
    
    elif model_type == 'mlp':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('model', MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size='auto',
                max_iter=200,
                random_state=42
            ))
        ])
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def train_feature_only_model(model, X_train, y_train, model_save_path=None):
    """
    Train a feature-only machine learning model.
    
    Args:
        model: Scikit-learn model or pipeline
        X_train: Training features
        y_train: Training labels
        model_save_path (str): Path to save the trained model (optional)
        
    Returns:
        Trained model
    """
    # Train the model
    model.fit(X_train, y_train)
    
    # Save the model if a path is provided
    if model_save_path:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        joblib.dump(model, model_save_path)
    
    return model

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

def evaluate_feature_enhanced_model(model, X_test_images, X_test_features, y_test):
    """
    Evaluate the feature-enhanced model on test data and generate performance metrics.
    
    Args:
        model: Trained model
        X_test_images: Test image data
        X_test_features: Test extracted features
        y_test: Test labels
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Get predictions
    y_pred_prob = model.predict({'image_input': X_test_images, 'feature_input': X_test_features})
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Calculate metrics
    report = classification_report(y_test, y_pred, target_names=['Real', 'AI'])
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return {
        'classification_report': report,
        'confusion_matrix': conf_matrix
    }

def evaluate_feature_only_model(model, X_test, y_test):
    """
    Evaluate a feature-only model on test data and generate performance metrics.
    
    Args:
        model: Trained scikit-learn model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    report = classification_report(y_test, y_pred, target_names=['Real', 'AI'])
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return {
        'classification_report': report,
        'confusion_matrix': conf_matrix
    }

def get_feature_importance(model):
    """
    Get feature importance for feature-only models that support it.
    
    Args:
        model: Trained scikit-learn model
        
    Returns:
        dict: Dictionary mapping feature names to importance scores, or None if not supported
    """
    # Get the actual model from the pipeline
    if hasattr(model, 'named_steps') and 'model' in model.named_steps:
        actual_model = model.named_steps['model']
    else:
        actual_model = model
    
    # Check if the model supports feature importance
    if hasattr(actual_model, 'feature_importances_'):
        return actual_model.feature_importances_
    elif hasattr(actual_model, 'coef_'):
        return actual_model.coef_
    else:
        return None

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

def plot_confusion_matrix(conf_matrix, class_names, title='Confusion Matrix'):
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        conf_matrix: Confusion matrix
        class_names: Names of the classes
        title: Title for the plot
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')
    plt.close()

def plot_feature_importance(importances, feature_names, title='Feature Importance', top_n=20):
    """
    Plot feature importance.
    
    Args:
        importances: Array of feature importance scores
        feature_names: Names of the features
        title: Title for the plot
        top_n: Number of top features to show
    """
    # Create DataFrame with features and importances
    features_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance and take top N
    top_features = features_df.sort_values('Importance', ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.barh(top_features['Feature'], top_features['Importance'])
    plt.xlabel('Importance')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')
    plt.close() 