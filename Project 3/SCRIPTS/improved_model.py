import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import joblib
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime

# Get logger
logger = logging.getLogger('ai_detector')

def create_model():
    """
    Create a CNN model based on EfficientNetB0 for binary classification.
    
    Returns:
        tf.keras.Model: The compiled CNN model
    """
    logger.info("Creating CNN model based on EfficientNetB0...")
    
    # Create base model from EfficientNetB0
    base_model = EfficientNetB0(weights='imagenet', include_top=False)
    
    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    # Create model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

def create_feature_enhanced_model(feature_dims=100):
    """
    Create a model that combines CNN features with extracted features.
    
    Args:
        feature_dims (int): Dimension of extracted features
        
    Returns:
        tf.keras.Model: The compiled hybrid model
    """
    logger.info(f"Creating hybrid model with CNN + {feature_dims} extracted features...")
    
    # CNN branch
    base_model = EfficientNetB0(weights='imagenet', include_top=False)
    img_input = base_model.input
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    cnn_features = Dense(128, activation='relu')(x)
    
    # Feature branch
    feature_input = Input(shape=(feature_dims,), name='feature_input')
    features = Dense(64, activation='relu')(feature_input)
    features = Dropout(0.3)(features)
    
    # Combine branches
    combined = Concatenate()([cnn_features, features])
    combined = Dense(64, activation='relu')(combined)
    combined = Dropout(0.3)(combined)
    predictions = Dense(1, activation='sigmoid')(combined)
    
    # Create model
    model = Model(inputs=[img_input, feature_input], outputs=predictions)
    
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

def create_feature_only_model(model_type='random_forest'):
    """
    Create a machine learning model that uses only extracted features.
    
    Args:
        model_type (str): Type of model to create ('random_forest', 'gradient_boosting', or 'svm')
        
    Returns:
        object: The created model
    """
    logger.info(f"Creating feature-only model: {model_type}")
    
    if model_type == 'random_forest':
        return RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            random_state=42, 
            n_jobs=-1, 
            verbose=1
        )
    elif model_type == 'gradient_boosting':
        return GradientBoostingClassifier(
            n_estimators=100, 
            max_depth=5, 
            random_state=42, 
            verbose=1
        )
    elif model_type == 'svm':
        return SVC(
            kernel='rbf', 
            probability=True, 
            random_state=42, 
            verbose=True
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def train_model(model, train_generator, validation_generator, epochs=20, batch_size=32, 
               checkpoint_path=None, log_dir=None):
    """
    Train a CNN model.
    
    Args:
        model (tf.keras.Model): The model to train
        train_generator: Training data generator
        validation_generator: Validation data generator
        epochs (int): Number of epochs to train
        batch_size (int): Batch size for training
        checkpoint_path (str): Path to save model checkpoints
        log_dir (str): Directory to save TensorBoard logs
        
    Returns:
        tf.keras.callbacks.History: Training history
    """
    logger.info(f"Training CNN model for {epochs} epochs with batch size {batch_size}")
    
    # Create callbacks
    callbacks = []
    
    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)
    
    # Model checkpoint callback
    if checkpoint_path:
        model_checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        )
        callbacks.append(model_checkpoint)
        logger.info(f"Model checkpoints will be saved to: {checkpoint_path}")
    
    # TensorBoard callback
    if log_dir:
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_dir = os.path.join(log_dir, f'logs/{current_time}')
        tensorboard_callback = TensorBoard(
            log_dir=tensorboard_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
        callbacks.append(tensorboard_callback)
        logger.info(f"TensorBoard logs will be saved to: {tensorboard_dir}")
    
    # Progress callback to track epochs
    class ProgressCallback(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            logger.info(f"Starting epoch {epoch+1}/{epochs}")
            
        def on_epoch_end(self, epoch, logs=None):
            log_str = f"Epoch {epoch+1}/{epochs} completed - "
            log_str += " - ".join(f"{k}: {v:.4f}" for k, v in logs.items())
            logger.info(log_str)
            
            # Save intermediate model after every 5 epochs
            if (epoch + 1) % 5 == 0 and log_dir:
                intermediate_path = os.path.join(log_dir, f'intermediate_model_epoch_{epoch+1}.h5')
                self.model.save(intermediate_path)
                logger.info(f"Saved intermediate model to {intermediate_path}")
    
    progress_callback = ProgressCallback()
    callbacks.append(progress_callback)
    
    # Calculate steps per epoch
    steps_per_epoch = train_generator.samples // batch_size
    validation_steps = validation_generator.samples // batch_size
    
    # Ensure at least 1 step
    steps_per_epoch = max(1, steps_per_epoch)
    validation_steps = max(1, validation_steps)
    
    logger.info(f"Training with {steps_per_epoch} steps per epoch and {validation_steps} validation steps")
    
    # Train model
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=1
    )
    
    logger.info("CNN model training completed")
    
    return history

def train_feature_enhanced_model(model, X_train_images, X_train_features, y_train,
                                X_val_images, X_val_features, y_val,
                                epochs=20, batch_size=32,
                                checkpoint_path=None, log_dir=None):
    """
    Train a hybrid model with CNN features and extracted features.
    
    Args:
        model (tf.keras.Model): The model to train
        X_train_images: Training image data
        X_train_features: Training extracted features
        y_train: Training labels
        X_val_images: Validation image data
        X_val_features: Validation extracted features
        y_val: Validation labels
        epochs (int): Number of epochs to train
        batch_size (int): Batch size for training
        checkpoint_path (str): Path to save model checkpoints
        log_dir (str): Directory to save TensorBoard logs
        
    Returns:
        tf.keras.callbacks.History: Training history
    """
    logger.info(f"Training hybrid model for {epochs} epochs with batch size {batch_size}")
    
    # Create callbacks
    callbacks = []
    
    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)
    
    # Model checkpoint callback
    if checkpoint_path:
        model_checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        )
        callbacks.append(model_checkpoint)
        logger.info(f"Model checkpoints will be saved to: {checkpoint_path}")
    
    # TensorBoard callback
    if log_dir:
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_dir = os.path.join(log_dir, f'logs/{current_time}')
        tensorboard_callback = TensorBoard(
            log_dir=tensorboard_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
        callbacks.append(tensorboard_callback)
        logger.info(f"TensorBoard logs will be saved to: {tensorboard_dir}")
    
    # Progress callback to track epochs
    class ProgressCallback(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            logger.info(f"Starting epoch {epoch+1}/{epochs}")
            
        def on_epoch_end(self, epoch, logs=None):
            log_str = f"Epoch {epoch+1}/{epochs} completed - "
            log_str += " - ".join(f"{k}: {v:.4f}" for k, v in logs.items())
            logger.info(log_str)
            
            # Save intermediate model after every 5 epochs
            if (epoch + 1) % 5 == 0 and log_dir:
                intermediate_path = os.path.join(log_dir, f'intermediate_model_epoch_{epoch+1}.h5')
                self.model.save(intermediate_path)
                logger.info(f"Saved intermediate model to {intermediate_path}")
    
    progress_callback = ProgressCallback()
    callbacks.append(progress_callback)
    
    # Train model
    logger.info(f"Starting training with {X_train_images.shape[0]} samples and {X_val_images.shape[0]} validation samples")
    
    # Create small batches for demonstration
    def batch_generator(X_img, X_feat, y, batch_size):
        num_samples = X_img.shape[0]
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        
        start_idx = 0
        while True:
            batch_indices = indices[start_idx:start_idx + batch_size]
            batch_X_img = X_img[batch_indices]
            batch_X_feat = X_feat[batch_indices]
            batch_y = y[batch_indices]
            
            yield [batch_X_img, batch_X_feat], batch_y
            
            start_idx += batch_size
            if start_idx >= num_samples:
                start_idx = 0
                np.random.shuffle(indices)
    
    # Calculate steps per epoch
    steps_per_epoch = len(X_train_images) // batch_size
    validation_steps = len(X_val_images) // batch_size
    
    # Ensure at least 1 step
    steps_per_epoch = max(1, steps_per_epoch)
    validation_steps = max(1, validation_steps)
    
    logger.info(f"Training with {steps_per_epoch} steps per epoch and {validation_steps} validation steps")
    
    # Train model
    history = model.fit(
        [X_train_images, X_train_features],
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=([X_val_images, X_val_features], y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    logger.info("Hybrid model training completed")
    
    return history

def train_feature_only_model(model, X_train_features, y_train, model_save_path=None):
    """
    Train a feature-only model.
    
    Args:
        model: The model to train
        X_train_features: Training extracted features
        y_train: Training labels
        model_save_path (str): Path to save the trained model
        
    Returns:
        object: The trained model
    """
    logger.info(f"Training feature-only model with {X_train_features.shape[0]} samples")
    
    try:
        # For scikit-learn models with verbose parameter
        if hasattr(model, 'verbose'):
            model.fit(X_train_features, y_train)
        else:
            # Display progress for models without built-in verbosity
            logger.info("Training model...")
            model.fit(X_train_features, y_train)
        
        # Save model if path provided
        if model_save_path:
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            joblib.dump(model, model_save_path)
            logger.info(f"Model saved to {model_save_path}")
        
        logger.info("Feature model training completed")
        return model
        
    except Exception as e:
        logger.error(f"Error training feature-only model: {e}")
        raise

def evaluate_model(model, test_generator):
    """
    Evaluate a CNN model.
    
    Args:
        model (tf.keras.Model): The trained model
        test_generator: Test data generator
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    logger.info("Evaluating CNN model...")
    
    # Get the number of batches in the test generator
    num_batches = test_generator.samples // test_generator.batch_size + 1
    
    # Evaluate model
    metrics = model.evaluate(test_generator, steps=num_batches, verbose=1)
    metric_names = model.metrics_names
    
    metrics_dict = {name: value for name, value in zip(metric_names, metrics)}
    logger.info(f"Evaluation metrics: {metrics_dict}")
    
    # Get predictions
    y_pred_prob = model.predict(test_generator, steps=num_batches, verbose=1)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Get true labels
    y_true = test_generator.classes
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    logger.info(f"Confusion matrix:\n{cm}")
    
    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=['Real', 'AI'])
    logger.info(f"Classification report:\n{report}")
    
    return {
        'metrics': metrics_dict,
        'confusion_matrix': cm,
        'classification_report': report,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_prob': y_pred_prob
    }

def evaluate_feature_enhanced_model(model, X_test_images, X_test_features, y_test):
    """
    Evaluate a hybrid model.
    
    Args:
        model (tf.keras.Model): The trained model
        X_test_images: Test image data
        X_test_features: Test extracted features
        y_test: Test labels
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    logger.info("Evaluating hybrid model...")
    
    # Evaluate model
    metrics = model.evaluate([X_test_images, X_test_features], y_test, verbose=1)
    metric_names = model.metrics_names
    
    metrics_dict = {name: value for name, value in zip(metric_names, metrics)}
    logger.info(f"Evaluation metrics: {metrics_dict}")
    
    # Get predictions
    y_pred_prob = model.predict([X_test_images, X_test_features], verbose=1)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"Confusion matrix:\n{cm}")
    
    # Generate classification report
    report = classification_report(y_test, y_pred, target_names=['Real', 'AI'])
    logger.info(f"Classification report:\n{report}")
    
    return {
        'metrics': metrics_dict,
        'confusion_matrix': cm,
        'classification_report': report,
        'y_true': y_test,
        'y_pred': y_pred,
        'y_pred_prob': y_pred_prob
    }

def evaluate_feature_only_model(model, X_test_features, y_test):
    """
    Evaluate a feature-only model.
    
    Args:
        model: The trained model
        X_test_features: Test extracted features
        y_test: Test labels
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    logger.info(f"Evaluating feature-only model with {X_test_features.shape[0]} test samples")
    
    # Get predictions
    try:
        y_pred_prob = model.predict_proba(X_test_features)[:, 1]
        y_pred = model.predict(X_test_features)
    except:
        # Some models don't have predict_proba
        y_pred = model.predict(X_test_features)
        y_pred_prob = y_pred
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"Confusion matrix:\n{cm}")
    
    # Generate classification report
    report = classification_report(y_test, y_pred, target_names=['Real', 'AI'])
    logger.info(f"Classification report:\n{report}")
    
    return {
        'confusion_matrix': cm,
        'classification_report': report,
        'y_true': y_test,
        'y_pred': y_pred,
        'y_pred_prob': y_pred_prob
    }

def get_feature_importance(model):
    """
    Get feature importance from a model if available.
    
    Args:
        model: The trained model
        
    Returns:
        array: Feature importance values or None if not available
    """
    try:
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        elif hasattr(model, 'coef_'):
            return np.abs(model.coef_[0])
        else:
            return None
    except:
        return None

def plot_training_history(history, save_path=None):
    """
    Plot training history.
    
    Args:
        history (tf.keras.callbacks.History): Training history
        save_path (str): Path to save the plot
    """
    logger.info("Plotting training history...")
    
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Training history plot saved to {save_path}")
    
    plt.close()

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        cm (array): Confusion matrix
        classes (list): Class names
        title (str): Plot title
        save_path (str): Path to save the plot
    """
    logger.info("Plotting confusion matrix...")
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Confusion matrix plot saved to {save_path}")
    
    plt.close()

def plot_feature_importance(importances, feature_names, title='Feature Importance', save_path=None, top_n=20):
    """
    Plot feature importance.
    
    Args:
        importances (array): Feature importance values
        feature_names (list): Feature names
        title (str): Plot title
        save_path (str): Path to save the plot
        top_n (int): Number of top features to show
    """
    logger.info("Plotting feature importance...")
    
    # Create DataFrame for feature importance
    df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    df = df.sort_values('Importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=df)
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Feature importance plot saved to {save_path}")
    
    plt.close() 