import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping
import torch
import json
from datetime import datetime

def get_checkpoint_path(output_dir, model_name='model'):
    """
    Generate a checkpoint path for saving model states.
    
    Args:
        output_dir (str): Directory to save checkpoints
        model_name (str): Name of the model
        
    Returns:
        str: Path to checkpoint directory
    """
    # Create timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(output_dir, 'checkpoints', f"{model_name}_{timestamp}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    return checkpoint_dir

def get_tf_callbacks(checkpoint_dir, model_name='model', patience=10, monitor='val_loss'):
    """
    Get TensorFlow callbacks for model training.
    
    Args:
        checkpoint_dir (str): Directory to save checkpoints
        model_name (str): Name of the model
        patience (int): Number of epochs with no improvement after which training will be stopped
        monitor (str): Metric to monitor for early stopping and model checkpointing
        
    Returns:
        list: List of TensorFlow callbacks
    """
    # Create directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create TensorBoard directory
    tensorboard_dir = os.path.join(checkpoint_dir, 'tensorboard')
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    # Create logs directory
    logs_dir = os.path.join(checkpoint_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Model checkpoint callback
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_best.h5")
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor=monitor,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        verbose=1
    )
    
    # CSV logger callback
    csv_path = os.path.join(logs_dir, f"{model_name}_training.csv")
    csv_logger = CSVLogger(csv_path, append=True, separator=',')
    
    # TensorBoard callback
    tensorboard_callback = TensorBoard(
        log_dir=tensorboard_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        update_freq='epoch',
        profile_batch=0
    )
    
    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor=monitor,
        patience=patience,
        verbose=1,
        restore_best_weights=True,
        mode='auto'
    )
    
    return [checkpoint_callback, csv_logger, tensorboard_callback, early_stopping]

def save_torch_model(model, checkpoint_dir, model_name='model', epoch=None, metrics=None):
    """
    Save PyTorch model and training metrics.
    
    Args:
        model (torch.nn.Module): PyTorch model to save
        checkpoint_dir (str): Directory to save model
        model_name (str): Name of the model
        epoch (int): Current epoch number
        metrics (dict): Dictionary of metrics to save
        
    Returns:
        str: Path to saved model
    """
    # Create directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Add epoch to model name if provided
    if epoch is not None:
        model_file = f"{model_name}_epoch_{epoch}.pt"
    else:
        model_file = f"{model_name}_final.pt"
    
    model_path = os.path.join(checkpoint_dir, model_file)
    
    # Save model state
    torch.save(model.state_dict(), model_path)
    
    # Save metrics if provided
    if metrics:
        metrics_file = os.path.join(checkpoint_dir, f"{model_name}_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
    
    return model_path

def load_torch_model(model, model_path):
    """
    Load PyTorch model from saved state.
    
    Args:
        model (torch.nn.Module): PyTorch model to load weights into
        model_path (str): Path to saved model state
        
    Returns:
        torch.nn.Module: Loaded model
    """
    # Load model state
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set model to evaluation mode
    
    return model

def load_tf_model(model_path):
    """
    Load TensorFlow model from saved file.
    
    Args:
        model_path (str): Path to saved model
        
    Returns:
        tf.keras.Model: Loaded model
    """
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    return model 