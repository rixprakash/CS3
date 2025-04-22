import logging
import os
import sys
from datetime import datetime

def init_logger(log_file=None, log_level=logging.INFO, name='ai_detector'):
    """
    Initialize a logger for the AI image detection project.
    
    Args:
        log_file (str): Path to log file. If None, a default path will be used.
        log_level (int): Logging level (e.g., logging.INFO, logging.DEBUG)
        name (str): Logger name
        
    Returns:
        logging.Logger: Configured logger object
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Clear existing handlers
    if logger.handlers:
        logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is specified
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Add timestamp to log filename if not already provided
        if not log_file.endswith('.log'):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f"{log_file}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")
    
    # Log system information
    logger.info(f"Logger initialized with level: {logging.getLevelName(log_level)}")
    logger.info(f"Python version: {sys.version}")
    
    return logger

def get_logger(name='ai_detector'):
    """
    Get an existing logger or create a new one if it doesn't exist.
    
    Args:
        name (str): Logger name
        
    Returns:
        logging.Logger: Logger object
    """
    logger = logging.getLogger(name)
    
    # If logger is not configured, configure it with default settings
    if not logger.handlers:
        return init_logger(name=name)
    
    return logger 