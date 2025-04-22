import os
import numpy as np
import pandas as pd
import json
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    classification_report, cohen_kappa_score, matthews_corrcoef
)
from datetime import datetime

def evaluate_binary_classification(y_true, y_pred, y_prob=None, threshold=0.5):
    """
    Evaluate binary classification model performance
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted class labels (0 or 1)
        y_prob: Predicted probabilities for the positive class (optional)
        threshold: Classification threshold for probabilities
        
    Returns:
        Dictionary of metrics
    """
    # Convert probabilities to class predictions if provided
    if y_prob is not None and y_pred is None:
        y_pred = (y_prob > threshold).astype(int)
    
    # Calculate metrics
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
        'specificity': float(specificity_score(y_true, y_pred)),
        'mcc': float(matthews_corrcoef(y_true, y_pred)),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }
    
    # Add probability-based metrics if probabilities are provided
    if y_prob is not None:
        metrics.update({
            'threshold': threshold,
            'auc_roc': float(roc_auc_score(y_true, y_prob)),
            'auc_pr': float(average_precision_score(y_true, y_prob)),
        })
    
    return metrics

def evaluate_multiclass_classification(y_true, y_pred, y_prob=None, class_names=None):
    """
    Evaluate multiclass classification model performance
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted class labels
        y_prob: Predicted probabilities for each class (optional)
        class_names: List of class names (optional)
        
    Returns:
        Dictionary of metrics
    """
    # Create class names if not provided
    if class_names is None:
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        class_names = [f"Class {i}" for i in unique_classes]
    
    # Calculate overall metrics
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'kappa': float(cohen_kappa_score(y_true, y_pred)),
        'mcc': float(matthews_corrcoef(y_true, y_pred)),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }
    
    # Calculate per-class metrics
    class_report = classification_report(y_true, y_pred, 
                                         output_dict=True, 
                                         zero_division=0)
    
    # Extract per-class metrics
    class_metrics = {}
    for i, class_name in enumerate(class_names):
        if str(i) in class_report:
            class_metrics[class_name] = {
                'precision': float(class_report[str(i)]['precision']),
                'recall': float(class_report[str(i)]['recall']),
                'f1_score': float(class_report[str(i)]['f1-score']),
                'support': int(class_report[str(i)]['support'])
            }
    
    metrics['class_metrics'] = class_metrics
    
    # Add probability-based metrics if probabilities are provided
    if y_prob is not None:
        if y_prob.shape[1] > 2:  # Multiclass
            metrics.update({
                'weighted_auc_roc': float(roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')),
                'macro_auc_roc': float(roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')),
            })
    
    return metrics

def specificity_score(y_true, y_pred):
    """
    Calculate specificity score (true negative rate)
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted class labels
        
    Returns:
        Specificity score
    """
    cm = confusion_matrix(y_true, y_pred)
    if len(cm) <= 1:
        return 1.0
    tn = cm[0, 0]
    fp = cm[0, 1]
    if (tn + fp) == 0:
        return 0.0
    return tn / (tn + fp)

def calculate_confidence_intervals(metric_values, confidence=0.95):
    """
    Calculate confidence intervals for a list of metric values
    
    Args:
        metric_values: List of metric values from different runs
        confidence: Confidence level (default: 0.95)
        
    Returns:
        Dictionary with mean, lower and upper bounds
    """
    if len(metric_values) < 2:
        return {
            'mean': float(np.mean(metric_values)),
            'lower_bound': None,
            'upper_bound': None
        }
    
    # Calculate mean and standard error
    mean = np.mean(metric_values)
    std_err = np.std(metric_values, ddof=1) / np.sqrt(len(metric_values))
    
    # Calculate t-value for the given confidence level
    import scipy.stats as stats
    t_value = stats.t.ppf((1 + confidence) / 2, len(metric_values) - 1)
    
    # Calculate confidence intervals
    lower_bound = mean - t_value * std_err
    upper_bound = mean + t_value * std_err
    
    return {
        'mean': float(mean),
        'lower_bound': float(lower_bound),
        'upper_bound': float(upper_bound)
    }

def save_metrics(metrics, output_dir, filename='metrics.json'):
    """
    Save metrics to a JSON file
    
    Args:
        metrics: Dictionary of metrics
        output_dir: Directory to save the results
        filename: Name of the output file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Add timestamp to metrics
    metrics['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Save metrics to JSON file
    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Metrics saved to {output_path}")
    return output_path

def load_metrics(metrics_path):
    """
    Load metrics from a JSON file
    
    Args:
        metrics_path: Path to the metrics JSON file
        
    Returns:
        Dictionary of metrics
    """
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    return metrics

def aggregate_cross_validation_results(cv_results):
    """
    Aggregate metrics from cross-validation results
    
    Args:
        cv_results: List of dictionaries containing metrics from each fold
        
    Returns:
        Dictionary with aggregated metrics
    """
    # Initialize aggregated metrics
    aggregated = {}
    
    # Get all metric names from the first result
    if not cv_results:
        return aggregated
    
    metric_names = get_leaf_metric_names(cv_results[0])
    
    # Collect values for each metric across folds
    metric_values = {name: [] for name in metric_names}
    for result in cv_results:
        for name in metric_names:
            value = get_metric_by_name(result, name)
            if value is not None:
                metric_values[name].append(value)
    
    # Calculate statistics for each metric
    for name, values in metric_values.items():
        if values and all(isinstance(v, (int, float)) for v in values):
            ci = calculate_confidence_intervals(values)
            set_metric_by_name(aggregated, name, ci)
    
    return aggregated

def get_leaf_metric_names(metrics_dict, prefix=''):
    """
    Get a list of all leaf metric names in a nested dictionary
    
    Args:
        metrics_dict: Dictionary of metrics
        prefix: Prefix for nested metrics
        
    Returns:
        List of metric names
    """
    names = []
    for key, value in metrics_dict.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and not any(k in ['mean', 'lower_bound', 'upper_bound'] for k in value.keys()):
            names.extend(get_leaf_metric_names(value, full_key))
        elif not isinstance(value, (list, dict)):
            names.append(full_key)
    return names

def get_metric_by_name(metrics_dict, name):
    """
    Get a metric value from a nested dictionary using dot notation
    
    Args:
        metrics_dict: Dictionary of metrics
        name: Metric name in dot notation
        
    Returns:
        Metric value
    """
    parts = name.split('.')
    current = metrics_dict
    
    for part in parts:
        if part in current:
            current = current[part]
        else:
            return None
    
    return current if isinstance(current, (int, float)) else None

def set_metric_by_name(metrics_dict, name, value):
    """
    Set a metric value in a nested dictionary using dot notation
    
    Args:
        metrics_dict: Dictionary of metrics
        name: Metric name in dot notation
        value: Metric value
    """
    parts = name.split('.')
    current = metrics_dict
    
    for i, part in enumerate(parts[:-1]):
        if part not in current:
            current[part] = {}
        current = current[part]
    
    current[parts[-1]] = value

def compare_models(models_metrics, metric_keys=None):
    """
    Compare metrics across multiple models
    
    Args:
        models_metrics: Dictionary mapping model names to their metrics
        metric_keys: List of metric keys to compare (default: accuracy, f1_score, auc_roc)
        
    Returns:
        DataFrame with model comparison
    """
    if metric_keys is None:
        metric_keys = ['accuracy', 'f1_score', 'auc_roc']
    
    # Initialize comparison data
    comparison_data = []
    
    for model_name, metrics in models_metrics.items():
        model_data = {'model': model_name}
        
        for key in metric_keys:
            value = get_metric_by_name(metrics, key)
            if isinstance(value, dict) and 'mean' in value:
                model_data[key] = value['mean']
                model_data[f"{key}_ci"] = f"({value['lower_bound']:.4f}, {value['upper_bound']:.4f})"
            elif value is not None:
                model_data[key] = value
        
        comparison_data.append(model_data)
    
    # Create DataFrame
    return pd.DataFrame(comparison_data)

def calculate_feature_importance_metrics(feature_importances, feature_names):
    """
    Calculate metrics based on feature importances
    
    Args:
        feature_importances: Array of feature importance values
        feature_names: List of feature names
        
    Returns:
        Dictionary with feature importance metrics
    """
    # Normalize importances
    normalized = feature_importances / np.sum(feature_importances)
    
    # Sort features by importance
    sorted_indices = np.argsort(normalized)[::-1]
    sorted_importances = normalized[sorted_indices]
    sorted_names = [feature_names[i] for i in sorted_indices]
    
    # Calculate cumulative importance
    cumulative_importance = np.cumsum(sorted_importances)
    
    # Find how many features needed for X% of importance
    thresholds = [0.5, 0.8, 0.9, 0.95, 0.99]
    features_for_threshold = {}
    
    for threshold in thresholds:
        n_features = np.argmax(cumulative_importance >= threshold) + 1
        features_for_threshold[f"features_for_{int(threshold * 100)}pct"] = int(n_features)
    
    # Calculate entropy of feature importance distribution
    entropy = -np.sum(normalized * np.log2(normalized + 1e-10))
    max_entropy = np.log2(len(feature_names))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    # Create metrics dictionary
    metrics = {
        'top_features': [
            {
                'name': name,
                'importance': float(imp)
            } for name, imp in zip(sorted_names[:20], sorted_importances[:20])
        ],
        'total_features': len(feature_names),
        'entropy': float(entropy),
        'normalized_entropy': float(normalized_entropy),
        'feature_thresholds': features_for_threshold
    }
    
    return metrics 