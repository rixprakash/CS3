import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import pandas as pd
from datetime import datetime

def setup_plot_style():
    """Setup common plotting style for consistent visualizations"""
    sns.set(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12

def save_figure(plt, output_dir, filename, timestamp=True):
    """
    Save a matplotlib figure to the specified output directory
    
    Args:
        plt: matplotlib.pyplot instance
        output_dir (str): Directory to save the figure
        filename (str): Filename for the figure
        timestamp (bool): Whether to add timestamp to filename
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Add timestamp to filename if requested
    if timestamp:
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(filename)
        filename = f"{base}_{time_str}{ext}"
    
    # Save figure
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')

def plot_training_history(history, output_dir=None, figsize=(18, 10)):
    """
    Plot training history for a Keras model
    
    Args:
        history: Keras history object or dictionary containing training metrics
        output_dir (str, optional): Directory to save plots
        figsize (tuple): Figure size (width, height)
    """
    setup_plot_style()
    
    # Convert history object to dict if needed
    if not isinstance(history, dict):
        history = history.history
    
    metrics = [m for m in history.keys() if not m.startswith('val_')]
    
    fig, axs = plt.subplots(len(metrics), 1, figsize=figsize, sharex=True)
    if len(metrics) == 1:
        axs = [axs]
    
    for i, metric in enumerate(metrics):
        ax = axs[i]
        ax.plot(history[metric], label=f'Training {metric}')
        
        # Plot validation metric if available
        val_metric = f'val_{metric}'
        if val_metric in history:
            ax.plot(history[val_metric], label=f'Validation {metric}')
        
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_xlabel('Epoch' if i == len(metrics) - 1 else '')
        ax.legend(loc='best')
        ax.grid(True)
    
    plt.tight_layout()
    
    if output_dir:
        save_figure(plt, output_dir, 'training_history.png')
    else:
        plt.show()

def plot_confusion_matrix(y_true, y_pred, classes, output_dir=None, figsize=(10, 8), cmap=plt.cm.Blues):
    """
    Plot confusion matrix for classification results
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels (not probabilities)
        classes: List of class names
        output_dir (str, optional): Directory to save plot
        figsize (tuple): Figure size (width, height)
        cmap: Colormap for the plot
    """
    setup_plot_style()
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize the confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot raw counts
    im1 = ax1.imshow(cm, interpolation='nearest', cmap=cmap)
    ax1.set_title('Confusion Matrix (counts)')
    plt.colorbar(im1, ax=ax1)
    
    # Plot normalized values
    im2 = ax2.imshow(cm_norm, interpolation='nearest', cmap=cmap)
    ax2.set_title('Confusion Matrix (normalized)')
    plt.colorbar(im2, ax=ax2)
    
    # Configure both axes
    for ax in [ax1, ax2]:
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.set_yticklabels(classes)
        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')
    
    # Add text annotations
    fmt_raw = 'd'
    fmt_norm = '.2f'
    
    thresh1 = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax1.text(j, i, format(cm[i, j], fmt_raw),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh1 else "black")
    
    thresh2 = 0.5
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax2.text(j, i, format(cm_norm[i, j], fmt_norm),
                    horizontalalignment="center",
                    color="white" if cm_norm[i, j] > thresh2 else "black")
    
    plt.tight_layout()
    
    if output_dir:
        save_figure(plt, output_dir, 'confusion_matrix.png')
    else:
        plt.show()

def plot_roc_curve(y_true, y_prob, output_dir=None, figsize=(10, 8)):
    """
    Plot ROC curve for binary classification
    
    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities for the positive class
        output_dir (str, optional): Directory to save plot
        figsize (tuple): Figure size (width, height)
    """
    setup_plot_style()
    
    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    if output_dir:
        save_figure(plt, output_dir, 'roc_curve.png')
    else:
        plt.show()

def plot_precision_recall_curve(y_true, y_prob, output_dir=None, figsize=(10, 8)):
    """
    Plot Precision-Recall curve for binary classification
    
    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities for the positive class
        output_dir (str, optional): Directory to save plot
        figsize (tuple): Figure size (width, height)
    """
    setup_plot_style()
    
    # Compute Precision-Recall curve and average precision
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    average_precision = average_precision_score(y_true, y_prob)
    
    # Plot Precision-Recall curve
    plt.figure(figsize=figsize)
    plt.plot(recall, precision, color='darkorange', lw=2, 
             label=f'Precision-Recall curve (AP = {average_precision:.2f})')
    plt.axhline(y=sum(y_true)/len(y_true), color='navy', lw=2, linestyle='--', 
                label=f'Baseline (class balance = {sum(y_true)/len(y_true):.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True)
    
    if output_dir:
        save_figure(plt, output_dir, 'precision_recall_curve.png')
    else:
        plt.show()

def plot_feature_importance(feature_names, importance_values, output_dir=None, figsize=(12, 10), top_n=20):
    """
    Plot feature importance for tree-based models
    
    Args:
        feature_names: List of feature names
        importance_values: Array of feature importance values
        output_dir (str, optional): Directory to save plot
        figsize (tuple): Figure size (width, height)
        top_n (int): Number of top features to display
    """
    setup_plot_style()
    
    # Create dataframe for sorting
    feature_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_values
    }).sort_values('Importance', ascending=False)
    
    # Get top N features
    if top_n and len(feature_df) > top_n:
        feature_df = feature_df.head(top_n)
    
    # Plot feature importance
    plt.figure(figsize=figsize)
    sns.barplot(x='Importance', y='Feature', data=feature_df, palette='viridis')
    plt.title(f'Top {len(feature_df)} Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    if output_dir:
        save_figure(plt, output_dir, 'feature_importance.png')
    else:
        plt.show()

def plot_class_distribution(y, classes=None, output_dir=None, figsize=(12, 8)):
    """
    Plot class distribution in a dataset
    
    Args:
        y: Array-like of class labels
        classes (list, optional): List of class names
        output_dir (str, optional): Directory to save plot
        figsize (tuple): Figure size (width, height)
    """
    setup_plot_style()
    
    # Count class occurrences
    class_counts = pd.Series(y).value_counts().sort_index()
    
    # Use provided class names if available
    if classes:
        class_counts.index = [classes[i] for i in class_counts.index]
    
    # Plot distribution
    plt.figure(figsize=figsize)
    ax = sns.barplot(x=class_counts.index, y=class_counts.values, palette='viridis')
    
    # Add count and percentage annotations
    total = len(y)
    for i, count in enumerate(class_counts.values):
        percentage = 100 * count / total
        ax.text(i, count/2, f"{count}\n({percentage:.1f}%)", 
                ha='center', va='center', color='white', fontweight='bold')
    
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if output_dir:
        save_figure(plt, output_dir, 'class_distribution.png')
    else:
        plt.show()

def plot_image_samples(images, labels=None, predictions=None, class_names=None, 
                      n_samples=5, output_dir=None, figsize=(15, 10)):
    """
    Plot sample images with optional labels and predictions
    
    Args:
        images: Array of images (n_samples, height, width, channels)
        labels (optional): Array of true labels
        predictions (optional): Array of predicted labels or probabilities
        class_names (optional): List of class names
        n_samples (int): Number of samples to display
        output_dir (str, optional): Directory to save plot
        figsize (tuple): Figure size (width, height)
    """
    setup_plot_style()
    
    # Limit to n_samples
    if len(images) > n_samples:
        indices = np.random.choice(len(images), n_samples, replace=False)
        images = images[indices]
        if labels is not None:
            labels = labels[indices]
        if predictions is not None:
            predictions = predictions[indices]
    
    # Create figure
    n_cols = min(5, n_samples)
    n_rows = (n_samples + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_samples > 1 else [axes]
    
    # Plot each image
    for i, (img, ax) in enumerate(zip(images, axes)):
        ax.imshow(img)
        ax.axis('off')
        
        title = ""
        if labels is not None and class_names is not None:
            label_idx = labels[i] if not isinstance(labels[i], np.ndarray) else np.argmax(labels[i])
            title += f"True: {class_names[label_idx]}"
        
        if predictions is not None and class_names is not None:
            if isinstance(predictions[i], np.ndarray) and len(predictions[i]) > 1:
                # Softmax probabilities
                pred_idx = np.argmax(predictions[i])
                prob = predictions[i][pred_idx]
                title += f"\nPred: {class_names[pred_idx]} ({prob:.2f})"
            else:
                # Class index
                pred_idx = predictions[i] if not isinstance(predictions[i], np.ndarray) else np.argmax(predictions[i])
                title += f"\nPred: {class_names[pred_idx]}"
        
        ax.set_title(title)
    
    # Hide unused axes
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
        axes[j].set_visible(False)
    
    plt.tight_layout()
    
    if output_dir:
        save_figure(plt, output_dir, 'image_samples.png')
    else:
        plt.show()

def plot_calibration_curve(y_true, y_prob, n_bins=10, output_dir=None, figsize=(10, 8)):
    """
    Plot calibration curve for probability predictions
    
    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities for the positive class
        n_bins (int): Number of bins for histogram
        output_dir (str, optional): Directory to save plot
        figsize (tuple): Figure size (width, height)
    """
    setup_plot_style()
    
    # Create equal-width bins for probabilities
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges) - 1
    bin_indices = np.minimum(bin_indices, n_bins - 1)  # Ensure we don't exceed the last bin
    
    # Calculate mean predicted probability and true positive rate in each bin
    bin_probs = np.zeros(n_bins)
    bin_true_probs = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    
    for i in range(n_bins):
        bin_mask = (bin_indices == i)
        if np.any(bin_mask):
            bin_probs[i] = np.mean(y_prob[bin_mask])
            bin_true_probs[i] = np.mean(y_true[bin_mask])
            bin_counts[i] = np.sum(bin_mask)
    
    # Plot calibration curve
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
    
    # Calibration curve (top plot)
    ax1.plot(bin_probs, bin_true_probs, marker='o', linewidth=2, label='Calibration curve')
    ax1.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
    ax1.set_xlabel('Mean predicted probability')
    ax1.set_ylabel('Fraction of positives')
    ax1.set_title('Calibration Curve')
    ax1.legend(loc='best')
    ax1.grid(True)
    
    # Histogram of predictions (bottom plot)
    ax2.bar(bin_edges[:-1], bin_counts, width=1/n_bins, align='edge', alpha=0.8)
    ax2.set_xlabel('Predicted probability')
    ax2.set_ylabel('Count')
    ax2.set_title('Histogram of predicted probabilities')
    ax2.grid(True)
    
    plt.tight_layout()
    
    if output_dir:
        save_figure(plt, output_dir, 'calibration_curve.png')
    else:
        plt.show() 