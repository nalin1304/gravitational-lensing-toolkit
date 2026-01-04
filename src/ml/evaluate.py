"""
Model Evaluation and Metrics for Physics-Informed Neural Network

This module provides comprehensive evaluation metrics and visualizations
for the trained PINN model.
"""

import torch
import numpy as np
from typing import Dict, Tuple, List, Optional
import matplotlib.pyplot as plt
try:
    from sklearn.metrics import confusion_matrix, classification_report
    from sklearn.calibration import calibration_curve
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    def confusion_matrix(*args, **kwargs): return []
    def classification_report(*args, **kwargs): return ""
    def calibration_curve(*args, **kwargs): return [], []


def compute_metrics(
    pred_params: np.ndarray,
    true_params: np.ndarray,
    pred_classes: np.ndarray,
    true_classes: np.ndarray,
    param_names: List[str] = None
) -> Dict[str, any]:
    """
    Compute comprehensive evaluation metrics.
    
    Parameters
    ----------
    pred_params : np.ndarray
        Predicted parameters (N, 5)
    true_params : np.ndarray
        True parameters (N, 5)
    pred_classes : np.ndarray
        Predicted class labels (N,)
    true_classes : np.ndarray
        True class labels (N,)
    param_names : list, optional
        Names of parameters for reporting
    
    Returns
    -------
    metrics : dict
        Dictionary containing all metrics
    """
    if param_names is None:
        param_names = ['M_vir', 'r_s', 'beta_x', 'beta_y', 'H0']
    
    metrics = {}
    
    if not SKLEARN_AVAILABLE:
        print("Warning: scikit-learn not installed. Some metrics disabled.")
    
    # Parameter regression metrics
    mae = np.mean(np.abs(pred_params - true_params), axis=0)
    rmse = np.sqrt(np.mean((pred_params - true_params)**2, axis=0))
    mape = np.mean(np.abs((pred_params - true_params) / (true_params + 1e-10)) * 100, axis=0)
    
    metrics['parameter_metrics'] = {}
    for i, name in enumerate(param_names):
        metrics['parameter_metrics'][name] = {
            'MAE': float(mae[i]),
            'RMSE': float(rmse[i]),
            'MAPE': float(mape[i])
        }
    
    # Overall parameter metrics
    metrics['overall_MAE'] = float(np.mean(mae))
    metrics['overall_RMSE'] = float(np.mean(rmse))
    metrics['overall_MAPE'] = float(np.mean(mape))
    
    # Classification metrics
    accuracy = np.mean(pred_classes == true_classes)
    metrics['classification_accuracy'] = float(accuracy)
    
    # Per-class accuracy
    class_names = ['CDM', 'WDM', 'SIDM']
    metrics['per_class_accuracy'] = {}
    for i, name in enumerate(class_names):
        mask = true_classes == i
        if np.sum(mask) > 0:
            class_acc = np.mean(pred_classes[mask] == true_classes[mask])
            metrics['per_class_accuracy'][name] = float(class_acc)
    
    # Confusion matrix
    cm = confusion_matrix(true_classes, pred_classes)
    metrics['confusion_matrix'] = cm.tolist()
    
    return metrics


def evaluate_model(
    model,
    dataloader,
    device: str = 'cpu',
    return_predictions: bool = True
) -> Dict:
    """
    Evaluate model on a dataset.
    
    Parameters
    ----------
    model : PhysicsInformedNN
        Trained model
    dataloader : torch.utils.data.DataLoader
        DataLoader for evaluation data
    device : str
        Device for computation
    return_predictions : bool
        Whether to return all predictions
    
    Returns
    -------
    results : dict
        Evaluation results including metrics and predictions
    """
    model.eval()
    model.to(device)
    
    all_pred_params = []
    all_true_params = []
    all_pred_classes = []
    all_true_classes = []
    all_class_probs = []
    
    with torch.no_grad():
        for images, params, labels in dataloader:
            images = images.to(device)
            params = params.to(device)
            labels = labels.to(device)
            
            # Forward pass
            pred_params, pred_class_logits = model(images)
            
            # Get class predictions
            class_probs = torch.softmax(pred_class_logits, dim=1)
            pred_classes = torch.argmax(class_probs, dim=1)
            
            # Store results
            all_pred_params.append(pred_params.cpu().numpy())
            all_true_params.append(params.cpu().numpy())
            all_pred_classes.append(pred_classes.cpu().numpy())
            all_true_classes.append(labels.cpu().numpy())
            all_class_probs.append(class_probs.cpu().numpy())
    
    # Concatenate all batches
    pred_params = np.concatenate(all_pred_params, axis=0)
    true_params = np.concatenate(all_true_params, axis=0)
    pred_classes = np.concatenate(all_pred_classes, axis=0)
    true_classes = np.concatenate(all_true_classes, axis=0)
    class_probs = np.concatenate(all_class_probs, axis=0)
    
    # Compute metrics
    metrics = compute_metrics(pred_params, true_params, pred_classes, true_classes)
    
    results = {
        'metrics': metrics
    }
    
    if return_predictions:
        results['predictions'] = {
            'pred_params': pred_params,
            'true_params': true_params,
            'pred_classes': pred_classes,
            'true_classes': true_classes,
            'class_probs': class_probs
        }
    
    return results


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = None,
    normalize: bool = False,
    title: str = 'Confusion Matrix',
    cmap: str = 'Blues'
) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix (n_classes, n_classes)
    class_names : list, optional
        Names of classes
    normalize : bool
        Whether to normalize by row
    title : str
        Plot title
    cmap : str
        Colormap name
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    if class_names is None:
        class_names = ['CDM', 'WDM', 'SIDM']
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    
    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=14, fontweight='bold')
    
    fig.tight_layout()
    return fig


def plot_parameter_errors(
    pred_params: np.ndarray,
    true_params: np.ndarray,
    param_names: List[str] = None
) -> plt.Figure:
    """
    Plot parameter prediction errors.
    
    Parameters
    ----------
    pred_params : np.ndarray
        Predicted parameters (N, 5)
    true_params : np.ndarray
        True parameters (N, 5)
    param_names : list, optional
        Names of parameters
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    if param_names is None:
        param_names = ['M_vir', 'r_s', 'β_x', 'β_y', 'H0']
    
    errors = pred_params - true_params
    relative_errors = errors / (true_params + 1e-10) * 100
    
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    
    for i, name in enumerate(param_names):
        # Absolute error histogram
        ax = axes[0, i]
        ax.hist(errors[:, i], bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Error', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(f'{name}\nMAE={np.abs(errors[:, i]).mean():.3f}', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Relative error histogram
        ax = axes[1, i]
        ax.hist(relative_errors[:, i], bins=50, alpha=0.7, color='green', edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Relative Error (%)', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(f'MAPE={np.abs(relative_errors[:, i]).mean():.1f}%', fontsize=11)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_calibration_curve(
    class_probs: np.ndarray,
    true_classes: np.ndarray,
    n_bins: int = 10
) -> plt.Figure:
    """
    Plot calibration curve for classification probabilities.
    
    Parameters
    ----------
    class_probs : np.ndarray
        Predicted class probabilities (N, 3)
    true_classes : np.ndarray
        True class labels (N,)
    n_bins : int
        Number of bins for calibration curve
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    class_names = ['CDM', 'WDM', 'SIDM']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, name in enumerate(class_names):
        # Binary problem: class i vs rest
        y_true_binary = (true_classes == i).astype(int)
        y_prob = class_probs[:, i]
        
        # Compute calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true_binary, y_prob, n_bins=n_bins, strategy='uniform'
        )
        
        ax = axes[i]
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        ax.plot(mean_predicted_value, fraction_of_positives, 's-', label=f'{name}')
        ax.set_xlabel('Mean predicted probability', fontsize=11)
        ax.set_ylabel('Fraction of positives', fontsize=11)
        ax.set_title(f'Calibration: {name}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_parameter_scatter(
    pred_params: np.ndarray,
    true_params: np.ndarray,
    param_names: List[str] = None,
    max_samples: int = 1000
) -> plt.Figure:
    """
    Plot predicted vs true parameters as scatter plots.
    
    Parameters
    ----------
    pred_params : np.ndarray
        Predicted parameters (N, 5)
    true_params : np.ndarray
        True parameters (N, 5)
    param_names : list, optional
        Names of parameters
    max_samples : int
        Maximum number of samples to plot
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    if param_names is None:
        param_names = ['M_vir', 'r_s', 'β_x', 'β_y', 'H0']
    
    # Subsample if too many points
    if len(pred_params) > max_samples:
        indices = np.random.choice(len(pred_params), max_samples, replace=False)
        pred_params = pred_params[indices]
        true_params = true_params[indices]
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    for i, name in enumerate(param_names):
        ax = axes[i]
        
        # Scatter plot
        ax.scatter(true_params[:, i], pred_params[:, i], alpha=0.3, s=10)
        
        # Perfect prediction line
        min_val = min(true_params[:, i].min(), pred_params[:, i].min())
        max_val = max(true_params[:, i].max(), pred_params[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
        
        # Compute R²
        ss_res = np.sum((true_params[:, i] - pred_params[:, i])**2)
        ss_tot = np.sum((true_params[:, i] - true_params[:, i].mean())**2)
        r2 = 1 - ss_res / ss_tot
        
        ax.set_xlabel(f'True {name}', fontsize=11)
        ax.set_ylabel(f'Predicted {name}', fontsize=11)
        ax.set_title(f'{name}\nR²={r2:.3f}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    
    plt.tight_layout()
    return fig


def print_evaluation_summary(metrics: Dict):
    """
    Print formatted evaluation summary.
    
    Parameters
    ----------
    metrics : dict
        Metrics dictionary from compute_metrics
    """
    print("="*70)
    print(" "*20 + "MODEL EVALUATION SUMMARY")
    print("="*70)
    
    print("\n1. CLASSIFICATION METRICS:")
    print(f"   Overall Accuracy: {metrics['classification_accuracy']*100:.2f}%")
    print("\n   Per-class Accuracy:")
    for cls, acc in metrics['per_class_accuracy'].items():
        print(f"     {cls}: {acc*100:.2f}%")
    
    print("\n2. PARAMETER REGRESSION METRICS:")
    print(f"   Overall MAE:  {metrics['overall_MAE']:.4f}")
    print(f"   Overall RMSE: {metrics['overall_RMSE']:.4f}")
    print(f"   Overall MAPE: {metrics['overall_MAPE']:.2f}%")
    
    print("\n   Per-parameter metrics:")
    for param, values in metrics['parameter_metrics'].items():
        print(f"\n   {param}:")
        print(f"     MAE:  {values['MAE']:.4f}")
        print(f"     RMSE: {values['RMSE']:.4f}")
        print(f"     MAPE: {values['MAPE']:.2f}%")
    
    print("\n3. CONFUSION MATRIX:")
    cm = np.array(metrics['confusion_matrix'])
    class_names = ['CDM', 'WDM', 'SIDM']
    print("\n        " + "  ".join(f"{name:>6s}" for name in class_names))
    for i, name in enumerate(class_names):
        print(f"   {name:>4s} " + "  ".join(f"{cm[i, j]:6d}" for j in range(3)))
    
    print("\n" + "="*70)
