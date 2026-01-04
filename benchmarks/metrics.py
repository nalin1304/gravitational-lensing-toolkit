"""
Scientific Validation Metrics

Provides metrics for comparing predictions with ground truth or established codes

Author: Phase 13 Implementation
Date: October 2025
"""

import numpy as np
from typing import Tuple, Dict, Optional
from scipy import stats
try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    def ssim(*args, **kwargs): return 0.0
    def psnr(*args, **kwargs): return 0.0


def calculate_relative_error(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    epsilon: float = 1e-10
) -> float:
    """
    Calculate relative error between prediction and ground truth
    
    Args:
        predicted: Predicted values
        ground_truth: Ground truth values
        epsilon: Small value to avoid division by zero
        
    Returns:
        float: Mean relative error
    """
    relative_error = np.abs(predicted - ground_truth) / (np.abs(ground_truth) + epsilon)
    return np.mean(relative_error)


def calculate_chi_squared(
    predicted: np.ndarray,
    observed: np.ndarray,
    uncertainty: Optional[np.ndarray] = None
) -> Tuple[float, float]:
    """
    Calculate chi-squared statistic and p-value
    
    Args:
        predicted: Predicted values
        observed: Observed values
        uncertainty: Measurement uncertainties (if None, uses sqrt(observed))
        
    Returns:
        tuple: (chi_squared, p_value)
    """
    if uncertainty is None:
        uncertainty = np.sqrt(np.abs(observed) + 1e-10)
    
    chi2 = np.sum(((observed - predicted) / uncertainty) ** 2)
    dof = len(observed.flatten()) - 1
    p_value = 1 - stats.chi2.cdf(chi2, dof)
    
    return chi2, p_value


def calculate_rmse(
    predicted: np.ndarray,
    ground_truth: np.ndarray
) -> float:
    """
    Calculate Root Mean Squared Error
    
    Args:
        predicted: Predicted values
        ground_truth: Ground truth values
        
    Returns:
        float: RMSE
    """
    return np.sqrt(np.mean((predicted - ground_truth) ** 2))


def calculate_mae(
    predicted: np.ndarray,
    ground_truth: np.ndarray
) -> float:
    """
    Calculate Mean Absolute Error
    
    Args:
        predicted: Predicted values
        ground_truth: Ground truth values
        
    Returns:
        float: MAE
    """
    return np.mean(np.abs(predicted - ground_truth))


def calculate_structural_similarity(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    data_range: Optional[float] = None
) -> float:
    """
    Calculate Structural Similarity Index (SSIM)
    
    Useful for comparing 2D convergence maps
    
    Args:
        predicted: Predicted 2D map
        ground_truth: Ground truth 2D map
        data_range: Data range (if None, calculated automatically)
        
    Returns:
        float: SSIM score (0 to 1, higher is better)
    """
    if data_range is None:
        data_range = ground_truth.max() - ground_truth.min()
    
    if not SKIMAGE_AVAILABLE:
        print("Warning: scikit-image not installed. SSIM matching disabled.")
        return 0.0
        
    return ssim(ground_truth, predicted, data_range=data_range)


def calculate_peak_signal_noise_ratio(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    data_range: Optional[float] = None
) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR)
    
    Args:
        predicted: Predicted values
        ground_truth: Ground truth values
        data_range: Data range (if None, calculated automatically)
        
    Returns:
        float: PSNR in dB (higher is better)
    """
    if data_range is None:
        data_range = ground_truth.max() - ground_truth.min()
    
    if not SKIMAGE_AVAILABLE:
        print("Warning: scikit-image not installed. PSNR calculation disabled.")
        return 0.0
        
    return psnr(ground_truth, predicted, data_range=data_range)


def calculate_pearson_correlation(
    predicted: np.ndarray,
    ground_truth: np.ndarray
) -> Tuple[float, float]:
    """
    Calculate Pearson correlation coefficient
    
    Args:
        predicted: Predicted values
        ground_truth: Ground truth values
        
    Returns:
        tuple: (correlation, p_value)
    """
    correlation, p_value = stats.pearsonr(predicted.flatten(), ground_truth.flatten())
    return correlation, p_value


def calculate_fractional_bias(
    predicted: np.ndarray,
    ground_truth: np.ndarray
) -> float:
    """
    Calculate fractional bias
    
    Args:
        predicted: Predicted values
        ground_truth: Ground truth values
        
    Returns:
        float: Fractional bias
    """
    return 2 * np.mean(predicted - ground_truth) / (np.mean(predicted) + np.mean(ground_truth))


def calculate_residuals(
    predicted: np.ndarray,
    ground_truth: np.ndarray
) -> Dict[str, float]:
    """
    Calculate various residual statistics
    
    Args:
        predicted: Predicted values
        ground_truth: Ground truth values
        
    Returns:
        dict: Dictionary of residual statistics
    """
    residuals = predicted - ground_truth
    
    return {
        'mean': np.mean(residuals),
        'std': np.std(residuals),
        'median': np.median(residuals),
        'min': np.min(residuals),
        'max': np.max(residuals),
        'q25': np.percentile(residuals, 25),
        'q75': np.percentile(residuals, 75),
        'iqr': np.percentile(residuals, 75) - np.percentile(residuals, 25)
    }


def calculate_confidence_interval(
    data: np.ndarray,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate confidence interval for data
    
    Args:
        data: Data array
        confidence: Confidence level (default 0.95 for 95%)
        
    Returns:
        tuple: (lower_bound, upper_bound)
    """
    mean = np.mean(data)
    sem = stats.sem(data)
    interval = sem * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
    
    return mean - interval, mean + interval


def calculate_normalized_cross_correlation(
    predicted: np.ndarray,
    ground_truth: np.ndarray
) -> float:
    """
    Calculate normalized cross-correlation
    
    Args:
        predicted: Predicted values
        ground_truth: Ground truth values
        
    Returns:
        float: Normalized cross-correlation (-1 to 1)
    """
    pred_norm = (predicted - np.mean(predicted)) / np.std(predicted)
    truth_norm = (ground_truth - np.mean(ground_truth)) / np.std(ground_truth)
    
    return np.mean(pred_norm * truth_norm)


def calculate_all_metrics(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    uncertainty: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate all available metrics
    
    Args:
        predicted: Predicted values
        ground_truth: Ground truth values
        uncertainty: Measurement uncertainties
        
    Returns:
        dict: Dictionary of all metrics
    """
    metrics = {
        'relative_error': calculate_relative_error(predicted, ground_truth),
        'rmse': calculate_rmse(predicted, ground_truth),
        'mae': calculate_mae(predicted, ground_truth),
        'fractional_bias': calculate_fractional_bias(predicted, ground_truth),
    }
    
    # Add chi-squared if uncertainty provided
    if uncertainty is not None:
        chi2, p_value = calculate_chi_squared(predicted, ground_truth, uncertainty)
        metrics['chi_squared'] = chi2
        metrics['chi_squared_pvalue'] = p_value
    
    # Add correlation
    correlation, corr_pvalue = calculate_pearson_correlation(predicted, ground_truth)
    metrics['pearson_correlation'] = correlation
    metrics['pearson_pvalue'] = corr_pvalue
    
    # Add normalized cross-correlation
    metrics['normalized_cross_correlation'] = calculate_normalized_cross_correlation(
        predicted, ground_truth
    )
    
    # Add 2D image metrics if data is 2D
    if predicted.ndim == 2 and ground_truth.ndim == 2:
        metrics['ssim'] = calculate_structural_similarity(predicted, ground_truth)
        metrics['psnr'] = calculate_peak_signal_noise_ratio(predicted, ground_truth)
    
    # Add residual statistics
    residuals = calculate_residuals(predicted, ground_truth)
    for key, value in residuals.items():
        metrics[f'residual_{key}'] = value
    
    return metrics


def print_metrics_report(metrics: Dict[str, float], title: str = "Metrics Report"):
    """
    Print formatted metrics report
    
    Args:
        metrics: Dictionary of metrics
        title: Report title
    """
    print("=" * 60)
    print(f"{title:^60}")
    print("=" * 60)
    
    # Group metrics
    error_metrics = ['relative_error', 'rmse', 'mae', 'fractional_bias']
    correlation_metrics = ['pearson_correlation', 'normalized_cross_correlation']
    image_metrics = ['ssim', 'psnr']
    chi2_metrics = ['chi_squared', 'chi_squared_pvalue']
    
    # Print error metrics
    print("\nError Metrics:")
    print("-" * 60)
    for key in error_metrics:
        if key in metrics:
            print(f"  {key:30s}: {metrics[key]:12.6e}")
    
    # Print correlation metrics
    print("\nCorrelation Metrics:")
    print("-" * 60)
    for key in correlation_metrics:
        if key in metrics:
            print(f"  {key:30s}: {metrics[key]:12.6f}")
    
    # Print image metrics if available
    if any(key in metrics for key in image_metrics):
        print("\nImage Quality Metrics:")
        print("-" * 60)
        for key in image_metrics:
            if key in metrics:
                print(f"  {key:30s}: {metrics[key]:12.6f}")
    
    # Print chi-squared if available
    if any(key in metrics for key in chi2_metrics):
        print("\nChi-Squared Test:")
        print("-" * 60)
        for key in chi2_metrics:
            if key in metrics:
                print(f"  {key:30s}: {metrics[key]:12.6e}")
    
    # Print residual statistics
    print("\nResidual Statistics:")
    print("-" * 60)
    residual_keys = [k for k in metrics.keys() if k.startswith('residual_')]
    for key in residual_keys:
        label = key.replace('residual_', '')
        print(f"  {label:30s}: {metrics[key]:12.6e}")
    
    print("=" * 60)
