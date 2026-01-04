"""
Common Utilities Module

Shared utility functions used by both the API and Streamlit app.
This module provides:
- Model loading/caching
- Data preprocessing
- Common helpers

Import from here instead of app.utils to avoid circular dependencies.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def load_pretrained_model(model_path: Optional[str] = None):
    """
    Load a pretrained PINN model from disk.
    
    Args:
        model_path: Path to model checkpoint (.pth file)
                   If None, looks in default locations
    
    Returns:
        Loaded PyTorch model or None if not found
    """
    from src.ml.pinn import PhysicsInformedNN
    
    if model_path is None:
        # Try default locations
        possible_paths = [
            Path("models/pinn_best.pth"),
            Path("models/pinn_final.pth"),
            Path("results/pinn_demo/model_final.pth"),
            Path("../models/pinn_best.pth"),
        ]
        
        for path in possible_paths:
            if path.exists():
                model_path = str(path)
                break
    
    if model_path is None or not Path(model_path).exists():
        logger.warning(f"Model not found at {model_path}")
        return None
    
    try:
        # Initialize model
        model = PhysicsInformedNN(input_size=64)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        logger.info(f"Model loaded successfully from {model_path}")
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None


def prepare_model_input(
    convergence_map: np.ndarray,
    target_size: int = 64
) -> torch.Tensor:
    """
    Prepare convergence map for model input.
    
    Args:
        convergence_map: 2D numpy array
        target_size: Target size for resizing (default: 64)
    
    Returns:
        Preprocessed tensor of shape (1, 1, target_size, target_size)
    """
    # Ensure 2D
    if convergence_map.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {convergence_map.shape}")
    
    # Resize if needed
    if convergence_map.shape[0] != target_size or convergence_map.shape[1] != target_size:
        from scipy.ndimage import zoom
        zoom_factors = (target_size / convergence_map.shape[0], 
                       target_size / convergence_map.shape[1])
        convergence_map = zoom(convergence_map, zoom_factors, order=1)
    
    # Normalize to [0, 1]
    vmin, vmax = convergence_map.min(), convergence_map.max()
    if vmax > vmin:
        convergence_map = (convergence_map - vmin) / (vmax - vmin)
    
    # Convert to tensor (batch, channel, height, width)
    tensor = torch.from_numpy(convergence_map).float()
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    
    return tensor


def compute_classification_entropy(class_probs: np.ndarray) -> float:
    """
    Compute Shannon entropy of classification probabilities.
    Higher entropy = more uncertainty.
    
    Args:
        class_probs: Array of class probabilities (must sum to 1)
    
    Returns:
        Entropy value (0 to log(n_classes))
    """
    # Clip to avoid log(0)
    probs = np.clip(class_probs, 1e-10, 1.0)
    entropy = -np.sum(probs * np.log(probs))
    return entropy


def denormalize_parameters(
    normalized_params: np.ndarray,
    param_ranges: dict
) -> dict:
    """
    Denormalize predicted parameters back to physical units.
    
    Args:
        normalized_params: Array of normalized parameters [0, 1]
        param_ranges: Dictionary with min/max for each parameter
    
    Returns:
        Dictionary of denormalized parameters
    """
    denormalized = {}
    param_names = ['M_vir', 'r_s', 'ellipticity']
    
    for i, name in enumerate(param_names):
        if i < len(normalized_params):
            norm_val = normalized_params[i]
            min_val = param_ranges[name]['min']
            max_val = param_ranges[name]['max']
            denormalized[name] = norm_val * (max_val - min_val) + min_val
    
    return denormalized


def check_gpu_availability() -> Tuple[bool, str]:
    """
    Check if GPU is available for PyTorch.
    
    Returns:
        Tuple of (is_available, device_name)
    """
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        return True, device_name
    else:
        return False, "CPU"


def set_random_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
