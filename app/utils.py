"""
Core utility functions for the Streamlit web interface (Phase 10).

This module contains testable functions separated from Streamlit-specific code.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Tuple, List, Optional
from pathlib import Path

# Import project modules
from src.lens_models import LensSystem, NFWProfile, EllipticalNFWProfile
from src.ml.pinn import PhysicsInformedNN
from src.ml.generate_dataset import generate_convergence_map_vectorized


def generate_synthetic_convergence(
    profile_type: str,
    mass: float,
    scale_radius: float,
    ellipticity: float,
    grid_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic convergence map.
    
    Parameters
    ----------
    profile_type : str
        Type of lens profile ('NFW' or 'Elliptical NFW')
    mass : float
        Virial mass in solar masses
    scale_radius : float
        Scale radius in kpc
    ellipticity : float
        Ellipticity (0 to 0.5, only for Elliptical NFW)
    grid_size : int
        Resolution of the convergence map
    
    Returns
    -------
    convergence_map : np.ndarray
        Generated convergence map (grid_size, grid_size)
    X : np.ndarray
        X coordinate grid
    Y : np.ndarray
        Y coordinate grid
    """
    # Create lens system
    lens_system = LensSystem(z_lens=0.5, z_source=1.5)
    
    # Create lens profile
    if profile_type == "NFW":
        lens = NFWProfile(
            M_vir=mass,
            concentration=10.0,
            lens_system=lens_system
        )
    elif profile_type == "Elliptical NFW":
        lens = EllipticalNFWProfile(
            M_vir=mass,
            concentration=10.0,
            lens_sys=lens_system,
            ellipticity=ellipticity,
            position_angle=45.0
        )
    else:
        raise ValueError(f"Unknown profile type: {profile_type}")
    
    # Generate convergence map
    convergence_map = generate_convergence_map_vectorized(
        lens,
        grid_size=grid_size,
        extent=2.0  # ±2 arcseconds
    )
    
    # Create coordinate grids
    extent = 2.0
    x = np.linspace(-extent, extent, grid_size)
    y = np.linspace(-extent, extent, grid_size)
    X, Y = np.meshgrid(x, y)
    
    return convergence_map, X, Y


def plot_convergence_map(
    convergence_map: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    title: str = "Convergence Map",
    cmap: str = "viridis"
) -> plt.Figure:
    """
    Create convergence map visualization.
    
    Parameters
    ----------
    convergence_map : np.ndarray
        Convergence values
    X, Y : np.ndarray
        Coordinate grids
    title : str
        Plot title
    cmap : str
        Matplotlib colormap name
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    
    im = ax.contourf(X, Y, convergence_map, levels=20, cmap=cmap)
    ax.contour(X, Y, convergence_map, levels=10, colors='white', 
              alpha=0.3, linewidths=0.5)
    
    ax.set_xlabel('x (arcsec)', fontsize=12)
    ax.set_ylabel('y (arcsec)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('κ (convergence)', fontsize=12)
    
    plt.tight_layout()
    return fig


def plot_uncertainty_bars(
    param_names: List[str],
    means: np.ndarray,
    stds: np.ndarray
) -> plt.Figure:
    """
    Create uncertainty visualization with error bars.
    
    Parameters
    ----------
    param_names : list of str
        Parameter names
    means : np.ndarray
        Mean values
    stds : np.ndarray
        Standard deviations
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(len(param_names))
    
    # Normalize for better visualization
    normalized_means = means / (np.abs(means) + 1e-10)
    normalized_stds = stds / (np.abs(means) + 1e-10)
    
    ax.bar(x_pos, normalized_means, yerr=normalized_stds, 
           capsize=5, alpha=0.7, color='steelblue', edgecolor='black')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(param_names, fontsize=11)
    ax.set_ylabel('Normalized Value ± Uncertainty', fontsize=12)
    ax.set_title('Parameter Estimates with Uncertainty', 
                fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    return fig


def plot_classification_probs(
    class_names: List[str],
    probabilities: np.ndarray,
    entropy: float
) -> plt.Figure:
    """
    Visualize classification probabilities.
    
    Parameters
    ----------
    class_names : list of str
        Class names
    probabilities : np.ndarray
        Probability for each class
    entropy : float
        Predictive entropy
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar chart
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    bars = ax1.bar(class_names, probabilities, color=colors, 
                   alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Probability', fontsize=12)
    ax1.set_title('Dark Matter Classification', fontsize=13, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, prob in zip(bars, probabilities):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{prob:.1%}', ha='center', va='bottom', fontsize=10)
    
    # Pie chart
    ax2.pie(probabilities, labels=class_names, autopct='%1.1f%%',
            colors=colors, startangle=90, textprops={'fontsize': 11})
    ax2.set_title(f'Confidence (Entropy: {entropy:.3f})', 
                 fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_comparison(
    original: np.ndarray,
    processed: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray
) -> plt.Figure:
    """
    Create side-by-side comparison of original and processed data.
    
    Parameters
    ----------
    original : np.ndarray
        Original data
    processed : np.ndarray
        Processed data
    X, Y : np.ndarray
        Coordinate grids
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Original
    im1 = ax1.contourf(X, Y, original, levels=20, cmap='viridis')
    ax1.set_xlabel('x (arcsec)', fontsize=11)
    ax1.set_ylabel('y (arcsec)', fontsize=11)
    ax1.set_title('Original Image', fontsize=12, fontweight='bold')
    ax1.set_aspect('equal')
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    
    # Processed
    im2 = ax2.contourf(X, Y, processed, levels=20, cmap='viridis')
    ax2.set_xlabel('x (arcsec)', fontsize=11)
    ax2.set_ylabel('y (arcsec)', fontsize=11)
    ax2.set_title('Processed (64×64, Normalized)', fontsize=12, fontweight='bold')
    ax2.set_aspect('equal')
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    plt.tight_layout()
    return fig


def load_pretrained_model(model_path: Optional[str] = None) -> PhysicsInformedNN:
    """
    Load pre-trained PINN model.
    
    Parameters
    ----------
    model_path : str, optional
        Path to model weights file
    
    Returns
    -------
    model : PhysicsInformedNN
        Loaded model
    """
    model = PhysicsInformedNN(input_size=64, dropout_rate=0.2)
    
    if model_path and Path(model_path).exists():
        try:
            try:
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
            except TypeError:
                checkpoint = torch.load(model_path, map_location='cpu')
            model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
            model.eval()
        except Exception:
            # Silently continue with untrained model
            pass
    
    return model


def prepare_model_input(
    convergence_map: np.ndarray,
    target_size: int = 64
) -> torch.Tensor:
    """
    Prepare convergence map for model input.
    
    Parameters
    ----------
    convergence_map : np.ndarray
        Input convergence map
    target_size : int
        Target resolution
    
    Returns
    -------
    tensor : torch.Tensor
        Prepared tensor (1, 1, target_size, target_size)
    """
    # Resize if needed
    if convergence_map.shape[0] != target_size:
        from scipy.ndimage import zoom
        scale = target_size / convergence_map.shape[0]
        convergence_map = zoom(convergence_map, scale, order=1)
    
    # Normalize
    normalized = (convergence_map - convergence_map.min()) / \
                (convergence_map.max() - convergence_map.min() + 1e-10)
    
    # Convert to tensor
    tensor = torch.from_numpy(normalized).float()
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    
    return tensor


def compute_classification_entropy(probabilities: np.ndarray) -> float:
    """
    Compute predictive entropy for classification.
    
    Parameters
    ----------
    probabilities : np.ndarray
        Class probabilities
    
    Returns
    -------
    entropy : float
        Predictive entropy
    """
    return -np.sum(probabilities * np.log(probabilities + 1e-10))


def format_parameter_value(param_name: str, value: float) -> str:
    """
    Format parameter value for display.
    
    Parameters
    ----------
    param_name : str
        Parameter name
    value : float
        Parameter value
    
    Returns
    -------
    formatted : str
        Formatted string
    """
    if 'M_vir' in param_name:
        return f"{value:.2e}"
    elif 'H' in param_name:
        return f"{value:.2f}"
    else:
        return f"{value:.4f}"
