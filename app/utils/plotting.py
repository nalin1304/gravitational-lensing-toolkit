"""
Plotting Utilities

Centralized plotting functions for consistent visualization across pages.
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from typing import Optional, Tuple, Dict, Any, List
import matplotlib.colors as mcolors
import logging


def setup_publication_style():
    """Configure matplotlib for publication-quality plots."""
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 13,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'axes.linewidth': 1.0,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.5,
    })


def _validate_plot_inputs(
    data: np.ndarray,
    X: np.ndarray = None,
    Y: np.ndarray = None
) -> Tuple[bool, str, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Validate and prepare inputs for plotting.

    Returns:
        (is_valid, error_message, safe_data, safe_X, safe_Y)
    """
    if data is None:
        return False, "No data available", None, None, None

    try:
        # Convert to numpy arrays if needed
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)

        # Check for non-finite values (Infs)
        # We replace Infs with NaNs so plotting can continue, but warn if too many
        if not np.isfinite(data).all():
            if np.isinf(data).any():
                logging.warning("Input data contains infinite values. Replacing with NaN.")
                data = data.copy() # Avoid modifying original
                data[np.isinf(data)] = np.nan

        # Validate data dimensions (assuming 2D for convergence map)
        if data.ndim != 2:
            return False, f"Invalid data shape: {data.shape} (must be 2D)", None, None, None

        # Validate coordinates if provided
        if X is not None and Y is not None:
            if not isinstance(X, np.ndarray): X = np.asarray(X)
            if not isinstance(Y, np.ndarray): Y = np.asarray(Y)

            # Handle 1D coordinates by creating meshgrid
            if X.ndim == 1 and Y.ndim == 1:
                if X.shape[0] == data.shape[1] and Y.shape[0] == data.shape[0]:
                    X, Y = np.meshgrid(X, Y)
                else:
                    return False, f"1D Coordinates mismatch: X{X.shape}!=cols{data.shape[1]}, Y{Y.shape}!=rows{data.shape[0]}", None, None, None

            # Validate shapes match
            if X.shape != data.shape or Y.shape != data.shape:
                return False, f"Shape mismatch: Data {data.shape}, X {X.shape}, Y {Y.shape}", None, None, None

        return True, "", data, X, Y

    except Exception as e:
        logging.error(f"Error validating inputs: {e}")
        return False, f"Validation error: {str(e)}", None, None, None


def plot_convergence_map(
    convergence: np.ndarray,
    fov: float,
    title: str = "Convergence Map (κ)",
    cmap: str = 'viridis',
    show_colorbar: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    fig_size: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot convergence map with proper formatting.
    
    Parameters
    ----------
    convergence : np.ndarray
        Convergence map data
    fov : float
        Field of view in arcseconds
    title : str
        Plot title
    cmap : str
        Colormap name
    show_colorbar : bool
        Whether to show colorbar
    vmin, vmax : float, optional
        Value range for colormap
    fig_size : tuple
        Figure size (width, height) in inches
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    """
    # Validate inputs
    is_valid, error_msg, convergence, _, _ = _validate_plot_inputs(convergence)

    if not is_valid:
        fig, ax = plt.subplots(figsize=fig_size)
        ax.text(0.5, 0.5, error_msg, ha='center', va='center')
        ax.set_title(title, fontsize=12, pad=10)
        return fig

    fig, ax = plt.subplots(figsize=fig_size)
    
    extent = [-fov/2, fov/2, -fov/2, fov/2]
    
    # Check for all NaNs (after Inf replacement)
    if np.all(np.isnan(convergence)):
        ax.text(0.5, 0.5, "Data contains only NaN values", ha='center', va='center')
        ax.set_title(title, fontsize=12, pad=10)
        return fig

    im = ax.imshow(
        convergence,
        origin='lower',
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect='auto'
    )
    
    ax.set_xlabel('θₓ (arcsec)', fontsize=11)
    ax.set_ylabel('θᵧ (arcsec)', fontsize=11)
    ax.set_title(title, fontsize=12, pad=10)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('κ', rotation=0, labelpad=15, fontsize=11)
    
    plt.tight_layout()
    return fig


def plot_magnification_map(
    magnification: np.ndarray,
    fov: float,
    title: str = "Magnification Map (μ)",
    log_scale: bool = False,
    show_critical_curves: bool = True,
    fig_size: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot magnification map with critical curve highlighting.
    
    Parameters
    ----------
    magnification : np.ndarray
        Magnification map data
    fov : float
        Field of view in arcseconds
    title : str
        Plot title
    log_scale : bool
        Use logarithmic color scale
    show_critical_curves : bool
        Highlight critical curves (|μ| > 10)
    fig_size : tuple
        Figure size
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # Validate inputs
    is_valid, error_msg, magnification, _, _ = _validate_plot_inputs(magnification)

    if not is_valid:
        fig, ax = plt.subplots(figsize=fig_size)
        ax.text(0.5, 0.5, error_msg, ha='center', va='center')
        ax.set_title(title, fontsize=12, pad=10)
        return fig

    fig, ax = plt.subplots(figsize=fig_size)
    
    extent = [-fov/2, fov/2, -fov/2, fov/2]
    
    # Handle log scale
    if log_scale:
        plot_data = np.log10(np.abs(magnification) + 1e-6)
        vmin, vmax = None, None
    else:
        plot_data = magnification
        # Clip extreme values for better visualization
        vmin = np.percentile(plot_data, 1)
        vmax = np.percentile(plot_data, 99)
    
    im = ax.imshow(
        plot_data,
        origin='lower',
        extent=extent,
        cmap='RdBu_r',
        vmin=vmin,
        vmax=vmax,
        aspect='auto'
    )
    
    # Overlay critical curves
    if show_critical_curves:
        critical = np.abs(magnification) > 10
        if critical.any():
            ax.contour(
                critical,
                levels=[0.5],
                colors='yellow',
                linewidths=2,
                extent=extent,
                linestyles='--'
            )
            ax.plot([], [], 'y--', linewidth=2, label='Critical curves')
            ax.legend(loc='upper right')
    
    ax.set_xlabel('θₓ (arcsec)', fontsize=11)
    ax.set_ylabel('θᵧ (arcsec)', fontsize=11)
    
    if log_scale:
        ax.set_title(f"{title} (log scale)", fontsize=12, pad=10)
        cbar_label = 'log₁₀(|μ|)'
    else:
        ax.set_title(title, fontsize=12, pad=10)
        cbar_label = 'μ'
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, rotation=0, labelpad=15, fontsize=11)
    
    plt.tight_layout()
    return fig


def plot_comparison(
    data1: np.ndarray,
    data2: np.ndarray,
    fov: float,
    title1: str = "Model 1",
    title2: str = "Model 2",
    title_diff: str = "Difference",
    cmap: str = 'viridis',
    fig_size: Tuple[int, int] = (15, 5)
) -> plt.Figure:
    """
    Plot side-by-side comparison of two maps with difference.
    
    Parameters
    ----------
    data1, data2 : np.ndarray
        Data to compare
    fov : float
        Field of view
    title1, title2, title_diff : str
        Subplot titles
    cmap : str
        Colormap
    fig_size : tuple
        Figure size
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # Validate inputs
    is_valid, error_msg, data1, _, _ = _validate_plot_inputs(data1)
    if is_valid:
        is_valid, error_msg_2, data2, _, _ = _validate_plot_inputs(data2)
        if not is_valid:
            error_msg = error_msg_2

    if not is_valid:
        fig, ax = plt.subplots(figsize=fig_size)
        ax.text(0.5, 0.5, error_msg, ha='center', va='center')
        ax.set_title("Comparison Error", fontsize=12, pad=10)
        return fig

    fig, axes = plt.subplots(1, 3, figsize=fig_size)
    
    extent = [-fov/2, fov/2, -fov/2, fov/2]
    
    # Handle NaNs in data
    has_nan1 = np.all(np.isnan(data1))
    has_nan2 = np.all(np.isnan(data2))

    if not has_nan1 and not has_nan2:
        # Find common colorbar limits if valid
        vmin = min(np.nanmin(data1), np.nanmin(data2))
        vmax = max(np.nanmax(data1), np.nanmax(data2))
    else:
        vmin, vmax = 0, 1 # Default fallback
    
    # Plot data1
    if has_nan1:
        axes[0].text(0.5, 0.5, "All NaN Data", ha='center', va='center')
    else:
        im1 = axes[0].imshow(data1, origin='lower', extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    axes[0].set_title(title1, fontsize=12)
    axes[0].set_xlabel('θₓ (arcsec)')
    axes[0].set_ylabel('θᵧ (arcsec)')
    
    # Plot data2
    if has_nan2:
        axes[1].text(0.5, 0.5, "All NaN Data", ha='center', va='center')
    else:
        im2 = axes[1].imshow(data2, origin='lower', extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    axes[1].set_title(title2, fontsize=12)
    axes[1].set_xlabel('θₓ (arcsec)')
    axes[1].set_ylabel('θᵧ (arcsec)')
    
    # Plot difference
    diff = data1 - data2
    im3 = axes[2].imshow(diff, origin='lower', extent=extent, cmap='RdBu_r',
                        vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
    axes[2].set_title(title_diff, fontsize=12)
    axes[2].set_xlabel('θₓ (arcsec)')
    axes[2].set_ylabel('θᵧ (arcsec)')
    cbar3 = plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    
    # Add statistics text
    rms = np.sqrt(np.mean(diff**2))
    max_diff = np.abs(diff).max()
    axes[2].text(0.02, 0.98, f'RMS: {rms:.3e}\nMax: {max_diff:.3e}',
                transform=axes[2].transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    for ax in axes:
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    return fig


def plot_radial_profile(
    data: np.ndarray,
    fov: float,
    label: str = "Profile",
    log_scale: bool = False,
    fig_size: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot azimuthally-averaged radial profile.
    
    Parameters
    ----------
    data : np.ndarray
        2D map data
    fov : float
        Field of view in arcseconds
    label : str
        Data label for legend
    log_scale : bool
        Use log scale for y-axis
    fig_size : tuple
        Figure size
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Create coordinate grid
    n = data.shape[0]
    y, x = np.ogrid[:n, :n]
    center = n // 2
    r_pixels = np.sqrt((x - center)**2 + (y - center)**2)
    
    # Convert to arcseconds
    pixel_scale = fov / n
    r_arcsec = r_pixels * pixel_scale
    
    # Bin radially
    r_max = fov / 2
    r_bins = np.linspace(0, r_max, 50)
    r_centers = (r_bins[:-1] + r_bins[1:]) / 2
    
    profile = []
    profile_std = []
    
    for i in range(len(r_bins) - 1):
        mask = (r_arcsec >= r_bins[i]) & (r_arcsec < r_bins[i+1])
        if mask.sum() > 0:
            profile.append(np.mean(data[mask]))
            profile_std.append(np.std(data[mask]))
        else:
            profile.append(np.nan)
            profile_std.append(np.nan)
    
    profile = np.array(profile)
    profile_std = np.array(profile_std)
    
    # Plot
    ax.plot(r_centers, profile, '-', label=label, linewidth=2)
    ax.fill_between(r_centers, profile - profile_std, profile + profile_std,
                     alpha=0.3, label='± 1σ')
    
    ax.set_xlabel('Radius (arcsec)', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title('Radial Profile', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend()
    
    if log_scale:
        ax.set_yscale('log')
    
    plt.tight_layout()
    return fig


def plot_training_history(
    history: Dict[str, list],
    metrics: list = ['loss', 'val_loss'],
    fig_size: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot training history.
    
    Parameters
    ----------
    history : dict
        Training history with metric names as keys
    metrics : list
        Metrics to plot
    fig_size : tuple
        Figure size
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=fig_size)
    
    if len(metrics) == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        if metric in history:
            epochs = range(1, len(history[metric]) + 1)
            ax.plot(epochs, history[metric], '-o', markersize=3, linewidth=1.5)
            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
            ax.set_title(f'{metric.replace("_", " ").title()} vs Epoch', fontsize=12)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Add best value annotation
            best_val = min(history[metric]) if 'loss' in metric else max(history[metric])
            best_epoch = history[metric].index(best_val) + 1
            ax.axhline(y=best_val, color='r', linestyle='--', alpha=0.5, linewidth=1)
            ax.text(0.98, 0.02, f'Best: {best_val:.4f} (Epoch {best_epoch})',
                   transform=ax.transAxes, fontsize=9,
                   verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig


def display_figure(fig: plt.Figure, use_container_width: bool = True):
    """Display matplotlib figure in Streamlit with proper cleanup."""
    st.pyplot(fig, use_container_width=use_container_width)
    plt.close(fig)  # Prevent memory leaks
