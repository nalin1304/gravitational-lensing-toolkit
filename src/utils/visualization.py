"""
Visualization Functions for Gravitational Lensing

This module provides plotting functions to visualize lens systems,
convergence maps, deflection fields, and other lensing quantities.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from typing import Optional, Tuple
import matplotlib.colors as colors


def setup_dark_style():
    """Set up a dark background style for astronomy plots."""
    plt.style.use('dark_background')
    

def plot_lens_system(lens_model,
                    source_pos: Tuple[float, float],
                    image_pos: np.ndarray,
                    magnifications: Optional[np.ndarray] = None,
                    convergence_map: Optional[np.ndarray] = None,
                    grid_x: Optional[np.ndarray] = None,
                    grid_y: Optional[np.ndarray] = None,
                    show_einstein_radius: bool = True,
                    figsize: Tuple[float, float] = (12, 10),
                    save_path: Optional[str] = None):
    """
    Plot a complete lens system visualization.
    
    This creates a comprehensive plot showing:
    - Convergence map as background
    - Lens position at origin
    - Source position
    - Image positions labeled A, B, C, D
    - Einstein radius (optional)
    - Critical curves (optional)
    
    Parameters
    ----------
    lens_model : MassProfile
        The lens model (e.g., PointMassProfile, NFWProfile)
    source_pos : tuple of float
        Source position (x, y) in arcseconds
    image_pos : np.ndarray
        Array of image positions, shape (N, 2)
    magnifications : np.ndarray, optional
        Array of magnifications for each image
    convergence_map : np.ndarray, optional
        2D convergence map for background
    grid_x : np.ndarray, optional
        x-coordinates of grid
    grid_y : np.ndarray, optional
        y-coordinates of grid
    show_einstein_radius : bool, optional
        Whether to show Einstein radius circle (default: True)
    figsize : tuple, optional
        Figure size (default: (12, 10))
    save_path : str, optional
        Path to save figure (if None, figure is displayed)
        
    Examples
    --------
    >>> from lens_models import LensSystem, PointMassProfile
    >>> from optics import ray_trace
    >>> from utils import plot_lens_system
    >>> lens_sys = LensSystem(0.5, 1.5)
    >>> lens = PointMassProfile(1e12, lens_sys)
    >>> results = ray_trace((0.5, 0.0), lens)
    >>> plot_lens_system(lens, (0.5, 0.0), results['image_positions'],
    ...                  results['magnifications'], results['convergence_map'],
    ...                  results['grid_x'], results['grid_y'])
    """
    setup_dark_style()
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Plot convergence map if provided
    if convergence_map is not None and grid_x is not None and grid_y is not None:
        extent = [grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()]
        
        # Use log scale for better visualization
        vmin = np.percentile(convergence_map[convergence_map > 0], 1)
        vmax = np.percentile(convergence_map[convergence_map > 0], 99)
        
        im = ax.imshow(convergence_map, extent=extent, origin='lower',
                      cmap='viridis', alpha=0.8,
                      norm=colors.LogNorm(vmin=max(vmin, 1e-4), vmax=vmax))
        
        cbar = plt.colorbar(im, ax=ax, label='Convergence κ')
        cbar.ax.tick_params(labelsize=10)
    
    # Plot lens at origin
    ax.plot(0, 0, 'o', color='gold', markersize=15, 
            label='Lens', zorder=10, markeredgecolor='white', markeredgewidth=1.5)
    
    # Plot source position
    ax.plot(source_pos[0], source_pos[1], '*', color='red', 
            markersize=20, label='Source', zorder=11,
            markeredgecolor='white', markeredgewidth=1.5)
    
    # Plot image positions
    if len(image_pos) > 0:
        labels = ['A', 'B', 'C', 'D', 'E', 'F']
        for i, (img, label) in enumerate(zip(image_pos, labels[:len(image_pos)])):
            color = 'cyan'
            marker_size = 12
            
            # Make size proportional to magnification if available
            if magnifications is not None and i < len(magnifications):
                mag = np.abs(magnifications[i])
                marker_size = 8 + min(mag * 2, 20)
            
            ax.plot(img[0], img[1], 'o', color=color, markersize=marker_size,
                   label=f'Image {label}', zorder=12,
                   markeredgecolor='white', markeredgewidth=1)
            
            # Add label text
            offset = 0.15
            ax.text(img[0] + offset, img[1] + offset, label,
                   color='white', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    
    # Plot Einstein radius if applicable
    if show_einstein_radius and hasattr(lens_model, 'einstein_radius'):
        theta_E = lens_model.einstein_radius
        einstein_circle = Circle((0, 0), theta_E, fill=False, 
                                edgecolor='white', linestyle='--',
                                linewidth=2, label=f'Einstein Radius ({theta_E:.3f}")',
                                alpha=0.7)
        ax.add_patch(einstein_circle)
    
    # Formatting
    ax.set_xlabel('x [arcsec]', fontsize=14, fontweight='bold')
    ax.set_ylabel('y [arcsec]', fontsize=14, fontweight='bold')
    ax.set_title('Gravitational Lens System', fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    
    # Add info text
    info_text = f"Lens: z = {lens_model.lens_system.z_l:.3f}\n"
    info_text += f"Source: z = {lens_model.lens_system.z_s:.3f}\n"
    info_text += f"Images found: {len(image_pos)}"
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='black', alpha=0.8),
           color='white')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    return fig, ax


def plot_radial_profile(lens_model,
                       r_max: float = 5.0,
                       n_points: int = 200,
                       figsize: Tuple[float, float] = (12, 5),
                       save_path: Optional[str] = None):
    """
    Plot radial profiles of surface density and convergence.
    
    Parameters
    ----------
    lens_model : MassProfile
        The lens model
    r_max : float, optional
        Maximum radius to plot in arcseconds (default: 5.0)
    n_points : int, optional
        Number of points to compute (default: 200)
    figsize : tuple, optional
        Figure size (default: (12, 5))
    save_path : str, optional
        Path to save figure
        
    Examples
    --------
    >>> from lens_models import LensSystem, NFWProfile
    >>> from utils import plot_radial_profile
    >>> lens_sys = LensSystem(0.5, 1.5)
    >>> halo = NFWProfile(1e12, 5, lens_sys)
    >>> plot_radial_profile(halo)
    """
    setup_dark_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Create radial array
    r = np.logspace(-2, np.log10(r_max), n_points)
    
    # Compute profiles
    sigma = lens_model.surface_density(r)
    kappa = lens_model.convergence(r, np.zeros_like(r))
    
    # Plot surface density
    ax1.loglog(r, sigma, linewidth=2.5, color='cyan')
    ax1.set_xlabel('Radius [arcsec]', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Surface Density Σ [M$_\\odot$/pc²]', fontsize=12, fontweight='bold')
    ax1.set_title('Surface Density Profile', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, which='both', linestyle=':', linewidth=0.5)
    
    # Mark Einstein radius if applicable
    if hasattr(lens_model, 'einstein_radius'):
        theta_E = lens_model.einstein_radius
        ax1.axvline(theta_E, color='red', linestyle='--', linewidth=2,
                   label=f'θ$_E$ = {theta_E:.3f}"', alpha=0.7)
        ax1.legend(fontsize=10)
    
    # Plot convergence
    ax2.loglog(r, kappa, linewidth=2.5, color='lime')
    ax2.set_xlabel('Radius [arcsec]', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Convergence κ', fontsize=12, fontweight='bold')
    ax2.set_title('Convergence Profile', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both', linestyle=':', linewidth=0.5)
    ax2.axhline(1.0, color='yellow', linestyle=':', linewidth=1.5,
               label='κ = 1 (critical)', alpha=0.7)
    
    # Mark Einstein radius
    if hasattr(lens_model, 'einstein_radius'):
        theta_E = lens_model.einstein_radius
        ax2.axvline(theta_E, color='red', linestyle='--', linewidth=2,
                   label=f'θ$_E$ = {theta_E:.3f}"', alpha=0.7)
    
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    return fig, (ax1, ax2)


def plot_deflection_field(lens_model,
                         extent: float = 3.0,
                         n_arrows: int = 20,
                         figsize: Tuple[float, float] = (10, 10),
                         save_path: Optional[str] = None):
    """
    Plot the deflection angle field as a quiver plot.
    
    Parameters
    ----------
    lens_model : MassProfile
        The lens model
    extent : float, optional
        Half-width of region to plot in arcseconds (default: 3.0)
    n_arrows : int, optional
        Number of arrows per dimension (default: 20)
    figsize : tuple, optional
        Figure size (default: (10, 10))
    save_path : str, optional
        Path to save figure
        
    Examples
    --------
    >>> from lens_models import LensSystem, PointMassProfile
    >>> from utils import plot_deflection_field
    >>> lens_sys = LensSystem(0.5, 1.5)
    >>> lens = PointMassProfile(1e12, lens_sys)
    >>> plot_deflection_field(lens)
    """
    setup_dark_style()
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Create grid
    x = np.linspace(-extent, extent, n_arrows)
    y = np.linspace(-extent, extent, n_arrows)
    xx, yy = np.meshgrid(x, y)
    
    # Compute deflection angles
    alpha_x, alpha_y = lens_model.deflection_angle(xx.ravel(), yy.ravel())
    alpha_x = alpha_x.reshape(xx.shape)
    alpha_y = alpha_y.reshape(yy.shape)
    
    # Compute magnitude
    alpha_mag = np.sqrt(alpha_x**2 + alpha_y**2)
    
    # Plot as quiver with color by magnitude
    quiver = ax.quiver(xx, yy, alpha_x, alpha_y, alpha_mag,
                      cmap='plasma', scale=extent*5, width=0.004,
                      alpha=0.9, edgecolors='white', linewidths=0.5)
    
    cbar = plt.colorbar(quiver, ax=ax, label='Deflection Angle [arcsec]')
    cbar.ax.tick_params(labelsize=10)
    
    # Plot lens at origin
    ax.plot(0, 0, 'o', color='gold', markersize=15, 
            label='Lens', zorder=10, markeredgecolor='white', markeredgewidth=2)
    
    # Plot Einstein radius if applicable
    if hasattr(lens_model, 'einstein_radius'):
        theta_E = lens_model.einstein_radius
        einstein_circle = Circle((0, 0), theta_E, fill=False,
                                edgecolor='cyan', linestyle='--',
                                linewidth=2.5, label=f'Einstein Radius ({theta_E:.3f}")',
                                alpha=0.8)
        ax.add_patch(einstein_circle)
    
    ax.set_xlabel('x [arcsec]', fontsize=14, fontweight='bold')
    ax.set_ylabel('y [arcsec]', fontsize=14, fontweight='bold')
    ax.set_title('Deflection Angle Field', fontsize=16, fontweight='bold', pad=20)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    return fig, ax


def plot_magnification_map(lens_model,
                          extent: float = 3.0,
                          resolution: int = 200,
                          figsize: Tuple[float, float] = (10, 10),
                          save_path: Optional[str] = None):
    """
    Plot the magnification map showing caustics and critical curves.
    
    Parameters
    ----------
    lens_model : MassProfile
        The lens model
    extent : float, optional
        Half-width of region to plot in arcseconds (default: 3.0)
    resolution : int, optional
        Grid resolution (default: 200)
    figsize : tuple, optional
        Figure size (default: (10, 10))
    save_path : str, optional
        Path to save figure
        
    Examples
    --------
    >>> from lens_models import LensSystem, NFWProfile
    >>> from utils import plot_magnification_map
    >>> lens_sys = LensSystem(0.5, 1.5)
    >>> halo = NFWProfile(1e12, 5, lens_sys)
    >>> plot_magnification_map(halo)
    """
    from optics.ray_tracing import compute_magnification
    
    setup_dark_style()
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Create grid
    x = np.linspace(-extent, extent, resolution)
    y = np.linspace(-extent, extent, resolution)
    xx, yy = np.meshgrid(x, y)
    
    # Compute magnifications
    mag_map = np.zeros_like(xx)
    dx = x[1] - x[0]
    
    for i in range(resolution):
        for j in range(resolution):
            try:
                mag_map[i, j] = compute_magnification(xx[i, j], yy[i, j], 
                                                      lens_model, dx)
            except Exception:
                mag_map[i, j] = np.nan
    
    # Plot with symmetric log scale
    vmax = np.nanpercentile(np.abs(mag_map), 95)
    im = ax.imshow(mag_map, extent=[-extent, extent, -extent, extent],
                  origin='lower', cmap='RdBu_r',
                  norm=colors.SymLogNorm(linthresh=1, vmin=-vmax, vmax=vmax),
                  alpha=0.9)
    
    cbar = plt.colorbar(im, ax=ax, label='Magnification μ')
    cbar.ax.tick_params(labelsize=10)
    
    # Plot critical curves (where |μ| is very large)
    critical_mask = np.abs(mag_map) > 100
    ax.contour(xx, yy, critical_mask.astype(int), levels=[0.5],
              colors='yellow', linewidths=2.5, linestyles='--',
              alpha=0.8)
    
    # Plot lens
    ax.plot(0, 0, 'o', color='gold', markersize=15,
           label='Lens', zorder=10, markeredgecolor='white', markeredgewidth=2)
    
    ax.set_xlabel('x [arcsec]', fontsize=14, fontweight='bold')
    ax.set_ylabel('y [arcsec]', fontsize=14, fontweight='bold')
    ax.set_title('Magnification Map & Critical Curves', fontsize=16, 
                fontweight='bold', pad=20)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    return fig, ax


def plot_source_plane_mapping(beta_x, beta_y, grid_x, grid_y,
                              source_pos: Tuple[float, float],
                              figsize: Tuple[float, float] = (10, 10),
                              save_path: Optional[str] = None):
    """
    Plot the source plane showing the mapping from image plane.
    
    This visualizes how the image plane maps to the source plane,
    revealing the caustic structure.
    
    Parameters
    ----------
    beta_x : np.ndarray
        Source plane x-coordinates (2D)
    beta_y : np.ndarray
        Source plane y-coordinates (2D)
    grid_x : np.ndarray
        Image plane x-coordinates (1D)
    grid_y : np.ndarray
        Image plane y-coordinates (1D)
    source_pos : tuple
        Actual source position (x, y)
    figsize : tuple, optional
        Figure size (default: (10, 10))
    save_path : str, optional
        Path to save figure
    """
    setup_dark_style()
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Downsample for visualization
    step = max(1, len(grid_x) // 50)
    
    # Plot grid mapping
    for i in range(0, len(grid_x), step):
        ax.plot(beta_x[i, :], beta_y[i, :], 'c-', alpha=0.3, linewidth=0.5)
    for j in range(0, len(grid_y), step):
        ax.plot(beta_x[:, j], beta_y[:, j], 'c-', alpha=0.3, linewidth=0.5)
    
    # Plot source
    ax.plot(source_pos[0], source_pos[1], '*', color='red',
           markersize=25, label='Source', zorder=10,
           markeredgecolor='white', markeredgewidth=2)
    
    ax.set_xlabel('β$_x$ [arcsec]', fontsize=14, fontweight='bold')
    ax.set_ylabel('β$_y$ [arcsec]', fontsize=14, fontweight='bold')
    ax.set_title('Source Plane Mapping & Caustics', fontsize=16,
                fontweight='bold', pad=20)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    return fig, ax
