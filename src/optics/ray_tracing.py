"""
Ray Tracing Engine for Gravitational Lensing

This module implements the ray shooting algorithm to find multiple images
of a source lensed by a given mass distribution.
"""

import numpy as np
from typing import Tuple, Optional, Dict
from scipy import interpolate
from scipy.ndimage import label


def ray_trace(source_position: Tuple[float, float], 
              lens_model,
              grid_extent: float = 3.0,
              grid_resolution: int = 300,
              threshold: float = 0.05,
              return_maps: bool = True) -> Dict:
    """
    Perform ray tracing to find lensed images of a source.
    
    This function implements the ray shooting algorithm:
    1. Creates a grid on the image plane (θ space)
    2. Computes deflection angles α(θ) for all grid points
    3. Maps to source plane: β(θ) = θ - α(θ)
    4. Finds locations where β matches the source position
    5. Refines image positions and calculates magnifications
    
    Parameters
    ----------
    source_position : tuple of float
        Source position (x, y) in arcseconds on the source plane
    lens_model : MassProfile
        The lens model (e.g., PointMassProfile, NFWProfile)
    grid_extent : float, optional
        Half-width of the image plane grid in arcseconds (default: 3.0)
    grid_resolution : int, optional
        Number of grid points per dimension (default: 300)
    threshold : float, optional
        Distance threshold in arcseconds for identifying images (default: 0.05)
    return_maps : bool, optional
        Whether to return full convergence and caustic maps (default: True)
        
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'image_positions': ndarray of shape (N, 2), positions of N images
        - 'magnifications': ndarray of shape (N,), magnification of each image
        - 'convergence_map': 2D array (if return_maps=True)
        - 'source_plane_map': 2D array of source plane mapping (if return_maps=True)
        - 'grid_x': 1D array of x-coordinates
        - 'grid_y': 1D array of y-coordinates
        
    Examples
    --------
    >>> from lens_models import LensSystem, PointMassProfile
    >>> from optics import ray_trace
    >>> lens_sys = LensSystem(0.5, 1.5)
    >>> lens = PointMassProfile(1e12, lens_sys)
    >>> results = ray_trace((0.5, 0.0), lens)
    >>> print(f"Found {len(results['image_positions'])} images")
    
    Notes
    -----
    The algorithm uses a threshold to identify image positions. For better
    accuracy, decrease the threshold or increase the grid_resolution, but
    this will increase computation time.
    """
    source_x, source_y = source_position
    
    # Step 1: Create grid on image plane
    x = np.linspace(-grid_extent, grid_extent, grid_resolution)
    y = np.linspace(-grid_extent, grid_extent, grid_resolution)
    xx, yy = np.meshgrid(x, y)
    
    # Step 2: Compute deflection angles (vectorized)
    alpha_x, alpha_y = lens_model.deflection_angle(xx.ravel(), yy.ravel())
    alpha_x = alpha_x.reshape(xx.shape)
    alpha_y = alpha_y.reshape(yy.shape)
    
    # Step 3: Map to source plane: β = θ - α
    beta_x = xx - alpha_x
    beta_y = yy - alpha_y
    
    # Step 4: Find pixels where |β - β_source| < threshold
    distance_to_source = np.sqrt((beta_x - source_x)**2 + (beta_y - source_y)**2)
    image_mask = distance_to_source < threshold
    
    # Step 5: Identify connected regions (separate images)
    labeled_array, num_images = label(image_mask)
    
    image_positions = []
    magnifications = []
    
    if num_images > 0:
        # For each image, find the centroid position
        for img_id in range(1, num_images + 1):
            img_pixels = labeled_array == img_id
            
            # Centroid in pixel coordinates
            y_indices, x_indices = np.where(img_pixels)
            
            if len(x_indices) > 0:
                # Weighted centroid using inverse distance
                weights = 1.0 / (distance_to_source[img_pixels] + 1e-10)
                weights_sum = weights.sum()
                if weights_sum > 1e-10:
                    weights /= weights_sum
                else:
                    # Uniform weights if sum is too small
                    weights = np.ones_like(weights) / len(weights)
                
                x_centroid = np.sum(x[x_indices] * weights)
                y_centroid = np.sum(y[y_indices] * weights)
                
                # Refine position with subpixel interpolation
                # Use local gradient to fine-tune
                dx = x[1] - x[0]
                
                # Calculate magnification at this position
                mag = compute_magnification(x_centroid, y_centroid, lens_model, dx)
                
                image_positions.append([x_centroid, y_centroid])
                magnifications.append(mag)
    
    # Convert to arrays
    if len(image_positions) > 0:
        image_positions = np.array(image_positions)
        magnifications = np.array(magnifications)
    else:
        image_positions = np.array([]).reshape(0, 2)
        magnifications = np.array([])
    
    # Prepare results dictionary
    results = {
        'image_positions': image_positions,
        'magnifications': magnifications,
        'grid_x': x,
        'grid_y': y
    }
    
    # Add maps if requested
    if return_maps:
        convergence_map = lens_model.convergence(xx.ravel(), yy.ravel())
        convergence_map = convergence_map.reshape(xx.shape)
        
        results['convergence_map'] = convergence_map
        results['source_plane_map'] = distance_to_source
        results['beta_x'] = beta_x
        results['beta_y'] = beta_y
    
    return results


def compute_magnification(x: float, y: float, lens_model, dx: float = 0.01) -> float:
    """
    Compute the magnification at a position using the Jacobian matrix.
    
    The magnification is μ = 1/det(A), where A is the Jacobian matrix
    of the lens mapping: A_ij = ∂β_i/∂θ_j
    
    For the lens equation β = θ - α(θ), we have:
    A = I - ∂α/∂θ
    
    The determinant is:
    det(A) = (1 - κ)² - γ²
    
    where κ is convergence and γ is shear.
    
    Parameters
    ----------
    x : float
        x-coordinate in arcseconds
    y : float
        y-coordinate in arcseconds
    lens_model : MassProfile
        The lens model
    dx : float, optional
        Step size for numerical differentiation (default: 0.01 arcsec)
        
    Returns
    -------
    mu : float
        Magnification (absolute value). Positive for positive parity,
        negative for negative parity images.
        
    Notes
    -----
    The magnification can be infinite at critical curves where det(A) = 0.
    We clip extreme values for numerical stability.
    """
    # Compute deflection angles at (x, y) and nearby points
    # Central differences for better accuracy
    # Compute deflection angles at (x, y) not strictly needed for derivatives
    # alpha_x0, alpha_y0 = lens_model.deflection_angle(x, y)
    
    # Partial derivatives ∂α_x/∂x
    alpha_xp, _ = lens_model.deflection_angle(x + dx, y)
    alpha_xm, _ = lens_model.deflection_angle(x - dx, y)
    dalpha_x_dx = (alpha_xp - alpha_xm) / (2 * dx)
    
    # ∂α_x/∂y
    alpha_xp, _ = lens_model.deflection_angle(x, y + dx)
    alpha_xm, _ = lens_model.deflection_angle(x, y - dx)
    dalpha_x_dy = (alpha_xp - alpha_xm) / (2 * dx)
    
    # ∂α_y/∂x
    _, alpha_yp = lens_model.deflection_angle(x + dx, y)
    _, alpha_ym = lens_model.deflection_angle(x - dx, y)
    dalpha_y_dx = (alpha_yp - alpha_ym) / (2 * dx)
    
    # ∂α_y/∂y
    _, alpha_yp = lens_model.deflection_angle(x, y + dx)
    _, alpha_ym = lens_model.deflection_angle(x, y - dx)
    dalpha_y_dy = (alpha_yp - alpha_ym) / (2 * dx)
    
    # Jacobian matrix: A = I - ∂α/∂θ
    A11 = 1 - dalpha_x_dx
    A12 = -dalpha_x_dy
    A21 = -dalpha_y_dx
    A22 = 1 - dalpha_y_dy
    
    # Determinant
    det_A = A11 * A22 - A12 * A21
    
    # Magnification
    if np.abs(det_A) < 1e-10:
        # Near critical curve - set to large value
        mu = np.sign(det_A) * 1000.0
    else:
        mu = 1.0 / det_A
    
    # Clip extreme values
    mu = np.clip(mu, -1000, 1000)
    
    # Ensure it's a scalar float
    return float(np.asarray(mu).item())


def find_einstein_radius(lens_model, tolerance: float = 0.01, 
                         max_radius: float = 5.0) -> float:
    """
    Find the Einstein radius of a lens by tracing rays.
    
    The Einstein radius is where a source at the center produces
    images in a ring.
    
    Parameters
    ----------
    lens_model : MassProfile
        The lens model
    tolerance : float, optional
        Tolerance for the search (default: 0.01 arcsec)
    max_radius : float, optional
        Maximum radius to search (default: 5.0 arcsec)
        
    Returns
    -------
    theta_E : float
        Einstein radius in arcseconds
        
    Notes
    -----
    This function uses a bisection search to find the radius where
    the deflection angle equals the radius itself.
    """
    # For a source at the center, Einstein radius satisfies: θ = α(θ)
    # We search for |θ| where this holds
    
    def equation(r):
        """Function that should be zero at Einstein radius."""
        alpha_x, alpha_y = lens_model.deflection_angle(r, 0)
        return r - np.sqrt(alpha_x**2 + alpha_y**2)
    
    # Bisection search
    r_min = 0.01
    r_max = max_radius
    
    while r_max - r_min > tolerance:
        r_mid = (r_min + r_max) / 2
        val = equation(r_mid)
        
        if val > 0:
            r_max = r_mid
        else:
            r_min = r_mid
    
    return (r_min + r_max) / 2


def compute_time_delay(theta_x: float, theta_y: float,
                      source_x: float, source_y: float,
                      lens_model) -> float:
    """
    Compute the time delay for light traveling from source to image.
    
    The time delay is given by:
    Δt = (D_l D_s)/(c D_ls) × [(θ - β)²/2 - ψ(θ)]
    
    where ψ is the lensing potential.
    
    Parameters
    ----------
    theta_x : float
        Image x-position in arcseconds
    theta_y : float
        Image y-position in arcseconds
    source_x : float
        Source x-position in arcseconds
    source_y : float
        Source y-position in arcseconds
    lens_model : MassProfile
        The lens model
        
    Returns
    -------
    time_delay : float
        Time delay in days
        
    Notes
    -----
    This is the Fermat potential, which determines the light travel time.
    Multiple images have different time delays, enabling time-delay
    cosmography for measuring H0.
    """
    from astropy import constants as const
    from astropy import units as u
    
    # Geometric term: |θ - β|²/2
    dx = theta_x - source_x
    dy = theta_y - source_y
    geometric_term = (dx**2 + dy**2) / 2
    
    # Lensing potential
    psi = lens_model.lensing_potential(theta_x, theta_y)
    # Handle array return
    if isinstance(psi, np.ndarray):
        psi = psi[0] if len(psi) > 0 else psi.item()
    
    # Fermat potential
    fermat = geometric_term - psi
    
    # Convert to time delay
    lens_sys = lens_model.lens_system
    D_l = lens_sys.angular_diameter_distance_lens()
    D_s = lens_sys.angular_diameter_distance_source()
    D_ls = lens_sys.angular_diameter_distance_lens_source()
    
    # Time delay factor: (1+z_l) D_l D_s / (c D_ls)
    factor = (1 + lens_sys.z_l) * D_l * D_s / (const.c * D_ls)
    
    # Convert arcsec² to radians²
    if isinstance(fermat, np.ndarray):
        fermat = fermat[0] if len(fermat) > 0 else fermat.item()
    fermat_rad2 = float(fermat) * (u.arcsec.to(u.rad))**2
    
    # Time delay in seconds
    time_delay_s = (factor * fermat_rad2).to(u.s).value
    
    # Convert to days
    time_delay_days = time_delay_s / (24 * 3600)
    
    return float(time_delay_days)
