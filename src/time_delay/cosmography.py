"""
Cosmography Module for Time Delay Analysis

This module implements time delay cosmography for inferring the Hubble constant (H0)
from gravitationally lensed systems with measured time delays.

The Fermat potential Φ includes both geometric and gravitational time delays:
    Φ(θ) = (1/2) |θ - β|² - ψ(θ)
where θ is the image position, β is the source position, and ψ is the lensing potential.

Time delay between images i and j:
    Δt_ij = (1 + z_l) × D_Δt / c × [Φ(θ_i) - Φ(θ_j)]

where D_Δt is the time delay distance that depends on H0:
    D_Δt = D_l × D_s / D_ls
"""

import numpy as np
from typing import Tuple, List, Dict, Union, Optional
from astropy import units as u
from astropy import constants as const
from astropy.cosmology import FlatLambdaCDM
import scipy.optimize as optimize
from scipy.interpolate import interp1d

from ..lens_models import LensSystem, MassProfile


def calculate_time_delays(
    image_positions: List[Tuple[float, float]],
    source_position: Tuple[float, float],
    lens_model: MassProfile,
    cosmology: Optional[FlatLambdaCDM] = None
) -> Dict[str, Union[np.ndarray, float]]:
    """
    Calculate time delays between multiple images of a lensed source.
    
    The time delay has two components:
    1. Geometric delay: light travel time difference due to path length
    2. Gravitational delay (Shapiro delay): extra time due to gravitational potential
    
    Parameters
    ----------
    image_positions : list of tuples
        List of (x, y) image positions in arcseconds
        Example: [(1.0, 0.0), (-0.8, 0.5), (-0.3, -0.9), (0.2, 1.1)]
    source_position : tuple
        Source position (β_x, β_y) in arcseconds
    lens_model : MassProfile
        Mass profile of the lens (contains lens_system with cosmology)
    cosmology : FlatLambdaCDM, optional
        Cosmology for time delay distance calculation
        If None, uses lens_model.lens_system cosmology
        
    Returns
    -------
    result : dict
        Dictionary containing:
        - 'time_delay_matrix': N×N matrix of delays Δt_ij in days
        - 'time_delay_distance': D_Δt in Mpc
        - 'fermat_potentials': Φ(θ_i) for each image
        - 'geometric_delays': Geometric component of delays (days)
        - 'gravitational_delays': Gravitational component (days)
        
    Notes
    -----
    The Fermat potential is:
        Φ(θ) = (1/2) |θ - β|² - ψ(θ)
    where ψ is the lensing potential from the deflection angle:
        α(θ) = ∇ψ(θ)
        
    Time delay:
        Δt_ij = (1 + z_l) / c × D_Δt × [Φ(θ_i) - Φ(θ_j)]
        
    References
    ----------
    Schneider, Ehlers & Falco (1992), "Gravitational Lenses"
    Suyu et al. (2010), ApJ, 711, 201 - H0 from time delays
    """
    # Get cosmology
    if cosmology is None:
        lens_sys = lens_model.lens_system
        # Extract H0 and Om0 from the cosmology object
        H0_val = lens_sys.cosmology.H0.value
        Om0_val = lens_sys.cosmology.Om0
        cosmology = FlatLambdaCDM(H0=H0_val, Om0=Om0_val)
    else:
        lens_sys = lens_model.lens_system
    
    # Get redshifts
    z_l = lens_sys.z_l
    z_s = lens_sys.z_s
    
    # Calculate time delay distance D_Δt = D_l × D_s / D_ls
    D_l = lens_sys.angular_diameter_distance_lens().to(u.Mpc).value
    D_s = lens_sys.angular_diameter_distance_source().to(u.Mpc).value
    D_ls = lens_sys.angular_diameter_distance_lens_source().to(u.Mpc).value
    
    D_dt = D_l * D_s / D_ls  # Time delay distance in Mpc
    
    # Convert to SI units for time delay calculation
    D_dt_m = D_dt * 1e6 * const.pc.to(u.m).value  # meters (extract value)
    c_mps = const.c.to(u.m / u.s).value  # m/s
    
    # Convert positions from arcsec to radians
    arcsec_to_rad = np.pi / 180.0 / 3600.0
    
    # Number of images
    n_images = len(image_positions)
    
    # Calculate Fermat potential for each image
    fermat_potentials = np.zeros(n_images)
    
    for i, (x_img, y_img) in enumerate(image_positions):
        # Convert to radians
        theta_x = x_img * arcsec_to_rad
        theta_y = y_img * arcsec_to_rad
        beta_x = source_position[0] * arcsec_to_rad
        beta_y = source_position[1] * arcsec_to_rad
        
        # Geometric term: (1/2) |θ - β|²
        geometric_term = 0.5 * ((theta_x - beta_x)**2 + (theta_y - beta_y)**2)
        
        # Gravitational potential ψ(θ)
        # For a given deflection angle α(θ), we have α = ∇ψ
        # We can estimate ψ by integrating: ψ(r) ≈ ∫₀ʳ α(r') dr'
        # For simplicity, use the lensing potential if available
        # Otherwise approximate from deflection angle
        
        if hasattr(lens_model, 'lensing_potential'):
            psi = lens_model.lensing_potential(x_img, y_img)
            # Convert from arcsec² to rad²
            psi_rad2 = psi * (arcsec_to_rad**2)
        else:
            # Approximate from deflection angle
            # ψ(θ) ≈ θ · α(θ) for radial profiles
            alpha_x, alpha_y = lens_model.deflection_angle(x_img, y_img)
            alpha_x_rad = alpha_x * arcsec_to_rad
            alpha_y_rad = alpha_y * arcsec_to_rad
            psi_rad2 = (theta_x * alpha_x_rad + theta_y * alpha_y_rad)
        
        # Fermat potential (dimensionless, in rad²)
        # Extract scalar if it's an array
        psi_value = psi_rad2.item() if hasattr(psi_rad2, 'item') else psi_rad2
        fermat_potentials[i] = geometric_term - psi_value
    
    # Calculate time delay matrix
    time_delay_matrix = np.zeros((n_images, n_images))
    geometric_delays = np.zeros((n_images, n_images))
    gravitational_delays = np.zeros((n_images, n_images))
    
    for i in range(n_images):
        for j in range(n_images):
            if i != j:
                # Time delay in seconds
                delta_phi = fermat_potentials[i] - fermat_potentials[j]
                time_delay_sec = (1.0 + z_l) * D_dt_m / c_mps * delta_phi
                
                # Convert to days
                time_delay_days = time_delay_sec / 86400.0
                time_delay_matrix[i, j] = time_delay_days
                
                # Store components for analysis
                # (Approximate geometric vs gravitational split)
                theta_i = np.array(image_positions[i]) * arcsec_to_rad
                theta_j = np.array(image_positions[j]) * arcsec_to_rad
                beta = np.array(source_position) * arcsec_to_rad
                
                geom_i = 0.5 * np.sum((theta_i - beta)**2)
                geom_j = 0.5 * np.sum((theta_j - beta)**2)
                
                geometric_delays[i, j] = (1.0 + z_l) * D_dt_m / c_mps * (geom_i - geom_j) / 86400.0
                gravitational_delays[i, j] = time_delay_days - geometric_delays[i, j]
    
    return {
        'time_delay_matrix': time_delay_matrix,
        'time_delay_distance': D_dt,
        'fermat_potentials': fermat_potentials,
        'geometric_delays': geometric_delays,
        'gravitational_delays': gravitational_delays,
        'redshift_lens': z_l,
        'redshift_source': z_s
    }


def infer_h0(
    observed_delays: Dict[Tuple[int, int], Tuple[float, float]],
    image_positions: List[Tuple[float, float]],
    source_position: Tuple[float, float],
    lens_model: MassProfile,
    h0_range: Tuple[float, float] = (50.0, 90.0),
    n_grid: int = 200,
    Om0: float = 0.3
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Infer the Hubble constant H0 from observed time delays.
    
    This function performs a chi-squared fit to find the value of H0 that
    best matches the observed time delays, given a lens model.
    
    Parameters
    ----------
    observed_delays : dict
        Dictionary mapping image pairs (i, j) to (delay, uncertainty) in days
        Example: {(0, 1): (10.5, 0.8), (0, 2): (25.3, 1.2), (1, 2): (14.8, 0.9)}
    image_positions : list of tuples
        List of (x, y) image positions in arcseconds
    source_position : tuple
        Source position (β_x, β_y) in arcseconds
    lens_model : MassProfile
        Mass profile of the lens
    h0_range : tuple, optional
        Range of H0 values to search (min, max) in km/s/Mpc
    n_grid : int, optional
        Number of grid points for H0 search
    Om0 : float, optional
        Matter density parameter (default: 0.3)
        
    Returns
    -------
    result : dict
        Dictionary containing:
        - 'h0_best': Best-fit H0 value (km/s/Mpc)
        - 'h0_uncertainty': 1-sigma uncertainty on H0
        - 'h0_grid': Grid of H0 values tested
        - 'chi2_grid': Chi-squared values at each H0
        - 'posterior': Posterior probability distribution (normalized)
        - 'reduced_chi2': Reduced chi-squared at best fit
        - 'n_dof': Number of degrees of freedom
        
    Notes
    -----
    The chi-squared is computed as:
        χ² = Σ [(Δt_obs - Δt_pred(H0))² / σ²]
    where the sum is over all observed image pairs.
    
    The time delay distance scales as D_Δt ∝ 1/H0, so time delays scale as:
        Δt ∝ 1/H0
        
    References
    ----------
    Refsdal (1964), MNRAS, 128, 307 - Original proposal
    Suyu et al. (2017), MNRAS, 468, 2590 - H0LiCOW measurement
    """
    # Create H0 grid
    h0_grid = np.linspace(h0_range[0], h0_range[1], n_grid)
    chi2_grid = np.zeros(n_grid)
    
    # Original cosmology parameters
    z_l = lens_model.lens_system.z_l
    z_s = lens_model.lens_system.z_s
    
    # Loop over H0 values
    for idx, h0_test in enumerate(h0_grid):
        # Create cosmology with this H0
        cosmo_test = FlatLambdaCDM(H0=h0_test, Om0=Om0)
        
        # Calculate predicted delays with this H0
        # We need to update the lens system's cosmology
        # Create a temporary lens system with new H0
        from ..lens_models import LensSystem
        temp_lens_sys = LensSystem(z_l, z_s, H0=h0_test, Om0=Om0)
        
        # Create temporary lens model with updated cosmology
        # Copy lens model parameters
        if hasattr(lens_model, 'M_vir'):
            # NFW-like profile
            from ..lens_models import NFWProfile
            temp_lens = type(lens_model)(
                lens_model.M_vir,
                lens_model.c,
                temp_lens_sys
            )
            # Copy additional parameters if present
            if hasattr(lens_model, 'm_wdm'):
                temp_lens.m_wdm = lens_model.m_wdm
            if hasattr(lens_model, 'sigma_SIDM'):
                temp_lens.sigma_SIDM = lens_model.sigma_SIDM
        elif hasattr(lens_model, 'M'):
            # Point mass
            from ..lens_models import PointMassProfile
            temp_lens = PointMassProfile(lens_model.M, temp_lens_sys)
        else:
            raise ValueError("Unsupported lens model type")
        
        # Calculate predicted delays
        delay_result = calculate_time_delays(
            image_positions,
            source_position,
            temp_lens,
            cosmo_test
        )
        
        predicted_matrix = delay_result['time_delay_matrix']
        
        # Compute chi-squared
        chi2 = 0.0
        for (i, j), (obs_delay, sigma) in observed_delays.items():
            pred_delay = predicted_matrix[i, j]
            chi2 += ((obs_delay - pred_delay) / sigma)**2
        
        chi2_grid[idx] = chi2
    
    # Find best-fit H0
    idx_best = np.argmin(chi2_grid)
    h0_best = h0_grid[idx_best]
    chi2_min = chi2_grid[idx_best]
    
    # Calculate uncertainty (1-sigma = Δχ² = 1 for 1 parameter)
    # Find where χ² = χ²_min + 1
    chi2_threshold = chi2_min + 1.0
    within_1sigma = chi2_grid < chi2_threshold
    
    if np.sum(within_1sigma) > 1:
        h0_1sigma_range = h0_grid[within_1sigma]
        h0_uncertainty = (h0_1sigma_range.max() - h0_1sigma_range.min()) / 2.0
    else:
        # Estimate from curvature
        # χ² ≈ χ²_min + (H0 - H0_best)² / σ²
        # Fit parabola near minimum
        mask = np.abs(h0_grid - h0_best) < (h0_range[1] - h0_range[0]) / 10.0
        if np.sum(mask) >= 3:
            from scipy.optimize import curve_fit
            def parabola(h, a, b, c):
                return a * (h - b)**2 + c
            try:
                popt, _ = curve_fit(parabola, h0_grid[mask], chi2_grid[mask],
                                   p0=[1.0, h0_best, chi2_min])
                h0_uncertainty = 1.0 / np.sqrt(popt[0])
            except Exception:
                h0_uncertainty = (h0_range[1] - h0_range[0]) / 20.0
        else:
            h0_uncertainty = (h0_range[1] - h0_range[0]) / 20.0
    
    # Compute posterior (assume flat prior)
    # P(H0|data) ∝ exp(-χ²/2)
    posterior = np.exp(-0.5 * (chi2_grid - chi2_min))
    posterior /= np.trapezoid(posterior, h0_grid)  # Normalize
    
    # Degrees of freedom
    n_obs = len(observed_delays)
    n_params = 1  # Just H0
    n_dof = n_obs - n_params
    reduced_chi2 = chi2_min / n_dof if n_dof > 0 else chi2_min
    
    return {
        'h0_best': h0_best,
        'h0_uncertainty': h0_uncertainty,
        'h0_grid': h0_grid,
        'chi2_grid': chi2_grid,
        'posterior': posterior,
        'reduced_chi2': reduced_chi2,
        'n_dof': n_dof,
        'chi2_min': chi2_min
    }


def monte_carlo_h0_uncertainty(
    observed_delays: Dict[Tuple[int, int], Tuple[float, float]],
    image_positions: List[Tuple[float, float]],
    source_position: Tuple[float, float],
    lens_model: MassProfile,
    lens_uncertainties: Dict[str, float],
    n_realizations: int = 1000,
    h0_true: float = 70.0,
    Om0: float = 0.3
) -> Dict[str, Union[np.ndarray, float]]:
    """
    Estimate H0 uncertainty using Monte Carlo sampling of lens model parameters.
    
    This function propagates uncertainties in the lens model (mass, concentration, etc.)
    to the inferred H0 value by:
    1. Drawing random realizations of lens parameters from their uncertainties
    2. Inferring H0 for each realization
    3. Computing the distribution of inferred H0 values
    
    Parameters
    ----------
    observed_delays : dict
        Dictionary mapping image pairs to (delay, uncertainty) in days
    image_positions : list of tuples
        List of (x, y) image positions in arcseconds
    source_position : tuple
        Source position (β_x, β_y) in arcseconds
    lens_model : MassProfile
        Fiducial lens model
    lens_uncertainties : dict
        Uncertainties on lens parameters
        Example: {'M_vir': 1e11, 'c': 1.0, 'source_x': 0.01, 'source_y': 0.01}
    n_realizations : int, optional
        Number of Monte Carlo realizations
    h0_true : float, optional
        True H0 value for cosmology (default: 70.0)
    Om0 : float, optional
        Matter density parameter
        
    Returns
    -------
    result : dict
        Dictionary containing:
        - 'h0_samples': Array of inferred H0 values from each realization
        - 'h0_mean': Mean of H0 distribution
        - 'h0_std': Standard deviation of H0 distribution
        - 'h0_median': Median H0
        - 'h0_16': 16th percentile (lower 1σ)
        - 'h0_84': 84th percentile (upper 1σ)
        
    Notes
    -----
    This method accounts for systematic uncertainties in the lens model that
    affect the inferred H0. Common sources of uncertainty:
    - Lens mass and concentration
    - Source position
    - External convergence from line-of-sight structure
    - Kinematic constraints
    """
    h0_samples = np.zeros(n_realizations)
    
    for i in range(n_realizations):
        # Draw random lens parameters
        if hasattr(lens_model, 'M_vir'):
            # NFW-like profile
            M_vir_sample = lens_model.M_vir + np.random.randn() * lens_uncertainties.get('M_vir', 0)
            c_sample = lens_model.c + np.random.randn() * lens_uncertainties.get('c', 0)
            
            # Ensure positive values
            M_vir_sample = max(M_vir_sample, 1e10)
            c_sample = max(c_sample, 1.0)
            
            # Create perturbed lens model
            from ..lens_models import LensSystem
            lens_sys = LensSystem(lens_model.lens_system.z_l,
                                 lens_model.lens_system.z_s,
                                 H0=h0_true, Om0=Om0)
            
            perturbed_lens = type(lens_model)(M_vir_sample, c_sample, lens_sys)
            
            # Copy additional parameters
            if hasattr(lens_model, 'm_wdm'):
                perturbed_lens.m_wdm = lens_model.m_wdm
            if hasattr(lens_model, 'sigma_SIDM'):
                perturbed_lens.sigma_SIDM = lens_model.sigma_SIDM
                
        elif hasattr(lens_model, 'mass'):
            # Point mass
            mass_sample = lens_model.mass + np.random.randn() * lens_uncertainties.get('mass', 0)
            mass_sample = max(mass_sample, 1e10)
            
            from ..lens_models import PointMassProfile, LensSystem
            lens_sys = LensSystem(lens_model.lens_system.z_l,
                                 lens_model.lens_system.z_s,
                                 H0=h0_true, Om0=Om0)
            perturbed_lens = PointMassProfile(mass_sample, lens_sys)
        else:
            raise ValueError("Unsupported lens model type")
        
        # Perturb source position
        source_x = source_position[0] + np.random.randn() * lens_uncertainties.get('source_x', 0)
        source_y = source_position[1] + np.random.randn() * lens_uncertainties.get('source_y', 0)
        source_sample = (source_x, source_y)
        
        # Infer H0 for this realization
        try:
            result = infer_h0(
                observed_delays,
                image_positions,
                source_sample,
                perturbed_lens,
                h0_range=(h0_true - 20, h0_true + 20),
                n_grid=100,
                Om0=Om0
            )
            h0_samples[i] = result['h0_best']
        except Exception as e:
            # If inference fails, use fiducial value
            h0_samples[i] = h0_true
    
    # Remove outliers (> 3σ)
    h0_median = np.median(h0_samples)
    h0_std_robust = 1.4826 * np.median(np.abs(h0_samples - h0_median))
    mask = np.abs(h0_samples - h0_median) < 3 * h0_std_robust
    h0_samples_clean = h0_samples[mask]
    
    return {
        'h0_samples': h0_samples_clean,
        'h0_mean': np.mean(h0_samples_clean),
        'h0_std': np.std(h0_samples_clean),
        'h0_median': np.median(h0_samples_clean),
        'h0_16': np.percentile(h0_samples_clean, 16),
        'h0_84': np.percentile(h0_samples_clean, 84)
    }


class TimeDelayCosmography:
    """
    Comprehensive time delay cosmography analysis class.
    
    This class provides a complete workflow for H0 inference from time delays:
    1. Calculate time delays from lens model
    2. Infer H0 from observed delays
    3. Estimate systematic uncertainties
    4. Generate diagnostic plots
    
    Examples
    --------
    >>> from src.lens_models import LensSystem, NFWProfile
    >>> from src.time_delay import TimeDelayCosmography
    >>> 
    >>> # Create lens system and model
    >>> lens_sys = LensSystem(0.5, 1.5, H0=70)
    >>> lens = NFWProfile(1e12, 10.0, lens_sys)
    >>> 
    >>> # Create cosmography analyzer
    >>> cosmo = TimeDelayCosmography(lens)
    >>> 
    >>> # Calculate predicted delays
    >>> images = [(1.0, 0.0), (-0.8, 0.5), (-0.3, -0.9)]
    >>> source = (0.1, 0.05)
    >>> delays = cosmo.calculate_delays(images, source)
    >>> 
    >>> # Add noise and infer H0 back
    >>> obs_delays = {(0,1): (delays[0,1], 0.5), (0,2): (delays[0,2], 0.8)}
    >>> result = cosmo.infer_h0(obs_delays, images, source)
    >>> print(f"H0 = {result['h0_best']:.1f} ± {result['h0_uncertainty']:.1f}")
    """
    
    def __init__(self, lens_model: MassProfile):
        """
        Initialize cosmography analyzer.
        
        Parameters
        ----------
        lens_model : MassProfile
            Gravitational lens model
        """
        self.lens_model = lens_model
        self.lens_system = lens_model.lens_system
    
    def calculate_delays(
        self,
        image_positions: List[Tuple[float, float]],
        source_position: Tuple[float, float]
    ) -> np.ndarray:
        """
        Calculate time delay matrix for given image configuration.
        
        Parameters
        ----------
        image_positions : list of tuples
            Image positions in arcseconds
        source_position : tuple
            Source position in arcseconds
            
        Returns
        -------
        delays : np.ndarray
            N×N matrix of time delays in days
        """
        result = calculate_time_delays(
            image_positions,
            source_position,
            self.lens_model
        )
        return result['time_delay_matrix']
    
    def infer_h0(
        self,
        observed_delays: Dict[Tuple[int, int], Tuple[float, float]],
        image_positions: List[Tuple[float, float]],
        source_position: Tuple[float, float],
        **kwargs
    ) -> Dict:
        """
        Infer H0 from observed time delays.
        
        Parameters
        ----------
        observed_delays : dict
            Observed delays with uncertainties
        image_positions : list of tuples
            Image positions in arcseconds
        source_position : tuple
            Source position in arcseconds
        **kwargs
            Additional arguments passed to infer_h0()
            
        Returns
        -------
        result : dict
            H0 inference results
        """
        return infer_h0(
            observed_delays,
            image_positions,
            source_position,
            self.lens_model,
            **kwargs
        )
    
    def monte_carlo_uncertainty(
        self,
        observed_delays: Dict[Tuple[int, int], Tuple[float, float]],
        image_positions: List[Tuple[float, float]],
        source_position: Tuple[float, float],
        lens_uncertainties: Dict[str, float],
        **kwargs
    ) -> Dict:
        """
        Estimate H0 uncertainty via Monte Carlo.
        
        Parameters
        ----------
        observed_delays : dict
            Observed delays with uncertainties
        image_positions : list of tuples
            Image positions
        source_position : tuple
            Source position
        lens_uncertainties : dict
            Lens parameter uncertainties
        **kwargs
            Additional arguments passed to monte_carlo_h0_uncertainty()
            
        Returns
        -------
        result : dict
            Monte Carlo H0 distribution
        """
        return monte_carlo_h0_uncertainty(
            observed_delays,
            image_positions,
            source_position,
            self.lens_model,
            lens_uncertainties,
            **kwargs
        )
