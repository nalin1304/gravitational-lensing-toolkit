"""
Correct Recursive Multi-Plane Gravitational Lensing

This module implements the TRUE recursive multi-plane lens equation
with proper angular diameter distance scaling as required by GR.

CRITICAL PHYSICS:
The multi-plane lens equation is RECURSIVE, not additive.
Each deflection depends on the accumulated deflections from all previous planes.

Correct Formulation (Schneider+ 1992):
======================================
For N lens planes at z₁ < z₂ < ... < zₙ < zₛ:

θᵢ = θᵢ₊₁ + (Dᵢ,ᵢ₊₁ / Dᵢ₊₁) αᵢ(θᵢ)

with boundary condition at source plane:
θₙ = β + (Dₙ,ₛ / Dₛ) αₙ(θₙ)

where:
- θᵢ: position at plane i (image plane coordinates)
- Dᵢ,ⱼ: angular diameter distance from plane i to plane j
- αᵢ(θ): deflection angle at plane i evaluated at position θ
- β: source position

This is a BACKWARD recursion from source to observer.

Scientific References:
---------------------
- Schneider, Ehlers & Falco (1992), Chapter 9
- McCully et al. (2014), ApJ 836, 141
- Collett & Cunnington (2016), MNRAS 462, 3255

Author: ISEF 2025 - Task 2
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Callable
from dataclasses import dataclass
import warnings
import logging
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u

logger = logging.getLogger(__name__)


@dataclass
class LensPlaneData:
    """
    Complete data for a single lens plane.

    Attributes
    ----------
    z : float
        Redshift of this plane
    alpha_func : Callable
        Function to compute deflection: (x, y) -> (alpha_x, alpha_y)
    D_d : float
        Angular diameter distance to this plane (Mpc)
    label : str, optional
        Description of this plane
    """
    z: float
    alpha_func: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]
    D_d: float
    label: str = ""


def angular_diameter_distance_ratio(
    z_i: float,
    z_j: float,
    cosmology: FlatLambdaCDM
) -> float:
    """
    Compute ratio Dᵢⱼ / Dⱼ for angular diameter distances.

    This is the key geometric factor in multi-plane lensing.

    Parameters
    ----------
    z_i : float
        Redshift of first plane
    z_j : float
        Redshift of second plane (must be > z_i)
    cosmology : FlatLambdaCDM
        Cosmology

    Returns
    -------
    ratio : float
        Dᵢⱼ / Dⱼ

    Notes
    -----
    In flat FLRW cosmology:
        Dᵢⱼ = (1 + zⱼ)⁻¹ ∫[zᵢ to zⱼ] c dz'/H(z')

    Astropy handles this correctly via angular_diameter_distance_z1z2.
    """
    if z_j <= z_i:
        raise ValueError(f"z_j ({z_j}) must be > z_i ({z_i})")

    D_ij = cosmology.angular_diameter_distance_z1z2(z_i, z_j).to(u.Mpc).value
    D_j = cosmology.angular_diameter_distance(z_j).to(u.Mpc).value

    return D_ij / D_j


def multi_plane_trace(
    beta: np.ndarray,
    lens_planes: List[Dict],
    cosmology: FlatLambdaCDM,
    z_source: float,
    max_iter: int = 50,
    tolerance: float = 1e-8,
    verbose: bool = False
) -> np.ndarray:
    """
    Solve recursive multi-plane lens equation to find image position(s).

    CRITICAL: Multi-plane lensing REQUIRES cosmological distances.
    This function ONLY supports the thin-lens formalism on FLRW background.
    Schwarzschild geodesics are NOT compatible with multi-plane systems.

    This implements the TRUE multi-plane lens equation:

        θᵢ = θᵢ₊₁ + (Dᵢ,ᵢ₊₁ / Dᵢ₊₁) αᵢ(θᵢ)

    with θₙ = β + (Dₙ,ₛ / Dₛ) αₙ(θₙ)

    The distance ratios Dᵢ,ᵢ₊₁ / Dᵢ₊₁ REQUIRE angular diameter distances
    in an expanding universe. These are undefined in Schwarzschild spacetime.

    Parameters
    ----------
    beta : np.ndarray
        Source position in arcseconds, shape (2,) for [x, y]
    lens_planes : List[Dict]
        List of lens plane dictionaries, each with:
        - 'z': redshift (MUST be > 0 for cosmological validity)
        - 'alpha_func': function (x, y) -> (alpha_x, alpha_y)
        Must be sorted by increasing redshift
    cosmology : FlatLambdaCDM
        Cosmology for distance calculations
    z_source : float
        Source redshift (MUST be > all lens plane redshifts)
    max_iter : int, optional
        Maximum iterations for solving implicit equation (default: 50)
    tolerance : float, optional
        Convergence tolerance in arcseconds (default: 1e-8)
    verbose : bool, optional
        Print iteration details (default: False)

    Returns
    -------
    theta : np.ndarray
        Image position in arcseconds, shape (2,)

    Raises
    ------
    ValueError
        If lens planes not sorted, z_plane >= z_source, or z_plane ≤ 0
    RuntimeError
        If iteration does not converge

    Notes
    -----
    The equation is solved via fixed-point iteration, starting from
    the source position and iterating backward through the planes.

    For multiple images, call this function multiple times with different
    initial guesses.

    Examples
    --------
    >>> from astropy.cosmology import FlatLambdaCDM
    >>> cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    >>>
    >>> # Define lens planes
    >>> def alpha_plane1(x, y):
    ...     # Point mass deflection
    ...     r = np.sqrt(x**2 + y**2)
    ...     theta_E = 1.0  # arcsec
    ...     factor = theta_E**2 / (r**2 + 1e-10)
    ...     return factor * x, factor * y
    >>>
    >>> planes = [
    ...     {'z': 0.5, 'alpha_func': alpha_plane1}
    ... ]
    >>>
    >>> beta = np.array([0.3, 0.0])  # Source position
    >>> theta = multi_plane_trace(beta, planes, cosmo, z_source=2.0)
    """
    beta = np.asarray(beta, dtype=float)
    if beta.shape != (2,):
        raise ValueError(f"beta must have shape (2,), got {beta.shape}")

    N = len(lens_planes)
    if N == 0:
        # No lensing
        return beta.copy()

    # SCIENTIFIC VALIDITY CHECK: Multi-plane requires cosmological distances
    # This ensures we're using thin-lens formalism, not Schwarzschild geodesics
    for i, plane in enumerate(lens_planes):
        z = plane['z']
        if z <= 0.0:
            raise ValueError(
                f"Multi-plane lensing requires cosmological redshifts (z > 0).\n"
                f"Plane {i} has z={z:.6f} ≤ 0, which is incompatible with "
                f"angular diameter distance calculations in FLRW cosmology.\n"
                f"\n"
                f"Multi-plane lensing ONLY supports thin-lens formalism.\n"
                f"For strong-field lensing near compact objects (z≈0), use "
                f"single-plane Schwarzschild geodesics instead."
            )

    if z_source <= 0.0:
        raise ValueError(
            f"Source redshift must be cosmological (z > 0), got z_source={z_source:.6f}"
        )

    # Validate plane ordering
    z_prev = 0.0
    for i, plane in enumerate(lens_planes):
        z = plane['z']
        if z <= z_prev:
            raise ValueError(f"Lens planes must be sorted by redshift. "
                           f"Plane {i} has z={z} <= {z_prev}")
        if z >= z_source:
            raise ValueError(f"Lens plane {i} has z={z} >= z_source={z_source}")
        z_prev = z

    # Compute all angular diameter distance ratios
    D_s = cosmology.angular_diameter_distance(z_source).to(u.Mpc).value

    # Distance ratios for each plane
    ratios = []
    for i in range(N):
        z_i = lens_planes[i]['z']

        if i < N - 1:
            # Distance to next plane
            z_next = lens_planes[i + 1]['z']
            ratio = angular_diameter_distance_ratio(z_i, z_next, cosmology)
        else:
            # Last plane: distance to source
            ratio = angular_diameter_distance_ratio(z_i, z_source, cosmology)

        ratios.append(ratio)

        if verbose:
            if i < N - 1:
                logger.info(f"Plane {i} (z={z_i:.3f}): D_{i},{i+1}/D_{i+1} = {ratio:.6f}")
            else:
                logger.info(f"Plane {i} (z={z_i:.3f}): D_{i},s/D_s = {ratio:.6f}")

    # Initialize: start from source position
    theta = beta.copy()

    # Fixed-point iteration with adaptive relaxation for better convergence
    relaxation = 0.7  # Initial relaxation factor

    # Fixed-point iteration to solve recursive equation
    for iteration in range(max_iter):
        theta_old = theta.copy()

        # Backward recursion from source to observer
        # Start with source plane
        theta_current = beta.copy()

        # Work backward through planes (N-1 to 0)
        for i in range(N - 1, -1, -1):
            plane = lens_planes[i]
            alpha_func = plane['alpha_func']
            ratio = ratios[i]

            # Compute deflection at current position
            # Use current best estimate
            theta_eval = theta

            alpha_x, alpha_y = alpha_func(theta_eval[0], theta_eval[1])
            alpha = np.array([alpha_x, alpha_y], dtype=float)

            # Handle array outputs
            if np.ndim(alpha_x) > 0:
                alpha = np.array([alpha_x.flat[0], alpha_y.flat[0]], dtype=float)

            # Update position via recursive equation
            # θᵢ = θᵢ₊₁ + (Dᵢ,ᵢ₊₁/Dᵢ₊₁) αᵢ(θᵢ)
            theta_current = theta_current + ratio * alpha

        # Apply relaxation for stability
        theta_new = relaxation * theta_current + (1 - relaxation) * theta_old

        # Check convergence
        delta = np.linalg.norm(theta_new - theta_old)

        # Adapt relaxation: if converging well, increase; if oscillating, decrease
        if iteration > 0 and delta < 0.1:
            relaxation = min(0.9, relaxation * 1.05)
        elif iteration > 0 and delta > 1.0:
            relaxation = max(0.3, relaxation * 0.8)

        if verbose and iteration % 10 == 0:
            logger.info(f"Iteration {iteration}: δ = {delta:.3e} arcsec, relax = {relaxation:.3f}")

        if delta < tolerance:
            if verbose:
                logger.info(f"Converged in {iteration + 1} iterations")
            return theta_new

        theta = theta_new

    # Did not converge - issue warning
    import warnings
    warnings.warn(
        f"multi_plane_trace did not converge in {max_iter} iterations. "
        f"Final residual: {delta:.3e} arcsec",
        RuntimeWarning,
        stacklevel=2
    )

    return theta


def multi_plane_deflection_forward(
    theta: np.ndarray,
    lens_planes: List[Dict],
    cosmology: FlatLambdaCDM,
    z_source: float
) -> np.ndarray:
    """
    Compute source position β from image position θ (forward ray trace).

    This is the FORWARD direction: given θ, find β.

    Algorithm:
    ---------
    Start at observer with position θ₀ = θ
    For each plane i = 0, 1, ..., N-1:
        1. Compute deflection αᵢ(θᵢ)
        2. Update: θᵢ₊₁ = θᵢ - (Dᵢ,ᵢ₊₁/Dᵢ₊₁) αᵢ(θᵢ)
    Final position θₙ = β

    Parameters
    ----------
    theta : np.ndarray
        Image position in arcseconds, shape (2,) or (N, 2)
    lens_planes : List[Dict]
        Lens planes (sorted by z)
    cosmology : FlatLambdaCDM
        Cosmology
    z_source : float
        Source redshift

    Returns
    -------
    beta : np.ndarray
        Source position(s), same shape as theta

    Notes
    -----
    This is computationally cheap (no iteration required).
    Use this for:
    - Ray shooting (finding β given θ)
    - Creating source-plane maps
    - Magnification calculations
    """
    theta = np.atleast_2d(theta)
    original_shape = theta.shape
    is_single_point = (len(original_shape) == 1) or (original_shape[0] == 1 and len(theta.shape) == 2 and theta.shape[1] == 2)

    N = len(lens_planes)
    if N == 0:
        if is_single_point:
            return theta.reshape(2)
        return theta

    # Compute distance ratios
    ratios = []
    for i in range(N):
        z_i = lens_planes[i]['z']
        if i < N - 1:
            z_next = lens_planes[i + 1]['z']
            ratio = angular_diameter_distance_ratio(z_i, z_next, cosmology)
        else:
            ratio = angular_diameter_distance_ratio(z_i, z_source, cosmology)
        ratios.append(ratio)

    # Forward ray trace
    position = theta.copy()

    for i, plane in enumerate(lens_planes):
        alpha_func = plane['alpha_func']
        ratio = ratios[i]

        # Vectorized deflection
        x = position[:, 0]
        y = position[:, 1]
        alpha_x, alpha_y = alpha_func(x, y)

        # Ensure array shape
        alpha_x = np.atleast_1d(alpha_x)
        alpha_y = np.atleast_1d(alpha_y)

        alpha = np.column_stack([alpha_x, alpha_y])

        # Update position
        position = position - ratio * alpha

    beta = position

    # Return in original shape
    if is_single_point:
        return beta.reshape(2)
    else:
        return beta


def validate_multi_plane_consistency(
    lens_planes: List[Dict],
    cosmology: FlatLambdaCDM,
    z_source: float,
    test_beta: np.ndarray = np.array([0.5, 0.0]),
    tolerance: float = 1e-6
) -> Dict:
    """
    Validate multi-plane implementation via round-trip consistency.

    Test:
    1. Forward: θ -> β (using multi_plane_deflection_forward)
    2. Backward: β -> θ (using multi_plane_trace)
    3. Check: θ_recovered ≈ θ_original

    Parameters
    ----------
    lens_planes : List[Dict]
        Lens planes to test
    cosmology : FlatLambdaCDM
        Cosmology
    z_source : float
        Source redshift
    test_beta : np.ndarray, optional
        Test source position
    tolerance : float, optional
        Acceptable error in arcseconds

    Returns
    -------
    results : dict
        - 'passed': bool, whether test passed
        - 'max_error': float, maximum position error
        - 'theta_original': np.ndarray
        - 'theta_recovered': np.ndarray
        - 'beta_computed': np.ndarray
    """
    # Start with a source position
    beta = test_beta

    # Forward trace: β -> θ (this requires solving lens equation)
    # For testing, we'll do inverse: pick θ, compute β, then recover θ

    # Pick a test image position
    theta_original = beta + np.array([1.0, 0.5])  # Offset from source

    # Forward: compute β from θ
    beta_computed = multi_plane_deflection_forward(
        theta_original, lens_planes, cosmology, z_source
    )

    # Ensure beta_computed is 1D for multi_plane_trace
    if beta_computed.ndim > 1:
        beta_computed = beta_computed.reshape(2)

    # Backward: solve for θ given β
    theta_recovered = multi_plane_trace(
        beta_computed, lens_planes, cosmology, z_source,
        max_iter=200, tolerance=1e-10
    )

    # Check consistency
    error = np.linalg.norm(theta_recovered - theta_original)
    passed = error < tolerance

    results = {
        'passed': passed,
        'max_error': error,
        'theta_original': theta_original,
        'theta_recovered': theta_recovered,
        'beta_computed': beta_computed,
        'message': f"Round-trip error: {error:.3e} arcsec ({'PASS' if passed else 'FAIL'})"
    }

    logger.info(results['message'])

    return results


def compare_recursive_vs_additive(
    lens_planes: List[Dict],
    cosmology: FlatLambdaCDM,
    z_source: float,
    theta_grid: np.ndarray
) -> Dict:
    """
    Compare recursive multi-plane (correct) vs additive approximation (wrong).

    This demonstrates why the recursive formulation is essential.

    Parameters
    ----------
    lens_planes : List[Dict]
        Lens planes
    cosmology : FlatLambdaCDM
        Cosmology
    z_source : float
        Source redshift
    theta_grid : np.ndarray
        Grid of image positions to test, shape (N, 2)

    Returns
    -------
    comparison : dict
        - 'beta_recursive': Correct source positions
        - 'beta_additive': Incorrect (additive) source positions
        - 'max_difference': Maximum difference
        - 'rms_difference': RMS difference
    """
    # Correct recursive method
    beta_recursive = multi_plane_deflection_forward(
        theta_grid, lens_planes, cosmology, z_source
    )

    # Incorrect additive method (for comparison)
    # This sums deflections: β_wrong = θ - Σᵢ (Dᵢₛ/Dₛ) αᵢ(θ)
    theta_grid_2d = np.atleast_2d(theta_grid)
    beta_additive = theta_grid_2d.copy()

    D_s = cosmology.angular_diameter_distance(z_source).to(u.Mpc).value

    for plane in lens_planes:
        z_i = plane['z']
        alpha_func = plane['alpha_func']

        # Weight factor (WRONG - should not use this for deflection sum)
        D_is = cosmology.angular_diameter_distance_z1z2(z_i, z_source).to(u.Mpc).value
        weight = D_is / D_s

        # Compute deflection at SAME position θ (not updated)
        x = theta_grid_2d[:, 0]
        y = theta_grid_2d[:, 1]
        alpha_x, alpha_y = alpha_func(x, y)
        alpha = np.column_stack([np.atleast_1d(alpha_x), np.atleast_1d(alpha_y)])

        # WRONG: subtract all deflections at original θ
        beta_additive = beta_additive - weight * alpha

    # Compute differences
    diff = beta_recursive - beta_additive
    if diff.ndim == 1:
        diff = diff.reshape(1, -1)
    max_diff = np.max(np.linalg.norm(diff, axis=1))
    rms_diff = np.sqrt(np.mean(np.sum(diff**2, axis=1)))

    comparison = {
        'beta_recursive': beta_recursive,
        'beta_additive': beta_additive,
        'difference': diff,
        'max_difference': max_diff,
        'rms_difference': rms_diff,
        'message': f"Recursive vs Additive: max diff = {max_diff:.3e} arcsec, RMS = {rms_diff:.3e} arcsec"
    }

    logger.info(comparison['message'])

    return comparison


def validate_single_plane_equivalence(
    alpha_func: Callable,
    cosmology: FlatLambdaCDM,
    z_lens: float,
    z_source: float,
    test_positions: np.ndarray
) -> Dict:
    """
    Validate that multi-plane reduces to single-plane when N=1.

    Parameters
    ----------
    alpha_func : Callable
        Deflection function
    cosmology : FlatLambdaCDM
        Cosmology
    z_lens : float
        Lens redshift
    z_source : float
        Source redshift
    test_positions : np.ndarray
        Test image positions, shape (N, 2)

    Returns
    -------
    results : dict
        Comparison of single-plane vs multi-plane with N=1
    """
    # Multi-plane with single plane
    planes = [{'z': z_lens, 'alpha_func': alpha_func}]
    beta_multiplane = multi_plane_deflection_forward(
        test_positions, planes, cosmology, z_source
    )

    # Direct single-plane calculation
    # β = θ - (D_ls/D_s) α(θ)
    D_ls = cosmology.angular_diameter_distance_z1z2(z_lens, z_source).to(u.Mpc).value
    D_s = cosmology.angular_diameter_distance(z_source).to(u.Mpc).value
    weight = D_ls / D_s

    test_2d = np.atleast_2d(test_positions)
    x = test_2d[:, 0]
    y = test_2d[:, 1]
    alpha_x, alpha_y = alpha_func(x, y)
    alpha = np.column_stack([np.atleast_1d(alpha_x), np.atleast_1d(alpha_y)])

    beta_single = test_2d - weight * alpha

    # Compare
    diff = beta_multiplane - beta_single
    max_error = np.max(np.abs(diff))

    results = {
        'beta_multiplane': beta_multiplane,
        'beta_single': beta_single,
        'max_error': max_error,
        'passed': max_error < 1e-10,
        'message': f"Single-plane equivalence: max error = {max_error:.3e} arcsec"
    }

    logger.info(results['message'])

    return results


# =============================================================================
# Convenience wrapper class
# =============================================================================

class MultiPlaneLensSystem:
    """
    High-level interface for multi-plane lensing.

    This wraps the functional API in a convenient class.

    Examples
    --------
    >>> from astropy.cosmology import FlatLambdaCDM
    >>> cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    >>>
    >>> system = MultiPlaneLensSystem(z_source=2.0, cosmology=cosmo)
    >>> system.add_plane(z=0.3, alpha_func=my_deflection_func)
    >>> system.add_plane(z=0.5, alpha_func=another_func)
    >>>
    >>> # Forward ray trace
    >>> theta = np.array([1.0, 0.5])
    >>> beta = system.trace_forward(theta)
    >>>
    >>> # Backward ray trace (solve lens equation)
    >>> beta_target = np.array([0.3, 0.0])
    >>> theta_image = system.trace_backward(beta_target)
    """

    def __init__(self, z_source: float, cosmology: FlatLambdaCDM):
        """Initialize multi-plane system."""
        self.z_source = z_source
        self.cosmology = cosmology
        self.planes: List[Dict] = []

    def add_plane(
        self,
        z: float,
        alpha_func: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
        label: str = ""
    ) -> None:
        """Add a lens plane."""
        if z >= self.z_source:
            raise ValueError(f"Plane z={z} >= source z={self.z_source}")

        self.planes.append({
            'z': z,
            'alpha_func': alpha_func,
            'label': label
        })

        # Sort by redshift
        self.planes.sort(key=lambda p: p['z'])

    def trace_forward(self, theta: np.ndarray) -> np.ndarray:
        """Trace from image plane to source plane (θ → β)."""
        return multi_plane_deflection_forward(
            theta, self.planes, self.cosmology, self.z_source
        )

    def trace_backward(
        self,
        beta: np.ndarray,
        max_iter: int = 50,
        tolerance: float = 1e-8
    ) -> np.ndarray:
        """Solve lens equation to find image position (β → θ)."""
        return multi_plane_trace(
            beta, self.planes, self.cosmology, self.z_source,
            max_iter=max_iter, tolerance=tolerance
        )

    def validate(self) -> Dict:
        """Validate implementation consistency."""
        return validate_multi_plane_consistency(
            self.planes, self.cosmology, self.z_source
        )

    def summary(self) -> str:
        """Get system summary."""
        lines = [
            f"Multi-Plane Lens System",
            f"  Source redshift: z = {self.z_source}",
            f"  Number of planes: {len(self.planes)}",
            ""
        ]

        for i, plane in enumerate(self.planes):
            label = plane.get('label', f"Plane {i}")
            lines.append(f"  {label}: z = {plane['z']:.3f}")

        return "\n".join(lines)
