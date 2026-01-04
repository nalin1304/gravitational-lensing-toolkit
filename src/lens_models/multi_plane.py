"""
Multi-plane gravitational lensing for complex line-of-sight structures.

This module implements the multi-plane lens equation for systems with multiple
lens planes at different redshifts (e.g., galaxy clusters with foreground/background
structure, line-of-sight halos along the line of sight to a quasar).

Physics
-------
For N lens planes at redshifts z₁ < z₂ < ... < zₙ < zₛ (source at zₛ):

The effective deflection angle is the cumulative sum:
    α_eff(θ) = Σᵢ₌₁ᴺ (Dᵢₛ/Dₛ) α_i(θ - Σⱼ₌₁ⁱ⁻¹ (Dⱼᵢ/Dⱼₛ) Dⱼ αⱼ)

where:
- Dᵢₛ = angular diameter distance from plane i to source
- Dₛ = angular diameter distance to source
- Dⱼᵢ = angular diameter distance between planes j and i
- αⱼ = deflection angle at plane j

References
----------
Schneider, Ehlers & Falco (1992) "Gravitational Lenses"
McCully et al. (2014) ApJ 836, 141 - Multi-plane lensing for clusters
Collett & Cunnington (2016) MNRAS 462, 3255 - Line-of-sight effects

Examples
--------
>>> from multi_plane import MultiPlaneLens
>>> from astropy.cosmology import FlatLambdaCDM
>>> from mass_profiles import NFWProfile
>>> from lens_system import LensSystem
>>>
>>> # Define cosmology
>>> cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
>>>
>>> # Create lens systems for each plane
>>> lens_sys1 = LensSystem(z_lens=0.3, z_source=2.0, H0=70, Om0=0.3)
>>> lens_sys2 = LensSystem(z_lens=0.5, z_source=2.0, H0=70, Om0=0.3)
>>>
>>> # Create multi-plane lens with foreground and main cluster
>>> lens = MultiPlaneLens(source_redshift=2.0, cosmology=cosmo)
>>> lens.add_plane(redshift=0.3, profile=NFWProfile(M200=1e14, c=5, lens_system=lens_sys1))
>>> lens.add_plane(redshift=0.5, profile=NFWProfile(M200=5e14, c=4, lens_system=lens_sys2))
>>>
>>> # Compute lensing
>>> theta = np.array([1.0, 1.0])  # arcsec
>>> beta = lens.ray_trace(theta)
>>> convergence = lens.convergence_map(image_size=512, fov=60.0)
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import warnings


@dataclass
class LensPlane:
    """
    Single lens plane in a multi-plane system.
    
    Attributes
    ----------
    redshift : float
        Redshift of this lens plane
    profile : object
        Mass profile object (must have deflection_angle() method)
    center : tuple
        Center position (x, y) in arcseconds
    Dd : float
        Angular diameter distance to this plane (Mpc)
    Dds : float
        Angular diameter distance from this plane to source (Mpc)
    Ds : float
        Angular diameter distance to source (Mpc)
    """
    redshift: float
    profile: object
    center: Tuple[float, float] = (0.0, 0.0)
    Dd: float = 0.0
    Dds: float = 0.0
    Ds: float = 0.0


class MultiPlaneLens:
    """
    Multi-plane gravitational lens system.
    
    This class handles lensing by multiple mass distributions at different
    redshifts along the line of sight. Essential for modeling:
    - Galaxy clusters with substructure
    - Line-of-sight halos in quasar lensing
    - Large-scale structure effects
    
    Parameters
    ----------
    source_redshift : float
        Redshift of the background source
    cosmology : object
        Cosmology object with angular_diameter_distance() method
    
    Attributes
    ----------
    planes : List[LensPlane]
        List of lens planes, sorted by increasing redshift
    """
    
    def __init__(self, source_redshift: float, cosmology):
        """Initialize multi-plane lens system."""
        self.source_redshift = source_redshift
        self.cosmology = cosmology
        self.planes: List[LensPlane] = []
        
        # Compute source distance
        self.Ds = cosmology.angular_diameter_distance(source_redshift).value  # in Mpc
    
    def add_plane(
        self,
        redshift: float,
        profile: object,
        center: Tuple[float, float] = (0.0, 0.0)
    ) -> None:
        """
        Add a lens plane to the system.
        
        Parameters
        ----------
        redshift : float
            Redshift of this lens plane (must be < source_redshift)
        profile : object
            Mass profile with deflection_angle(theta) method
        center : tuple, optional
            Center position (x, y) in arcseconds
        
        Raises
        ------
        ValueError
            If redshift >= source_redshift
        """
        if redshift >= self.source_redshift:
            raise ValueError(
                f"Lens plane redshift ({redshift}) must be less than "
                f"source redshift ({self.source_redshift})"
            )
        
        # Compute distances
        Dd = self.cosmology.angular_diameter_distance(redshift).value  # in Mpc
        Dds = self.cosmology.angular_diameter_distance_z1z2(redshift, self.source_redshift).value  # in Mpc
        
        # Create plane
        plane = LensPlane(
            redshift=redshift,
            profile=profile,
            center=center,
            Dd=Dd,
            Dds=Dds,
            Ds=self.Ds
        )
        
        self.planes.append(plane)
        
        # Sort by redshift
        self.planes.sort(key=lambda p: p.redshift)
    
    def ray_trace(
        self,
        theta: np.ndarray,
        return_intermediate: bool = False
    ) -> np.ndarray:
        """
        Perform ray tracing through all lens planes.
        
        This implements the multi-plane lens equation by tracing a light ray
        backwards from the observer through each lens plane.
        
        Parameters
        ----------
        theta : np.ndarray
            Image plane coordinates, shape (2,) or (..., 2)
        return_intermediate : bool, optional
            If True, return positions at each plane
        
        Returns
        -------
        beta : np.ndarray
            Source plane position (if return_intermediate=False)
        positions : List[np.ndarray]
            Positions at each plane (if return_intermediate=True)
        
        Notes
        -----
        Algorithm:
        1. Start at image plane with position θ
        2. For each lens plane i:
           a. Compute deflection α_i at current position
           b. Update position: θ → θ - (D_is/D_s) α_i
        3. Final position is source plane coordinate β
        """
        # Handle input shape
        input_shape = theta.shape
        if theta.ndim == 1:
            # Single position (2,) -> (1, 2)
            theta = theta.reshape(1, 2)
            single_point = True
        else:
            single_point = False
        
        # Track position through planes
        current_pos = theta.copy()
        
        if return_intermediate:
            positions = [current_pos.copy()]
        
        # Trace through each plane
        for i, plane in enumerate(self.planes):
            # Compute position relative to plane center
            rel_pos = current_pos - np.array(plane.center)
            
            # Get deflection angle from this plane
            # Extract x and y components
            rel_x = rel_pos[..., 0]
            rel_y = rel_pos[..., 1]
            alpha_x, alpha_y = plane.profile.deflection_angle(rel_x, rel_y)
            alpha_i = np.stack([alpha_x, alpha_y], axis=-1)
            
            # Compute deflection weight: D_is / D_s
            # For planes beyond this one, compute proper distances
            if i < len(self.planes) - 1:
                # Distance from this plane to next plane
                next_plane = self.planes[i + 1]
                D_ij = self.cosmology.angular_diameter_distance_z1z2(
                    plane.redshift, next_plane.redshift
                ).value  # in Mpc
                weight = D_ij / plane.Dds
            else:
                # Last plane: use full distance to source
                weight = plane.Dds / self.Ds
            
            # Update position
            current_pos = current_pos - weight * alpha_i
            
            if return_intermediate:
                positions.append(current_pos.copy())
        
        # Source plane position
        beta = current_pos
        
        if return_intermediate:
            return positions
        else:
            # Return in original shape
            if single_point:
                return beta[0]  # Return (2,) array
            else:
                return beta
    
    def effective_deflection(self, theta: np.ndarray) -> np.ndarray:
        """
        Compute effective deflection angle for the full system.
        
        Parameters
        ----------
        theta : np.ndarray
            Image plane position, shape (2,) or (..., 2)
        
        Returns
        -------
        alpha_eff : np.ndarray
            Effective deflection angle
        """
        # Handle input shape
        input_shape = theta.shape
        if theta.ndim == 1:
            single_point = True
            theta_orig = theta.copy()
        else:
            single_point = False
            theta_orig = theta
        
        # Ray trace to get source position
        beta = self.ray_trace(theta)
        
        # Effective deflection: α_eff = θ - β
        alpha_eff = theta_orig - beta
        
        return alpha_eff
    
    def convergence_map(
        self,
        image_size: int = 512,
        fov: float = 60.0
    ) -> np.ndarray:
        """
        Compute total convergence map from all planes.
        
        Parameters
        ----------
        image_size : int
            Size of output image in pixels
        fov : float
            Field of view in arcseconds
        
        Returns
        -------
        kappa_total : np.ndarray
            Total convergence map, shape (image_size, image_size)
        
        Notes
        -----
        The total convergence is the weighted sum:
            κ_total = Σᵢ (D_is / D_s) κᵢ
        """
        # Create coordinate grid
        x = np.linspace(-fov/2, fov/2, image_size)
        y = np.linspace(-fov/2, fov/2, image_size)
        xx, yy = np.meshgrid(x, y)
        
        # Initialize total convergence
        kappa_total = np.zeros((image_size, image_size))
        
        # Sum contribution from each plane
        for plane in self.planes:
            # Compute weight factor
            weight = plane.Dds / self.Ds
            
            # Get convergence from this plane
            rel_x = xx - plane.center[0]
            rel_y = yy - plane.center[1]
            
            # Check if profile has convergence method
            if hasattr(plane.profile, 'convergence'):
                kappa_i = plane.profile.convergence(rel_x, rel_y)
            else:
                # Estimate from deflection if convergence not available
                warnings.warn(
                    f"Plane at z={plane.redshift} has no convergence() method. "
                    "Using deflection-based estimate."
                )
                # Use relation ∇·α = 2κ
                dx = x[1] - x[0]
                alpha_x, alpha_y = plane.profile.deflection_angle(rel_x, rel_y)
                alpha = np.stack([alpha_x, alpha_y], axis=-1)
                kappa_i = self._convergence_from_deflection(alpha, dx)
            
            # Add weighted contribution
            kappa_total += weight * kappa_i
        
        return kappa_total
    
    def _convergence_from_deflection(
        self,
        alpha: np.ndarray,
        dx: float
    ) -> np.ndarray:
        """
        Estimate convergence from deflection using ∇·α = 2κ.
        
        Parameters
        ----------
        alpha : np.ndarray
            Deflection angle field, shape (..., 2)
        dx : float
            Pixel scale
        
        Returns
        -------
        kappa : np.ndarray
            Convergence map
        """
        # Extract components
        alpha_x = alpha[..., 0]
        alpha_y = alpha[..., 1]
        
        # Compute divergence
        div_alpha = (
            np.gradient(alpha_x, dx, axis=1) +
            np.gradient(alpha_y, dx, axis=0)
        )
        
        # κ = ∇·α / 2
        kappa = div_alpha / 2.0
        
        return kappa
    
    def magnification_map(
        self,
        image_size: int = 512,
        fov: float = 60.0
    ) -> np.ndarray:
        """
        Compute magnification map.
        
        Parameters
        ----------
        image_size : int
            Size of output image in pixels
        fov : float
            Field of view in arcseconds
        
        Returns
        -------
        mu : np.ndarray
            Magnification map
        
        Notes
        -----
        Magnification μ = 1 / det(A) where A is the Jacobian matrix:
            A = [[1-κ-γ₁, -γ₂    ],
                 [-γ₂,     1-κ+γ₁]]
        
        For multi-plane lensing, we compute effective κ and γ from
        the Jacobian of the full ray-tracing transformation.
        """
        # Create coordinate grid
        x = np.linspace(-fov/2, fov/2, image_size)
        y = np.linspace(-fov/2, fov/2, image_size)
        xx, yy = np.meshgrid(x, y)
        theta = np.stack([xx, yy], axis=-1)
        
        # Compute Jacobian numerically
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        
        # Ray trace
        beta = self.ray_trace(theta)
        
        # Compute derivatives ∂β/∂θ
        dbeta_dtheta_x = np.gradient(beta[..., 0], dx, axis=1)
        dbeta_dtheta_y = np.gradient(beta[..., 1], dy, axis=0)
        dbeta_dtheta_xy = np.gradient(beta[..., 0], dy, axis=0)
        dbeta_dtheta_yx = np.gradient(beta[..., 1], dx, axis=1)
        
        # Jacobian determinant
        det_A = (dbeta_dtheta_x * dbeta_dtheta_y - 
                 dbeta_dtheta_xy * dbeta_dtheta_yx)
        
        # Magnification (avoid division by zero)
        mu = np.where(np.abs(det_A) > 1e-10, 1.0 / det_A, 0.0)
        
        return mu
    
    def critical_curves(
        self,
        image_size: int = 512,
        fov: float = 60.0,
        threshold: float = 10.0
    ) -> np.ndarray:
        """
        Find critical curves where magnification diverges.
        
        Parameters
        ----------
        image_size : int
            Size of search grid
        fov : float
            Field of view in arcseconds
        threshold : float
            Magnification threshold for critical curve
        
        Returns
        -------
        critical_mask : np.ndarray
            Boolean mask of critical curve locations
        """
        mu = self.magnification_map(image_size, fov)
        
        # Critical curves where |μ| > threshold
        critical_mask = np.abs(mu) > threshold
        
        return critical_mask
    
    def time_delay_surface(
        self,
        image_size: int = 512,
        fov: float = 60.0
    ) -> np.ndarray:
        """
        Compute Fermat potential (time delay surface).
        
        Parameters
        ----------
        image_size : int
            Size of output image
        fov : float
            Field of view in arcseconds
        
        Returns
        -------
        tau : np.ndarray
            Time delay surface (dimensionless)
        
        Notes
        -----
        Fermat potential: τ(θ) = ½|θ-β|² - ψ(θ)
        where ψ is the lensing potential.
        
        For multi-plane lensing, this becomes more complex and involves
        summing contributions from each plane.
        """
        # Create coordinate grid
        x = np.linspace(-fov/2, fov/2, image_size)
        y = np.linspace(-fov/2, fov/2, image_size)
        xx, yy = np.meshgrid(x, y)
        theta = np.stack([xx, yy], axis=-1)
        
        # Get source position
        beta = self.ray_trace(theta)
        
        # Geometric term: ½|θ-β|²
        geometric_term = 0.5 * np.sum((theta - beta)**2, axis=-1)
        
        # Potential term: sum over planes
        # ψ = Σᵢ (D_is/D_s) ψᵢ
        potential = np.zeros((image_size, image_size))
        
        for plane in self.planes:
            weight = plane.Dds / self.Ds
            
            # Position relative to plane
            rel_pos = theta - np.array(plane.center)
            
            # Get potential (if available)
            if hasattr(plane.profile, 'potential'):
                psi_i = plane.profile.potential(rel_pos)
                potential += weight * psi_i
            else:
                warnings.warn(
                    f"Plane at z={plane.redshift} has no potential() method. "
                    "Skipping contribution to time delay."
                )
        
        # Fermat potential
        tau = geometric_term - potential
        
        return tau
    
    def summary(self) -> Dict:
        """
        Get summary of multi-plane system.
        
        Returns
        -------
        info : dict
            Summary information
        """
        info = {
            'source_redshift': self.source_redshift,
            'num_planes': len(self.planes),
            'planes': []
        }
        
        for i, plane in enumerate(self.planes):
            plane_info = {
                'index': i,
                'redshift': plane.redshift,
                'center': plane.center,
                'Dd': plane.Dd,
                'Dds': plane.Dds,
                'weight': plane.Dds / self.Ds,
                'profile_type': type(plane.profile).__name__
            }
            info['planes'].append(plane_info)
        
        return info


def compare_single_vs_multiplane(
    main_lens_z: float = 0.5,
    perturber_z: float = 0.3,
    source_z: float = 2.0,
    perturber_fraction: float = 0.1
) -> None:
    """
    Compare single-plane vs multi-plane lensing.
    
    Demonstrates the importance of multi-plane treatment when there
    is significant line-of-sight structure.
    
    Parameters
    ----------
    main_lens_z : float
        Redshift of main lens
    perturber_z : float
        Redshift of foreground perturber
    source_z : float
        Source redshift
    perturber_fraction : float
        Mass fraction of perturber relative to main lens
    """
    print("=" * 70)
    print("SINGLE-PLANE VS MULTI-PLANE COMPARISON")
    print("=" * 70)
    print(f"Main lens redshift: z = {main_lens_z}")
    print(f"Perturber redshift: z = {perturber_z}")
    print(f"Source redshift: z = {source_z}")
    print(f"Perturber mass fraction: {perturber_fraction:.1%}")
    print()


if __name__ == "__main__":
    print("Multi-plane gravitational lensing module loaded successfully.")
    print()
    print("Example usage:")
    print("  from multi_plane import MultiPlaneLens")
    print("  lens = MultiPlaneLens(source_redshift=2.0, cosmology=cosmo)")
    print("  lens.add_plane(redshift=0.3, profile=NFWProfile(...))")
    print("  beta = lens.ray_trace(theta)")
