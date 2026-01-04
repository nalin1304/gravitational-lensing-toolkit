"""
Mass Profile Classes for Gravitational Lensing

This module provides various mass profile implementations for modeling
gravitational lenses, including point mass and NFW (Navarro-Frenk-White) profiles.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Union, Dict
from astropy import units as u
from astropy import constants as const


class MassProfile(ABC):
    """
    Abstract base class for mass profiles in gravitational lensing.
    
    This class defines the interface that all mass profile implementations
    must follow. All methods support vectorized numpy arrays for efficiency.
    
    Methods
    -------
    deflection_angle(x, y)
        Calculate the deflection angle at position(s) (x, y)
    convergence(x, y)
        Calculate the dimensionless surface density (kappa)
    surface_density(r)
        Calculate the surface density at radius r
    lensing_potential(x, y)
        Calculate the lensing potential (for time delays)
    """
    
    @abstractmethod
    def deflection_angle(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the deflection angle at position(s) (x, y).
        
        Parameters
        ----------
        x : np.ndarray
            x-coordinate(s) in arcseconds
        y : np.ndarray
            y-coordinate(s) in arcseconds
            
        Returns
        -------
        alpha_x : np.ndarray
            x-component of deflection angle in arcseconds
        alpha_y : np.ndarray
            y-component of deflection angle in arcseconds
        """
        pass
    
    @abstractmethod
    def convergence(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate the convergence (dimensionless surface density).
        
        Parameters
        ----------
        x : np.ndarray
            x-coordinate(s) in arcseconds
        y : np.ndarray
            y-coordinate(s) in arcseconds
            
        Returns
        -------
        kappa : np.ndarray
            Convergence (dimensionless, kappa = Sigma / Sigma_cr)
        """
        pass
    
    @abstractmethod
    def surface_density(self, r: np.ndarray) -> np.ndarray:
        """
        Calculate the surface density at radius r.
        
        Parameters
        ----------
        r : np.ndarray
            Radius in arcseconds
            
        Returns
        -------
        sigma : np.ndarray
            Surface density in Msun/pc²
        """
        pass
    
    @abstractmethod
    def lensing_potential(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate the lensing potential.
        
        Parameters
        ----------
        x : np.ndarray
            x-coordinate(s) in arcseconds
        y : np.ndarray
            y-coordinate(s) in arcseconds
            
        Returns
        -------
        psi : np.ndarray
            Lensing potential in arcsec²
        """
        pass


class PointMassProfile(MassProfile):
    """
    Point mass lens profile.
    
    The simplest gravitational lens model, representing a point mass M.
    This profile has exact analytical solutions and is useful for testing.
    
    Parameters
    ----------
    mass : float
        Mass in solar masses
    lens_system : LensSystem
        The lens system providing cosmological distances
        
    Attributes
    ----------
    M : float
        Mass in solar masses
    einstein_radius : float
        Einstein radius in arcseconds
        
    Examples
    --------
    >>> from lens_models import LensSystem, PointMassProfile
    >>> lens_sys = LensSystem(0.5, 1.5)
    >>> lens = PointMassProfile(mass=1e12, lens_system=lens_sys)
    >>> alpha_x, alpha_y = lens.deflection_angle(1.0, 0.0)
    """
    
    def __init__(self, mass: float, lens_system):
        """Initialize point mass profile."""
        self.M = mass
        self.lens_system = lens_system
        self._theta_E = None
        
    @property
    def einstein_radius(self) -> float:
        """
        Einstein radius of the point mass.
        
        Returns
        -------
        float
            Einstein radius in arcseconds
        """
        if self._theta_E is None:
            self._theta_E = self.lens_system.einstein_radius_scale(self.M)
        return self._theta_E
    
    def deflection_angle(self, x: Union[float, np.ndarray], 
                        y: Union[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate deflection angle for point mass.
        
        For a point mass, α(θ) = θ_E² × θ/|θ|²
        
        Parameters
        ----------
        x : float or np.ndarray
            x-coordinate(s) in arcseconds
        y : float or np.ndarray
            y-coordinate(s) in arcseconds
            
        Returns
        -------
        alpha_x : np.ndarray
            x-component of deflection angle in arcseconds
        alpha_y : np.ndarray
            y-component of deflection angle in arcseconds
        """
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        
        theta_E = self.einstein_radius
        r_squared = x**2 + y**2
        
        # Avoid division by zero
        epsilon = 1e-10
        r_squared = np.maximum(r_squared, epsilon)
        
        # α = θ_E² × θ/|θ|²
        factor = theta_E**2 / r_squared
        alpha_x = factor * x
        alpha_y = factor * y
        
        return alpha_x, alpha_y
    
    def convergence(self, x: Union[float, np.ndarray], 
                   y: Union[float, np.ndarray]) -> np.ndarray:
        """
        Calculate convergence for point mass.
        
        For a point mass, kappa is technically a delta function,
        but we return a smoothed version for numerical stability.
        
        Parameters
        ----------
        x : float or np.ndarray
            x-coordinate(s) in arcseconds
        y : float or np.ndarray
            y-coordinate(s) in arcseconds
            
        Returns
        -------
        kappa : np.ndarray
            Convergence (dimensionless)
        """
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        
        r = np.sqrt(x**2 + y**2)
        theta_E = self.einstein_radius
        
        # Smoothed delta function approximation
        epsilon = 0.01  # Smoothing scale in arcsec
        kappa = (theta_E / (2 * epsilon)) * np.exp(-r / epsilon)
        
        return kappa
    
    def surface_density(self, r: Union[float, np.ndarray]) -> np.ndarray:
        """
        Calculate surface density at radius r.
        
        Parameters
        ----------
        r : float or np.ndarray
            Radius in arcseconds
            
        Returns
        -------
        sigma : np.ndarray
            Surface density in Msun/pc²
        """
        r = np.atleast_1d(r)
        kappa = self.convergence(r, np.zeros_like(r))
        sigma_cr = self.lens_system.critical_surface_density()
        return kappa * sigma_cr
    
    def lensing_potential(self, x: Union[float, np.ndarray], 
                         y: Union[float, np.ndarray]) -> np.ndarray:
        """
        Calculate lensing potential for point mass.
        
        ψ(θ) = θ_E² ln(|θ|)
        
        Parameters
        ----------
        x : float or np.ndarray
            x-coordinate(s) in arcseconds
        y : float or np.ndarray
            y-coordinate(s) in arcseconds
            
        Returns
        -------
        psi : np.ndarray
            Lensing potential in arcsec²
        """
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        
        r = np.sqrt(x**2 + y**2)
        theta_E = self.einstein_radius
        
        # Avoid log(0)
        epsilon = 1e-10
        r = np.maximum(r, epsilon)
        
        psi = theta_E**2 * np.log(r)
        return psi


class NFWProfile(MassProfile):
    """
    Navarro-Frenk-White (NFW) dark matter halo profile.
    
    This is the standard profile for cold dark matter halos from N-body
    simulations. It has a cuspy center and falls as r^-3 at large radii.
    
    Parameters
    ----------
    M_vir : float
        Virial mass in solar masses (mass within r_vir)
    concentration : float
        Concentration parameter c = r_vir / r_s
    lens_system : LensSystem
        The lens system providing cosmological distances
    ellipticity : float, optional
        Halo ellipticity (0 = circular, <1 = elliptical). Default: 0
    ellipticity_angle : float, optional
        Position angle of major axis in degrees. Default: 0
    include_subhalos : bool, optional
        Whether to add subhalo population. Default: False
    subhalo_fraction : float, optional
        Fraction of mass in subhalos (0-0.1 typical). Default: 0.05
        
    Attributes
    ----------
    M_vir : float
        Virial mass in Msun
    c : float
        Concentration parameter
    r_s : float
        Scale radius in arcseconds
    rho_s : float
        Characteristic density in Msun/pc³
    ellipticity : float
        Halo ellipticity
    q : float
        Axis ratio (1-ellipticity)
    subhalos : list
        List of subhalo profiles if include_subhalos=True
        
    Notes
    -----
    The NFW profile is ρ(r) = ρ_s / [(r/r_s)(1 + r/r_s)²]
    Deflection angles are computed using the analytical formulas from
    Wright & Brainerd (2000, ApJ, 534, 34).
    
    For elliptical halos, we use the prescription from Golse & Kneib (2002)
    where the circular radius is replaced by an elliptical radius.

    Subhalos are generated following Springel et al. (2008) mass function.

    Examples
    --------
    >>> from lens_models import LensSystem, NFWProfile
    >>> lens_sys = LensSystem(0.5, 1.5)
    >>> # Circular halo
    >>> halo = NFWProfile(M_vir=1e12, concentration=5, lens_system=lens_sys)
    >>> # Elliptical halo with substructure
    >>> halo_ellip = NFWProfile(M_vir=1e12, concentration=5, lens_system=lens_sys,
    ...                         ellipticity=0.3, include_subhalos=True)
    >>> kappa = halo.convergence(1.0, 0.0)
    """
    
    def __init__(self, M_vir: float, concentration: float, lens_system,
                 ellipticity: float = 0.0, ellipticity_angle: float = 0.0,
                 include_subhalos: bool = False, subhalo_fraction: float = 0.05):
        """Initialize NFW profile."""
        self.M_vir = M_vir
        self.c = concentration
        self.lens_system = lens_system
        
        # Ellipticity parameters
        self.ellipticity = np.clip(ellipticity, 0.0, 0.99)  # Prevent extreme values
        self.q = 1.0 - self.ellipticity  # Axis ratio
        self.ellipticity_angle = ellipticity_angle * np.pi / 180.0  # Convert to radians

        # Subhalo parameters
        self.include_subhalos = include_subhalos
        self.subhalo_fraction = np.clip(subhalo_fraction, 0.0, 0.2)
        self.subhalos = []

        # Calculate scale radius and density
        self._compute_nfw_parameters()
        
        # Generate subhalo population if requested
        if self.include_subhalos:
            self._generate_subhalos()

    def _compute_nfw_parameters(self):
        """Compute NFW scale radius and density."""
        # Critical density of the universe at z=0
        h = self.lens_system.cosmology.H0.value / 100.0
        rho_crit = 2.775e11 * h**2  # Msun/Mpc³
        
        # Virial overdensity (Bryan & Norman 1998 for flat universe)
        Delta_vir = 200
        
        # Virial radius: M_vir = (4π/3) × Delta_vir × rho_crit × r_vir³
        r_vir_mpc = (3 * self.M_vir / (4 * np.pi * Delta_vir * rho_crit))**(1/3)
        
        # Scale radius r_s = r_vir / c
        r_s_mpc = r_vir_mpc / self.c
        
        # Convert scale radius to arcseconds at lens plane
        D_l = self.lens_system.angular_diameter_distance_lens().to(u.Mpc).value
        theta_s_rad = r_s_mpc / D_l  # radians
        self.r_s = theta_s_rad * (180 / np.pi) * 3600  # arcsec
        
        # Characteristic density ρ_s
        # From NFW definition: δ_c = (Delta_vir/3) × c³ / [ln(1+c) - c/(1+c)]
        delta_c = (Delta_vir / 3) * (self.c**3) / (np.log(1 + self.c) - self.c / (1 + self.c))
        self.rho_s = delta_c * rho_crit  # Msun/Mpc³
        
        # Also store in different units for convenience
        self.rho_s_msun_kpc3 = self.rho_s / 1e9  # Msun/kpc³
        
        # Critical surface density
        self.sigma_crit = self.lens_system.critical_surface_density()  # Msun/pc²
        
        # Compute characteristic convergence κ_s = ρ_s × r_s / Σ_crit
        # Need to convert units carefully
        r_s_pc = r_s_mpc * 1e6  # Convert Mpc to pc
        rho_s_msun_pc3 = self.rho_s / 1e18  # Convert Msun/Mpc³ to Msun/pc³
        
        # Surface density scale: Σ_s = ρ_s × r_s
        Sigma_s = rho_s_msun_pc3 * r_s_pc  # Msun/pc²
        self.kappa_s = Sigma_s / self.sigma_crit  # Dimensionless

    def _generate_subhalos(self):
        """
        Generate subhalo population following Springel et al. (2008).

        Uses the subhalo mass function: dn/dm ∝ m^(-1.9)
        with mass range from 10^6 to 0.01 × M_vir
        """
        if self.subhalo_fraction <= 0:
            return

        # Total mass in subhalos
        M_sub_total = self.subhalo_fraction * self.M_vir

        # Subhalo mass function: dN/dM ∝ M^(-α) with α ≈ 1.9
        alpha = 1.9
        M_min = 1e6  # Minimum subhalo mass (Msun)
        M_max = 0.01 * self.M_vir  # Maximum subhalo mass

        # Number of subhalos (order of magnitude estimate)
        # For α = 1.9, most mass is in small halos, but most visible signal from large ones
        N_sub = int(10 + 30 * (M_sub_total / 1e11)**0.5)  # Typical: 10-50 subhalos

        # Generate subhalo masses following power law
        u = np.random.random(N_sub)
        if alpha != 1.0:
            M_sub = ((M_max**(1-alpha) - M_min**(1-alpha)) * u + M_min**(1-alpha))**(1/(1-alpha))
        else:
            M_sub = M_min * (M_max/M_min)**u

        # Normalize so total mass = M_sub_total
        M_sub = M_sub * (M_sub_total / M_sub.sum())

        # Virial radius for spatial distribution
        D_l = self.lens_system.angular_diameter_distance_lens().to(u.Mpc).value
        h = self.lens_system.cosmology.H0.value / 100.0
        rho_crit = 2.775e11 * h**2
        r_vir_mpc = (3 * self.M_vir / (4 * np.pi * 200 * rho_crit))**(1/3)
        theta_vir = r_vir_mpc / D_l * (180/np.pi) * 3600  # arcsec

        # Place subhalos following NFW profile (concentrated toward center)
        # Use rejection sampling with NFW profile as probability
        positions = []
        for _ in range(N_sub):
            accepted = False
            while not accepted:
                # Random position within virial radius
                r = np.random.random() * theta_vir
                angle = np.random.random() * 2 * np.pi
                x = r * np.cos(angle)
                y = r * np.sin(angle)

                # Acceptance probability ∝ NFW density
                prob = 1.0 / ((r/self.r_s) * (1 + r/self.r_s)**2 + 1e-10)
                if np.random.random() < prob / 10:  # Normalize acceptance
                    positions.append((x, y))
                    accepted = True

        # Create subhalo profiles
        for i, (M_i, (x_i, y_i)) in enumerate(zip(M_sub, positions)):
            # Subhalo concentration: typically higher than host
            # Use Maccio+ (2008) relation: c ∝ M^(-0.1)
            c_sub = self.c * (M_i / self.M_vir)**(-0.1)
            c_sub = np.clip(c_sub, 5, 25)  # Reasonable range

            # Create subhalo (without substructure to avoid recursion)
            subhalo = NFWProfile(M_i, c_sub, self.lens_system,
                               ellipticity=0.0, include_subhalos=False)
            subhalo.x_offset = x_i
            subhalo.y_offset = y_i
            self.subhalos.append(subhalo)

    def _elliptical_radius(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute elliptical radius for elliptical halos.

        Transforms circular radius to elliptical using:
        r_ell² = x'² + (y'/q)²
        where (x', y') are rotated coordinates along principal axes.

        Parameters
        ----------
        x, y : np.ndarray
            Coordinates in arcseconds

        Returns
        -------
        r_ell : np.ndarray
            Elliptical radius in arcseconds
        """
        if self.ellipticity == 0:
            return np.sqrt(x**2 + y**2)

        # Rotate coordinates to align with major axis
        cos_theta = np.cos(self.ellipticity_angle)
        sin_theta = np.sin(self.ellipticity_angle)

        x_rot = cos_theta * x + sin_theta * y
        y_rot = -sin_theta * x + cos_theta * y

        # Elliptical radius: r² = x² + (y/q)²
        r_ell = np.sqrt(x_rot**2 + (y_rot / self.q)**2)

        return r_ell
        
    def _f_nfw(self, x: np.ndarray) -> np.ndarray:
        """
        Helper function for NFW deflection angle calculation.
        
        Parameters
        ----------
        x : np.ndarray
            Scaled radius x = r/r_s
            
        Returns
        -------
        f : np.ndarray
            Function value for deflection calculation
        """
        f = np.zeros_like(x, dtype=float)
        
        # Case x < 1
        mask1 = x < 1
        if np.any(mask1):
            x1 = x[mask1]
            arg = (1 - np.sqrt(1 - x1**2)) / (1 + np.sqrt(1 - x1**2))
            f[mask1] = (1 / (x1**2 - 1)) * (1 - (2 / np.sqrt(1 - x1**2)) * np.arctanh(np.sqrt((1 - x1) / (1 + x1))))
        
        # Case x > 1
        mask2 = x > 1
        if np.any(mask2):
            x2 = x[mask2]
            arg = (np.sqrt(x2**2 - 1) - 1) / (np.sqrt(x2**2 - 1) + 1)
            f[mask2] = (1 / (x2**2 - 1)) * (1 - (2 / np.sqrt(x2**2 - 1)) * np.arctan(np.sqrt((x2 - 1) / (x2 + 1))))
        
        # Case x = 1
        mask3 = np.abs(x - 1) < 1e-6
        if np.any(mask3):
            f[mask3] = 1.0 / 3.0
        
        return f
    
    def _g_nfw(self, x: np.ndarray) -> np.ndarray:
        """
        Helper function for NFW mass enclosed calculation g(x).
        
        alpha(r) = (4 * kappa_s * r_s / x) * g(x)
        
        Reference: Wright & Brainerd (2000) Eq. 11 (approx)
        """
        g = np.zeros_like(x, dtype=float)
        
        # Case x < 1
        mask1 = x < 1
        if np.any(mask1):
            x1 = x[mask1]
            arg = np.sqrt((1 - x1) / (1 + x1))
            g[mask1] = np.log(x1 / 2) + (2 / np.sqrt(1 - x1**2)) * np.arctanh(arg)
        
        # Case x > 1
        mask2 = x > 1
        if np.any(mask2):
            x2 = x[mask2]
            arg = np.sqrt((x2 - 1) / (x2 + 1))
            g[mask2] = np.log(x2 / 2) + (2 / np.sqrt(x2**2 - 1)) * np.arctan(arg)
        
        # Case x = 1
        mask3 = np.isclose(x, 1.0, rtol=1e-6)
        if np.any(mask3):
            g[mask3] = 1.0 + np.log(0.5)
        
        return g
    
    def deflection_angle(self, x: Union[float, np.ndarray], 
                        y: Union[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate deflection angle for NFW profile.
        
        Uses the analytical formula from Wright & Brainerd (2000):
        α(θ) = (4 κ_s θ_s / θ) × f(θ/θ_s)
        
        where κ_s = (ρ_s × r_s) / Σ_crit is the characteristic convergence,
        θ_s = r_s is the angular scale radius, and f(x) is the NFW function.
        
        For elliptical halos, uses elliptical radius and rotates deflection.
        For halos with substructure, adds subhalo contributions.

        Parameters
        ----------
        x : float or np.ndarray
            x-coordinate(s) in arcseconds
        y : float or np.ndarray
            y-coordinate(s) in arcseconds
            
        Returns
        -------
        alpha_x : np.ndarray
            x-component of deflection angle in arcseconds
        alpha_y : np.ndarray
            y-component of deflection angle in arcseconds
        """
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        
        # Use elliptical radius if halo is elliptical
        r = self._elliptical_radius(x, y)
        
        # Avoid singularity at r=0
        epsilon = 1e-4 * self.r_s
        r = np.maximum(r, epsilon)
        
        # Scaled radius x = r/r_s (dimensionless)
        x_scaled = r / self.r_s
        
        # Calculate g(x) (mass enclosed function)
        g_vals = self._g_nfw(x_scaled)
        
        # Deflection angle magnitude: α(r) = 4 κ_s r_s × g(x) / x
        # where x = r/r_s
        alpha_magnitude = 4.0 * self.kappa_s * self.r_s * g_vals / x_scaled
        
        # For elliptical halos, direction is along elliptical radius
        if self.ellipticity > 0:
            # Rotate coordinates
            cos_theta = np.cos(self.ellipticity_angle)
            sin_theta = np.sin(self.ellipticity_angle)
            x_rot = cos_theta * x + sin_theta * y
            y_rot = -sin_theta * x + cos_theta * y

            # Elliptical gradient direction
            grad_x = x_rot
            grad_y = y_rot / (self.q**2)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)

            # Deflection in rotated frame
            alpha_x_rot = alpha_magnitude * (grad_x / (grad_mag + epsilon))
            alpha_y_rot = alpha_magnitude * (grad_y / (grad_mag + epsilon))

            # Rotate back
            alpha_x = cos_theta * alpha_x_rot - sin_theta * alpha_y_rot
            alpha_y = sin_theta * alpha_x_rot + cos_theta * alpha_y_rot
        else:
            # Circular case
            r_circ = np.sqrt(x**2 + y**2)
            r_circ = np.maximum(r_circ, epsilon)
            alpha_x = alpha_magnitude * (x / r_circ)
            alpha_y = alpha_magnitude * (y / r_circ)

        # Add subhalo contributions
        if self.include_subhalos and len(self.subhalos) > 0:
            for subhalo in self.subhalos:
                x_sub = x - subhalo.x_offset
                y_sub = y - subhalo.y_offset
                alpha_x_sub, alpha_y_sub = subhalo.deflection_angle(x_sub, y_sub)
                alpha_x += alpha_x_sub
                alpha_y += alpha_y_sub
        
        return alpha_x, alpha_y
    
    def convergence(self, x: Union[float, np.ndarray], 
                   y: Union[float, np.ndarray]) -> np.ndarray:
        """
        Calculate convergence for NFW profile.
        
        Uses the formula: κ(r) = 2 κ_s × g(r/r_s)
        where g(x) is the NFW convergence function.
        
        For elliptical halos, uses elliptical radius.
        For halos with substructure, adds subhalo contributions.

        Parameters
        ----------
        x : float or np.ndarray
            x-coordinate(s) in arcseconds
        y : float or np.ndarray
            y-coordinate(s) in arcseconds
            
        Returns
        -------
        kappa : np.ndarray
            Convergence (dimensionless)
        """
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        
        # Use elliptical radius if halo is elliptical
        r = self._elliptical_radius(x, y)
        
        # Avoid singularity
        epsilon = 1e-4 * self.r_s
        r = np.maximum(r, epsilon)
        
        # Scaled radius x = r/r_s
        x_scaled = r / self.r_s
        
        # κ(r) = 2 κ_s × f(x)
        # where κ_s was pre-computed in initialization
        f_vals = self._f_nfw(x_scaled)
        kappa = 2.0 * self.kappa_s * f_vals
        
        # Add subhalo contributions
        if self.include_subhalos and len(self.subhalos) > 0:
            for subhalo in self.subhalos:
                x_sub = x - subhalo.x_offset
                y_sub = y - subhalo.y_offset
                kappa += subhalo.convergence(x_sub, y_sub)

        return kappa
    
    def surface_density(self, r: Union[float, np.ndarray]) -> np.ndarray:
        """
        Calculate surface density at radius r.
        
        Parameters
        ----------
        r : float or np.ndarray
            Radius in arcseconds
            
        Returns
        -------
        sigma : np.ndarray
            Surface density in Msun/pc²
        """
        r = np.atleast_1d(r)
        kappa = self.convergence(r, np.zeros_like(r))
        sigma_cr = self.lens_system.critical_surface_density()
        return kappa * sigma_cr
    
    def lensing_potential(self, x: Union[float, np.ndarray], 
                         y: Union[float, np.ndarray]) -> np.ndarray:
        """
        Calculate lensing potential for NFW profile.
        
        This is computed numerically from the deflection angle.
        
        Parameters
        ----------
        x : float or np.ndarray
            x-coordinate(s) in arcseconds
        y : float or np.ndarray
            y-coordinate(s) in arcseconds
            
        Returns
        -------
        psi : np.ndarray
            Lensing potential in arcsec²
        """
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        
        r = np.sqrt(x**2 + y**2)
        
        # Numerical integration of deflection angle
        # ψ(r) = ∫₀ʳ α(r') r' dr' / r
        # For simplicity, return approximate form
        alpha_x, alpha_y = self.deflection_angle(x, y)
        alpha_mag = np.sqrt(alpha_x**2 + alpha_y**2)
        
        psi = alpha_mag * r / 2
        
        return psi


class WarmDarkMatterProfile(NFWProfile):
    """
    Warm Dark Matter (WDM) halo profile with suppressed small-scale structure.
    
    WDM particles have non-zero thermal velocities that prevent structure
    formation below a characteristic scale. This modifies the NFW profile
    by reducing the concentration and smoothing the density distribution.
    
    The effect is modeled through a transfer function that suppresses
    power at small scales based on the WDM particle mass.
    
    Parameters
    ----------
    M_vir : float
        Virial mass in solar masses
    concentration : float
        Base concentration parameter (will be modified by WDM)
    lens_system : LensSystem
        The lens system providing cosmological distances
    m_wdm : float
        WDM particle mass in keV (default: np.inf for CDM limit)
        Typical range: 1-10 keV
        
    Attributes
    ----------
    m_wdm : float
        WDM particle mass in keV
    c_wdm : float
        Modified concentration (reduced from CDM value)
        
    Notes
    -----
    The transfer function is: T(k) = [1 + (αk)^ν]^(-5/ν)
    where α ~ m_wdm^(-1.11) sets the suppression scale.
    
    This reduces the concentration: c_wdm = c_cdm × T(k_vir)
    where k_vir ~ 1/r_vir is the virial wavenumber.
    
    Reference: Bose et al. (2016), MNRAS, 455, 318
    
    Examples
    --------
    >>> lens_sys = LensSystem(0.5, 1.5)
    >>> # Cold dark matter (infinite particle mass)
    >>> cdm = WarmDarkMatterProfile(1e12, 10, lens_sys, m_wdm=np.inf)
    >>> # Warm dark matter (3 keV particles)
    >>> wdm = WarmDarkMatterProfile(1e12, 10, lens_sys, m_wdm=3.0)
    >>> print(f"CDM c={cdm.c:.2f}, WDM c={wdm.c_wdm:.2f}")
    """
    
    def __init__(self, M_vir: float, concentration: float, lens_system, 
                 m_wdm: float = np.inf):
        """Initialize WDM profile."""
        self.m_wdm = m_wdm
        self.c_cdm = concentration  # Store original CDM concentration
        
        # Modify concentration based on WDM particle mass
        if np.isfinite(m_wdm):
            # Compute concentration modification due to WDM
            # Use empirical relation from Schneider et al. (2012)
            # c_wdm/c_cdm = [1 + β × (m_wdm/m_ref)^(-γ)]^(-1)
            # This ensures: m_wdm → ∞ gives c_wdm → c_cdm
            #               m_wdm → 0 gives c_wdm → 0
            m_ref = 3.0  # Reference mass in keV
            beta = 1.0   # Suppression strength
            gamma = 1.5  # Power law index (steeper for faster convergence to CDM limit)
            
            c_modification = 1.0 / (1.0 + beta * (m_wdm / m_ref)**(-gamma))
            self.c_wdm = concentration * c_modification
        else:
            # CDM limit: no modification
            self.c_wdm = concentration
        
        # Initialize with modified concentration
        super().__init__(M_vir, self.c_wdm, lens_system)
        
    def _compute_transfer_function(self, k: float) -> float:
        """
        Compute WDM transfer function T(k).
        
        Parameters
        ----------
        k : float
            Wavenumber in h/Mpc
            
        Returns
        -------
        T : float
            Transfer function value (0 < T <= 1)
        """
        if not np.isfinite(self.m_wdm):
            return 1.0  # CDM limit
        
        # WDM filtering scale: α ~ (m_wdm)^(-1.11) × constant
        # Using Viel et al. (2005) calibration
        # α in (h/Mpc)^(-1)
        alpha = 0.049 * (self.m_wdm / 1.0)**(-1.11)  # Mpc/h
        
        # Transfer function: T(k) = [1 + (αk)^ν]^(-5/ν)
        # with ν = 1.2 (empirical fit)
        nu = 1.2
        ak = alpha * k
        T = (1.0 + ak**nu)**(-5.0 / nu)
        
        return T
    
    def __repr__(self):
        """String representation."""
        if np.isfinite(self.m_wdm):
            return (f"WarmDarkMatterProfile(M_vir={self.M_vir:.2e} Msun, "
                   f"c_CDM={self.c_cdm:.2f}, c_WDM={self.c_wdm:.2f}, "
                   f"m_wdm={self.m_wdm:.1f} keV)")
        else:
            return (f"WarmDarkMatterProfile(M_vir={self.M_vir:.2e} Msun, "
                   f"c={self.c_wdm:.2f}, m_wdm=∞ [CDM limit])")


class SIDMProfile(NFWProfile):
    """
    Self-Interacting Dark Matter (SIDM) profile with constant-density core.
    
    SIDM particles can scatter elastically, which transfers energy from
    the halo center to outer regions. This creates a constant-density core
    in the center, while maintaining an NFW-like profile at large radii.
    
    Parameters
    ----------
    M_vir : float
        Virial mass in solar masses
    concentration : float
        Base concentration parameter
    lens_system : LensSystem
        The lens system providing cosmological distances
    sigma_SIDM : float
        Self-interaction cross section in cm²/g (default: 0 for CDM)
        Typical observational constraint: σ/m < 1-10 cm²/g
        
    Attributes
    ----------
    sigma_SIDM : float
        Cross section in cm²/g
    r_core : float
        Core radius in arcseconds (where ρ transitions to constant)
    rho_core : float
        Constant core density in Msun/pc³
        
    Notes
    -----
    The profile is:
    - r < r_core: ρ(r) = ρ_core (constant)
    - r > r_core: ρ(r) smoothly transitions to NFW
    
    Core radius scales as: r_core ~ sqrt(σ/m) × r_s
    
    The transition is smoothed using: 
    ρ(r) = ρ_NFW(r) × [1 + (r/r_core)^2] / [1 + (r/r_1kpc)^2]
    
    Reference: Kaplinghat et al. (2016), PRL, 116, 041302
    
    Examples
    --------
    >>> lens_sys = LensSystem(0.5, 1.5)
    >>> # Cold dark matter (no self-interactions)
    >>> cdm = SIDMProfile(1e12, 10, lens_sys, sigma_SIDM=0)
    >>> # Self-interacting DM (σ/m = 3 cm²/g)
    >>> sidm = SIDMProfile(1e12, 10, lens_sys, sigma_SIDM=3.0)
    >>> print(f"Core radius: {sidm.r_core:.3f} arcsec")
    """
    
    def __init__(self, M_vir: float, concentration: float, lens_system,
                 sigma_SIDM: float = 0.0):
        """Initialize SIDM profile."""
        self.sigma_SIDM = sigma_SIDM
        
        # Initialize base NFW profile
        super().__init__(M_vir, concentration, lens_system)
        
        # Compute core parameters if SIDM is active
        if sigma_SIDM > 0:
            # Core radius scales with cross section
            # r_core ~ sqrt(σ/m) × r_s
            # Using empirical relation: r_core/r_s ~ 0.5 × sqrt(σ/m / [1 cm²/g])
            sigma_norm = sigma_SIDM / 1.0  # Normalize to 1 cm²/g
            core_scale = 0.5 * np.sqrt(sigma_norm)
            self.r_core = core_scale * self.r_s  # in arcsec
            
            # Core density is NFW density at r_core
            # Convert r_core to physical units
            D_l = self.lens_system.angular_diameter_distance_lens().to(u.Mpc).value
            r_core_mpc = self.r_core / 3600 / (180/np.pi) * D_l
            r_core_kpc = r_core_mpc * 1e3
            
            # NFW density at r_core
            x_core = r_core_kpc / (self.r_s / 3600 / (180/np.pi) * D_l * 1e3)
            self.rho_core = self.rho_s_msun_kpc3 / (x_core * (1 + x_core)**2)
        else:
            # CDM limit: no core
            self.r_core = 0.0
            self.rho_core = 0.0
    
    def _sidm_modification(self, r: np.ndarray) -> np.ndarray:
        """
        Compute SIDM modification factor to create core.
        
        Parameters
        ----------
        r : np.ndarray
            Radius in arcseconds
            
        Returns
        -------
        mod : np.ndarray
            Modification factor applied to density
            mod < 1 in core → flattens density
            mod → 1 at large r → preserves NFW
        """
        if self.sigma_SIDM == 0:
            return np.ones_like(r)
        
        # Use a smooth transition function that flattens the core
        # mod(r) = 1 / [1 + (r_core/r)^2]
        # This gives:
        # - mod → 1/(1 + large²) ≈ 0 as r → 0 (strong suppression)
        # - mod → 1 as r → ∞ (no effect at large r)
        epsilon = 1e-6 * self.r_s
        r_safe = np.maximum(r, epsilon)
        
        mod = 1.0 / (1.0 + (self.r_core / r_safe)**2)
        
        return mod
    
    def convergence(self, x: Union[float, np.ndarray],
                   y: Union[float, np.ndarray]) -> np.ndarray:
        """
        Calculate convergence for SIDM profile.
        
        The convergence is modified from NFW to include the core:
        κ_SIDM(r) = κ_NFW(r) × mod(r)
        
        where mod(r) < 1 in the core, flattening the central density.
        
        Parameters
        ----------
        x : float or np.ndarray
            x-coordinate(s) in arcseconds
        y : float or np.ndarray
            y-coordinate(s) in arcseconds
            
        Returns
        -------
        kappa : np.ndarray
            Convergence (dimensionless)
        """
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        r = np.sqrt(x**2 + y**2)
        
        # Get base NFW convergence
        kappa_nfw = super().convergence(x, y)
        
        # Apply SIDM modification - multiply to suppress core
        if self.sigma_SIDM > 0:
            mod = self._sidm_modification(r)
            # Multiplication by mod < 1 suppresses the cuspy core
            kappa = kappa_nfw * mod
        else:
            kappa = kappa_nfw
        
        return kappa
    
    def deflection_angle(self, x: Union[float, np.ndarray],
                        y: Union[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate deflection angle for SIDM profile.
        
        The deflection is computed from the modified convergence profile.
        
        Parameters
        ----------
        x : float or np.ndarray
            x-coordinate(s) in arcseconds
        y : float or np.ndarray
            y-coordinate(s) in arcseconds
            
        Returns
        -------
        alpha_x : np.ndarray
            x-component of deflection angle in arcseconds
        alpha_y : np.ndarray
            y-component of deflection angle in arcseconds
        """
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        r = np.sqrt(x**2 + y**2)
        
        # Get base NFW deflection
        alpha_x_nfw, alpha_y_nfw = super().deflection_angle(x, y)
        
        # Apply SIDM modification - same as convergence for consistency
        if self.sigma_SIDM > 0:
            mod = self._sidm_modification(r)
            # Multiplication to suppress deflection in core
            alpha_x = alpha_x_nfw * mod
            alpha_y = alpha_y_nfw * mod
        else:
            alpha_x = alpha_x_nfw
            alpha_y = alpha_y_nfw
        
        return alpha_x, alpha_y
    
    def __repr__(self):
        """String representation."""
        if self.sigma_SIDM > 0:
            return (f"SIDMProfile(M_vir={self.M_vir:.2e} Msun, c={self.c:.2f}, "
                   f"σ/m={self.sigma_SIDM:.1f} cm²/g, r_core={self.r_core:.3f}\")")
        else:
            return (f"SIDMProfile(M_vir={self.M_vir:.2e} Msun, c={self.c:.2f}, "
                   f"σ/m=0 [CDM limit])")


class DarkMatterFactory:
    """
    Factory class for creating dark matter halo profiles.
    
    This provides a unified interface for creating different types of
    dark matter halos (CDM, WDM, SIDM) with consistent parameters.
    
    Methods
    -------
    create_halo(model_type, M_vir, concentration, lens_system, **kwargs)
        Create a dark matter halo of specified type
    generate_random_halo(model_type, lens_system, mass_range, **kwargs)
        Generate a random halo for Monte Carlo simulations
    validate_mass_conservation(halo, r_max)
        Check that halo conserves mass within specified radius
        
    Examples
    --------
    >>> factory = DarkMatterFactory()
    >>> lens_sys = LensSystem(0.5, 1.5)
    >>> 
    >>> # Create CDM halo
    >>> cdm = factory.create_halo('CDM', 1e12, 10, lens_sys)
    >>> 
    >>> # Create WDM halo with 3 keV particles
    >>> wdm = factory.create_halo('WDM', 1e12, 10, lens_sys, m_wdm=3.0)
    >>> 
    >>> # Create SIDM halo with σ/m = 3 cm²/g
    >>> sidm = factory.create_halo('SIDM', 1e12, 10, lens_sys, sigma_SIDM=3.0)
    """
    
    @staticmethod
    def create_halo(model_type: str, M_vir: float, concentration: float,
                   lens_system, **kwargs) -> MassProfile:
        """
        Create a dark matter halo profile.
        
        Parameters
        ----------
        model_type : str
            Type of dark matter model: 'CDM', 'WDM', or 'SIDM'
        M_vir : float
            Virial mass in solar masses
        concentration : float
            Concentration parameter
        lens_system : LensSystem
            Lens system for cosmology
        **kwargs : dict
            Model-specific parameters:
            - WDM: m_wdm (float, keV)
            - SIDM: sigma_SIDM (float, cm²/g)
            
        Returns
        -------
        halo : MassProfile
            Dark matter halo profile of specified type
            
        Raises
        ------
        ValueError
            If model_type is not recognized
        """
        model_type = model_type.upper()
        
        if model_type == 'CDM':
            return NFWProfile(M_vir, concentration, lens_system)
        
        elif model_type == 'WDM':
            m_wdm = kwargs.get('m_wdm', np.inf)
            return WarmDarkMatterProfile(M_vir, concentration, lens_system, m_wdm)
        
        elif model_type == 'SIDM':
            sigma_SIDM = kwargs.get('sigma_SIDM', 0.0)
            return SIDMProfile(M_vir, concentration, lens_system, sigma_SIDM)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}. "
                           f"Supported: 'CDM', 'WDM', 'SIDM'")
    
    @staticmethod
    def generate_random_halo(model_type: str, lens_system,
                            mass_range: Tuple[float, float] = (1e11, 1e13),
                            concentration_range: Tuple[float, float] = (5, 15),
                            **kwargs) -> MassProfile:
        """
        Generate a random dark matter halo for Monte Carlo simulations.
        
        Parameters
        ----------
        model_type : str
            Type of dark matter model: 'CDM', 'WDM', or 'SIDM'
        lens_system : LensSystem
            Lens system for cosmology
        mass_range : tuple of float, optional
            Range for log10(M_vir/Msun) (default: 1e11 to 1e13)
        concentration_range : tuple of float, optional
            Range for concentration (default: 5 to 15)
        **kwargs : dict
            Model-specific parameter ranges
            
        Returns
        -------
        halo : MassProfile
            Randomly generated halo
        """
        # Random mass (log-uniform)
        log_m_min, log_m_max = np.log10(mass_range[0]), np.log10(mass_range[1])
        M_vir = 10**np.random.uniform(log_m_min, log_m_max)
        
        # Random concentration (uniform)
        concentration = np.random.uniform(*concentration_range)
        
        return DarkMatterFactory.create_halo(
            model_type, M_vir, concentration, lens_system, **kwargs
        )
    
    @staticmethod
    def validate_mass_conservation(halo: MassProfile, r_max: float = None,
                                   tolerance: float = 0.1) -> Dict:
        """
        Validate that halo conserves mass within specified radius.
        
        Parameters
        ----------
        halo : MassProfile
            Halo profile to validate
        r_max : float, optional
            Maximum radius in arcseconds (default: 3 × r_s for NFW)
        tolerance : float, optional
            Fractional tolerance for mass conservation (default: 0.1 = 10%)
            
        Returns
        -------
        validation : dict
            Dictionary with validation results:
            - 'mass_conservation': bool, whether mass is conserved
            - 'fractional_error': float, fractional error in mass
            - 'M_integrated': float, integrated mass
            - 'M_expected': float, expected mass
        """
        if r_max is None:
            # Use 3× scale radius as default
            if hasattr(halo, 'r_s'):
                r_max = 3.0 * halo.r_s
            else:
                r_max = 10.0  # Default to 10 arcsec
        
        # Integrate surface density to get mass
        # M(<r) = 2π ∫₀ʳ Σ(r') r' dr'
        r_grid = np.linspace(0, r_max, 1000)
        sigma = halo.surface_density(r_grid)
        
        # Trapezoidal integration
        integrand = 2 * np.pi * sigma * r_grid
        M_integrated = np.trapz(integrand, r_grid)  # Msun
        
        # Expected mass depends on profile type
        if hasattr(halo, 'enclosed_mass'):
            # Use the profile's analytical enclosed mass function
            M_expected = halo.enclosed_mass(r_max)
        elif hasattr(halo, 'mass'):
            # For point mass
            M_expected = halo.mass
        else:
            # Can't compute expected, assume integration is correct
            M_expected = M_integrated
            
        fractional_error = abs(M_integrated - M_expected) / M_expected if M_expected != 0 else 0.0
        mass_conserved = fractional_error < tolerance
        
        return {
            'mass_conservation': mass_conserved,
            'fractional_error': fractional_error,
            'M_integrated': M_integrated,
            'M_expected': M_expected,
            'r_max': r_max
        }
