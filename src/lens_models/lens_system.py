"""
Lens System Class for Cosmological Calculations

This module defines the LensSystem class which handles cosmological
distance calculations and conversions for gravitational lensing.
"""

import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy import constants as const


class LensSystem:
    """
    A gravitational lens system with cosmological distance calculations.
    
    This class computes angular diameter distances and derived quantities
    for a lens-source configuration in a flat ΛCDM cosmology.
    
    Parameters
    ----------
    z_lens : float
        Redshift of the lensing galaxy (must be > 0)
    z_source : float
        Redshift of the background source (must be > z_lens)
    H0 : float, optional
        Hubble constant in km/s/Mpc (default: 70)
    Om0 : float, optional
        Matter density parameter (default: 0.3)
        
    Attributes
    ----------
    cosmology : astropy.cosmology.FlatLambdaCDM
        The cosmological model used for calculations
    z_l : float
        Lens redshift
    z_s : float
        Source redshift
        
    Examples
    --------
    >>> lens_sys = LensSystem(z_lens=0.5, z_source=1.5)
    >>> sigma_cr = lens_sys.critical_surface_density()
    >>> print(f"Critical surface density: {sigma_cr:.2e} Msun/pc²")
    """
    
    def __init__(self, z_lens: float, z_source: float, H0: float = 70.0, Om0: float = 0.3):
        """Initialize the lens system with redshifts and cosmology."""
        if z_lens <= 0:
            raise ValueError("Lens redshift must be positive")
        if z_source <= z_lens:
            raise ValueError("Source redshift must be greater than lens redshift")
            
        self.z_l = z_lens
        self.z_s = z_source
        self.cosmology = FlatLambdaCDM(H0=H0, Om0=Om0)
        
        # Pre-compute distances
        self._D_l = None
        self._D_s = None
        self._D_ls = None
        self._sigma_cr = None
        
    def angular_diameter_distance_lens(self) -> u.Quantity:
        """
        Angular diameter distance to the lens.
        
        Returns
        -------
        astropy.units.Quantity
            Distance in Mpc
        """
        if self._D_l is None:
            self._D_l = self.cosmology.angular_diameter_distance(self.z_l)
        return self._D_l
    
    def angular_diameter_distance_source(self) -> u.Quantity:
        """
        Angular diameter distance to the source.
        
        Returns
        -------
        astropy.units.Quantity
            Distance in Mpc
        """
        if self._D_s is None:
            self._D_s = self.cosmology.angular_diameter_distance(self.z_s)
        return self._D_s
    
    def angular_diameter_distance_lens_source(self) -> u.Quantity:
        """
        Angular diameter distance from lens to source.
        
        Returns
        -------
        astropy.units.Quantity
            Distance in Mpc
        """
        if self._D_ls is None:
            self._D_ls = self.cosmology.angular_diameter_distance_z1z2(self.z_l, self.z_s)
        return self._D_ls
    
    def critical_surface_density(self) -> float:
        """
        Calculate the critical surface density for lensing.
        
        The critical surface density is given by:
        Σ_cr = c²/(4πG) × D_s/(D_l × D_ls)
        
        Returns
        -------
        float
            Critical surface density in solar masses per pc²
            
        Notes
        -----
        This is the surface density that produces unit convergence.
        Typical values are ~10^9 Msun/pc² for cosmological lenses.
        """
        if self._sigma_cr is None:
            D_l = self.angular_diameter_distance_lens()
            D_s = self.angular_diameter_distance_source()
            D_ls = self.angular_diameter_distance_lens_source()
            
            # Calculate critical density: c²/(4πG) × D_s/(D_l × D_ls)
            numerator = (const.c**2 / (4 * np.pi * const.G)) * D_s
            denominator = D_l * D_ls
            
            sigma_cr = (numerator / denominator).to(u.Msun / u.pc**2)
            self._sigma_cr = sigma_cr.value
            
        return self._sigma_cr
    
    def arcsec_to_kpc(self, arcsec: float) -> float:
        """
        Convert angular size in arcseconds to physical size at lens plane.
        
        Parameters
        ----------
        arcsec : float
            Angular size in arcseconds
            
        Returns
        -------
        float
            Physical size in kpc at the lens redshift
            
        Examples
        --------
        >>> lens_sys = LensSystem(0.5, 1.5)
        >>> size_kpc = lens_sys.arcsec_to_kpc(1.0)  # 1 arcsec in kpc
        """
        D_l = self.angular_diameter_distance_lens()
        theta_rad = (arcsec * u.arcsec).to(u.rad).value
        size_kpc = (D_l.to(u.kpc).value) * theta_rad
        return size_kpc
    
    def einstein_radius_scale(self, mass_msun: float) -> float:
        """
        Calculate the Einstein radius for a point mass.
        
        For a point mass M, the Einstein radius is:
        θ_E = sqrt(4GM/c² × D_ls/(D_l × D_s))
        
        Parameters
        ----------
        mass_msun : float
            Mass in solar masses
            
        Returns
        -------
        float
            Einstein radius in arcseconds
            
        Notes
        -----
        This provides a characteristic angular scale for the lens system.
        For extended mass distributions, the actual Einstein radius may differ.
        
        Examples
        --------
        >>> lens_sys = LensSystem(0.5, 1.5)
        >>> theta_E = lens_sys.einstein_radius_scale(1e12)
        >>> print(f"Einstein radius: {theta_E:.3f} arcsec")
        """
        D_l = self.angular_diameter_distance_lens()
        D_s = self.angular_diameter_distance_source()
        D_ls = self.angular_diameter_distance_lens_source()
        
        # θ_E = sqrt(4GM/c² × D_ls/(D_l × D_s))
        M = mass_msun * u.Msun
        factor = 4 * const.G * M / const.c**2
        distance_ratio = D_ls / (D_l * D_s)
        
        theta_E_rad = np.sqrt((factor * distance_ratio).to(u.dimensionless_unscaled).value)
        theta_E_arcsec = (theta_E_rad * u.rad).to(u.arcsec).value
        
        return theta_E_arcsec
    
    def __repr__(self) -> str:
        """String representation of the lens system."""
        return (f"LensSystem(z_lens={self.z_l:.3f}, z_source={self.z_s:.3f}, "
                f"H0={self.cosmology.H0.value:.1f})")
