"""
Physical Constants Module

Centralized physical constants for gravitational lensing calculations.
All values use SI units unless otherwise specified.

Author: Gravitational Lensing Project
Reference: CODATA 2018 recommended values
"""

import numpy as np

# ============================================================================
# Fundamental Physical Constants (SI units)
# ============================================================================

#: Gravitational constant [m³ kg⁻¹ s⁻²]
#: Reference: CODATA 2018
G_CONST = 6.67430e-11  # m³ kg⁻¹ s⁻²

#: Speed of light in vacuum [m s⁻¹]
#: Reference: Exact value by definition (SI 2019)
C_LIGHT = 299792458.0  # m s⁻¹

#: Planck constant [J s]
#: Reference: CODATA 2018
H_PLANCK = 6.62607015e-34  # J s

#: Reduced Planck constant (ℏ = h / 2π) [J s]
HBAR = H_PLANCK / (2.0 * np.pi)  # J s

#: Boltzmann constant [J K⁻¹]
#: Reference: CODATA 2018
K_BOLTZMANN = 1.380649e-23  # J K⁻¹

# ============================================================================
# Astronomical Constants
# ============================================================================

#: Solar mass [kg]
#: Reference: IAU 2015 nominal solar mass
M_SUN_KG = 1.98847e30  # kg

#: Solar mass [10¹² M☉] - convenient for galaxy cluster calculations
M_SUN_1E12 = M_SUN_KG / 1e12  # kg (for expressing masses in 10¹² M☉)

#: Parsec [m]
#: Reference: IAU definition (1 pc = 648000/π AU)
PARSEC = 3.0856775814913673e16  # m

#: Kiloparsec [m]
KPC = 1e3 * PARSEC  # m

#: Megaparsec [m]
MPC = 1e6 * PARSEC  # m

#: Astronomical Unit [m]
#: Reference: IAU 2012 (exact definition)
AU = 1.495978707e11  # m

#: Solar radius [m]
#: Reference: IAU 2015 nominal solar radius
R_SUN = 6.957e8  # m

#: Solar luminosity [W]
#: Reference: IAU 2015 nominal solar luminosity
L_SUN = 3.828e26  # W

# ============================================================================
# Angular Conversion Constants
# ============================================================================

#: Arcsecond to radians conversion factor
ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)  # rad/arcsec

#: Radians to arcsecond conversion factor
RAD_TO_ARCSEC = (180.0 * 3600.0) / np.pi  # arcsec/rad

#: Degree to radians conversion factor
DEG_TO_RAD = np.pi / 180.0  # rad/deg

#: Radians to degree conversion factor
RAD_TO_DEG = 180.0 / np.pi  # deg/rad

# ============================================================================
# Cosmological Constants (Planck 2018 Results)
# ============================================================================

#: Hubble constant (Planck 2018) [km s⁻¹ Mpc⁻¹]
#: Reference: Planck Collaboration (2020), A&A 641, A6
H0_PLANCK = 67.4  # km s⁻¹ Mpc⁻¹

#: Hubble constant in SI units [s⁻¹]
H0_SI = H0_PLANCK * 1000.0 / MPC  # s⁻¹

#: Matter density parameter (Planck 2018)
OMEGA_M_PLANCK = 0.315

#: Dark energy density parameter (Planck 2018)
OMEGA_LAMBDA_PLANCK = 0.685

#: Baryon density parameter (Planck 2018)
OMEGA_B_PLANCK = 0.049

#: Dark matter density parameter (derived)
OMEGA_DM_PLANCK = OMEGA_M_PLANCK - OMEGA_B_PLANCK

#: Curvature density parameter
OMEGA_K = 0.0  # Flat universe assumption

# ============================================================================
# Critical Density
# ============================================================================

def critical_density(H0: float = H0_PLANCK) -> float:
    """
    Calculate critical density of the universe.
    
    ρ_c = 3 H₀² / (8π G)
    
    Parameters
    ----------
    H0 : float, optional
        Hubble constant [km s⁻¹ Mpc⁻¹], default: Planck 2018 value
    
    Returns
    -------
    rho_c : float
        Critical density [kg m⁻³]
    
    Examples
    --------
    >>> rho_c = critical_density()
    >>> print(f"Critical density: {rho_c:.3e} kg/m³")
    """
    H0_si = H0 * 1000.0 / MPC  # Convert to s⁻¹
    return (3.0 * H0_si**2) / (8.0 * np.pi * G_CONST)


# Precompute critical density for default cosmology
RHO_CRIT = critical_density()  # kg m⁻³

# ============================================================================
# Gravitational Lensing Constants
# ============================================================================

#: Einstein radius coefficient [m]
#: Used in: θ_E = sqrt(4GM/c² × D_LS/(D_L D_S))
EINSTEIN_COEFF = 4.0 * G_CONST / (C_LIGHT**2)  # m kg⁻¹

#: Critical surface density coefficient [kg m⁻²]
#: Used in: Σ_crit = c²/(4πG) × D_S/(D_L D_LS)
SIGMA_CRIT_COEFF = (C_LIGHT**2) / (4.0 * np.pi * G_CONST)  # kg s² m⁻⁵

#: Schwarzschild radius coefficient [m kg⁻¹]
#: Used in: r_s = 2GM/c²
SCHWARZSCHILD_COEFF = 2.0 * G_CONST / (C_LIGHT**2)  # m kg⁻¹

# ============================================================================
# Unit Conversions for Gravitational Lensing
# ============================================================================

def schwarzschild_radius(mass_kg: float) -> float:
    """
    Calculate Schwarzschild radius for a given mass.

    r_s = 2GM/c²

    Parameters
    ----------
    mass_kg : float
        Mass in kilograms

    Returns
    -------
    r_s : float
        Schwarzschild radius in meters

    Examples
    --------
    >>> M_bh = 1e6 * M_SUN_KG  # Million solar mass black hole
    >>> r_s = schwarzschild_radius(M_bh)
    >>> print(f"r_s = {r_s/1e3:.2f} km")
    r_s = 2953.25 km
    """
    return SCHWARZSCHILD_COEFF * mass_kg

def einstein_radius_arcsec(M_kg: float, D_l_m: float, D_s_m: float, D_ls_m: float) -> float:
    """
    Calculate Einstein radius in arcseconds.
    
    θ_E = sqrt(4GM/c² × D_LS/(D_L D_S))
    
    Parameters
    ----------
    M_kg : float
        Lens mass [kg]
    D_l_m : float
        Angular diameter distance to lens [m]
    D_s_m : float
        Angular diameter distance to source [m]
    D_ls_m : float
        Angular diameter distance lens to source [m]
    
    Returns
    -------
    theta_E : float
        Einstein radius [arcsec]
    
    Examples
    --------
    >>> M = 1e14 * M_SUN_KG  # Galaxy cluster mass
    >>> D_l = 1000 * MPC  # 1 Gpc
    >>> D_s = 2000 * MPC  # 2 Gpc
    >>> D_ls = 1000 * MPC  # 1 Gpc
    >>> theta_E = einstein_radius_arcsec(M, D_l, D_s, D_ls)
    >>> print(f"Einstein radius: {theta_E:.2f} arcsec")
    """
    theta_rad = np.sqrt(EINSTEIN_COEFF * M_kg * D_ls_m / (D_l_m * D_s_m))
    return theta_rad * RAD_TO_ARCSEC


def critical_surface_density(D_l_m: float, D_s_m: float, D_ls_m: float) -> float:
    """
    Calculate critical surface density for lensing.
    
    Σ_crit = c²/(4πG) × D_S/(D_L D_LS)
    
    Parameters
    ----------
    D_l_m : float
        Angular diameter distance to lens [m]
    D_s_m : float
        Angular diameter distance to source [m]
    D_ls_m : float
        Angular diameter distance lens to source [m]
    
    Returns
    -------
    Sigma_crit : float
        Critical surface density [kg m⁻²]
    
    Examples
    --------
    >>> D_l = 1000 * MPC
    >>> D_s = 2000 * MPC
    >>> D_ls = 1000 * MPC
    >>> Sigma_c = critical_surface_density(D_l, D_s, D_ls)
    >>> print(f"Σ_crit: {Sigma_c:.3e} kg/m²")
    """
    return SIGMA_CRIT_COEFF * D_s_m / (D_l_m * D_ls_m)


# ============================================================================
# Common Physical Scales for Gravitational Lensing
# ============================================================================

#: Typical galaxy cluster mass [kg]
M_CLUSTER_TYPICAL = 1e14 * M_SUN_KG  # kg

#: Typical galaxy mass [kg]
M_GALAXY_TYPICAL = 1e12 * M_SUN_KG  # kg

#: Typical NFW scale radius for galaxy cluster [m]
R_S_CLUSTER_TYPICAL = 200.0 * KPC  # m

#: Typical NFW scale radius for galaxy [m]
R_S_GALAXY_TYPICAL = 20.0 * KPC  # m

#: Typical lens redshift
Z_LENS_TYPICAL = 0.5

#: Typical source redshift
Z_SOURCE_TYPICAL = 2.0

# ============================================================================
# Convenience Functions
# ============================================================================

def mass_to_solar(M_kg: float) -> float:
    """
    Convert mass from kg to solar masses.
    
    Parameters
    ----------
    M_kg : float
        Mass [kg]
    
    Returns
    -------
    M_sun : float
        Mass [M☉]
    
    Examples
    --------
    >>> M_kg = 1.98847e30
    >>> print(f"Mass: {mass_to_solar(M_kg):.1f} M☉")
    Mass: 1.0 M☉
    """
    return M_kg / M_SUN_KG


def solar_to_mass(M_sun: float) -> float:
    """
    Convert mass from solar masses to kg.
    
    Parameters
    ----------
    M_sun : float
        Mass [M☉]
    
    Returns
    -------
    M_kg : float
        Mass [kg]
    
    Examples
    --------
    >>> M_sun = 1.0
    >>> print(f"Mass: {solar_to_mass(M_sun):.3e} kg")
    Mass: 1.988e+30 kg
    """
    return M_sun * M_SUN_KG


def kpc_to_meters(kpc: float) -> float:
    """Convert kiloparsecs to meters."""
    return kpc * KPC


def meters_to_kpc(meters: float) -> float:
    """Convert meters to kiloparsecs."""
    return meters / KPC


def arcsec_to_radians(arcsec: float) -> float:
    """Convert arcseconds to radians."""
    return arcsec * ARCSEC_TO_RAD


def radians_to_arcsec(radians: float) -> float:
    """Convert radians to arcseconds."""
    return radians * RAD_TO_ARCSEC


# ============================================================================
# Module Documentation
# ============================================================================

__all__ = [
    # Fundamental constants
    'G_CONST', 'C_LIGHT', 'H_PLANCK', 'HBAR', 'K_BOLTZMANN',
    
    # Astronomical constants
    'M_SUN_KG', 'M_SUN_1E12', 'PARSEC', 'KPC', 'MPC', 'AU', 'R_SUN', 'L_SUN',
    
    # Angular conversions
    'ARCSEC_TO_RAD', 'RAD_TO_ARCSEC', 'DEG_TO_RAD', 'RAD_TO_DEG',
    
    # Cosmological parameters
    'H0_PLANCK', 'H0_SI', 'OMEGA_M_PLANCK', 'OMEGA_LAMBDA_PLANCK',
    'OMEGA_B_PLANCK', 'OMEGA_DM_PLANCK', 'OMEGA_K', 'RHO_CRIT',
    
    # Lensing constants
    'EINSTEIN_COEFF', 'SIGMA_CRIT_COEFF', 'SCHWARZSCHILD_COEFF',
    
    # Typical scales
    'M_CLUSTER_TYPICAL', 'M_GALAXY_TYPICAL', 'R_S_CLUSTER_TYPICAL',
    'R_S_GALAXY_TYPICAL', 'Z_LENS_TYPICAL', 'Z_SOURCE_TYPICAL',
    
    # Functions
    'critical_density', 'einstein_radius_arcsec', 'critical_surface_density',
    'schwarzschild_radius',
    'mass_to_solar', 'solar_to_mass', 'kpc_to_meters', 'meters_to_kpc',
    'arcsec_to_radians', 'radians_to_arcsec'
]

if __name__ == "__main__":
    # Print module constants for verification
    print("=" * 60)
    print("Physical Constants Module")
    print("=" * 60)
    print(f"\nFundamental Constants:")
    print(f"  G = {G_CONST:.5e} m³ kg⁻¹ s⁻²")
    print(f"  c = {C_LIGHT:.3e} m s⁻¹")
    print(f"  h = {H_PLANCK:.5e} J s")
    print(f"\nAstronomical Constants:")
    print(f"  M☉ = {M_SUN_KG:.5e} kg")
    print(f"  1 pc = {PARSEC:.5e} m")
    print(f"  1 kpc = {KPC:.5e} m")
    print(f"  1 Mpc = {MPC:.5e} m")
    print(f"\nCosmological Parameters (Planck 2018):")
    print(f"  H₀ = {H0_PLANCK} km s⁻¹ Mpc⁻¹")
    print(f"  Ωₘ = {OMEGA_M_PLANCK}")
    print(f"  ΩΛ = {OMEGA_LAMBDA_PLANCK}")
    print(f"  ρ_crit = {RHO_CRIT:.3e} kg m⁻³")
    print("=" * 60)
