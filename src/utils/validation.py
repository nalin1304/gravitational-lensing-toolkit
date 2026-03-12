"""
Input Validation Module for Gravitational Lensing

This module provides comprehensive input validation functions for all common
lens parameters, physical constraints, and array types used in gravitational
lensing calculations.

Example Usage
-------------
Basic parameter validation:

    >>> from src.utils.validation import (
    ...     validate_redshift,
    ...     validate_mass,
    ...     validate_ellipticity,
    ...     validate_lens_system
    ... )

    # Validate individual parameters
    >>> z_lens = validate_redshift(0.5, param_name="lens redshift")
    >>> z_source = validate_redshift(1.5, param_name="source redshift")
    >>> validate_source_lens_redshift(z_source, z_lens)

    # Validate mass
    >>> M_vir = validate_mass(1e12, units="Msun")

    # Validate ellipticity
    >>> e = validate_ellipticity(0.3)

    # Validate concentration
    >>> c = validate_concentration(5.0)

    # Validate position angle
    >>> pa = validate_position_angle(45.0)

    # Validate coordinate arrays
    >>> x, y = validate_coordinate_arrays(np.array([1.0, 2.0]), np.array([0.5, 1.5]))

    # Validate entire lens system configuration
    >>> config = {
    ...     'z_lens': 0.5,
    ...     'z_source': 2.0,
    ...     'M_vir': 1e12,
    ...     'concentration': 5.0,
    ...     'ellipticity': 0.3,
    ...     'position_angle': 45.0
    ... }
    >>> validated = validate_lens_configuration(config)

Validation with physical constraints:

    >>> from src.utils.validation import validate_lens_geometry

    # Validate lens geometry (includes redshift ordering)
    >>> validate_lens_geometry(z_lens=0.5, z_source=1.5, M_vir=1e12)

Custom validation with informative errors:

    >>> try:
    ...     validate_ellipticity(1.2)
    ... except ValueError as e:
    ...     print(f"Validation failed: {e}")
    Validation failed: Ellipticity must be in range [0, 1), got 1.2

Notes
-----
All validation functions:
- Use type hints for clarity and IDE support
- Return validated values (possibly normalized)
- Raise ValueError with descriptive messages on failure
- Support both scalar and array inputs where appropriate
- Include comprehensive docstrings with parameter descriptions

Author: Gravitational Lensing Project
Version: 1.0.0
"""

from typing import Union, Tuple, Optional, Dict, Any, List
import numpy as np
from numpy.typing import NDArray


# ============================================================================
# Custom Exceptions
# ============================================================================


class ValidationError(ValueError):
    """Base exception for validation errors with detailed context."""

    def __init__(self, message: str, param_name: str = None, param_value: Any = None):
        self.param_name = param_name
        self.param_value = param_value
        super().__init__(message)


class RedshiftError(ValidationError):
    """Exception for redshift validation errors."""

    pass


class MassError(ValidationError):
    """Exception for mass validation errors."""

    pass


class GeometryError(ValidationError):
    """Exception for geometric parameter validation errors."""

    pass


class ArrayError(ValidationError):
    """Exception for array validation errors."""

    pass


# ============================================================================
# Redshift Validation
# ============================================================================


def validate_redshift(
    z: Union[float, np.ndarray],
    param_name: str = "redshift",
    allow_zero: bool = False,
    max_z: float = 20.0,
) -> Union[float, np.ndarray]:
    """
    Validate redshift value(s).

    Parameters
    ----------
    z : float or np.ndarray
        Redshift value(s) to validate
    param_name : str, optional
        Name of the parameter for error messages (default: "redshift")
    allow_zero : bool, optional
        Whether to allow z = 0 (local universe). Default is False
        for lens redshifts (must be > 0) but may be True for sources
    max_z : float, optional
        Maximum allowed redshift (default: 20.0, beyond reionization)

    Returns
    -------
    float or np.ndarray
        Validated redshift value(s)

    Raises
    ------
    RedshiftError
        If redshift is negative, too large, or non-finite

    Examples
    --------
    >>> validate_redshift(0.5)
    0.5

    >>> validate_redshift(2.0, param_name="source redshift")
    2.0

    >>> validate_redshift(np.array([0.1, 0.5, 1.0]))
    array([0.1, 0.5, 1.0])

    >>> validate_redshift(0.0, allow_zero=True)
    0.0

    Notes
    -----
    Physical constraints:
    - Redshift must be non-negative (z >= 0)
    - For lens galaxies, typically z > 0 (not at observer)
    - For high-z sources, z < 20 is a reasonable upper limit
    - Arrays must contain all finite values
    """
    z_arr = np.atleast_1d(z)

    # Check for non-finite values
    if not np.all(np.isfinite(z_arr)):
        invalid_indices = np.where(~np.isfinite(z_arr))[0]
        raise RedshiftError(
            f"{param_name} contains non-finite values at indices: {invalid_indices}",
            param_name=param_name,
            param_value=z,
        )

    # Check minimum value
    min_z = 0.0 if allow_zero else 1e-10
    if np.any(z_arr < min_z):
        invalid_values = z_arr[z_arr < min_z]
        raise RedshiftError(
            f"{param_name} must be {'>= 0' if allow_zero else '> 0'}, "
            f"got invalid values: {invalid_values}",
            param_name=param_name,
            param_value=z,
        )

    # Check maximum value
    if np.any(z_arr > max_z):
        invalid_values = z_arr[z_arr > max_z]
        raise RedshiftError(
            f"{param_name} must be <= {max_z} (beyond reionization era), "
            f"got invalid values: {invalid_values}",
            param_name=param_name,
            param_value=z,
        )

    return z if np.isscalar(z) else z_arr


def validate_source_lens_redshift(
    z_source: Union[float, np.ndarray],
    z_lens: Union[float, np.ndarray],
    tolerance: float = 1e-6,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Validate that source redshift is greater than lens redshift.

    This is a fundamental physical constraint for gravitational lensing:
    the source must be behind the lens.

    Parameters
    ----------
    z_source : float or np.ndarray
        Source redshift(s)
    z_lens : float or np.ndarray
        Lens redshift(s)
    tolerance : float, optional
        Minimum separation required between z_source and z_lens (default: 1e-6)

    Returns
    -------
    tuple
        (validated z_source, validated z_lens)

    Raises
    ------
    RedshiftError
        If z_source <= z_lens or if shapes are incompatible

    Examples
    --------
    >>> validate_source_lens_redshift(1.5, 0.5)
    (1.5, 0.5)

    >>> z_s = np.array([1.0, 2.0, 3.0])
    >>> z_l = np.array([0.3, 0.5, 0.8])
    >>> validate_source_lens_redshift(z_s, z_l)
    (array([1., 2., 3.]), array([0.3, 0.5, 0.8]))

    Notes
    -----
    For strong lensing to occur:
    - Source must be behind lens: z_source > z_lens
    - This ensures positive angular diameter distance D_ls
    - Avoids pathological cases where D_ls = 0
    """
    # Validate individual redshifts
    z_source = validate_redshift(z_source, param_name="source redshift")
    z_lens = validate_redshift(z_lens, param_name="lens redshift")

    # Convert to arrays for comparison
    z_s_arr = np.atleast_1d(z_source)
    z_l_arr = np.atleast_1d(z_lens)

    # Check shape compatibility
    if z_s_arr.shape != z_l_arr.shape and z_l_arr.size != 1:
        raise RedshiftError(
            f"Shape mismatch: z_source {z_s_arr.shape} vs z_lens {z_l_arr.shape}",
            param_name="redshift pair",
            param_value=(z_source, z_lens),
        )

    # Check physical constraint: source must be behind lens
    if np.any(z_s_arr <= z_l_arr + tolerance):
        invalid_pairs = np.column_stack(
            [
                z_s_arr[z_s_arr <= z_l_arr + tolerance],
                z_l_arr[z_s_arr <= z_l_arr + tolerance],
            ]
        )
        raise RedshiftError(
            f"Source redshift must be greater than lens redshift. "
            f"Invalid (z_source, z_lens) pairs: {invalid_pairs}",
            param_name="redshift geometry",
            param_value=(z_source, z_lens),
        )

    return z_source, z_lens


# ============================================================================
# Mass Validation
# ============================================================================


def validate_mass(
    mass: Union[float, np.ndarray],
    units: str = "Msun",
    param_name: str = "mass",
    allow_zero: bool = False,
    max_mass: Optional[float] = None,
) -> Union[float, np.ndarray]:
    """
    Validate mass value(s).

    Parameters
    ----------
    mass : float or np.ndarray
        Mass value(s) to validate
    units : str, optional
        Units of mass: "Msun" (solar masses), "kg", or "Msun_1e12"
        (10^12 solar masses). Default is "Msun"
    param_name : str, optional
        Name of the parameter for error messages (default: "mass")
    allow_zero : bool, optional
        Whether to allow zero mass. Default is False for physical lenses
    max_mass : float, optional
        Maximum allowed mass in specified units. Default is None (no limit)

    Returns
    -------
    float or np.ndarray
        Validated mass value(s)

    Raises
    ------
    MassError
        If mass is negative, zero (when not allowed), or exceeds max_mass

    Examples
    --------
    >>> validate_mass(1e12)
    1000000000000.0

    >>> validate_mass(1.0, units="Msun_1e12")
    1.0

    >>> validate_mass(5e41, units="kg", param_name="virial mass")
    5e+41

    >>> masses = np.array([1e11, 1e12, 1e13])
    >>> validate_mass(masses)
    array([1.e+11, 1.e+12, 1.e+13])

    Notes
    -----
    Typical mass ranges:
    - Galaxy: 10^10 - 10^12 Msun
    - Galaxy cluster: 10^13 - 10^15 Msun
    - Supermassive black hole: 10^6 - 10^10 Msun
    """
    mass_arr = np.atleast_1d(mass)

    # Check for non-finite values
    if not np.all(np.isfinite(mass_arr)):
        invalid_indices = np.where(~np.isfinite(mass_arr))[0]
        raise MassError(
            f"{param_name} contains non-finite values at indices: {invalid_indices}",
            param_name=param_name,
            param_value=mass,
        )

    # Check minimum value
    min_mass = 0.0 if allow_zero else np.finfo(float).eps
    if np.any(mass_arr <= min_mass):
        invalid_values = mass_arr[mass_arr <= min_mass]
        raise MassError(
            f"{param_name} must be {'>= 0' if allow_zero else '> 0'}, "
            f"got invalid values: {invalid_values} {units}",
            param_name=param_name,
            param_value=mass,
        )

    # Check maximum value
    if max_mass is not None and np.any(mass_arr > max_mass):
        invalid_values = mass_arr[mass_arr > max_mass]
        raise MassError(
            f"{param_name} must be <= {max_mass} {units}, "
            f"got invalid values: {invalid_values}",
            param_name=param_name,
            param_value=mass,
        )

    return mass if np.isscalar(mass) else mass_arr


def validate_mass_concentration_ratio(
    mass: Union[float, np.ndarray],
    concentration: Union[float, np.ndarray],
    mass_units: str = "Msun",
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Validate the mass-concentration relation consistency.

    While there is no strict constraint, very high concentration with
    very high mass (or vice versa) can indicate unrealistic parameters.

    Parameters
    ----------
    mass : float or np.ndarray
        Virial mass value(s)
    concentration : float or np.ndarray
        Concentration parameter value(s)
    mass_units : str, optional
        Units of mass (default: "Msun")

    Returns
    -------
    tuple
        (validated mass, validated concentration)

    Raises
    ------
    ValidationError
        If parameters are physically inconsistent

    Examples
    --------
    >>> validate_mass_concentration_ratio(1e12, 5.0)
    (1e+12, 5.0)

    Notes
    -----
    Typical concentration values decrease with mass:
    - Low mass halos: c ~ 15-20
    - Galaxy halos: c ~ 5-15
    - Cluster halos: c ~ 2-8
    """
    mass = validate_mass(mass, units=mass_units, param_name="virial mass")
    concentration = validate_concentration(concentration)

    # Warn about unusual combinations (not strictly errors)
    mass_arr = np.atleast_1d(mass)
    conc_arr = np.atleast_1d(concentration)

    # Very high concentration with very high mass
    if np.any((mass_arr > 1e15) & (conc_arr > 20)):
        import warnings

        warnings.warn(
            "Unusually high concentration (>20) for cluster mass (>1e15 Msun). "
            "Typical cluster concentrations are 2-8.",
            UserWarning,
        )

    # Very low concentration with very low mass
    if np.any((mass_arr < 1e10) & (conc_arr < 3)):
        import warnings

        warnings.warn(
            "Unusually low concentration (<3) for low mass halo (<1e10 Msun). "
            "Typical dwarf galaxy concentrations are 10-20.",
            UserWarning,
        )

    return mass, concentration


# ============================================================================
# Geometric Parameter Validation
# ============================================================================


def validate_ellipticity(
    ellipticity: Union[float, np.ndarray], param_name: str = "ellipticity"
) -> Union[float, np.ndarray]:
    """
    Validate ellipticity parameter.

    The ellipticity is defined as ε = (a-b)/(a+b) where a and b are the
    semi-major and semi-minor axes. This definition ensures 0 <= ε < 1.

    Parameters
    ----------
    ellipticity : float or np.ndarray
        Ellipticity value(s) to validate
    param_name : str, optional
        Name of the parameter for error messages (default: "ellipticity")

    Returns
    -------
    float or np.ndarray
        Validated ellipticity value(s)

    Raises
    ------
    GeometryError
        If ellipticity is outside valid range [0, 1) or non-finite

    Examples
    --------
    >>> validate_ellipticity(0.0)  # Circular
    0.0

    >>> validate_ellipticity(0.5)  # Moderate ellipticity
    0.5

    >>> validate_ellipticity(0.9)  # Highly elliptical
    0.9

    >>> eps = np.array([0.1, 0.3, 0.5])
    >>> validate_ellipticity(eps)
    array([0.1, 0.3, 0.5])

    Notes
    -----
    Physical interpretation:
    - ε = 0: Circular (axis ratio q = 1)
    - ε = 0.3: Moderate ellipticity (q ≈ 0.54)
    - ε → 1: Highly elongated (q → 0, approaches singularity)

    The axis ratio q is related to ellipticity by: q = (1-ε)/(1+ε)
    """
    e_arr = np.atleast_1d(ellipticity)

    # Check for non-finite values
    if not np.all(np.isfinite(e_arr)):
        invalid_indices = np.where(~np.isfinite(e_arr))[0]
        raise GeometryError(
            f"{param_name} contains non-finite values at indices: {invalid_indices}",
            param_name=param_name,
            param_value=ellipticity,
        )

    # Check range: [0, 1)
    if np.any(e_arr < 0):
        invalid_values = e_arr[e_arr < 0]
        raise GeometryError(
            f"{param_name} must be >= 0, got: {invalid_values}",
            param_name=param_name,
            param_value=ellipticity,
        )

    if np.any(e_arr >= 1):
        invalid_values = e_arr[e_arr >= 1]
        raise GeometryError(
            f"{param_name} must be < 1 (would require infinite axis ratio), "
            f"got: {invalid_values}",
            param_name=param_name,
            param_value=ellipticity,
        )

    return ellipticity if np.isscalar(ellipticity) else e_arr


def validate_concentration(
    concentration: Union[float, np.ndarray],
    param_name: str = "concentration",
    min_c: float = 1.0,
    max_c: float = 50.0,
) -> Union[float, np.ndarray]:
    """
    Validate NFW concentration parameter.

    Parameters
    ----------
    concentration : float or np.ndarray
        Concentration parameter c = r_vir / r_s
    param_name : str, optional
        Name of the parameter for error messages (default: "concentration")
    min_c : float, optional
        Minimum allowed concentration (default: 1.0)
    max_c : float, optional
        Maximum allowed concentration (default: 50.0)

    Returns
    -------
    float or np.ndarray
        Validated concentration value(s)

    Raises
    ------
    GeometryError
        If concentration is outside valid range or non-finite

    Examples
    --------
    >>> validate_concentration(5.0)
    5.0

    >>> validate_concentration(15.0)
    15.0

    >>> c = np.array([3.0, 5.0, 10.0])
    >>> validate_concentration(c)
    array([ 3.,  5., 10.])

    Notes
    -----
    Physical interpretation of concentration:
    - c = r_vir / r_s (virial radius / scale radius)
    - Low c (~2-5): Extended halos (clusters)
    - Moderate c (~5-15): Galaxy halos
    - High c (~15-30): Dwarf galaxies
    - Very high c (>50): Suspicious/unphysical

    Typical ranges from simulations:
    - CDM simulations: c ~ 3-25 depending on mass and redshift
    - Lower mass halos are more concentrated at fixed redshift
    """
    c_arr = np.atleast_1d(concentration)

    # Check for non-finite values
    if not np.all(np.isfinite(c_arr)):
        invalid_indices = np.where(~np.isfinite(c_arr))[0]
        raise GeometryError(
            f"{param_name} contains non-finite values at indices: {invalid_indices}",
            param_name=param_name,
            param_value=concentration,
        )

    # Check minimum
    if np.any(c_arr <= min_c):
        invalid_values = c_arr[c_arr <= min_c]
        raise GeometryError(
            f"{param_name} must be > {min_c} (r_vir must be greater than r_s), "
            f"got: {invalid_values}",
            param_name=param_name,
            param_value=concentration,
        )

    # Check maximum
    if np.any(c_arr >= max_c):
        invalid_values = c_arr[c_arr >= max_c]
        raise GeometryError(
            f"{param_name} must be < {max_c} (suspiciously high concentration), "
            f"got: {invalid_values}. Typical values are 2-20.",
            param_name=param_name,
            param_value=concentration,
        )

    return concentration if np.isscalar(concentration) else c_arr


def validate_position_angle(
    angle: Union[float, np.ndarray],
    param_name: str = "position_angle",
    degrees: bool = True,
    wrap: bool = True,
) -> Union[float, np.ndarray]:
    """
    Validate position angle.

    Position angle is measured from North (or +x) towards East (or +y),
    counter-clockwise.

    Parameters
    ----------
    angle : float or np.ndarray
        Position angle value(s)
    param_name : str, optional
        Name of the parameter for error messages (default: "position_angle")
    degrees : bool, optional
        Whether angle is in degrees (True) or radians (False). Default: True
    wrap : bool, optional
        Whether to wrap angle to standard range. Default: True

    Returns
    -------
    float or np.ndarray
        Validated (and optionally wrapped) position angle value(s)

    Raises
    ------
    GeometryError
        If angle is non-finite

    Examples
    --------
    >>> validate_position_angle(45.0)
    45.0

    >>> validate_position_angle(400.0)  # Wrapped to 40.0
    40.0

    >>> validate_position_angle(-30.0)  # Wrapped to 330.0
    330.0

    >>> validate_position_angle(np.pi/4, degrees=False)
    0.7853981633974483

    Notes
    -----
    Standard convention:
    - 0°: North (or aligned with +x axis)
    - 90°: East (or aligned with +y axis)
    - 180°: South
    - 270°: West

    By default, angles are wrapped to [0, 360) degrees or [0, 2π) radians.
    """
    angle_arr = np.atleast_1d(angle)

    # Check for non-finite values
    if not np.all(np.isfinite(angle_arr)):
        invalid_indices = np.where(~np.isfinite(angle_arr))[0]
        raise GeometryError(
            f"{param_name} contains non-finite values at indices: {invalid_indices}",
            param_name=param_name,
            param_value=angle,
        )

    # Wrap to standard range if requested
    if wrap:
        if degrees:
            angle_arr = angle_arr % 360.0
        else:
            angle_arr = angle_arr % (2 * np.pi)

    # Return scalar if input was scalar, wrapped value
    if np.isscalar(angle):
        return float(angle_arr[0])
    return angle_arr


def validate_axis_ratio(
    q: Union[float, np.ndarray], param_name: str = "axis_ratio"
) -> Union[float, np.ndarray]:
    """
    Validate axis ratio (b/a) for elliptical profiles.

    Parameters
    ----------
    q : float or np.ndarray
        Axis ratio (semi-minor / semi-major axis)
    param_name : str, optional
        Name of the parameter for error messages (default: "axis_ratio")

    Returns
    -------
    float or np.ndarray
        Validated axis ratio value(s)

    Raises
    ------
    GeometryError
        If q is outside range (0, 1] or non-finite

    Examples
    --------
    >>> validate_axis_ratio(1.0)  # Circular
    1.0

    >>> validate_axis_ratio(0.5)  # Elliptical
    0.5

    >>> validate_axis_ratio(np.array([0.3, 0.7, 1.0]))
    array([0.3, 0.7, 1.0])

    Notes
    -----
    Relation to ellipticity: q = (1-ε)/(1+ε)
    - q = 1: Circular (ε = 0)
    - q = 0.5: ε = 1/3
    - q → 0: Highly elongated (ε → 1)
    """
    q_arr = np.atleast_1d(q)

    # Check for non-finite values
    if not np.all(np.isfinite(q_arr)):
        invalid_indices = np.where(~np.isfinite(q_arr))[0]
        raise GeometryError(
            f"{param_name} contains non-finite values at indices: {invalid_indices}",
            param_name=param_name,
            param_value=q,
        )

    # Check range: (0, 1]
    if np.any(q_arr <= 0):
        invalid_values = q_arr[q_arr <= 0]
        raise GeometryError(
            f"{param_name} must be > 0, got: {invalid_values}",
            param_name=param_name,
            param_value=q,
        )

    if np.any(q_arr > 1):
        invalid_values = q_arr[q_arr > 1]
        raise GeometryError(
            f"{param_name} must be <= 1, got: {invalid_values}",
            param_name=param_name,
            param_value=q,
        )

    return q if np.isscalar(q) else q_arr


# ============================================================================
# Array Validation
# ============================================================================


def validate_coordinate_arrays(
    x: np.ndarray,
    y: np.ndarray,
    require_same_shape: bool = True,
    allow_scalar: bool = True,
    units: str = "arcsec",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate coordinate arrays for lensing calculations.

    Parameters
    ----------
    x : np.ndarray
        X coordinate array
    y : np.ndarray
        Y coordinate array
    require_same_shape : bool, optional
        Whether x and y must have the same shape (default: True)
    allow_scalar : bool, optional
        Whether to allow scalar inputs (converted to 0-d arrays)
    units : str, optional
        Units of coordinates for error messages (default: "arcsec")

    Returns
    -------
    tuple
        (validated x array, validated y array)

    Raises
    ------
    ArrayError
        If arrays have incompatible shapes or contain non-finite values

    Examples
    --------
    >>> x = np.array([1.0, 2.0, 3.0])
    >>> y = np.array([0.5, 1.0, 1.5])
    >>> validate_coordinate_arrays(x, y)
    (array([1., 2., 3.]), array([0.5, 1. , 1.5]))

    >>> # 2D grid
    >>> x = np.linspace(-5, 5, 100)
    >>> y = np.linspace(-5, 5, 100)
    >>> X, Y = np.meshgrid(x, y)
    >>> validate_coordinate_arrays(X, Y)

    Notes
    -----
    Validation checks:
    - Arrays must be numpy arrays (or convertible)
    - Arrays must contain only finite values (no NaN, inf)
    - Arrays should have compatible shapes for broadcasting
    - Typical units are arcseconds from lens center
    """
    # Convert to arrays
    if allow_scalar:
        x_arr = np.atleast_1d(np.asarray(x))
        y_arr = np.atleast_1d(np.asarray(y))
    else:
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)

    # Check for non-finite values in x
    if not np.all(np.isfinite(x_arr)):
        invalid_count = np.sum(~np.isfinite(x_arr))
        raise ArrayError(
            f"x coordinates contain {invalid_count} non-finite values (NaN or inf)",
            param_name="x",
            param_value=x,
        )

    # Check for non-finite values in y
    if not np.all(np.isfinite(y_arr)):
        invalid_count = np.sum(~np.isfinite(y_arr))
        raise ArrayError(
            f"y coordinates contain {invalid_count} non-finite values (NaN or inf)",
            param_name="y",
            param_value=y,
        )

    # Check shape compatibility
    if require_same_shape and x_arr.shape != y_arr.shape:
        raise ArrayError(
            f"Coordinate arrays must have same shape: x {x_arr.shape} vs y {y_arr.shape}",
            param_name="coordinates",
            param_value=(x, y),
        )

    return x_arr, y_arr


def validate_image_array(
    image: np.ndarray,
    param_name: str = "image",
    require_2d: bool = True,
    min_size: int = 2,
    allow_nan: bool = False,
    expected_shape: Optional[Tuple[int, ...]] = None,
) -> np.ndarray:
    """
    Validate image array for lensing analysis.

    Parameters
    ----------
    image : np.ndarray
        Image array to validate
    param_name : str, optional
        Name of the parameter for error messages (default: "image")
    require_2d : bool, optional
        Whether image must be 2D (default: True)
    min_size : int, optional
        Minimum size along each dimension (default: 2)
    allow_nan : bool, optional
        Whether to allow NaN values (default: False)
    expected_shape : tuple, optional
        Expected shape if known (default: None)

    Returns
    -------
    np.ndarray
        Validated image array

    Raises
    ------
    ArrayError
        If image has invalid shape, size, or values

    Examples
    --------
    >>> img = np.random.randn(100, 100)
    >>> validate_image_array(img)
    array([...])

    >>> # Batch of images
    >>> imgs = np.random.randn(10, 64, 64)
    >>> validate_image_array(imgs, require_2d=False)
    array([...])

    Notes
    -----
    Typical image properties:
    - 2D arrays for single convergence/kappa maps
    - 3D arrays for batches (batch_size, height, width)
    - Pixel values typically represent surface mass density
    """
    img_arr = np.asarray(image)

    # Check dimensionality
    if require_2d and img_arr.ndim != 2:
        raise ArrayError(
            f"{param_name} must be 2D, got {img_arr.ndim}D array with shape {img_arr.shape}",
            param_name=param_name,
            param_value=image,
        )

    # Check minimum size
    if np.any(np.array(img_arr.shape) < min_size):
        raise ArrayError(
            f"{param_name} dimensions must be >= {min_size}, got shape {img_arr.shape}",
            param_name=param_name,
            param_value=image,
        )

    # Check expected shape if provided
    if expected_shape is not None and img_arr.shape != expected_shape:
        raise ArrayError(
            f"{param_name} has wrong shape: expected {expected_shape}, got {img_arr.shape}",
            param_name=param_name,
            param_value=image,
        )

    # Check for NaN values
    if not allow_nan and np.any(np.isnan(img_arr)):
        nan_count = np.sum(np.isnan(img_arr))
        raise ArrayError(
            f"{param_name} contains {nan_count} NaN values",
            param_name=param_name,
            param_value=image,
        )

    # Check for infinite values
    if np.any(np.isinf(img_arr)):
        inf_count = np.sum(np.isinf(img_arr))
        raise ArrayError(
            f"{param_name} contains {inf_count} infinite values",
            param_name=param_name,
            param_value=image,
        )

    return img_arr


def validate_finite_array(arr: np.ndarray, param_name: str = "array") -> np.ndarray:
    """
    Validate that array contains only finite values.

    Parameters
    ----------
    arr : np.ndarray
        Array to validate
    param_name : str, optional
        Name of the parameter for error messages

    Returns
    -------
    np.ndarray
        Validated array

    Raises
    ------
    ArrayError
        If array contains non-finite values

    Examples
    --------
    >>> arr = np.array([1.0, 2.0, 3.0])
    >>> validate_finite_array(arr)
    array([1., 2., 3.])
    """
    arr = np.asarray(arr)

    if not np.all(np.isfinite(arr)):
        nan_count = np.sum(np.isnan(arr))
        inf_count = np.sum(np.isinf(arr))
        raise ArrayError(
            f"{param_name} contains {nan_count} NaN and {inf_count} infinite values",
            param_name=param_name,
            param_value=arr,
        )

    return arr


# ============================================================================
# Cosmological Parameter Validation
# ============================================================================


def validate_hubble_constant(
    H0: Union[float, np.ndarray], param_name: str = "H0"
) -> Union[float, np.ndarray]:
    """
    Validate Hubble constant value.

    Parameters
    ----------
    H0 : float or np.ndarray
        Hubble constant in km/s/Mpc
    param_name : str, optional
        Name of the parameter for error messages

    Returns
    -------
    float or np.ndarray
        Validated H0 value(s)

    Raises
    ------
    ValidationError
        If H0 is outside reasonable range

    Examples
    --------
    >>> validate_hubble_constant(70.0)
    70.0

    >>> validate_hubble_constant(67.4)  # Planck 2018
    67.4

    Notes
    -----
    Reasonable range for H0:
    - Planck CMB: 67.4 ± 0.5 km/s/Mpc
    - SH0ES (Cepheids): 73.04 ± 1.04 km/s/Mpc
    - Typical range: 50-100 km/s/Mpc
    """
    H0_arr = np.atleast_1d(H0)

    if not np.all(np.isfinite(H0_arr)):
        raise ValidationError(
            f"{param_name} contains non-finite values",
            param_name=param_name,
            param_value=H0,
        )

    if np.any(H0_arr <= 0):
        raise ValidationError(
            f"{param_name} must be positive, got: {H0_arr[H0_arr <= 0]}",
            param_name=param_name,
            param_value=H0,
        )

    if np.any(H0_arr < 50) or np.any(H0_arr > 100):
        import warnings

        warnings.warn(
            f"{param_name} = {H0} km/s/Mpc is outside typical range [50, 100]. "
            f"Planck: 67.4, SH0ES: 73.0",
            UserWarning,
        )

    return H0 if np.isscalar(H0) else H0_arr


def validate_matter_density(
    Omega_m: Union[float, np.ndarray], param_name: str = "Omega_m"
) -> Union[float, np.ndarray]:
    """
    Validate matter density parameter.

    Parameters
    ----------
    Omega_m : float or np.ndarray
        Matter density parameter Ω_m
    param_name : str, optional
        Name of the parameter for error messages

    Returns
    -------
    float or np.ndarray
        Validated Omega_m value(s)

    Raises
    ------
    ValidationError
        If Omega_m is outside valid range [0, 1]

    Examples
    --------
    >>> validate_matter_density(0.3)
    0.3

    >>> validate_matter_density(0.315)  # Planck 2018
    0.315

    Notes
    -----
    Standard values:
    - Planck 2018: Ω_m = 0.315 ± 0.007
    - WMAP9: Ω_m = 0.279 ± 0.025
    - Typical range: 0.2 - 0.4
    """
    om_arr = np.atleast_1d(Omega_m)

    if not np.all(np.isfinite(om_arr)):
        raise ValidationError(
            f"{param_name} contains non-finite values",
            param_name=param_name,
            param_value=Omega_m,
        )

    if np.any(om_arr < 0) or np.any(om_arr > 1):
        invalid_values = om_arr[(om_arr < 0) | (om_arr > 1)]
        raise ValidationError(
            f"{param_name} must be in range [0, 1], got: {invalid_values}",
            param_name=param_name,
            param_value=Omega_m,
        )

    return Omega_m if np.isscalar(Omega_m) else om_arr


def validate_cosmology(
    H0: float = 70.0,
    Omega_m: float = 0.3,
    Omega_lambda: Optional[float] = None,
    flat: bool = True,
) -> Dict[str, float]:
    """
    Validate complete cosmological parameter set.

    Parameters
    ----------
    H0 : float, optional
        Hubble constant in km/s/Mpc (default: 70.0)
    Omega_m : float, optional
        Matter density parameter (default: 0.3)
    Omega_lambda : float, optional
        Dark energy density parameter. If None and flat=True, computed as 1-Omega_m
    flat : bool, optional
        Whether to enforce flat universe (Omega_k = 0)

    Returns
    -------
    dict
        Dictionary with validated cosmological parameters

    Raises
    ------
    ValidationError
        If cosmological parameters are inconsistent

    Examples
    --------
    >>> validate_cosmology(H0=70.0, Omega_m=0.3)
    {'H0': 70.0, 'Omega_m': 0.3, 'Omega_lambda': 0.7, 'Omega_k': 0.0}

    >>> validate_cosmology(H0=67.4, Omega_m=0.315)  # Planck 2018
    {'H0': 67.4, 'Omega_m': 0.315, 'Omega_lambda': 0.685, 'Omega_k': 0.0}

    Notes
    -----
    For flat ΛCDM: Ω_m + Ω_Λ = 1
    """
    # Validate individual parameters
    H0 = validate_hubble_constant(H0)
    Omega_m = validate_matter_density(Omega_m)

    # Determine Omega_lambda
    if Omega_lambda is None:
        if flat:
            Omega_lambda = 1.0 - Omega_m
        else:
            raise ValidationError(
                "Omega_lambda must be specified if flat=False",
                param_name="Omega_lambda",
            )
    else:
        Omega_lambda = float(Omega_lambda)
        if Omega_lambda < 0 or Omega_lambda > 1:
            raise ValidationError(
                f"Omega_lambda must be in [0, 1], got {Omega_lambda}",
                param_name="Omega_lambda",
                param_value=Omega_lambda,
            )

    # Check flatness if required
    if flat:
        Omega_k = 1.0 - Omega_m - Omega_lambda
        if abs(Omega_k) > 1e-6:
            raise ValidationError(
                f"Flat universe requires Ω_m + Ω_Λ = 1, got "
                f"{Omega_m} + {Omega_lambda} = {Omega_m + Omega_lambda}",
                param_name="cosmology",
            )
        Omega_k = 0.0
    else:
        Omega_k = 1.0 - Omega_m - Omega_lambda

    return {
        "H0": float(H0),
        "Omega_m": float(Omega_m),
        "Omega_lambda": float(Omega_lambda),
        "Omega_k": float(Omega_k),
    }


# ============================================================================
# Composite Validation Functions
# ============================================================================


def validate_lens_geometry(
    z_lens: float,
    z_source: float,
    M_vir: Optional[float] = None,
    concentration: Optional[float] = None,
    ellipticity: Optional[float] = None,
    position_angle: Optional[float] = None,
    center_x: Optional[float] = None,
    center_y: Optional[float] = None,
    H0: float = 70.0,
    Omega_m: float = 0.3,
) -> Dict[str, Any]:
    """
    Validate complete lens system geometry.

    This function performs comprehensive validation of all lens parameters
    and checks physical consistency between them.

    Parameters
    ----------
    z_lens : float
        Lens redshift
    z_source : float
        Source redshift
    M_vir : float, optional
        Virial mass in solar masses
    concentration : float, optional
        NFW concentration parameter
    ellipticity : float, optional
        Ellipticity parameter
    position_angle : float, optional
        Position angle in degrees
    center_x : float, optional
        Center x-coordinate in arcseconds
    center_y : float, optional
        Center y-coordinate in arcseconds
    H0 : float, optional
        Hubble constant (default: 70.0)
    Omega_m : float, optional
        Matter density (default: 0.3)

    Returns
    -------
    dict
        Dictionary with all validated parameters

    Raises
    ------
    ValidationError
        If any parameter is invalid or physically inconsistent

    Examples
    --------
    >>> params = validate_lens_geometry(
    ...     z_lens=0.5,
    ...     z_source=1.5,
    ...     M_vir=1e12,
    ...     concentration=5.0,
    ...     ellipticity=0.3,
    ...     position_angle=45.0
    ... )

    Notes
    -----
    Validation order:
    1. Cosmological parameters
    2. Redshift geometry (source > lens)
    3. Mass parameters
    4. Geometric parameters (ellipticity, PA)
    5. Consistency checks
    """
    results = {}

    # Validate cosmology
    results["cosmology"] = validate_cosmology(H0, Omega_m)

    # Validate redshift geometry
    z_source, z_lens = validate_source_lens_redshift(z_source, z_lens)
    results["z_source"] = z_source
    results["z_lens"] = z_lens

    # Validate mass parameters
    if M_vir is not None:
        results["M_vir"] = validate_mass(M_vir, units="Msun", param_name="virial mass")

    if concentration is not None:
        results["concentration"] = validate_concentration(concentration)

    # Validate mass-concentration consistency
    if M_vir is not None and concentration is not None:
        validate_mass_concentration_ratio(M_vir, concentration)

    # Validate geometric parameters
    if ellipticity is not None:
        results["ellipticity"] = validate_ellipticity(ellipticity)

    if position_angle is not None:
        results["position_angle"] = validate_position_angle(position_angle)

    # Validate center coordinates
    if center_x is not None:
        if not np.isfinite(center_x):
            raise GeometryError(
                f"center_x must be finite, got {center_x}",
                param_name="center_x",
                param_value=center_x,
            )
        results["center_x"] = float(center_x)

    if center_y is not None:
        if not np.isfinite(center_y):
            raise GeometryError(
                f"center_y must be finite, got {center_y}",
                param_name="center_y",
                param_value=center_y,
            )
        results["center_y"] = float(center_y)

    return results


def validate_lens_configuration(
    config: Dict[str, Any], strict: bool = True
) -> Dict[str, Any]:
    """
    Validate lens configuration from dictionary.

    Parameters
    ----------
    config : dict
        Dictionary containing lens parameters
    strict : bool, optional
        If True, raise error for unexpected keys (default: True)

    Returns
    -------
    dict
        Dictionary with validated parameters

    Raises
    ------
    ValidationError
        If configuration is invalid

    Examples
    --------
    >>> config = {
    ...     'z_lens': 0.5,
    ...     'z_source': 1.5,
    ...     'M_vir': 1e12,
    ...     'concentration': 5.0,
    ...     'ellipticity': 0.3,
    ...     'position_angle': 45.0,
    ...     'center_x': 0.0,
    ...     'center_y': 0.0,
    ...     'H0': 70.0,
    ...     'Omega_m': 0.3
    ... }
    >>> validated = validate_lens_configuration(config)

    Notes
    -----
    Expected keys in config:
    - Required: z_lens, z_source
    - Optional: M_vir, concentration, ellipticity, position_angle
    - Optional: center_x, center_y, H0, Omega_m
    """
    # Check required keys
    required_keys = ["z_lens", "z_source"]
    missing_keys = [k for k in required_keys if k not in config]
    if missing_keys:
        raise ValidationError(
            f"Missing required keys: {missing_keys}", param_name="configuration"
        )

    # Extract parameters
    z_lens = config.get("z_lens")
    z_source = config.get("z_source")
    M_vir = config.get("M_vir")
    concentration = config.get("concentration")
    ellipticity = config.get("ellipticity")
    position_angle = config.get("position_angle")
    center_x = config.get("center_x")
    center_y = config.get("center_y")
    H0 = config.get("H0", 70.0)
    Omega_m = config.get("Omega_m", 0.3)

    # Validate geometry
    results = validate_lens_geometry(
        z_lens=z_lens,
        z_source=z_source,
        M_vir=M_vir,
        concentration=concentration,
        ellipticity=ellipticity,
        position_angle=position_angle,
        center_x=center_x,
        center_y=center_y,
        H0=H0,
        Omega_m=Omega_m,
    )

    # Check for unexpected keys in strict mode
    if strict:
        expected_keys = {
            "z_lens",
            "z_source",
            "M_vir",
            "concentration",
            "ellipticity",
            "position_angle",
            "center_x",
            "center_y",
            "H0",
            "Omega_m",
            "Omega_lambda",
            "flat",
        }
        unexpected = set(config.keys()) - expected_keys
        if unexpected:
            raise ValidationError(
                f"Unexpected keys in configuration: {unexpected}",
                param_name="configuration",
            )

    return results


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Exceptions
    "ValidationError",
    "RedshiftError",
    "MassError",
    "GeometryError",
    "ArrayError",
    # Redshift validation
    "validate_redshift",
    "validate_source_lens_redshift",
    # Mass validation
    "validate_mass",
    "validate_mass_concentration_ratio",
    # Geometric validation
    "validate_ellipticity",
    "validate_concentration",
    "validate_position_angle",
    "validate_axis_ratio",
    # Array validation
    "validate_coordinate_arrays",
    "validate_image_array",
    "validate_finite_array",
    # Cosmological validation
    "validate_hubble_constant",
    "validate_matter_density",
    "validate_cosmology",
    # Composite validation
    "validate_lens_geometry",
    "validate_lens_configuration",
]


if __name__ == "__main__":
    # Run validation examples
    print("=" * 70)
    print("Input Validation Module - Test Examples")
    print("=" * 70)

    print("\n1. Redshift Validation:")
    print(f"   validate_redshift(0.5) = {validate_redshift(0.5)}")
    print(f"   validate_redshift(2.0) = {validate_redshift(2.0)}")

    print("\n2. Source-Lens Geometry:")
    zs, zl = validate_source_lens_redshift(1.5, 0.5)
    print(f"   z_source={zs}, z_lens={zl}")

    print("\n3. Mass Validation:")
    print(f"   validate_mass(1e12) = {validate_mass(1e12):.2e} Msun")

    print("\n4. Ellipticity Validation:")
    print(f"   validate_ellipticity(0.3) = {validate_ellipticity(0.3)}")

    print("\n5. Concentration Validation:")
    print(f"   validate_concentration(5.0) = {validate_concentration(5.0)}")

    print("\n6. Position Angle Validation:")
    print(f"   validate_position_angle(45.0) = {validate_position_angle(45.0)}")
    print(
        f"   validate_position_angle(400.0) = {validate_position_angle(400.0)} (wrapped)"
    )

    print("\n7. Coordinate Arrays:")
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([0.5, 1.0, 1.5])
    xv, yv = validate_coordinate_arrays(x, y)
    print(f"   Input: x={x}, y={y}")
    print(f"   Validated: shape={xv.shape}, all finite={np.all(np.isfinite(xv))}")

    print("\n8. Cosmology Validation:")
    cosmo = validate_cosmology(H0=67.4, Omega_m=0.315)
    print(
        f"   Planck 2018: H0={cosmo['H0']}, Ω_m={cosmo['Omega_m']}, Ω_Λ={cosmo['Omega_lambda']}"
    )

    print("\n9. Complete Lens Geometry:")
    params = validate_lens_geometry(
        z_lens=0.5,
        z_source=1.5,
        M_vir=1e12,
        concentration=5.0,
        ellipticity=0.3,
        position_angle=45.0,
    )
    print(f"   Validated {len(params)} parameters")
    print(f"   Keys: {list(params.keys())}")

    print("\n" + "=" * 70)
    print("All validations passed!")
    print("=" * 70)
