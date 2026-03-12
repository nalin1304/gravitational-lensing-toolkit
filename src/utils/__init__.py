"""
Utilities Module for Gravitational Lensing

This module provides visualization, validation, and helper functions.
"""

from .visualization import (
    plot_lens_system,
    plot_radial_profile,
    plot_deflection_field,
    plot_magnification_map,
)

from .validation import (
    # Exceptions
    ValidationError,
    RedshiftError,
    MassError,
    GeometryError,
    ArrayError,
    # Redshift validation
    validate_redshift,
    validate_source_lens_redshift,
    # Mass validation
    validate_mass,
    validate_mass_concentration_ratio,
    # Geometric validation
    validate_ellipticity,
    validate_concentration,
    validate_position_angle,
    validate_axis_ratio,
    # Array validation
    validate_coordinate_arrays,
    validate_image_array,
    validate_finite_array,
    # Cosmological validation
    validate_hubble_constant,
    validate_matter_density,
    validate_cosmology,
    # Composite validation
    validate_lens_geometry,
    validate_lens_configuration,
)

__all__ = [
    # Visualization
    "plot_lens_system",
    "plot_radial_profile",
    "plot_deflection_field",
    "plot_magnification_map",
    # Validation exceptions
    "ValidationError",
    "RedshiftError",
    "MassError",
    "GeometryError",
    "ArrayError",
    # Redshift
    "validate_redshift",
    "validate_source_lens_redshift",
    # Mass
    "validate_mass",
    "validate_mass_concentration_ratio",
    # Geometric
    "validate_ellipticity",
    "validate_concentration",
    "validate_position_angle",
    "validate_axis_ratio",
    # Arrays
    "validate_coordinate_arrays",
    "validate_image_array",
    "validate_finite_array",
    # Cosmology
    "validate_hubble_constant",
    "validate_matter_density",
    "validate_cosmology",
    # Composite
    "validate_lens_geometry",
    "validate_lens_configuration",
]
