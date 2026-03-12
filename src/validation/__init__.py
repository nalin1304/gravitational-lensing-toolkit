"""
Scientific Validation Framework

Comprehensive validation tools for gravitational lensing predictions.
"""

from .scientific_validator import (
    ValidationLevel,
    ValidationResult,
    ScientificValidator,
    quick_validate,
    rigorous_validate,
)

from .uncertainty_quantification import (
    # Main functions
    monte_carlo_error_propagation,
    compute_confidence_intervals,
    bootstrap_errors,
    propagate_parameter_errors,
    compute_covariance_matrix,
    correlation_from_covariance,
    gaussian_error_propagation,
    compute_prediction_uncertainty_map,
    hierarchical_bootstrap,
    jackknife_errors,
    weighted_bootstrap,
    lens_parameter_uncertainty,
    convergence_map_uncertainty,
    # Data classes
    UncertaintyResult,
    PropagatedErrors,
)

__all__ = [
    # Scientific validation
    "ValidationLevel",
    "ValidationResult",
    "ScientificValidator",
    "quick_validate",
    "rigorous_validate",
    # Uncertainty quantification
    "monte_carlo_error_propagation",
    "compute_confidence_intervals",
    "bootstrap_errors",
    "propagate_parameter_errors",
    "compute_covariance_matrix",
    "correlation_from_covariance",
    "gaussian_error_propagation",
    "compute_prediction_uncertainty_map",
    "hierarchical_bootstrap",
    "jackknife_errors",
    "weighted_bootstrap",
    "lens_parameter_uncertainty",
    "convergence_map_uncertainty",
    "UncertaintyResult",
    "PropagatedErrors",
]
