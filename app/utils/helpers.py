"""
Helper Utilities for Streamlit Application

Provides validation, dependency checking, and common utility functions:
- Parameter validation (positive numbers, ranges, grid sizes)
- Dependency checking (torch, numpy, scipy, matplotlib)
- User action logging
- Computation time estimation
- Safe operations (division, imports)

Author: Refactored from app/error_handler.py
"""

import logging
import sys
import os
from typing import Any, Optional, Dict
from datetime import datetime

# Ensure logs directory exists
def ensure_logs_directory() -> None:
    """Ensure logs directory exists."""
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

# Configure logging
def setup_logging():
    """Setup logging configuration."""
    ensure_logs_directory()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/app_{datetime.now().strftime("%Y%m%d")}.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

# Initialize logger (configuration happens when setup_logging is called)
logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class ComputationError(Exception):
    """Custom exception for computation errors."""
    pass


def validate_positive_number(value: float, name: str = "Value") -> None:
    """
    Validate that a number is positive.
    
    Args:
        value: Number to validate
        name: Parameter name for error message
    
    Raises:
        ValidationError: If value is not positive
    
    Example:
        >>> validate_positive_number(1.5, "Mass")  # OK
        >>> validate_positive_number(-1.0, "Mass")  # Raises ValidationError
    """
    if value <= 0:
        raise ValidationError(f"{name} must be positive. Got: {value}")


def validate_range(value: float, min_val: float, max_val: float, name: str = "Value") -> None:
    """
    Validate that a number is within a specific range.
    
    Args:
        value: Number to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        name: Parameter name for error message
    
    Raises:
        ValidationError: If value is outside the range
    
    Example:
        >>> validate_range(0.5, 0.0, 1.0, "Redshift")  # OK
        >>> validate_range(2.0, 0.0, 1.0, "Redshift")  # Raises ValidationError
    """
    if not (min_val <= value <= max_val):
        raise ValidationError(
            f"{name} must be between {min_val} and {max_val}. Got: {value}"
        )


def validate_grid_size(size: int, max_size: int = 512) -> None:
    """
    Validate grid size for computational efficiency.
    
    Args:
        size: Grid size to validate
        max_size: Maximum allowed grid size (default: 512)
    
    Raises:
        ValidationError: If grid size is invalid
    
    Example:
        >>> validate_grid_size(128)  # OK
        >>> validate_grid_size(15)  # Raises ValidationError (too small)
        >>> validate_grid_size(1024)  # Raises ValidationError (too large)
    """
    if size < 16:
        raise ValidationError(f"Grid size too small (minimum: 16). Got: {size}")
    if size > max_size:
        raise ValidationError(
            f"Grid size too large (maximum: {max_size}). Got: {size}"
        )
    if size % 2 != 0:
        raise ValidationError(f"Grid size must be even. Got: {size}")


def validate_file_path(path: str, allowed_extensions: Optional[list] = None) -> None:
    """
    Validate file path and extension.
    
    Args:
        path: File path to validate
        allowed_extensions: List of allowed file extensions (e.g., ['.fits', '.csv'])
    
    Raises:
        ValidationError: If path is empty or extension is not allowed
    
    Example:
        >>> validate_file_path("data.fits", [".fits", ".csv"])  # OK
        >>> validate_file_path("data.txt", [".fits", ".csv"])  # Raises ValidationError
    """
    if not path:
        raise ValidationError("File path cannot be empty")
    
    if allowed_extensions:
        if not any(path.endswith(ext) for ext in allowed_extensions):
            raise ValidationError(
                f"Invalid file extension. Allowed: {', '.join(allowed_extensions)}"
            )


def validate_array_shape(array, expected_shape: tuple, name: str = "Array") -> None:
    """
    Validate array shape matches expected dimensions.
    
    Args:
        array: NumPy or PyTorch array to validate
        expected_shape: Expected shape tuple
        name: Array name for error message
    
    Raises:
        ValidationError: If shape doesn't match
    
    Example:
        >>> import numpy as np
        >>> arr = np.zeros((64, 64))
        >>> validate_array_shape(arr, (64, 64), "Convergence map")  # OK
    """
    if array.shape != expected_shape:
        raise ValidationError(
            f"{name} shape mismatch. Expected: {expected_shape}, Got: {array.shape}"
        )


def validate_computation_parameters(params: Dict) -> None:
    """
    Validate common computation parameters for lens simulations.
    
    Args:
        params: Dictionary containing computation parameters
            Required keys: grid_size, fov, redshift_lens, redshift_source
    
    Raises:
        ValidationError: If any parameter is invalid
    
    Example:
        >>> params = {
        ...     "grid_size": 128,
        ...     "fov": 10.0,
        ...     "redshift_lens": 0.5,
        ...     "redshift_source": 2.0
        ... }
        >>> validate_computation_parameters(params)  # OK
    """
    required_keys = ['grid_size', 'fov', 'redshift_lens', 'redshift_source']
    
    # Check required keys
    for key in required_keys:
        if key not in params:
            raise ValidationError(f"Missing required parameter: {key}")
    
    # Validate grid size
    validate_grid_size(params['grid_size'])
    
    # Validate FOV
    validate_positive_number(params['fov'], "Field of View")
    validate_range(params['fov'], 0.1, 100.0, "Field of View")
    
    # Validate redshifts
    validate_positive_number(params['redshift_lens'], "Lens Redshift")
    validate_positive_number(params['redshift_source'], "Source Redshift")
    
    if params['redshift_source'] <= params['redshift_lens']:
        raise ValidationError(
            "Source redshift must be greater than lens redshift. "
            f"Got: z_source={params['redshift_source']}, z_lens={params['redshift_lens']}"
        )


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers with default fallback.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value if division fails (default: 0.0)
    
    Returns:
        Result of division or default value
    
    Example:
        >>> safe_divide(10, 2)  # Returns 5.0
        >>> safe_divide(10, 0)  # Returns 0.0 (default)
    """
    try:
        if denominator == 0:
            logger.warning("Division by zero attempted")
            return default
        return numerator / denominator
    except Exception as e:
        logger.error(f"Error in division: {e}")
        return default


def check_dependencies() -> Dict:
    """
    Check if all required dependencies are available.
    
    Returns:
        Dictionary with availability and version info for each dependency
    
    Example:
        >>> deps = check_dependencies()
        >>> if deps['torch']['available']:
        ...     print(f"PyTorch version: {deps['torch']['version']}")
    """
    dependencies = {}
    
    # Check PyTorch
    try:
        import torch
        dependencies['torch'] = {
            'available': True,
            'version': torch.__version__,
            'cuda_available': torch.cuda.is_available()
        }
    except ImportError:
        dependencies['torch'] = {'available': False}
    
    # Check NumPy
    try:
        import numpy
        dependencies['numpy'] = {
            'available': True,
            'version': numpy.__version__
        }
    except ImportError:
        dependencies['numpy'] = {'available': False}
    
    # Check Matplotlib
    try:
        import matplotlib
        dependencies['matplotlib'] = {
            'available': True,
            'version': matplotlib.__version__
        }
    except ImportError:
        dependencies['matplotlib'] = {'available': False}
    
    # Check SciPy
    try:
        import scipy
        dependencies['scipy'] = {
            'available': True,
            'version': scipy.__version__
        }
    except ImportError:
        dependencies['scipy'] = {'available': False}
    
    # Check Astropy
    try:
        import astropy
        dependencies['astropy'] = {
            'available': True,
            'version': astropy.__version__
        }
    except ImportError:
        dependencies['astropy'] = {'available': False}
    
    return dependencies


def safe_import(module_name: str, package_name: Optional[str] = None) -> Optional[Any]:
    """
    Safely import a module with error handling.
    
    Args:
        module_name: Name of the module to import
        package_name: Optional package name for display
    
    Returns:
        Imported module or None if import fails
    
    Example:
        >>> numpy = safe_import('numpy')
        >>> if numpy:
        ...     arr = numpy.zeros((10, 10))
    """
    try:
        module = __import__(module_name)
        return module
    except ImportError:
        display_name = package_name or module_name
        logger.error(
            f"Module '{display_name}' not found. "
            f"Install it with: pip install {display_name}"
        )
        return None


def estimate_computation_time(grid_size: int, num_iterations: int = 1000) -> str:
    """
    Estimate computation time based on parameters.
    
    Args:
        grid_size: Size of computational grid
        num_iterations: Number of iterations (for iterative solvers)
    
    Returns:
        Human-readable time estimate
    
    Example:
        >>> estimate = estimate_computation_time(grid_size=128, num_iterations=500)
        >>> print(f"Estimated time: {estimate}")
    """
    # Rough estimates based on typical performance
    base_time = (grid_size / 100) ** 2 * num_iterations * 0.001  # seconds
    
    if base_time < 1:
        return "< 1 second"
    elif base_time < 60:
        return f"~{int(base_time)} seconds"
    elif base_time < 3600:
        return f"~{int(base_time / 60)} minutes"
    else:
        return f"~{base_time / 3600:.1f} hours"


def log_user_action(action: str, details: Optional[Dict] = None) -> None:
    """
    Log user actions for analytics and debugging.
    
    Args:
        action: Description of the action
        details: Optional dictionary with additional details
    
    Example:
        >>> log_user_action("generate_convergence_map", {"M_vir": 1e14, "grid_size": 128})
    """
    log_message = f"User action: {action}"
    if details:
        log_message += f" | Details: {details}"
    logger.info(log_message)
