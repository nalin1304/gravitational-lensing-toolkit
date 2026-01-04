"""
Production-ready error handling and validation
Comprehensive error management for Streamlit application
"""

import streamlit as st
import traceback
import logging
from typing import Optional, Callable, Any
from functools import wraps
import sys
import os
from datetime import datetime

# Ensure logs directory exists
def ensure_logs_directory():
    """Ensure logs directory exists"""
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
        
# Create logs directory first
ensure_logs_directory()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/app_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass


class ComputationError(Exception):
    """Custom exception for computation errors"""
    pass


def handle_errors(func: Callable) -> Callable:
    """
    Decorator for comprehensive error handling
    
    Usage:
        @handle_errors
        def my_function():
            # Your code here
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValidationError as e:
            st.error(f"⚠️ **Validation Error:** {str(e)}")
            logger.warning(f"Validation error in {func.__name__}: {str(e)}")
            return None
        except ComputationError as e:
            st.error(f"🔴 **Computation Error:** {str(e)}")
            logger.error(f"Computation error in {func.__name__}: {str(e)}")
            return None
        except FileNotFoundError as e:
            st.error(f"📁 **File Not Found:** {str(e)}")
            logger.error(f"File not found in {func.__name__}: {str(e)}")
            return None
        except PermissionError as e:
            st.error(f"🔒 **Permission Denied:** {str(e)}")
            logger.error(f"Permission error in {func.__name__}: {str(e)}")
            return None
        except MemoryError as e:
            st.error("💾 **Memory Error:** Insufficient memory. Try reducing parameters.")
            logger.critical(f"Memory error in {func.__name__}: {str(e)}")
            return None
        except KeyboardInterrupt:
            st.warning("⏸️ **Operation Cancelled** by user.")
            logger.info(f"Operation cancelled in {func.__name__}")
            return None
        except Exception as e:
            st.error(f"❌ **Unexpected Error:** {str(e)}")
            
            # Show detailed error in expander for debugging
            with st.expander("🔍 Error Details (for debugging)"):
                st.code(traceback.format_exc())
            
            logger.exception(f"Unexpected error in {func.__name__}: {str(e)}")
            return None
    return wrapper


def validate_positive_number(value: float, name: str = "Value") -> None:
    """Validate that a number is positive"""
    if value <= 0:
        raise ValidationError(f"{name} must be positive. Got: {value}")


def validate_range(value: float, min_val: float, max_val: float, name: str = "Value") -> None:
    """Validate that a number is within a range"""
    if not (min_val <= value <= max_val):
        raise ValidationError(
            f"{name} must be between {min_val} and {max_val}. Got: {value}"
        )


def validate_grid_size(size: int, max_size: int = 512) -> None:
    """Validate grid size for computational efficiency"""
    if size < 16:
        raise ValidationError(f"Grid size too small (minimum: 16). Got: {size}")
    if size > max_size:
        raise ValidationError(
            f"Grid size too large (maximum: {max_size}). Got: {size}"
        )
    if size % 2 != 0:
        raise ValidationError(f"Grid size must be even. Got: {size}")


def validate_file_path(path: str, allowed_extensions: Optional[list] = None) -> None:
    """Validate file path and extension"""
    if not path:
        raise ValidationError("File path cannot be empty")
    
    if allowed_extensions:
        if not any(path.endswith(ext) for ext in allowed_extensions):
            raise ValidationError(
                f"Invalid file extension. Allowed: {', '.join(allowed_extensions)}"
            )


def validate_array_shape(array, expected_shape: tuple, name: str = "Array") -> None:
    """Validate array shape"""
    if array.shape != expected_shape:
        raise ValidationError(
            f"{name} shape mismatch. Expected: {expected_shape}, Got: {array.shape}"
        )


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers"""
    try:
        if denominator == 0:
            logger.warning("Division by zero attempted")
            return default
        return numerator / denominator
    except Exception as e:
        logger.error(f"Error in division: {e}")
        return default


def show_success(message: str) -> None:
    """Show success message with consistent styling"""
    st.success(f"✅ {message}")
    logger.info(f"Success: {message}")


def show_warning(message: str) -> None:
    """Show warning message with consistent styling"""
    st.warning(f"⚠️ {message}")
    logger.warning(f"Warning: {message}")


def show_info(message: str) -> None:
    """Show info message with consistent styling"""
    st.info(f"ℹ️ {message}")
    logger.info(f"Info: {message}")


def show_error(message: str) -> None:
    """Show error message with consistent styling"""
    st.error(f"❌ {message}")
    logger.error(f"Error: {message}")


def with_spinner(message: str = "Processing..."):
    """
    Decorator to show spinner during long operations
    
    Usage:
        @with_spinner("Generating lens system...")
        def generate_lens():
            # Your code here
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            with st.spinner(message):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def check_dependencies() -> dict:
    """Check if all required dependencies are available"""
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
    
    return dependencies


def log_user_action(action: str, details: Optional[dict] = None) -> None:
    """Log user actions for analytics"""
    log_message = f"User action: {action}"
    if details:
        log_message += f" | Details: {details}"
    logger.info(log_message)


def create_download_button(
    data: Any,
    filename: str,
    button_text: str = "📥 Download",
    mime_type: str = "text/plain"
) -> None:
    """Create a styled download button"""
    try:
        st.download_button(
            label=button_text,
            data=data,
            file_name=filename,
            mime=mime_type
        )
        logger.info(f"Download button created for {filename}")
    except Exception as e:
        show_error(f"Failed to create download button: {str(e)}")


def validate_computation_parameters(params: dict) -> None:
    """
    Validate common computation parameters
    
    Args:
        params: Dictionary containing computation parameters
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


def estimate_computation_time(grid_size: int, num_iterations: int = 1000) -> str:
    """
    Estimate computation time based on parameters
    
    Returns:
        Human-readable time estimate
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


def create_parameter_summary(params: dict) -> str:
    """Create a formatted summary of parameters"""
    summary = "### 📋 Parameter Summary\n\n"
    for key, value in params.items():
        if isinstance(value, float):
            summary += f"- **{key.replace('_', ' ').title()}**: {value:.4f}\n"
        else:
            summary += f"- **{key.replace('_', ' ').title()}**: {value}\n"
    return summary


def safe_import(module_name: str, package_name: Optional[str] = None) -> Optional[Any]:
    """
    Safely import a module with error handling
    
    Args:
        module_name: Name of the module to import
        package_name: Optional package name for display
        
    Returns:
        Imported module or None if import fails
    """
    try:
        module = __import__(module_name)
        return module
    except ImportError:
        display_name = package_name or module_name
        show_error(
            f"Module '{display_name}' not found. "
            f"Install it with: pip install {display_name}"
        )
        logger.error(f"Failed to import {module_name}")
        return None



