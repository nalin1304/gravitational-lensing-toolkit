"""
Session State Management

Centralized management of Streamlit session state across multi-page app.
"""

import streamlit as st
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def init_session_state():
    """Initialize all session state variables with defaults."""
    
    defaults = {
        # Model selection
        'model_type': 'PINN',
        'mass_profile': 'NFW',
        
        # Lens parameters
        'M_vir': 5e12,
        'concentration': 5.0,
        'ellipticity': 0.2,
        'position_angle': 45.0,
        'theta_E': 1.0,
        
        # Source parameters
        'z_lens': 0.5,
        'z_source': 2.0,
        'beta_x': 0.1,
        'beta_y': 0.1,
        'source_amplitude': 1.0,
        'source_radius': 0.5,
        
        # Grid parameters
        'grid_size': 128,
        'fov': 10.0,
        
        # Computation results
        'convergence_map': None,
        'magnification_map': None,
        'predicted_params': None,
        'inference_time': None,
        
        # Validation metrics
        'validation_report': None,
        'bayesian_uncertainties': None,
        
        # FITS data
        'fits_loaded': False,
        'fits_data': None,
        'fits_metadata': None,
        
        # Multi-plane
        'multiplane_enabled': False,
        'num_planes': 2,
        'plane_redshifts': [0.3, 0.5],
        
        # Training
        'training_epochs': 100,
        'learning_rate': 0.001,
        'batch_size': 32,
        'training_history': None,
        
        # Comparison
        'compare_models': False,
        'comparison_results': None,
        
        # Advanced features
        'psf_model': 'gaussian',
        'psf_fwhm': 0.1,
        'include_noise': False,
        'noise_level': 0.01,
        'substructure_detection': False,
        
        # Export
        'export_format': 'PNG',
        'include_metadata': True,
    }
    
    # Initialize only if not already set
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_state(key: str, default: Any = None) -> Any:
    """Get a value from session state with optional default."""
    return st.session_state.get(key, default)


def set_state(key: str, value: Any) -> None:
    """Set a value in session state."""
    st.session_state[key] = value


def clear_state(*keys: str) -> None:
    """Clear specific keys from session state."""
    for key in keys:
        if key in st.session_state:
            del st.session_state[key]


def reset_computation_results():
    """Reset all computation results."""
    clear_state(
        'convergence_map',
        'magnification_map',
        'predicted_params',
        'inference_time',
        'validation_report',
        'bayesian_uncertainties',
        'comparison_results'
    )


def get_lens_parameters() -> Dict[str, Any]:
    """Get all current lens parameters as a dictionary."""
    return {
        'M_vir': get_state('M_vir'),
        'concentration': get_state('concentration'),
        'ellipticity': get_state('ellipticity'),
        'position_angle': get_state('position_angle'),
        'theta_E': get_state('theta_E'),
        'z_lens': get_state('z_lens'),
        'z_source': get_state('z_source'),
        'beta_x': get_state('beta_x'),
        'beta_y': get_state('beta_y'),
        'source_amplitude': get_state('source_amplitude'),
        'source_radius': get_state('source_radius'),
    }


def get_grid_parameters() -> Dict[str, Any]:
    """Get current grid configuration."""
    return {
        'grid_size': get_state('grid_size'),
        'fov': get_state('fov'),
    }


def parameter_changed() -> None:
    """Callback when parameters change - reset results."""
    reset_computation_results()
    logger.info("Parameters changed - computation results cleared")


def update_from_dict(params: Dict[str, Any]) -> None:
    """Update multiple session state values from dictionary."""
    for key, value in params.items():
        set_state(key, value)
