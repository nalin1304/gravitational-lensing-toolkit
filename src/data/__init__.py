"""
Data Module for Real Observatory Data

This module handles loading and processing of real observational data
from space telescopes (HST, JWST) and ground-based observatories.
"""

from .real_data_loader import (
    FITSDataLoader,
    PSFModel,
    ObservationMetadata,
    preprocess_real_data,
    load_real_data,
    ASTROPY_AVAILABLE,
    SCIPY_AVAILABLE
)

__all__ = [
    'FITSDataLoader',
    'PSFModel',
    'ObservationMetadata',
    'preprocess_real_data',
    'load_real_data',
    'ASTROPY_AVAILABLE',
    'SCIPY_AVAILABLE',
]
