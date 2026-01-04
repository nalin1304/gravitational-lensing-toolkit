"""
Optics Module for Gravitational Lensing

This module provides ray tracing and wave optics calculations.
"""

from .ray_tracing import ray_trace, compute_magnification, compute_time_delay
from .wave_optics import WaveOpticsEngine, plot_wave_vs_geometric

__all__ = [
    'ray_trace', 
    'compute_magnification', 
    'compute_time_delay',
    'WaveOpticsEngine',
    'plot_wave_vs_geometric'
]
