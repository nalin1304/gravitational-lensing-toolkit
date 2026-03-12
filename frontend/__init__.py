"""
Gravitational Lensing Frontend Package

This package provides a Streamlit-based frontend for gravitational
lensing analysis with physics-informed neural networks.

Modules
-------
app : Main Streamlit application
components : Reusable UI components
utils : Helper functions and utilities
"""

__version__ = "2.0.0"
__author__ = "Gravitational Lensing Team"

from .components import (
    render_header,
    render_card,
    render_info_box,
    render_warning_box,
    render_success_box,
    sidebar_navigation,
    apply_custom_css,
    lens_model_form,
    training_monitor,
    plot_3d_surface,
    plot_comparison,
)

from .utils import (
    initialize_session_state,
    get_lens_model,
    compute_convergence_map,
    compute_deflection_field,
    compute_lensing_potential,
    find_critical_curves,
    run_wave_optics_simulation,
    compare_wave_geometric,
    create_einstein_ring_animation,
    load_test_results,
    generate_validation_report,
    format_metric,
)

__all__ = [
    # Components
    "render_header",
    "render_card",
    "render_info_box",
    "render_warning_box",
    "render_success_box",
    "sidebar_navigation",
    "apply_custom_css",
    "lens_model_form",
    "training_monitor",
    "plot_3d_surface",
    "plot_comparison",
    # Utils
    "initialize_session_state",
    "get_lens_model",
    "compute_convergence_map",
    "compute_deflection_field",
    "compute_lensing_potential",
    "find_critical_curves",
    "run_wave_optics_simulation",
    "compare_wave_geometric",
    "create_einstein_ring_animation",
    "load_test_results",
    "generate_validation_report",
    "format_metric",
]
