"""
Utility Functions for Gravitational Lensing Frontend

This module provides helper functions that use the API client
to interact with the backend.
"""

import numpy as np
import streamlit as st
from typing import Tuple, Dict, Optional, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from api_client import get_api_client, format_api_error


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "current_lens" not in st.session_state:
        st.session_state["current_lens"] = None

    if "current_lens_id" not in st.session_state:
        st.session_state["current_lens_id"] = None

    if "current_lens_system" not in st.session_state:
        st.session_state["current_lens_system"] = None

    if "kappa_grid" not in st.session_state:
        st.session_state["kappa_grid"] = None

    if "x_grid" not in st.session_state:
        st.session_state["x_grid"] = None

    if "y_grid" not in st.session_state:
        st.session_state["y_grid"] = None

    if "training_history" not in st.session_state:
        st.session_state["training_history"] = {
            "epochs": [],
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
        }

    if "test_results" not in st.session_state:
        st.session_state["test_results"] = None


def get_lens_model(
    model_type: str,
    params: dict,
    z_lens: float,
    z_source: float,
    H0: float = 70.0,
    Omega_m: float = 0.3,
):
    """
    Create a lens model via API.

    Parameters
    ----------
    model_type : str
        Type of lens model
    params : dict
        Model parameters
    z_lens : float
        Lens redshift
    z_source : float
        Source redshift
    H0 : float
        Hubble constant
    Omega_m : float
        Matter density parameter

    Returns
    -------
    tuple
        (lens_model_info, lens_id) or (None, None) on error
    """
    client = get_api_client()

    try:
        if model_type == "Point Mass":
            result = client.create_point_mass_lens(
                mass=params["mass"],
                z_lens=z_lens,
                z_source=z_source,
                H0=H0,
                Omega_m=Omega_m,
            )
        elif model_type == "NFW (Navarro-Frenk-White)":
            result = client.create_nfw_lens(
                M_vir=params["M_vir"],
                concentration=params["concentration"],
                z_lens=z_lens,
                z_source=z_source,
                ellipticity=params.get("ellipticity", 0.0),
                position_angle=params.get("position_angle", 0.0),
                H0=H0,
                Omega_m=Omega_m,
            )
        else:
            st.error(f"Model type '{model_type}' not yet supported via API")
            return None, None

        lens_id = result.get("lens_id")
        return result, lens_id

    except Exception as e:
        st.error(format_api_error(e))
        return None, None


def compute_convergence_map(
    lens_id: str, grid_size: int = 256, grid_extent: float = 5.0, colormap: str = "hot"
):
    """
    Compute convergence map via API.

    Parameters
    ----------
    lens_id : str
        Lens model ID from API
    grid_size : int
        Grid resolution
    grid_extent : float
        Physical extent in arcseconds
    colormap : str
        Colormap name

    Returns
    -------
    tuple
        (fig, kappa_grid, x_grid, y_grid) or (None, None, None, None) on error
    """
    client = get_api_client()

    try:
        result = client.compute_convergence_map(
            lens_id=lens_id, grid_size=grid_size, grid_extent=grid_extent
        )

        # Extract data from API response
        kappa_grid = np.array(result["kappa_grid"])
        x_grid = np.array(result["x_grid"])
        y_grid = np.array(result["y_grid"])

        # Create Plotly figure
        fig = go.Figure(
            data=[
                go.Heatmap(
                    z=kappa_grid,
                    x=x_grid,
                    y=y_grid,
                    colorscale=colormap,
                    colorbar=dict(title="κ", titleside="right"),
                    hovertemplate="x: %{x:.2f}<br>y: %{y:.2f}<br>κ: %{z:.4f}<extra></extra>",
                )
            ]
        )

        fig.update_layout(
            title="Convergence Map (κ)",
            xaxis_title="x (arcsec)",
            yaxis_title="y (arcsec)",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
        )

        return fig, kappa_grid, x_grid, y_grid

    except Exception as e:
        st.error(format_api_error(e))
        return None, None, None, None


def compute_deflection_field(
    lens_id: str,
    grid_size: int = 20,
    grid_extent: float = 5.0,
):
    """
    Compute deflection field via API.

    Parameters
    ----------
    lens_id : str
        Lens model ID
    grid_size : int
        Grid resolution (number of arrows per dimension)
    grid_extent : float
        Physical extent in arcseconds

    Returns
    -------
    plotly.Figure or None
        Deflection field visualization
    """
    client = get_api_client()

    try:
        # Create grid of positions
        x = np.linspace(-grid_extent, grid_extent, grid_size)
        y = np.linspace(-grid_extent, grid_extent, grid_size)
        xx, yy = np.meshgrid(x, y)
        positions = list(zip(xx.ravel(), yy.ravel()))

        result = client.compute_deflection(lens_id=lens_id, positions=positions)

        # Extract deflection angles
        deflections = result["deflections"]
        alpha_x = np.array([d["alpha_x"] for d in deflections]).reshape(xx.shape)
        alpha_y = np.array([d["alpha_y"] for d in deflections]).reshape(xx.shape)

        # Create quiver plot
        fig = go.Figure(
            data=go.Quiver(
                x=xx.ravel(),
                y=yy.ravel(),
                u=alpha_x.ravel(),
                v=alpha_y.ravel(),
                scale=1.0,
                arrow=dict(size=8),
                line=dict(color="cyan", width=2),
            )
        )

        fig.update_layout(
            title="Deflection Angle Field",
            xaxis_title="x (arcsec)",
            yaxis_title="y (arcsec)",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
        )

        return fig

    except Exception as e:
        st.error(format_api_error(e))
        return None


def compute_lensing_potential(lens_id: str, positions: List[Tuple[float, float]]):
    """
    Compute lensing potential at positions via API.

    Parameters
    ----------
    lens_id : str
        Lens model ID
    positions : list of tuple
        List of (x, y) positions

    Returns
    -------
    numpy.ndarray or None
        Potential values
    """
    client = get_api_client()

    try:
        result = client.compute_lensing_potential(lens_id=lens_id, positions=positions)

        return np.array(result["potentials"])

    except Exception as e:
        st.error(format_api_error(e))
        return None


def find_critical_curves(lens_id: str, grid_size: int = 512, grid_extent: float = 5.0):
    """
    Find critical curves via API.

    Parameters
    ----------
    lens_id : str
        Lens model ID
    grid_size : int
        Grid resolution
    grid_extent : float
        Physical extent

    Returns
    -------
    tuple
        (critical_curves, caustics) or (None, None) on error
    """
    client = get_api_client()

    try:
        # First get convergence map
        result = client.compute_convergence_map(
            lens_id=lens_id, grid_size=grid_size, grid_extent=grid_extent
        )

        kappa_grid = np.array(result["kappa_grid"])
        x_grid = np.array(result["x_grid"])
        y_grid = np.array(result["y_grid"])

        # Find where κ ≈ 1 (critical curves)
        critical_mask = np.abs(kappa_grid - 1.0) < 0.1

        # Get critical curve points
        critical_curves = []
        if np.any(critical_mask):
            cy, cx = np.where(critical_mask)
            critical_curves = list(zip(x_grid[cx], y_grid[cy]))

        return critical_curves, []  # Caustics would need ray tracing

    except Exception as e:
        st.error(format_api_error(e))
        return None, None


def run_wave_optics_simulation(
    lens_id: str,
    source_position: Tuple[float, float],
    wavelength: float = 500.0,
    grid_size: int = 256,
    grid_extent: float = 3.0,
):
    """
    Run wave optics simulation via API.

    Parameters
    ----------
    lens_id : str
        Lens model ID
    source_position : tuple
        (x, y) source position
    wavelength : float
        Wavelength in nm
    grid_size : int
        Grid resolution
    grid_extent : float
        Physical extent

    Returns
    -------
    dict or None
        Wave optics results
    """
    client = get_api_client()

    try:
        result = client.compute_wave_amplification(
            lens_id=lens_id,
            source_position=source_position,
            wavelength=wavelength,
            grid_size=grid_size,
            grid_extent=grid_extent,
            return_geometric=True,
        )

        return result

    except Exception as e:
        st.error(format_api_error(e))
        return None


def compare_wave_geometric(
    lens_id: str,
    source_position: Tuple[float, float],
    wavelength: float = 500.0,
    grid_size: int = 256,
    grid_extent: float = 3.0,
):
    """
    Compare wave vs geometric optics via API.

    Parameters
    ----------
    lens_id : str
        Lens model ID
    source_position : tuple
        (x, y) source position
    wavelength : float
        Wavelength in nm
    grid_size : int
        Grid resolution
    grid_extent : float
        Physical extent

    Returns
    -------
    dict or None
        Comparison results
    """
    client = get_api_client()

    try:
        result = client.compare_wave_geometric(
            lens_id=lens_id,
            source_position=source_position,
            wavelength=wavelength,
            grid_size=grid_size,
            grid_extent=grid_extent,
        )

        return result

    except Exception as e:
        st.error(format_api_error(e))
        return None


def load_test_results() -> dict:
    """
    Load test results from API.

    Returns
    -------
    dict
        Test results dictionary
    """
    client = get_api_client()

    try:
        result = client.run_test_suite()
        return {
            "passed": result.get("passed", 0),
            "failed": result.get("failed", 0),
            "total": result.get("total", 0),
            "duration": result.get("duration", 0),
            "status": "completed" if result.get("passed", 0) > 0 else "failed",
        }
    except Exception as e:
        st.error(format_api_error(e))
        return {
            "passed": 0,
            "failed": 0,
            "total": 0,
            "status": "error",
            "message": str(e),
        }


def format_metric(value: float, unit: str = "", precision: int = 3) -> str:
    """
    Format a metric value with appropriate units.

    Parameters
    ----------
    value : float
        The value to format
    unit : str
        Unit string
    precision : int
        Number of decimal places

    Returns
    -------
    str
        Formatted string
    """
    if abs(value) < 1e-3 or abs(value) > 1e3:
        return f"{value:.{precision}e} {unit}"
    else:
        return f"{value:.{precision}f} {unit}"
