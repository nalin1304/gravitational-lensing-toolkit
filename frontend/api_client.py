"""
API Client for Gravitational Lensing Frontend

This module provides a client to interact with the FastAPI backend.
"""

import requests
import numpy as np
from typing import Dict, Optional, Tuple, List
import streamlit as st
from datetime import datetime

# API Configuration
API_BASE_URL = "http://localhost:8000"
API_VERSION = "v1"


class APIClient:
    """Client for interacting with the Gravitational Lensing API."""

    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def health_check(self) -> Dict:
        """Check if API is healthy."""
        try:
            response = self.session.get(
                f"{self.base_url}/api/{API_VERSION}/validate/health", timeout=5
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def create_point_mass_lens(
        self,
        mass: float,
        z_lens: float,
        z_source: float,
        H0: float = 70.0,
        Omega_m: float = 0.3,
    ) -> Dict:
        """Create a point mass lens model."""
        data = {
            "mass": mass,
            "z_lens": z_lens,
            "z_source": z_source,
            "H0": H0,
            "Omega_m": Omega_m,
        }

        response = self.session.post(
            f"{self.base_url}/api/{API_VERSION}/lens/point-mass",
            json=data,
            headers=self.headers,
        )
        response.raise_for_status()
        return response.json()

    def create_nfw_lens(
        self,
        M_vir: float,
        concentration: float,
        z_lens: float,
        z_source: float,
        ellipticity: float = 0.0,
        position_angle: float = 0.0,
        H0: float = 70.0,
        Omega_m: float = 0.3,
    ) -> Dict:
        """Create an NFW lens model."""
        data = {
            "M_vir": M_vir,
            "concentration": concentration,
            "z_lens": z_lens,
            "z_source": z_source,
            "ellipticity": ellipticity,
            "position_angle": position_angle,
            "H0": H0,
            "Omega_m": Omega_m,
        }

        response = self.session.post(
            f"{self.base_url}/api/{API_VERSION}/lens/nfw",
            json=data,
            headers=self.headers,
        )
        response.raise_for_status()
        return response.json()

    def compute_deflection(
        self, lens_id: str, positions: List[Tuple[float, float]]
    ) -> Dict:
        """Compute deflection angles at given positions."""
        data = {
            "lens_id": lens_id,
            "positions": [{"x": x, "y": y} for x, y in positions],
        }

        response = self.session.post(
            f"{self.base_url}/api/{API_VERSION}/compute/deflection",
            json=data,
            headers=self.headers,
        )
        response.raise_for_status()
        return response.json()

    def compute_convergence_map(
        self, lens_id: str, grid_size: int = 256, grid_extent: float = 5.0
    ) -> Dict:
        """Compute convergence map on a grid."""
        data = {"lens_id": lens_id, "grid_size": grid_size, "grid_extent": grid_extent}

        response = self.session.post(
            f"{self.base_url}/api/{API_VERSION}/compute/convergence",
            json=data,
            headers=self.headers,
        )
        response.raise_for_status()
        return response.json()

    def compute_lensing_potential(
        self, lens_id: str, positions: List[Tuple[float, float]]
    ) -> Dict:
        """Compute lensing potential at given positions."""
        data = {
            "lens_id": lens_id,
            "positions": [{"x": x, "y": y} for x, y in positions],
        }

        response = self.session.post(
            f"{self.base_url}/api/{API_VERSION}/compute/potential",
            json=data,
            headers=self.headers,
        )
        response.raise_for_status()
        return response.json()

    def find_lensed_images(
        self,
        lens_id: str,
        source_position: Tuple[float, float],
        grid_size: int = 512,
        grid_extent: float = 5.0,
    ) -> Dict:
        """Find lensed image positions for a given source."""
        data = {
            "lens_id": lens_id,
            "source_position": {"x": source_position[0], "y": source_position[1]},
            "grid_size": grid_size,
            "grid_extent": grid_extent,
        }

        response = self.session.post(
            f"{self.base_url}/api/{API_VERSION}/compute/images",
            json=data,
            headers=self.headers,
        )
        response.raise_for_status()
        return response.json()

    def compute_time_delay(
        self,
        lens_id: str,
        image_positions: List[Tuple[float, float]],
        source_position: Tuple[float, float],
        H0: float = 70.0,
        Omega_m: float = 0.3,
    ) -> Dict:
        """Compute time delays between images."""
        data = {
            "lens_id": lens_id,
            "image_positions": [{"x": x, "y": y} for x, y in image_positions],
            "source_position": {"x": source_position[0], "y": source_position[1]},
            "H0": H0,
            "Omega_m": Omega_m,
        }

        response = self.session.post(
            f"{self.base_url}/api/{API_VERSION}/compute/timedelay",
            json=data,
            headers=self.headers,
        )
        response.raise_for_status()
        return response.json()

    def compute_wave_amplification(
        self,
        lens_id: str,
        source_position: Tuple[float, float],
        wavelength: float = 500.0,
        grid_size: int = 512,
        grid_extent: float = 3.0,
        return_geometric: bool = True,
    ) -> Dict:
        """Compute wave optical amplification factor."""
        data = {
            "lens_id": lens_id,
            "source_position": {"x": source_position[0], "y": source_position[1]},
            "wavelength": wavelength,
            "grid_size": grid_size,
            "grid_extent": grid_extent,
            "return_geometric": return_geometric,
        }

        response = self.session.post(
            f"{self.base_url}/api/{API_VERSION}/wave/amplification",
            json=data,
            headers=self.headers,
        )
        response.raise_for_status()
        return response.json()

    def compare_wave_geometric(
        self,
        lens_id: str,
        source_position: Tuple[float, float],
        wavelength: float = 500.0,
        grid_size: int = 512,
        grid_extent: float = 3.0,
    ) -> Dict:
        """Compare wave vs geometric optics."""
        data = {
            "lens_id": lens_id,
            "source_position": {"x": source_position[0], "y": source_position[1]},
            "wavelength": wavelength,
            "grid_size": grid_size,
            "grid_extent": grid_extent,
        }

        response = self.session.post(
            f"{self.base_url}/api/{API_VERSION}/wave/compare",
            json=data,
            headers=self.headers,
        )
        response.raise_for_status()
        return response.json()

    def pinn_predict(self, images: List, model_path: Optional[str] = None) -> Dict:
        """Run PINN prediction on images."""
        # For file uploads, we'd use multipart/form-data
        # This is a simplified version
        data = {"use_pretrained": model_path is None, "model_path": model_path}

        response = self.session.post(
            f"{self.base_url}/api/{API_VERSION}/pinn/predict",
            json=data,
            headers=self.headers,
        )
        response.raise_for_status()
        return response.json()

    def run_test_suite(self) -> Dict:
        """Run validation test suite."""
        response = self.session.get(
            f"{self.base_url}/api/{API_VERSION}/validate/tests",
            timeout=300,  # Tests may take a while
        )
        response.raise_for_status()
        return response.json()

    def get_benchmarks(self) -> Dict:
        """Get benchmark results."""
        response = self.session.get(
            f"{self.base_url}/api/{API_VERSION}/validate/benchmarks"
        )
        response.raise_for_status()
        return response.json()


# Global API client instance
@st.cache_resource
def get_api_client() -> APIClient:
    """Get or create API client (cached)."""
    return APIClient()


def check_api_connection() -> Tuple[bool, str]:
    """Check if API is available."""
    client = get_api_client()
    try:
        health = client.health_check()
        if health.get("status") == "healthy":
            return True, "Connected to API"
        else:
            return False, f"API unhealthy: {health.get('message', 'Unknown error')}"
    except Exception as e:
        return False, f"Cannot connect to API: {str(e)}"


def format_api_error(error: Exception) -> str:
    """Format API error for display."""
    if isinstance(error, requests.exceptions.ConnectionError):
        return (
            "⚠️ Cannot connect to backend API. Please ensure the API server is running."
        )
    elif isinstance(error, requests.exceptions.Timeout):
        return "⏱️ Request timed out. The calculation may be too complex."
    elif isinstance(error, requests.exceptions.HTTPError):
        return f"❌ API Error: {error.response.status_code} - {error.response.text}"
    else:
        return f"❌ Error: {str(error)}"
