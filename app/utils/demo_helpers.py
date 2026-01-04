"""
Demo Helper Utilities for Zero-Friction User Experience
========================================================
Handles asset loading, pipeline execution, and result management for one-click demos.
"""

import os
import yaml
import numpy as np
import streamlit as st
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
from io import BytesIO
import logging

logger = logging.getLogger(__name__)

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DEMOS_DIR = PROJECT_ROOT / "demos"
ASSETS_DIR = PROJECT_ROOT / "assets" / "demos"


def ensure_demo_asset(asset_name: str) -> Path:
    """
    Ensure demo asset exists, downloading if necessary.

    Args:
        asset_name: Built-in asset identifier (e.g., "einstein_cross_hst")

    Returns:
        Path to the asset file
    """
    asset_path = ASSETS_DIR / f"{asset_name}.npy"

    # If asset already exists, return path
    if asset_path.exists():
        logger.info(f"Found cached demo asset: {asset_name}")
        return asset_path

    # Create assets directory if needed
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    # Generate synthetic demo data (in production, this would fetch from HubbleSite)
    logger.info(f"Generating synthetic demo asset: {asset_name}")

    if asset_name == "einstein_cross_hst":
        # Create realistic Einstein Cross simulation
        image = _generate_einstein_cross_image()
    elif asset_name == "twin_quasar_hst":
        # Create twin quasar simulation
        image = _generate_twin_quasar_image()
    elif asset_name == "jwst_cluster_arc":
        # Create cluster arc simulation
        image = _generate_cluster_arc_image()
    elif asset_name == "substructure_test":
        # Create substructure test image
        image = _generate_substructure_image()
    else:
        # Generic point source
        image = _generate_generic_source()

    # Save to disk
    np.save(asset_path, image)
    logger.info(f"Saved demo asset to: {asset_path}")

    return asset_path


def _generate_einstein_cross_image(size: int = 128) -> np.ndarray:
    """Generate synthetic Einstein Cross (quadruple image)."""
    image = np.zeros((size, size))
    center = size // 2

    # Four point sources in cross pattern
    positions = [
        (center + 15, center + 15),  # NE
        (center - 15, center + 15),  # NW
        (center + 15, center - 15),  # SE
        (center - 15, center - 15),  # SW
    ]

    for y, x in positions:
        # Gaussian point source
        yy, xx = np.ogrid[:size, :size]
        gaussian = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * 2.5**2))
        image += gaussian * 1000.0

    # Add lens galaxy in center
    yy, xx = np.ogrid[:size, :size]
    lens = np.exp(-((xx - center)**2 + (yy - center)**2) / (2 * 8**2))
    image += lens * 500.0

    # Add noise
    image += np.random.normal(0, 10, (size, size))

    return image.astype(np.float32)


def _generate_twin_quasar_image(size: int = 256) -> np.ndarray:
    """Generate synthetic Twin Quasar (double image)."""
    image = np.zeros((size, size))
    center = size // 2

    # Two point sources
    positions = [
        (center, center + 30),  # Image A
        (center, center - 30),  # Image B
    ]

    for y, x in positions:
        yy, xx = np.ogrid[:size, :size]
        gaussian = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * 3**2))
        image += gaussian * 1200.0

    # Lens galaxy
    yy, xx = np.ogrid[:size, :size]
    lens = np.exp(-((xx - center)**2 + (yy - center)**2) / (2 * 15**2))
    image += lens * 400.0

    image += np.random.normal(0, 8, (size, size))

    return image.astype(np.float32)


def _generate_cluster_arc_image(size: int = 512) -> np.ndarray:
    """Generate synthetic galaxy cluster with arc."""
    image = np.zeros((size, size))
    center = size // 2

    # Generate arc (curved extended source)
    theta = np.linspace(0, np.pi, 100)
    radius = 60
    arc_x = center + radius * np.cos(theta)
    arc_y = center + radius * np.sin(theta) + 30

    for x, y in zip(arc_x, arc_y):
        yy, xx = np.ogrid[:size, :size]
        gaussian = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * 4**2))
        image += gaussian * 800.0

    # Cluster galaxies
    cluster_positions = [
        (center, center - 20),
        (center + 25, center - 10),
        (center - 30, center + 5),
    ]

    for y, x in cluster_positions:
        yy, xx = np.ogrid[:size, :size]
        galaxy = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * 10**2))
        image += galaxy * 600.0

    image += np.random.normal(0, 5, (size, size))

    return image.astype(np.float32)


def _generate_substructure_image(size: int = 256) -> np.ndarray:
    """Generate image with subtle substructure perturbations."""
    image = np.zeros((size, size))
    center = size // 2

    # Main arc
    theta = np.linspace(-np.pi/3, np.pi/3, 80)
    radius = 50
    arc_x = center + radius * np.cos(theta) + 20
    arc_y = center + radius * np.sin(theta)

    for x, y in zip(arc_x, arc_y):
        yy, xx = np.ogrid[:size, :size]
        gaussian = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * 3**2))
        image += gaussian * 900.0

    # Subtle perturbations from substructure
    perturb_positions = [
        (center + 40, center + 30),
        (center - 45, center - 25),
    ]

    for y, x in perturb_positions:
        yy, xx = np.ogrid[:size, :size]
        # Very subtle brightness variations
        perturb = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * 15**2))
        image += perturb * 50.0  # Weak signal

    # Main lens
    yy, xx = np.ogrid[:size, :size]
    lens = np.exp(-((xx - center)**2 + (yy - center)**2) / (2 * 12**2))
    image += lens * 500.0

    image += np.random.normal(0, 6, (size, size))

    return image.astype(np.float32)


def _generate_generic_source(size: int = 128) -> np.ndarray:
    """Generate generic point source."""
    image = np.zeros((size, size))
    center = size // 2

    yy, xx = np.ogrid[:size, :size]
    gaussian = np.exp(-((xx - center)**2 + (yy - center)**2) / (2 * 3**2))
    image = gaussian * 1000.0 + np.random.normal(0, 10, (size, size))

    return image.astype(np.float32)


def load_demo_config(demo_name: str) -> Dict[str, Any]:
    """
    Load demo configuration from YAML file.

    Args:
        demo_name: Name of demo (e.g., "einstein_cross")

    Returns:
        Configuration dictionary
    """
    config_path = DEMOS_DIR / f"{demo_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Demo config not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded demo config: {demo_name}")
    return config


def full_analysis_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute full gravitational lensing analysis pipeline.

    This function orchestrates:
    1. Asset loading/generation
    2. Ray tracing (thin_lens mode enforced)
    3. PINN inference
    4. Uncertainty quantification
    5. Result formatting

    Args:
        config: Demo configuration dictionary

    Returns:
        Results dictionary with images, parameters, and uncertainties
    """
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    from src.lens_models.lens_system import LensSystem
    from src.optics.ray_tracing_backends import RayTracingBackend, RayTracingMode
    from src.ml.pinn import PINN
    import torch

    logger.info(f"Starting analysis pipeline for: {config.get('name', 'Unknown')}")

    # Step 1: Load/generate observation data
    if config.get("data", {}).get("source_image", "").startswith("builtin:"):
        asset_name = config["data"]["source_image"].split(":")[1]
        asset_path = ensure_demo_asset(asset_name)
        observation = np.load(asset_path)
    else:
        # Generate from config parameters
        observation = _generate_from_config(config)

    logger.info(f"Loaded observation: shape {observation.shape}")

    # Step 2: Validate thin_lens mode enforcement
    ray_mode = config.get("ray_tracing", {}).get("mode", "thin_lens")
    if ray_mode != "thin_lens":
        raise ValueError(
            f"Demo configs must use thin_lens mode (got: {ray_mode}). "
            "Schwarzschild mode is disabled for cosmological demos."
        )

    # Step 3: Set up lens system
    lens_config = config["lens"]
    source_config = config["source"]

    lens_system = LensSystem(
        mass=lens_config["mass"],
        z_lens=lens_config["z"],
        z_source=source_config["z"],
        lens_model=lens_config["model"],
        ellipticity=lens_config.get("ellipticity", 0.0),
    )

    logger.info(f"Lens system: {lens_config['model']} at z={lens_config['z']}")

    # Step 4: Ray tracing
    backend = RayTracingBackend(mode=RayTracingMode.THIN_LENS)

    grid_res = config.get("ray_tracing", {}).get("grid_resolution", 256)
    fov = config.get("observation", {}).get("fov_size", 128) * config.get("observation", {}).get("pixel_scale", 0.05)

    # Generate grid
    x = np.linspace(-fov/2, fov/2, grid_res)
    y = np.linspace(-fov/2, fov/2, grid_res)
    xx, yy = np.meshgrid(x, y)

    # Compute deflection and convergence
    alpha_x, alpha_y = lens_system.deflection(xx, yy)
    convergence = lens_system.convergence(xx, yy)

    logger.info("Ray tracing complete")

    # Step 5: PINN inference (if enabled)
    pinn_results = None
    uncertainty_map = None

    if config.get("analysis", {}).get("run_pinn_inference", False):
        try:
            # Load pre-trained PINN model
            pinn_model_path = PROJECT_ROOT / "src" / "ml" / "models" / "pretrained" / "pinn_lens_v1.pth"

            if pinn_model_path.exists():
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                pinn = PINN(input_dim=2, hidden_dim=64, output_dim=2).to(device)
                pinn.load_state_dict(torch.load(pinn_model_path, map_location=device))
                pinn.eval()

                logger.info("PINN inference enabled (using pre-trained model)")

                # Run inference
                with torch.no_grad():
                    coords = torch.tensor(np.stack([xx.flatten(), yy.flatten()], axis=1), dtype=torch.float32).to(device)
                    predictions = pinn(coords).cpu().numpy()

                pinn_results = {
                    "deflection_pred": predictions.reshape(grid_res, grid_res, 2),
                    "convergence_pred": np.linalg.norm(predictions, axis=1).reshape(grid_res, grid_res),
                }

                # Generate uncertainty map (simplified Bayesian approximation)
                if config.get("analysis", {}).get("uncertainty_quantification", False):
                    # Monte Carlo dropout for uncertainty
                    pinn.train()  # Enable dropout
                    mc_samples = 20
                    samples = []

                    for _ in range(mc_samples):
                        with torch.no_grad():
                            pred = pinn(coords).cpu().numpy()
                            samples.append(pred)

                    samples = np.array(samples)
                    uncertainty_map = np.std(samples, axis=0).reshape(grid_res, grid_res, 2)
                    uncertainty_map = np.linalg.norm(uncertainty_map, axis=2)

                    pinn.eval()
                    logger.info("Uncertainty quantification complete")
            else:
                logger.warning(f"PINN model not found at {pinn_model_path}, skipping inference")
        except Exception as e:
            logger.error(f"PINN inference failed: {e}")
            pinn_results = None

    # Step 6: Generate result visualizations
    results = {
        "config": config,
        "observation": observation,
        "convergence_map": convergence,
        "deflection": (alpha_x, alpha_y),
        "pinn_results": pinn_results,
        "uncertainty_map": uncertainty_map,
        "lens_parameters": {
            "mass": lens_config["mass"],
            "z_lens": lens_config["z"],
            "z_source": source_config["z"],
            "model": lens_config["model"],
            "ellipticity": lens_config.get("ellipticity", 0.0),
        },
        "ray_tracing_mode": "thin_lens",
    }

    logger.info("Pipeline complete")
    return results


def _generate_from_config(config: Dict[str, Any]) -> np.ndarray:
    """Generate synthetic observation from config parameters."""
    obs_config = config.get("observation", {})
    size = obs_config.get("fov_size", 128)

    # Simple placeholder - in production would use full ray tracing
    image = np.random.normal(0, obs_config.get("noise_level", 0.02), (size, size))
    return image.astype(np.float32)


def run_demo_and_redirect(demo_name: str):
    """
    Execute demo pipeline and redirect to results page.

    Args:
        demo_name: Name of demo to run
    """
    try:
        # Load configuration
        config = load_demo_config(demo_name)

        # Run full analysis
        with st.spinner(f"🌌 Simulating light paths through curved spacetime ({config.get('name', demo_name)})..."):
            results = full_analysis_pipeline(config)

        # Store in session state
        st.session_state["demo_results"] = results
        st.session_state["demo_name"] = demo_name

        st.toast("✅ Simulation complete!", icon="✨")

        # Redirect to results page
        st.switch_page("pages/03_Results.py")

    except Exception as e:
        st.error(f"❌ Demo execution failed: {str(e)}")
        logger.error(f"Demo {demo_name} failed: {e}", exc_info=True)


def export_pdf_report(results: Dict[str, Any]) -> BytesIO:
    """
    Generate PDF report from results.

    Args:
        results: Analysis results dictionary

    Returns:
        BytesIO buffer containing PDF
    """
    from matplotlib.backends.backend_pdf import PdfPages

    buffer = BytesIO()

    with PdfPages(buffer) as pdf:
        # Page 1: Overview
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle(f"Gravitational Lensing Analysis: {results['config'].get('name', 'Demo')}", fontsize=16, fontweight='bold')

        # Observation
        axes[0, 0].imshow(results["observation"], cmap='hot', origin='lower')
        axes[0, 0].set_title("Observation")
        axes[0, 0].axis('off')

        # Convergence map
        axes[0, 1].imshow(results["convergence_map"], cmap='viridis', origin='lower')
        axes[0, 1].set_title("Convergence κ (Mass Map)")
        axes[0, 1].axis('off')

        # PINN reconstruction (if available)
        if results.get("pinn_results"):
            axes[1, 0].imshow(results["pinn_results"]["convergence_pred"], cmap='viridis', origin='lower')
            axes[1, 0].set_title("PINN Reconstruction")
            axes[1, 0].axis('off')

        # Uncertainty
        if results.get("uncertainty_map") is not None:
            axes[1, 1].imshow(results["uncertainty_map"], cmap='Reds', origin='lower')
            axes[1, 1].set_title("95% Uncertainty")
            axes[1, 1].axis('off')

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # Page 2: Parameters
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')

        params = results["lens_parameters"]
        param_text = f"""
        Lens Parameters:
        ───────────────────────────────
        Model: {params['model']}
        Mass: {params['mass']:.2e} M☉
        Lens Redshift (z_l): {params['z_lens']}
        Source Redshift (z_s): {params['z_source']}
        Ellipticity: {params['ellipticity']:.2f}

        Ray Tracing Mode: {results['ray_tracing_mode']}

        Analysis:
        ───────────────────────────────
        PINN Inference: {'✓' if results.get('pinn_results') else '✗'}
        Uncertainty Quantification: {'✓' if results.get('uncertainty_map') is not None else '✗'}
        """

        ax.text(0.1, 0.9, param_text, transform=ax.transAxes, fontsize=12, verticalalignment='top', fontfamily='monospace')

        pdf.savefig(fig)
        plt.close()

    buffer.seek(0)
    return buffer
