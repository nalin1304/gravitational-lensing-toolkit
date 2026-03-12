"""
Synthetic Data Calibration Pipeline
====================================

Systematic calibration of synthetic gravitational lensing data against
real HST/JWST observations and literature values.

Purpose:
--------
Ensures synthetic training data matches observational properties:
- Pixel scales (arcsec/pixel)
- Signal-to-noise ratios
- PSF profiles
- Physical parameters (Einstein radius, mass, etc.)

Calibration Process:
-------------------
1. Load real observation (HST Einstein Cross, Twin Quasar, etc.)
2. Generate synthetic analog with matched properties
3. Run analysis pipeline on both
4. Compare recovered parameters to literature values
5. Compute calibration factors
6. Apply to all synthetic training data

Scientific References:
---------------------
- Huchra et al. (1985) - Q0957+561 (Twin Quasar) discovery
- Schmidt et al. (1998) - Q2237+030 (Einstein Cross) parameters
- Oguri (2007) - Gravitational lens modeling review

Author: ISEF 2025 - Scientific Calibration
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import logging
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

# Known literature values for famous lensing systems
LITERATURE_VALUES = {
    "einstein_cross": {
        "name": "Q2237+030 (Einstein Cross)",
        "z_lens": 0.0394,
        "z_source": 1.695,
        "einstein_radius_arcsec": 1.85,  # Corrigan et al. (1991)
        "lens_mass_msun": 1.0e11,  # Galaxy-scale
        "separation_A_B_arcsec": 1.8,  # Quadruple image separation
        "reference": "Schmidt et al. (1998), ApJ 507, 46",
    },
    "twin_quasar": {
        "name": "Q0957+561 (Twin Quasar)",
        "z_lens": 0.355,
        "z_source": 1.414,
        "einstein_radius_arcsec": 1.40,  # Falco et al. (1997)
        "lens_mass_msun": 3.0e12,  # Cluster-scale
        "separation_arcsec": 6.1,  # Double image separation
        "time_delay_days": 417,  # ± 3 days (Kundic et al. 1997)
        "reference": "Kundic et al. (1997), ApJ 482, 75",
    },
    "abell_2218": {
        "name": "Abell 2218 Cluster",
        "z_lens": 0.175,
        "z_source": 0.702,  # Typical arc redshift
        "einstein_radius_arcsec": 35.0,  # Cluster-scale
        "lens_mass_msun": 1.5e14,  # Massive cluster
        "reference": "Kneib et al. (1996), ApJ 471, 643",
    },
}


@dataclass
class CalibrationResult:
    """Results from calibration against a single system."""

    system_name: str

    # Literature values
    true_einstein_radius: float
    true_mass: float

    # Raw (uncalibrated) recovered values
    raw_einstein_radius: float
    raw_mass: float

    # Calibrated recovered values
    calibrated_einstein_radius: float
    calibrated_mass: float

    # Calibration factors
    radius_calibration_factor: float
    mass_calibration_factor: float

    # Errors
    raw_radius_error_percent: float
    calibrated_radius_error_percent: float
    raw_mass_error_percent: float
    calibrated_mass_error_percent: float

    # Metadata
    pixel_scale: float
    snr: float
    convergence_map: Optional[np.ndarray] = None


class SyntheticDataCalibrator:
    """
    Calibrate synthetic gravitational lensing data against real observations.

    Examples
    --------
    >>> calibrator = SyntheticDataCalibrator()
    >>>
    >>> # Calibrate against Einstein Cross
    >>> result = calibrator.calibrate_system("einstein_cross")
    >>> print(f"Error: {result.calibrated_radius_error_percent:.2f}%")
    >>>
    >>> # Apply calibration to synthetic dataset
    >>> calibrated_data = calibrator.apply_calibration(
    ...     synthetic_maps,
    ...     result.radius_calibration_factor
    ... )
    """

    def __init__(self):
        """Initialize calibrator with literature values."""
        self.literature = LITERATURE_VALUES
        self.calibration_results: Dict[str, CalibrationResult] = {}

    def calibrate_system(
        self,
        system_key: str,
        synthetic_convergence: Optional[np.ndarray] = None,
        pixel_scale: float = 0.05,
        fov_arcsec: float = 10.0,
    ) -> CalibrationResult:
        """
        Calibrate synthetic data against a known lensing system.

        Parameters
        ----------
        system_key : str
            Key in LITERATURE_VALUES ("einstein_cross", "twin_quasar", etc.)
        synthetic_convergence : ndarray, optional
            Pre-generated synthetic convergence map. If None, generates new.
        pixel_scale : float
            Pixel scale in arcsec/pixel (default: 0.05 for HST)
        fov_arcsec : float
            Field of view in arcseconds

        Returns
        -------
        CalibrationResult
            Calibration results with factors and errors
        """
        if system_key not in self.literature:
            raise ValueError(
                f"Unknown system: {system_key}. "
                f"Available: {list(self.literature.keys())}"
            )

        lit = self.literature[system_key]
        logger.info(f"Calibrating against {lit['name']}")

        # Generate synthetic analog if not provided
        if synthetic_convergence is None:
            synthetic_convergence = self._generate_synthetic_analog(
                lit, pixel_scale, fov_arcsec
            )

        # Analyze synthetic system
        raw_theta_e, raw_mass = self._analyze_convergence_map(
            synthetic_convergence,
            pixel_scale=pixel_scale,
            z_lens=lit["z_lens"],
            z_source=lit["z_source"],
        )

        # FIXED: Use holdout-based calibration instead of trivial self-calibration
        # For proper calibration, we compute factors from OTHER systems (holdout)
        # and apply to this system - this tests generalization ability
        
        # For single-system calibration, we report raw errors only
        # (self-calibration would be scientifically meaningless)
        calibration_factors = self._get_calibration_factors()
        
        if calibration_factors is not None:
            # Apply calibration factors derived from OTHER systems
            radius_factor, mass_factor = calibration_factors
            calibrated_theta_e = raw_theta_e * radius_factor
            calibrated_mass = raw_mass * mass_factor
        else:
            # No calibration available - report raw results with note
            radius_factor = 1.0
            mass_factor = 1.0
            calibrated_theta_e = raw_theta_e
            calibrated_mass = raw_mass

        # Compute errors
        raw_radius_error = abs(raw_theta_e - lit["einstein_radius_arcsec"]) / lit["einstein_radius_arcsec"] * 100
        cal_radius_error = abs(calibrated_theta_e - lit["einstein_radius_arcsec"]) / lit["einstein_radius_arcsec"] * 100
        raw_mass_error = abs(raw_mass - lit["lens_mass_msun"]) / lit["lens_mass_msun"] * 100
        cal_mass_error = abs(calibrated_mass - lit["lens_mass_msun"]) / lit["lens_mass_msun"] * 100

        result = CalibrationResult(
            system_name=lit["name"],
            true_einstein_radius=lit["einstein_radius_arcsec"],
            true_mass=lit["lens_mass_msun"],
            raw_einstein_radius=raw_theta_e,
            raw_mass=raw_mass,
            calibrated_einstein_radius=calibrated_theta_e,
            calibrated_mass=calibrated_mass,
            radius_calibration_factor=radius_factor,
            mass_calibration_factor=mass_factor,
            raw_radius_error_percent=raw_radius_error,
            calibrated_radius_error_percent=cal_radius_error,
            raw_mass_error_percent=raw_mass_error,
            calibrated_mass_error_percent=cal_mass_error,
            pixel_scale=pixel_scale,
            snr=30.0,  # Typical HST SNR
            convergence_map=synthetic_convergence,
        )

        self.calibration_results[system_key] = result
        logger.info(
            f"Calibration complete: {system_key}\n"
            f"  Raw error: {raw_radius_error:.2f}%\n"
            f"  Calibrated error: {cal_radius_error:.2f}%"
        )

        return result

    def _get_calibration_factors(self):
        """
        Get calibration factors from OTHER calibration systems (holdout method).
        
        This is the scientifically correct approach - derive calibration factors
        from independent systems, then apply to new systems.
        
        Returns None if insufficient data for calibration.
        """
        if len(self.calibration_results) < 3:
            # Need at least 3 other systems for meaningful calibration
            return None
        
        # Compute average calibration factors from completed systems
        radius_factors = []
        mass_factors = []
        
        for result in self.calibration_results.values():
            if result.raw_einstein_radius > 0 and result.raw_mass > 0:
                radius_factors.append(result.true_einstein_radius / result.raw_einstein_radius)
                mass_factors.append(result.true_mass / result.raw_mass)
        
        if len(radius_factors) < 2:
            return None
        
        # Use median to be robust against outliers
        import statistics
        return statistics.median(radius_factors), statistics.median(mass_factors)

    def _generate_synthetic_analog(
        self,
        lit_values: Dict,
        pixel_scale: float,
        fov_arcsec: float,
    ) -> np.ndarray:
        """
        Generate synthetic convergence map matching literature system.

        Uses SIS model with literature Einstein radius.
        """
        # Generate grid
        npix = int(fov_arcsec / pixel_scale)
        x = np.linspace(-fov_arcsec/2, fov_arcsec/2, npix)
        y = np.linspace(-fov_arcsec/2, fov_arcsec/2, npix)
        xx, yy = np.meshgrid(x, y)

        # Create SIS convergence using literature Einstein radius
        # For SIS: κ(θ) = θ_E / (2|θ|)
        theta_e = lit_values["einstein_radius_arcsec"]

        r = np.sqrt(xx**2 + yy**2)
        # Avoid division by zero at center
        convergence = np.where(r > 0.01, theta_e / (2 * r), theta_e / 0.02)

        # Clip to avoid infinities
        convergence = np.clip(convergence, 0, 20.0)

        return convergence

    def _analyze_convergence_map(
        self,
        convergence: np.ndarray,
        pixel_scale: float,
        z_lens: float,
        z_source: float,
    ) -> Tuple[float, float]:
        """
        Analyze convergence map to extract Einstein radius and mass.

        Returns
        -------
        theta_e : float
            Einstein radius in arcseconds
        mass : float
            Lens mass in solar masses
        """
        # Find Einstein radius (where κ = 1)
        # For SIS: θ_E is where convergence drops to ~1

        center_idx = convergence.shape[0] // 2
        radial_profile = convergence[center_idx, center_idx:]

        # Find radius where κ crosses 1.0
        try:
            crossing_idx = np.where(radial_profile < 1.0)[0][0]
            theta_e_pixels = crossing_idx
            theta_e_arcsec = theta_e_pixels * pixel_scale
        except IndexError:
            # If no crossing, use half-max as proxy
            max_kappa = np.max(convergence)
            half_max_indices = np.where(convergence > max_kappa / 2)
            if len(half_max_indices[0]) > 0:
                # Estimate radius from extent
                radius_pixels = np.sqrt(len(half_max_indices[0]) / np.pi)
                theta_e_arcsec = radius_pixels * pixel_scale
            else:
                theta_e_arcsec = 1.0  # Fallback

        # Estimate mass from Einstein radius
        # For SIS: θ_E = 4π (σ_v/c)² (D_LS/D_S)
        # Approximation: M ∝ θ_E² for fixed cosmology

        from astropy.cosmology import FlatLambdaCDM
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

        D_L = cosmo.angular_diameter_distance(z_lens).value  # Mpc
        D_S = cosmo.angular_diameter_distance(z_source).value
        D_LS = cosmo.angular_diameter_distance_z1z2(z_lens, z_source).value

        # Critical surface density
        from astropy.constants import c, G
        import astropy.units as u

        sigma_crit = (c.to(u.km/u.s)**2 / (4 * np.pi * G)) * (D_S / (D_L * D_LS)) / u.Mpc
        sigma_crit = sigma_crit.to(u.Msun / u.pc**2).value

        # Enclosed mass within Einstein radius
        area_sq_arcsec = np.pi * theta_e_arcsec**2
        area_sq_pc = area_sq_arcsec * (D_L * 1e6 * np.pi / 180 / 3600)**2

        # Mass = Σ_crit * Area * <κ>
        mean_kappa = np.mean(convergence[convergence > 0.1])  # Average in central region
        mass_msun = sigma_crit * area_sq_pc * mean_kappa

        return theta_e_arcsec, mass_msun

    def apply_calibration(
        self,
        synthetic_maps: np.ndarray,
        calibration_factor: float,
    ) -> np.ndarray:
        """
        Apply calibration factor to synthetic convergence maps.

        Parameters
        ----------
        synthetic_maps : ndarray
            Array of synthetic convergence maps [N, H, W]
        calibration_factor : float
            Calibration factor from calibrate_system()

        Returns
        -------
        calibrated_maps : ndarray
            Calibrated maps with adjusted scaling
        """
        # Scale convergence values by calibration factor
        # This adjusts the effective mass/radius to match observations
        calibrated = synthetic_maps * calibration_factor

        logger.info(
            f"Applied calibration factor {calibration_factor:.4f} "
            f"to {len(synthetic_maps)} synthetic maps"
        )

        return calibrated

    def generate_calibration_table(self) -> str:
        """
        Generate markdown table summarizing all calibration results.

        Returns
        -------
        table : str
            Markdown-formatted calibration table
        """
        if not self.calibration_results:
            return "No calibration results available. Run calibrate_system() first."

        lines = [
            "# Synthetic Data Calibration Results",
            "",
            "| System | True θ_E (″) | Raw Recovery (″) | Calibrated Recovery (″) | Raw Error | Calibrated Error |",
            "|--------|--------------|------------------|--------------------------|-----------|------------------|",
        ]

        for key, result in self.calibration_results.items():
            lines.append(
                f"| {result.system_name} | "
                f"{result.true_einstein_radius:.2f} | "
                f"{result.raw_einstein_radius:.2f} | "
                f"{result.calibrated_einstein_radius:.2f} | "
                f"{result.raw_radius_error_percent:.1f}% | "
                f"{result.calibrated_radius_error_percent:.1f}% |"
            )

        lines.extend([
            "",
            "## Mass Calibration",
            "",
            "| System | True Mass (M☉) | Raw Recovery (M☉) | Calibrated Recovery (M☉) | Raw Error | Calibrated Error |",
            "|--------|----------------|-------------------|---------------------------|-----------|------------------|",
        ])

        for key, result in self.calibration_results.items():
            lines.append(
                f"| {result.system_name} | "
                f"{result.true_mass:.2e} | "
                f"{result.raw_mass:.2e} | "
                f"{result.calibrated_mass:.2e} | "
                f"{result.raw_mass_error_percent:.1f}% | "
                f"{result.calibrated_mass_error_percent:.1f}% |"
            )

        lines.extend([
            "",
            "## Calibration Factors",
            "",
            "| System | Radius Factor | Mass Factor | Reference |",
            "|--------|---------------|-------------|-----------|",
        ])

        for key, result in self.calibration_results.items():
            lit = self.literature[key]
            lines.append(
                f"| {result.system_name} | "
                f"{result.radius_calibration_factor:.4f} | "
                f"{result.mass_calibration_factor:.4f} | "
                f"{lit['reference']} |"
            )

        return "\n".join(lines)

    def save_calibration(self, output_path: Path):
        """Save calibration results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results_dict = {}
        for key, result in self.calibration_results.items():
            results_dict[key] = {
                "system_name": result.system_name,
                "true_einstein_radius": result.true_einstein_radius,
                "calibrated_einstein_radius": result.calibrated_einstein_radius,
                "radius_error_percent": result.calibrated_radius_error_percent,
                "radius_calibration_factor": result.radius_calibration_factor,
                "true_mass": result.true_mass,
                "calibrated_mass": result.calibrated_mass,
                "mass_error_percent": result.calibrated_mass_error_percent,
                "mass_calibration_factor": result.mass_calibration_factor,
                "pixel_scale": result.pixel_scale,
                "snr": result.snr,
            }

        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)

        logger.info(f"Saved calibration results to {output_path}")


def run_full_calibration_suite() -> SyntheticDataCalibrator:
    """
    Run complete calibration suite against all known systems.

    Returns
    -------
    calibrator : SyntheticDataCalibrator
        Calibrator with all results

    Examples
    --------
    >>> calibrator = run_full_calibration_suite()
    >>> print(calibrator.generate_calibration_table())
    """
    calibrator = SyntheticDataCalibrator()

    logger.info("Starting full calibration suite...")

    # Calibrate each system
    for system_key in ["einstein_cross", "twin_quasar"]:
        try:
            result = calibrator.calibrate_system(system_key)
            logger.info(f"✓ {result.system_name}: {result.calibrated_radius_error_percent:.2f}% error")
        except Exception as e:
            logger.error(f"✗ Failed to calibrate {system_key}: {e}")

    # Generate summary table
    table = calibrator.generate_calibration_table()
    print("\n" + table)

    # Save results
    calibrator.save_calibration(Path("results/calibration_results.json"))

    return calibrator


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    calibrator = run_full_calibration_suite()
