"""
Wave Optics Engine for Gravitational Lensing

This module implements physical optics calculations including diffraction
and interference effects in gravitational lensing, extending beyond the
geometric optics approximation.

Wave optics is important when:
- Wavelength comparable to Schwarzschild radius
- Interference fringes between multiple images
- Chromatic effects in lensing
"""

import numpy as np
from typing import Tuple, Dict, Optional
import matplotlib.pyplot as plt
from astropy import constants as const
from astropy import units as u
from scipy.ndimage import label, gaussian_filter


class WaveOpticsEngine:
    """
    Calculate diffraction and interference effects using physical optics.

    This class computes the wave optical amplification factor accounting for
    the finite wavelength of light, which can produce interference patterns
    and differs from geometric optics predictions.

    The key equation is:
    F(θ) = exp(i × k × Φ(θ))

    where Φ(θ) is the Fermat potential (time delay surface) and k = 2π/λ.

    Parameters
    ----------
    None (stateless calculator)

    Examples
    --------
    >>> from lens_models import LensSystem, PointMassProfile
    >>> from optics import WaveOpticsEngine
    >>> lens_sys = LensSystem(0.5, 1.5)
    >>> lens = PointMassProfile(1e12, lens_sys)
    >>> engine = WaveOpticsEngine()
    >>> result = engine.compute_amplification_factor(
    ...     lens, source_position=(0.5, 0.0), wavelength=500.0
    ... )
    >>> print(f"Wave optics shows interference fringes")
    """

    def __init__(self):
        """Initialize the wave optics engine."""
        pass

    def compute_amplification_factor(
        self,
        lens_model,
        source_position: Tuple[float, float] = (0.5, 0.0),
        wavelength: float = 500.0,
        grid_size: int = 512,
        grid_extent: float = 3.0,
        lens_system=None,  # REQUIRED for proper distance calculation
        return_geometric: bool = True,
    ) -> Dict:
        """
        Calculate wave optical amplification including diffraction/interference.

        Algorithm:
        1. Compute Fermat potential on lens plane grid:
           Φ(θ) = 0.5|θ - β|² - ψ(θ)
           where ψ is the lensing potential from lens_model

        2. Calculate wave phase:
           φ(θ) = (2π/λ) × Φ(θ) × (geometric scale factor)

        3. Complex amplification:
           F(θ) = exp(i × φ(θ))

        4. Propagate to observer using FFT:
           F_obs = FFT2D(F(θ))

        5. Compute intensity: |F_obs|²

        Parameters
        ----------
        lens_model : MassProfile
            The lens model (must have lensing_potential method)
        source_position : tuple of float, optional
            Source position (β_x, β_y) in arcseconds (default: (0.5, 0.0))
        wavelength : float, optional
            Observation wavelength in nanometers (default: 500 nm for optical)
        grid_size : int, optional
            Grid size for computation (default: 512, recommend power of 2 for FFT)
        grid_extent : float, optional
            Physical extent of grid in arcseconds (default: 3.0)
        lens_system : LensSystem, REQUIRED
            Cosmological lens system for proper distance calculations.
            MUST be provided - no fallbacks allowed.
        return_geometric : bool, optional
            Whether to compute geometric optics for comparison (default: True)

        Returns
        -------
        result : dict
            Dictionary containing:
            - 'amplitude_map': 2D array of intensity |F|²
            - 'phase_map': 2D array of phase angle(F) in radians
            - 'grid_x': 1D array of x-coordinates in arcsec
            - 'grid_y': 1D array of y-coordinates in arcsec
            - 'wavelength': wavelength used in nm
            - 'fermat_potential': 2D array of Φ(θ) in arcsec²
            - 'geometric_comparison': dict with geometric optics result (if requested)

        Raises
        ------
        ValueError: If lens_system is not provided - this is REQUIRED for scientifically
            valid wave optics calculations.

        Notes
        -----
        The geometric scale factor converts the dimensionless Fermat potential
        to physical time delay using cosmological distances. The wave phase
        accumulates as light travels along different paths.

        For a lens at z_l with source at z_s, the time delay is:
        Δt = (1 + z_l) × (D_l × D_s / D_ls) / c × Φ(θ)

        The wave phase is then: φ = 2π × Δt / (λ / c) = (2πc/λ) × Δt
        """
        # CRITICAL: Require lens_system for scientifically valid calculations
        if lens_system is None:
            raise ValueError(
                "lens_system parameter is REQUIRED for wave optics calculations. "
                "Cannot use default/hardcoded distances as this would produce "
                "scientifically invalid results. Please provide a LensSystem object "
                "with proper cosmological parameters."
            )

        beta_x, beta_y = source_position

        # Step 1: Create computational grid on image plane (θ space)
        x = np.linspace(-grid_extent, grid_extent, grid_size)
        y = np.linspace(-grid_extent, grid_extent, grid_size)
        xx, yy = np.meshgrid(x, y)

        # Step 2: Compute lensing potential ψ(θ)
        # Flatten for vectorized computation
        x_flat = xx.ravel()
        y_flat = yy.ravel()

        psi_flat = lens_model.lensing_potential(x_flat, y_flat)
        psi = psi_flat.reshape(xx.shape)

        # Step 3: Compute Fermat potential
        # Φ(θ) = (1/2)|θ - β|² - ψ(θ)
        # This is the arrival time surface (dimensionless, in arcsec²)
        theta_minus_beta_x = xx - beta_x
        theta_minus_beta_y = yy - beta_y
        geometric_term = 0.5 * (theta_minus_beta_x**2 + theta_minus_beta_y**2)

        fermat_potential = geometric_term - psi

        # Step 4: Convert to physical time delay and then to wave phase
        # Get cosmological distances from lens system (already have units)
        D_l = lens_model.lens_system.angular_diameter_distance_lens()  # Quantity in Mpc
        D_s = (
            lens_model.lens_system.angular_diameter_distance_source()
        )  # Quantity in Mpc
        D_ls = (
            lens_model.lens_system.angular_diameter_distance_lens_source()
        )  # Quantity in Mpc
        z_l = lens_model.lens_system.z_l

        # Geometric factor: (1 + z_l) × D_l × D_s / D_ls / c
        # This converts Φ [arcsec²] to time delay [seconds]
        from src.utils.constants import ARCSEC_TO_RAD, C_LIGHT

        # Distance factor in meters (distances already have units from astropy)
        D_l_m = D_l.to(u.m).value
        D_s_m = D_s.to(u.m).value
        D_ls_m = D_ls.to(u.m).value

        # Geometric factor [m/rad²]
        geometric_factor = (1.0 + z_l) * D_l_m * D_s_m / D_ls_m

        # Time delay in seconds
        # c_light = C_LIGHT  # m/s
        time_delay = geometric_factor * fermat_potential * (ARCSEC_TO_RAD**2) / C_LIGHT

        # Convert wavelength to meters
        wavelength_m = wavelength * 1e-9  # nm to m

        # Wave phase φ = 2π × Δt × (c/λ) = 2π × Δt / T where T = λ/c is period
        # φ = 2π × c × Δt / λ
        wave_phase = 2.0 * np.pi * C_LIGHT * time_delay / wavelength_m

        # Step 5: Complex amplification field
        F_lens = np.exp(1j * wave_phase)

        # Step 6: Propagate to observer plane using Fresnel diffraction
        # Proper Fresnel propagation requires quadratic phase factor
        # Using Fraunhofer approximation (valid for large z/λ):
        # U(x,y) ∝ FFT[U₀(x',y') × exp(i·k/(2z)·(x'^2 + y'^2))]

        # Create coordinate grids
        dx = 2 * grid_extent / grid_size  # Pixel size
        fx = np.fft.fftfreq(grid_size, dx)  # Frequency coordinates
        fy = np.fft.fftfreq(grid_size, dx)
        FX, FY = np.meshgrid(fx, fy, indexing="ij")

        # Calculate proper cosmological distance for wave optics
        # z = D_l * D_s / D_ls (in appropriate units)
        D_l = lens_system.angular_diameter_distance_lens().to_value(u.Mpc)
        D_s = lens_system.angular_diameter_distance_source().to_value(u.Mpc)
        D_ls = lens_system.angular_diameter_distance_lens_source().to_value(u.Mpc)

        # Physical distance for wave propagation scale
        # For wave optics, z affects the Fresnel scaling
        z = D_l * D_s / D_ls  # Effective distance in Mpc

        # Quadratic phase factor for Fresnel propagation
        k = 2 * np.pi / wavelength_m  # Wave number
        phase_fresnel = np.pi * wavelength_m * z * (FX**2 + FY**2)

        # Apply Fresnel phase and propagate
        F_with_phase = F_lens * np.exp(1j * phase_fresnel)
        F_obs = np.fft.fft2(F_with_phase)
        F_obs = np.fft.fftshift(F_obs)  # Center zero frequency

        # Step 7: Compute observables
        amplitude_map = np.abs(F_obs) ** 2
        phase_map = np.angle(F_obs)

        # Normalize amplitude map (total flux should be conserved)
        amplitude_map = amplitude_map / np.sum(amplitude_map) * grid_size**2

        result = {
            "amplitude_map": amplitude_map,
            "phase_map": phase_map,
            "grid_x": x,
            "grid_y": y,
            "wavelength": wavelength,
            "fermat_potential": fermat_potential,
            "wave_phase": wave_phase,
            "grid_extent": grid_extent,
        }

        # Step 8: Optionally compute geometric optics for comparison
        if return_geometric:
            from .ray_tracing import ray_trace

            geo_result = ray_trace(
                source_position,
                lens_model,
                grid_extent=grid_extent,
                grid_resolution=grid_size,
                threshold=0.05,
                return_maps=True,
            )

            result["geometric_comparison"] = {
                "image_positions": geo_result["image_positions"],
                "magnifications": geo_result["magnifications"],
                "convergence_map": geo_result["convergence_map"],
            }

        return result

    def detect_fringes(
        self, amplitude_map: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray
    ) -> Dict:
        """
        Detect and characterize interference fringes in amplitude map.

        Parameters
        ----------
        amplitude_map : np.ndarray
            2D intensity map from wave optics calculation
        grid_x : np.ndarray
            x-coordinates in arcsec
        grid_y : np.ndarray
            y-coordinates in arcsec

        Returns
        -------
        fringe_info : dict
            Dictionary containing:
            - 'fringe_spacing': average spacing in arcsec
            - 'n_fringes': number of distinct fringes detected
            - 'fringe_contrast': (I_max - I_min) / (I_max + I_min)
        """
        # Compute radial profile
        dx = grid_x[1] - grid_x[0]
        center_idx = len(grid_x) // 2
        y_center = amplitude_map.shape[0] // 2

        # Extract radial profile along x-axis through center
        radial_profile = amplitude_map[y_center, :]
        r = grid_x

        # Find peaks in radial profile
        # Smooth slightly to avoid noise
        from scipy.signal import find_peaks

        smoothed = gaussian_filter(radial_profile, sigma=2.0)
        peaks, properties = find_peaks(smoothed, prominence=0.1 * np.max(smoothed))

        if len(peaks) > 1:
            # Compute average spacing between peaks
            peak_positions = r[peaks]
            spacings = np.diff(peak_positions)
            avg_spacing = np.mean(np.abs(spacings))
        else:
            avg_spacing = 0.0

        # Compute fringe contrast
        I_max = np.max(amplitude_map)
        I_min = np.min(amplitude_map)
        contrast = (I_max - I_min) / (I_max + I_min) if (I_max + I_min) > 0 else 0.0

        return {
            "fringe_spacing": avg_spacing,
            "n_fringes": len(peaks),
            "fringe_contrast": contrast,
        }

    def compare_with_geometric(
        self, wave_result: Dict, fractional_threshold: float = 0.01
    ) -> Dict:
        """
        Compare wave optics result with geometric optics.

        Parameters
        ----------
        wave_result : dict
            Result dictionary from compute_amplification_factor
        fractional_threshold : float, optional
            Threshold for significant difference (default: 0.01 = 1%)

        Returns
        -------
        comparison : dict
            Dictionary containing:
            - 'fractional_difference_map': 2D array of |wave - geo|/geo
            - 'max_difference': maximum fractional difference
            - 'mean_difference': mean fractional difference
            - 'significant_pixels': fraction of pixels with difference > threshold
        """
        if "geometric_comparison" not in wave_result:
            raise ValueError("Wave result must include geometric comparison")

        amplitude_map = wave_result["amplitude_map"]
        convergence_map = wave_result["geometric_comparison"]["convergence_map"]

        # Normalize both maps for comparison
        amp_norm = amplitude_map / np.sum(amplitude_map)
        conv_norm = convergence_map / np.sum(convergence_map)

        # Compute fractional difference
        # Avoid division by zero
        epsilon = 1e-10
        frac_diff = np.abs(amp_norm - conv_norm) / (conv_norm + epsilon)

        # Mask regions where both are very small (not meaningful)
        mask = (amp_norm > 0.01 * np.max(amp_norm)) | (
            conv_norm > 0.01 * np.max(conv_norm)
        )
        frac_diff_masked = frac_diff * mask

        max_diff = np.max(frac_diff_masked)
        mean_diff = np.mean(frac_diff_masked[mask]) if np.sum(mask) > 0 else 0.0
        significant = (
            np.sum(frac_diff_masked > fractional_threshold) / np.sum(mask)
            if np.sum(mask) > 0
            else 0.0
        )

        return {
            "fractional_difference_map": frac_diff_masked,
            "max_difference": max_diff,
            "mean_difference": mean_diff,
            "significant_pixels": significant,
        }

    def plot_interference_pattern(
        self,
        wave_result: Dict,
        figsize: Tuple[int, int] = (12, 10),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create publication-quality figure of wave optics results.

        Parameters
        ----------
        wave_result : dict
            Result from compute_amplification_factor
        figsize : tuple, optional
            Figure size in inches (default: (12, 10))
        save_path : str, optional
            Path to save figure (default: None, display only)

        Returns
        -------
        fig : matplotlib.Figure
            The created figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize, facecolor="#1a1a1a")
        fig.suptitle(
            f"Wave Optics: λ = {wave_result['wavelength']:.0f} nm",
            fontsize=16,
            color="white",
            y=0.98,
        )

        amplitude_map = wave_result["amplitude_map"]
        phase_map = wave_result["phase_map"]
        fermat_potential = wave_result["fermat_potential"]
        extent = wave_result["grid_extent"]
        extent_plot = [-extent, extent, -extent, extent]

        # 1. Amplitude (intensity) map
        ax1 = axes[0, 0]
        im1 = ax1.imshow(
            amplitude_map, extent=extent_plot, origin="lower", cmap="hot", aspect="auto"
        )
        ax1.set_xlabel("θ_x (arcsec)", color="white")
        ax1.set_ylabel("θ_y (arcsec)", color="white")
        ax1.set_title("Intensity |F|²", color="white", fontsize=12)
        ax1.tick_params(colors="white")
        ax1.set_facecolor("#0a0a0a")
        plt.colorbar(im1, ax=ax1, label="Normalized Intensity")

        # 2. Phase map
        ax2 = axes[0, 1]
        im2 = ax2.imshow(
            phase_map,
            extent=extent_plot,
            origin="lower",
            cmap="twilight",
            aspect="auto",
            vmin=-np.pi,
            vmax=np.pi,
        )
        ax2.set_xlabel("θ_x (arcsec)", color="white")
        ax2.set_ylabel("θ_y (arcsec)", color="white")
        ax2.set_title("Phase ∠F (radians)", color="white", fontsize=12)
        ax2.tick_params(colors="white")
        ax2.set_facecolor("#0a0a0a")
        plt.colorbar(im2, ax=ax2, label="Phase (rad)")

        # 3. Fermat potential
        ax3 = axes[1, 0]
        im3 = ax3.imshow(
            fermat_potential,
            extent=extent_plot,
            origin="lower",
            cmap="viridis",
            aspect="auto",
        )
        ax3.set_xlabel("θ_x (arcsec)", color="white")
        ax3.set_ylabel("θ_y (arcsec)", color="white")
        ax3.set_title("Fermat Potential Φ(θ)", color="white", fontsize=12)
        ax3.tick_params(colors="white")
        ax3.set_facecolor("#0a0a0a")
        plt.colorbar(im3, ax=ax3, label="Φ (arcsec²)")

        # 4. Radial profile showing fringes
        ax4 = axes[1, 1]
        y_center = amplitude_map.shape[0] // 2
        radial_profile = amplitude_map[y_center, :]
        x_coords = wave_result["grid_x"]

        ax4.plot(x_coords, radial_profile, color="#00ff41", linewidth=2)
        ax4.set_xlabel("θ_x (arcsec)", color="white")
        ax4.set_ylabel("Intensity", color="white")
        ax4.set_title("Radial Intensity Profile", color="white", fontsize=12)
        ax4.tick_params(colors="white")
        ax4.set_facecolor("#0a0a0a")
        ax4.grid(True, alpha=0.2, color="white")

        # Detect and annotate fringes
        fringe_info = self.detect_fringes(
            amplitude_map, x_coords, wave_result["grid_y"]
        )

        textstr = (
            f"Fringes detected: {fringe_info['n_fringes']}\n"
            f"Avg spacing: {fringe_info['fringe_spacing']:.3f} arcsec\n"
            f"Contrast: {fringe_info['fringe_contrast']:.3f}"
        )
        ax4.text(
            0.05,
            0.95,
            textstr,
            transform=ax4.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.5),
            color="white",
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, facecolor="#1a1a1a")
            print(f"Saved wave optics plot to {save_path}")

        return fig


def plot_wave_vs_geometric(
    lens_model,
    source_position: Tuple[float, float],
    wavelength: float = 500.0,
    grid_size: int = 512,
    grid_extent: float = 3.0,
    save_path: Optional[str] = None,
    lens_system=None,
) -> plt.Figure:
    """
    Create side-by-side comparison of wave vs geometric optics.

    Parameters
    ----------
    lens_model : MassProfile
        The lens model
    source_position : tuple
        Source position (x, y) in arcsec
    wavelength : float, optional
        Wavelength in nm (default: 500)
    grid_size : int, optional
        Grid size (default: 512)
    grid_extent : float, optional
        Grid extent in arcsec (default: 3.0)
    save_path : str, optional
        Path to save figure

    Returns
    -------
    fig : matplotlib.Figure
        The comparison figure
    """
    # Compute wave optics
    engine = WaveOpticsEngine()
    wave_result = engine.compute_amplification_factor(
        lens_model,
        source_position=source_position,
        wavelength=wavelength,
        grid_size=grid_size,
        grid_extent=grid_extent,
        lens_system=lens_system if lens_system else lens_model.lens_system,
        return_geometric=True,
    )

    # Get comparison
    comparison = engine.compare_with_geometric(wave_result)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), facecolor="#1a1a1a")
    fig.suptitle(
        f"Wave Optics vs Geometric Optics (λ = {wavelength:.0f} nm)",
        fontsize=16,
        color="white",
        y=0.98,
    )

    extent_plot = [-grid_extent, grid_extent, -grid_extent, grid_extent]

    # 1. Geometric optics (convergence map)
    ax1 = axes[0, 0]
    geo_map = wave_result["geometric_comparison"]["convergence_map"]
    im1 = ax1.imshow(
        geo_map, extent=extent_plot, origin="lower", cmap="hot", aspect="auto"
    )
    ax1.set_xlabel("θ_x (arcsec)", color="white")
    ax1.set_ylabel("θ_y (arcsec)", color="white")
    ax1.set_title("Geometric Optics (Ray Tracing)", color="white", fontsize=12)
    ax1.tick_params(colors="white")
    ax1.set_facecolor("#0a0a0a")
    plt.colorbar(im1, ax=ax1, label="Convergence κ")

    # Mark image positions
    img_pos = wave_result["geometric_comparison"]["image_positions"]
    if len(img_pos) > 0:
        ax1.plot(
            img_pos[:, 0],
            img_pos[:, 1],
            "c*",
            markersize=15,
            markeredgecolor="white",
            markeredgewidth=1.5,
            label="Images",
        )
    ax1.plot(
        source_position[0],
        source_position[1],
        "r*",
        markersize=15,
        markeredgecolor="white",
        markeredgewidth=1.5,
        label="Source",
    )
    ax1.legend(loc="upper right", fontsize=8)

    # 2. Wave optics (amplitude map)
    ax2 = axes[0, 1]
    wave_map = wave_result["amplitude_map"]
    im2 = ax2.imshow(
        wave_map, extent=extent_plot, origin="lower", cmap="hot", aspect="auto"
    )
    ax2.set_xlabel("θ_x (arcsec)", color="white")
    ax2.set_ylabel("θ_y (arcsec)", color="white")
    ax2.set_title("Wave Optics (Interference)", color="white", fontsize=12)
    ax2.tick_params(colors="white")
    ax2.set_facecolor("#0a0a0a")
    plt.colorbar(im2, ax=ax2, label="Intensity |F|²")

    # 3. Fractional difference map
    ax3 = axes[1, 0]
    diff_map = comparison["fractional_difference_map"]
    im3 = ax3.imshow(
        diff_map,
        extent=extent_plot,
        origin="lower",
        cmap="RdYlBu_r",
        aspect="auto",
        vmin=0,
        vmax=min(1.0, np.percentile(diff_map, 99)),
    )
    ax3.set_xlabel("θ_x (arcsec)", color="white")
    ax3.set_ylabel("θ_y (arcsec)", color="white")
    ax3.set_title("Fractional Difference |Wave - Geo|/Geo", color="white", fontsize=12)
    ax3.tick_params(colors="white")
    ax3.set_facecolor("#0a0a0a")
    plt.colorbar(im3, ax=ax3, label="Fractional Diff")

    # Highlight regions with >1% difference
    significant_mask = diff_map > 0.01
    if np.sum(significant_mask) > 0:
        ax3.contour(
            diff_map,
            levels=[0.01],
            colors="lime",
            linewidths=2,
            extent=extent_plot,
            origin="lower",
        )
        ax3.text(
            0.05,
            0.95,
            ">1% difference\n(green contour)",
            transform=ax3.transAxes,
            fontsize=9,
            verticalalignment="top",
            color="lime",
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.5),
        )

    # 4. Statistics and summary
    ax4 = axes[1, 1]
    ax4.axis("off")

    stats_text = f"""
    COMPARISON STATISTICS
    {"=" * 40}
    
    Wavelength: {wavelength:.1f} nm
    Grid size: {grid_size} × {grid_size}
    Grid extent: ±{grid_extent:.2f} arcsec
    
    Geometric Optics:
      Images found: {len(img_pos)}
      Total |μ|: {np.sum(np.abs(wave_result["geometric_comparison"]["magnifications"])):.3f}
    
    Wave Optics:
      Max difference: {comparison["max_difference"]:.3f}
      Mean difference: {comparison["mean_difference"]:.3f}
      Pixels >1% diff: {comparison["significant_pixels"] * 100:.1f}%
    
    Fringe Detection:
    """

    fringe_info = engine.detect_fringes(
        wave_map, wave_result["grid_x"], wave_result["grid_y"]
    )
    stats_text += f"""  N fringes: {fringe_info["n_fringes"]}
      Avg spacing: {fringe_info["fringe_spacing"]:.3f} arcsec
      Contrast: {fringe_info["fringe_contrast"]:.3f}
    """

    ax4.text(
        0.1,
        0.9,
        stats_text,
        transform=ax4.transAxes,
        fontsize=11,
        verticalalignment="top",
        family="monospace",
        color="white",
        bbox=dict(boxstyle="round", facecolor="#0a0a0a", alpha=0.8),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, facecolor="#1a1a1a")
        print(f"Saved comparison plot to {save_path}")

    return fig
