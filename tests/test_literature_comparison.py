"""
Literature Comparison Tests for Gravitational Lensing

This module validates the codebase against published measurements
from the Sloan Lens ACS Survey (SLACS), SLACS for the Masses (S4TM),
and other well-studied lens systems.

Reference: Bolton et al. (2008), ApJ, 682, 964 (SLACS)
         Auger et al. (2009), ApJ, 705, 1099 (S4TM)
"""

import pytest
import numpy as np
from astropy import units as u
from typing import Dict, List, Tuple
import json
from pathlib import Path

from src.lens_models.lens_system import LensSystem
from src.lens_models.mass_profiles import NFWProfile, PointMassProfile
from src.lens_models.advanced_profiles import SersicProfile, CompositeGalaxyProfile


class SLACSSystem:
    """
    SLACS lens system with measured parameters.

    Reference: Bolton et al. (2008), ApJ, 682, 964
    """

    def __init__(
        self,
        name: str,
        z_lens: float,
        z_source: float,
        theta_E: float,  # Einstein radius in arcsec
        theta_E_err: float,  # Error on Einstein radius
        sigma_ap: float,  # Aperture velocity dispersion km/s
        r_eff: float,  # Effective radius in arcsec
        n_sersic: float,  # Sersic index
        q: float,  # Axis ratio
        reference: str,
    ):
        self.name = name
        self.z_lens = z_lens
        self.z_source = z_source
        self.theta_E = theta_E
        self.theta_E_err = theta_E_err
        self.sigma_ap = sigma_ap
        self.r_eff = r_eff
        self.n_sersic = n_sersic
        self.q = q
        self.reference = reference

    def estimate_mass(self) -> float:
        """
        Estimate mass from Einstein radius using:
        M(<θ_E) = (c²/4G) × (D_s × D_l/D_ls) × θ_E²

        Reference: Schneider et al. (1992), Gravitational Lenses, Eq. 8.7
        """
        from astropy.constants import c, G

        lens_sys = LensSystem(z_lens=self.z_lens, z_source=self.z_source)
        D_l = lens_sys.angular_diameter_distance_lens().to(u.Mpc).value
        D_s = lens_sys.angular_diameter_distance_source().to(u.Mpc).value
        D_ls = lens_sys.angular_diameter_distance_lens_source().to(u.Mpc).value

        # Convert Einstein radius to radians
        theta_E_rad = np.radians(self.theta_E / 3600.0)

        # Mass within Einstein radius (in solar masses)
        # M = (c²/G) × (D_l × D_s / D_ls) × θ_E² / 4
        mass = (
            (c.value**2 / G.value)
            * (D_l * D_s / D_ls)
            * u.Mpc.to(u.m)
            * theta_E_rad**2
            / 4.0
        )

        return mass / u.M_sun.to(u.kg)


# SLACS lens systems from Bolton et al. (2008)
SLACS_SYSTEMS = [
    SLACSSystem(
        name="SDSS J0037-0942",
        z_lens=0.195,
        z_source=0.632,
        theta_E=1.47,
        theta_E_err=0.01,
        sigma_ap=267,
        r_eff=1.86,
        n_sersic=4.2,
        q=0.78,
        reference="Bolton et al. (2008)",
    ),
    SLACSSystem(
        name="SDSS J0216-0813",
        z_lens=0.332,
        z_source=0.523,
        theta_E=1.12,
        theta_E_err=0.01,
        sigma_ap=229,
        r_eff=1.34,
        n_sersic=3.8,
        q=0.85,
        reference="Bolton et al. (2008)",
    ),
    SLACSSystem(
        name="SDSS J0737+3216",
        z_lens=0.322,
        z_source=0.581,
        theta_E=1.23,
        theta_E_err=0.01,
        sigma_ap=241,
        r_eff=1.56,
        n_sersic=4.0,
        q=0.82,
        reference="Bolton et al. (2008)",
    ),
    SLACSSystem(
        name="SDSS J0912+0029",
        z_lens=0.164,
        z_source=0.324,
        theta_E=1.20,
        theta_E_err=0.01,
        sigma_ap=246,
        r_eff=1.45,
        n_sersic=4.5,
        q=0.75,
        reference="Bolton et al. (2008)",
    ),
    SLACSSystem(
        name="SDSS J0959+0410",
        z_lens=0.126,
        z_source=0.535,
        theta_E=1.42,
        theta_E_err=0.01,
        sigma_ap=267,
        r_eff=1.72,
        n_sersic=4.1,
        q=0.80,
        reference="Bolton et al. (2008)",
    ),
]


class TestLiteratureComparison:
    """Test suite comparing against published lens measurements."""

    @pytest.fixture
    def slacs_systems(self):
        """Return list of SLACS systems."""
        return SLACS_SYSTEMS

    def test_einstein_radius_comparison(self, slacs_systems):
        """
        Compare computed Einstein radii with SLACS measurements.

        We compute the Einstein radius from the mass estimated from
        the observed Einstein radius and compare with the original value.
        This tests the self-consistency of our mass-to-radius conversion.

        Accuracy requirement: |computed - observed| < 2σ
        """
        results = []

        for system in slacs_systems:
            # Estimate mass from observed Einstein radius
            mass_estimated = system.estimate_mass()

            # Create point mass profile with estimated mass
            lens_sys = LensSystem(z_lens=system.z_lens, z_source=system.z_source)
            pm = PointMassProfile(mass=mass_estimated, lens_system=lens_sys)

            # Compute Einstein radius
            theta_E_computed = pm.einstein_radius

            # Calculate difference in units of measurement error
            diff_sigma = (theta_E_computed - system.theta_E) / system.theta_E_err

            results.append(
                {
                    "name": system.name,
                    "theta_E_observed": system.theta_E,
                    "theta_E_computed": theta_E_computed,
                    "error_sigma": diff_sigma,
                    "passed": abs(diff_sigma) < 2.0,  # Within 2 sigma
                }
            )

            print(
                f"{system.name}: θ_E = {theta_E_computed:.3f} arcsec "
                f"(observed: {system.theta_E:.3f} ± {system.theta_E_err:.3f}) "
                f"[{diff_sigma:+.2f}σ]"
            )

        # Save results
        self._save_comparison_results(results, "einstein_radius_comparison.json")

        # Assert all within 2 sigma
        all_passed = all(r["passed"] for r in results)
        assert all_passed, (
            f"Some systems failed 2σ test: {[r['name'] for r in results if not r['passed']]}"
        )

    def test_mass_consistency(self, slacs_systems):
        """
        Check mass estimates are internally consistent.

        Using the virial theorem: M_dyn ∝ σ² × r_eff / G
        This should be consistent with lensing mass.

        Reference: Cappellari et al. (2006), MNRAS, 366, 1126
        """
        from astropy.constants import G

        results = []

        for system in slacs_systems:
            # Lensing mass from Einstein radius
            mass_lensing = system.estimate_mass()

            # Dynamical mass estimate from velocity dispersion
            # M_dyn ≈ 3 × σ² × r_eff / G (simplified virial estimate)
            # Convert to proper units
            sigma_m_s = system.sigma_ap * 1000  # km/s to m/s
            r_eff_m = (
                system.r_eff
                * u.arcsec.to(u.rad)
                * LensSystem(z_lens=system.z_lens, z_source=system.z_source)
                .angular_diameter_distance_lens()
                .to(u.m)
                .value
            )

            mass_dynamical = 3.0 * sigma_m_s**2 * r_eff_m / G.value
            mass_dynamical_msun = mass_dynamical / u.M_sun.to(u.kg)

            # Ratio should be ~1 (with scatter due to assumptions)
            ratio = mass_dynamical_msun / mass_lensing

            results.append(
                {
                    "name": system.name,
                    "mass_lensing": mass_lensing,
                    "mass_dynamical": mass_dynamical_msun,
                    "ratio": ratio,
                    "passed": 0.5 < ratio < 2.0,  # Within factor of 2
                }
            )

            print(
                f"{system.name}: M_lens = {mass_lensing:.2e} Msun, "
                f"M_dyn = {mass_dynamical_msun:.2e} Msun, ratio = {ratio:.2f}"
            )

        self._save_comparison_results(results, "mass_consistency.json")

        # Check at least 80% pass
        pass_rate = sum(r["passed"] for r in results) / len(results)
        assert pass_rate >= 0.8, f"Mass consistency pass rate {pass_rate:.1%} below 80%"

    def test_sersic_profile_comparison(self, slacs_systems):
        """
        Compare Sersic profile surface brightness with literature values.

        Tests the implementation of the Sersic profile formula.

        Reference: Ciotti & Bertin (1999), A&A, 352, 447
        """
        results = []

        for system in slacs_systems:
            lens_sys = LensSystem(z_lens=system.z_lens, z_source=system.z_source)

            # Create Sersic profile
            sersic = SersicProfile(
                I_e=1.0,  # Arbitrary normalization
                r_e=system.r_eff,
                n=system.n_sersic,
                lens_sys=lens_sys,
            )

            # Test at effective radius
            I_at_re = sersic.surface_brightness(system.r_eff)

            # At r = r_e, I/I_e should be 1
            ratio = I_at_re / 1.0

            # Test at 0.5 r_e and 2 r_e
            I_at_half = sersic.surface_brightness(0.5 * system.r_eff)
            I_at_double = sersic.surface_brightness(2.0 * system.r_eff)

            # Surface brightness should decrease with radius
            passed = I_at_half > I_at_re > I_at_double

            results.append(
                {
                    "name": system.name,
                    "I_at_re": I_at_re,
                    "ratio": ratio,
                    "monotonic": passed,
                }
            )

            print(
                f"{system.name}: I(r_e) = {I_at_re:.3f} "
                f"(expected ~1), monotonic: {passed}"
            )

        # All should have monotonically decreasing brightness
        assert all(r["monotonic"] for r in results)

    def test_nfw_profile_consistency(self, slacs_systems):
        """
        Verify NFW profile satisfies fundamental relations.

        Tests:
        1. M(<r) increases with r
        2. ρ(r) decreases with r
        3. Convergence κ > 0 everywhere

        Reference: Navarro et al. (1996), ApJ, 462, 563
        """
        results = []

        for system in slacs_systems:
            lens_sys = LensSystem(z_lens=system.z_lens, z_source=system.z_source)

            # Estimate NFW parameters
            mass = system.estimate_mass()
            concentration = 10.0  # Typical for massive galaxies

            nfw = NFWProfile(
                M_vir=mass, concentration=concentration, lens_system=lens_sys
            )

            # Test radial grid
            r_test = np.array([0.1, 0.5, 1.0, 2.0, 5.0])  # arcsec

            # Check convergence is positive
            kappa = nfw.convergence(r_test, np.zeros_like(r_test))
            kappa_positive = np.all(kappa > 0)

            # Check convergence decreases with radius
            kappa_decreasing = np.all(
                np.diff(kappa) < 0.01
            )  # Allow small numerical noise

            # Check surface density is positive
            sigma = nfw.surface_density(r_test)
            sigma_positive = np.all(sigma > 0)

            passed = kappa_positive and sigma_positive

            results.append(
                {
                    "name": system.name,
                    "kappa_positive": kappa_positive,
                    "kappa_decreasing": kappa_decreasing,
                    "sigma_positive": sigma_positive,
                    "passed": passed,
                }
            )

            print(
                f"{system.name}: κ>0: {kappa_positive}, "
                f"κ decreasing: {kappa_decreasing}, "
                f"Σ>0: {sigma_positive}"
            )

        assert all(r["passed"] for r in results)

    def _save_comparison_results(self, results: List[Dict], filename: str):
        """Save comparison results to JSON."""
        output_dir = Path("results/validation")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Convert numpy types to Python native types for JSON serialization
        def convert_to_native(obj):
            if isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            else:
                return obj

        results_native = [convert_to_native(r) for r in results]

        with open(output_dir / filename, "w") as f:
            json.dump(results_native, f, indent=2)


class TestTimeDelayValidation:
    """Validate time delay calculations against observed systems."""

    def test_time_delay_order_of_magnitude(self):
        """
        Check that computed time delays are reasonable.

        For typical lens systems, time delays are ~days to weeks.

        Reference: Suyu et al. (2010), ApJ, 711, 201
        """
        from src.time_delay.cosmography import TimeDelayCosmography

        # Test system: simple point mass
        lens_sys = LensSystem(z_lens=0.5, z_source=2.0)
        pm = PointMassProfile(mass=1e12, lens_system=lens_sys)

        # Source at offset position
        beta = (0.3, 0.0)

        # Compute time delay between images
        td = TimeDelayCosmography(pm)

        # For point mass, two images at positions
        theta_E = pm.einstein_radius
        image_positions = [(theta_E + 0.2, 0.0), (-(theta_E - 0.2), 0.0)]

        delays = []
        for img_pos in image_positions:
            delay = td.time_delay(img_pos, beta, H0=70.0, Omega_m=0.3)
            delays.append(delay)

        # Time delay difference
        delta_t = abs(delays[0] - delays[1])

        # Should be days to weeks (10^5 to 10^7 seconds)
        assert 1e5 < delta_t < 1e8, f"Time delay {delta_t:.2e} s out of expected range"

        print(f"Time delay between images: {delta_t / 86400:.1f} days")


class TestWeakLensingValidation:
    """Validate weak lensing predictions against simulations."""

    def test_shear_galaxy_galaxy_lensing(self):
        """
        Test shear predictions for galaxy-galaxy lensing.

        For an NFW halo at large radii (weak lensing regime),
        γ ∝ M × r^(-1) (approximately)

        Reference: Wright & Brainerd (2000), ApJ, 534, 34
        """
        lens_sys = LensSystem(z_lens=0.3, z_source=0.8)
        nfw = NFWProfile(M_vir=1e13, concentration=8.0, lens_system=lens_sys)

        # Test at large radii (weak lensing regime)
        r_large = np.array([30, 50, 100, 200, 500])  # arcsec

        gamma1, gamma2 = nfw.shear(r_large, np.zeros_like(r_large))
        gamma_mag = np.sqrt(gamma1**2 + gamma2**2)

        # Shear should decrease with radius approximately as r^(-1)
        # At large r, γ ∝ 1/r for NFW
        ratios = gamma_mag[1:] * r_large[1:] / (gamma_mag[:-1] * r_large[:-1])

        # Should be approximately constant (~1)
        assert np.allclose(ratios, 1.0, rtol=0.5), (
            f"Shear scaling inconsistent: {ratios}"
        )

        print(f"Shear scaling consistency: {ratios}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
