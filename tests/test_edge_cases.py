"""
Edge Case and Physics Consistency Tests

Comprehensive tests for extreme parameters, boundary conditions,
and physical consistency checks.

Reference: Various - see individual test docstrings
"""

import pytest
import numpy as np
from astropy import units as u
import warnings

from src.lens_models.lens_system import LensSystem
from src.lens_models.mass_profiles import NFWProfile, PointMassProfile
from src.lens_models.advanced_profiles import (
    SersicProfile,
    EllipticalNFWProfile,
    CompositeGalaxyProfile,
)


class TestExtremeParameters:
    """Test edge cases with extreme physical parameters."""

    def test_very_low_mass_dwarf_galaxy(self):
        """
        Test NFW profile with very low mass (dwarf galaxy regime).

        M_vir ~ 1e10 Msun
        Reference: Wolf et al. (2010), MNRAS, 406, 1220
        """
        lens_sys = LensSystem(z_lens=0.1, z_source=0.5)

        # Dwarf galaxy mass
        nfw = NFWProfile(
            M_vir=1e10,  # 10 billion solar masses
            concentration=15.0,
            lens_system=lens_sys,
        )

        # Test at various radii
        r = np.array([0.1, 0.5, 1.0, 2.0])  # arcsec

        # Should not crash
        kappa = nfw.convergence(r, np.zeros_like(r))
        alpha_x, alpha_y = nfw.deflection_angle(r, np.zeros_like(r))
        psi = nfw.lensing_potential(r, np.zeros_like(r))

        # Physical checks
        assert np.all(kappa > 0), "Convergence should be positive"
        assert np.all(np.isfinite(alpha_x)), "Deflection should be finite"
        assert np.all(np.isfinite(psi)), "Potential should be finite"

        print(f"Dwarf galaxy: κ range = [{kappa.min():.4f}, {kappa.max():.4f}]")

    def test_very_high_mass_cluster(self):
        """
        Test NFW profile with very high mass (galaxy cluster regime).

        M_vir ~ 1e15 Msun
        Reference: Ragozzine et al. (2012), ApJ, 757, 1
        """
        lens_sys = LensSystem(z_lens=0.3, z_source=1.0)

        # Cluster mass
        nfw = NFWProfile(
            M_vir=1e15,  # 1 quadrillion solar masses
            concentration=4.0,  # Lower concentration for clusters
            lens_system=lens_sys,
        )

        # Test at larger radii for clusters
        r = np.array([10, 30, 60, 100])  # arcsec

        kappa = nfw.convergence(r, np.zeros_like(r))
        alpha_x, alpha_y = nfw.deflection_angle(r, np.zeros_like(r))

        assert np.all(kappa > 0), "Convergence should be positive"
        assert np.all(np.isfinite(alpha_x)), "Deflection should be finite"

        print(f"Cluster: Einstein radius = {nfw.einstein_radius:.2f} arcsec")

    def test_extreme_redshift_high_z(self):
        """
        Test with high redshift lens and source (z > 2).

        Tests cosmological distance calculations at high z.
        Reference: Hogg (1999), arXiv:astro-ph/9905116
        """
        # High redshift lens
        lens_sys = LensSystem(z_lens=2.0, z_source=6.0)

        pm = PointMassProfile(mass=1e12, lens_system=lens_sys)

        # Einstein radius should still be reasonable
        theta_E = pm.einstein_radius

        assert 0.1 < theta_E < 10.0, (
            f"Einstein radius {theta_E:.2f} out of reasonable range"
        )

        print(f"High-z system: θ_E = {theta_E:.3f} arcsec")

    def test_extreme_redshift_low_z(self):
        """
        Test with very low redshift (nearby lens).

        z_lens ~ 0.01 (nearby galaxy)
        """
        lens_sys = LensSystem(z_lens=0.01, z_source=0.1)

        pm = PointMassProfile(mass=1e11, lens_system=lens_sys)

        theta_E = pm.einstein_radius

        # At low z, Einstein radius should be large
        assert theta_E > 1.0, f"Low-z Einstein radius {theta_E:.2f} unexpectedly small"

        print(f"Low-z system: θ_E = {theta_E:.3f} arcsec")

    def test_extreme_ellipticity_disk_galaxy(self):
        """
        Test elliptical NFW with extreme ellipticity (disk-like).

        e = 0.9 corresponds to q = 0.05 (very flat)
        Reference: Kormendy (1977), ApJ, 218, 333
        """
        lens_sys = LensSystem(z_lens=0.3, z_source=1.0)

        # Very elliptical system
        ellip_nfw = EllipticalNFWProfile(
            M_vir=1e12,
            concentration=10.0,
            lens_sys=lens_sys,
            ellipticity=0.9,  # Very flat
        )

        # Test along major and minor axes
        r = np.array([1.0, 2.0, 3.0])

        # Along major axis (x-direction after rotation)
        kappa_major = ellip_nfw.convergence(r, np.zeros_like(r))

        # Along minor axis
        kappa_minor = ellip_nfw.convergence(np.zeros_like(r), r)

        # Major axis should have higher convergence (more mass)
        assert np.mean(kappa_major) > np.mean(kappa_minor), (
            "Major axis should have higher convergence"
        )

        print(
            f"Ellipticity e=0.9: κ_major/κ_minor = {np.mean(kappa_major) / np.mean(kappa_minor):.2f}"
        )

    def test_extreme_sersic_index(self):
        """
        Test Sersic profile with extreme indices.

        n = 0.5 (exponential disk) and n = 10 (very concentrated)
        Reference: Graham & Driver (2005), PASA, 22, 118
        """
        lens_sys = LensSystem(z_lens=0.3, z_source=1.0)

        # Exponential disk (n=1)
        sersic_exp = SersicProfile(I_e=1.0, r_e=5.0, n=1.0, lens_sys=lens_sys)

        # Very concentrated (n=10)
        sersic_conc = SersicProfile(I_e=1.0, r_e=5.0, n=10.0, lens_sys=lens_sys)

        # Test surface brightness profiles
        r = np.array([0.1, 1.0, 5.0, 10.0, 20.0])

        I_exp = sersic_exp.surface_brightness(r)
        I_conc = sersic_conc.surface_brightness(r)

        # Both should be monotonically decreasing
        assert np.all(np.diff(I_exp) < 0.01), (
            "Exponential profile should decrease monotonically"
        )
        assert np.all(np.diff(I_conc) < 0.01), (
            "Concentrated profile should decrease monotonically"
        )

        # Concentrated profile should be more peaked at center
        assert I_conc[0] / I_conc[2] > I_exp[0] / I_exp[2], (
            "Concentrated profile should have steeper central gradient"
        )

        print(f"n=1: I(0.1)/I(5) = {I_exp[0] / I_exp[2]:.2f}")
        print(f"n=10: I(0.1)/I(5) = {I_conc[0] / I_conc[2]:.2f}")


class TestPhysicalConsistency:
    """Test that physical laws are satisfied."""

    def test_deflection_toward_mass_center(self):
        """
        Deflection should always point toward the lens center.

        For a source at (x, y), the deflection angle should have
        the opposite sign of the position vector.
        """
        lens_sys = LensSystem(z_lens=0.5, z_source=2.0)
        pm = PointMassProfile(mass=1e12, lens_system=lens_sys)

        # Test at various positions
        positions = [
            (1.0, 0.0),
            (0.0, 1.0),
            (-1.0, 0.0),
            (0.0, -1.0),
            (1.0, 1.0),
            (-0.5, 0.8),
        ]

        for x, y in positions:
            alpha_x, alpha_y = pm.deflection_angle(x, y)

            # Deflection should be opposite to position
            # i.e., alpha_x * x < 0 and alpha_y * y < 0
            if abs(x) > 1e-10:
                assert alpha_x * x < 0, (
                    f"Deflection x-component should point toward center at ({x}, {y})"
                )
            if abs(y) > 1e-10:
                assert alpha_y * y < 0, (
                    f"Deflection y-component should point toward center at ({x}, {y})"
                )

    def test_mass_conservation_surface_density(self):
        """
        Mass from surface density integration should equal input mass.

        M = 2π ∫₀^∞ Σ(r) r dr
        Reference: Schneider et al. (1992), Gravitational Lenses, Eq. 8.3
        """
        lens_sys = LensSystem(z_lens=0.5, z_source=2.0)
        M_input = 1e12

        pm = PointMassProfile(mass=M_input, lens_system=lens_sys)

        # Integrate surface density
        from scipy.integrate import quad

        def integrand(r):
            sigma = pm.surface_density(r)
            return 2 * np.pi * r * sigma

        # Integrate out to large radius
        M_integrated, _ = quad(integrand, 0, 1000)

        # Should be close to input mass (within 1%)
        assert abs(M_integrated - M_input) / M_input < 0.01, (
            f"Mass conservation violated: {M_integrated:.2e} vs {M_input:.2e}"
        )

        print(f"Mass conservation: {M_integrated:.2e} vs {M_input:.2e} Msun")

    def test_convergence_positive(self):
        """
        Convergence κ must be non-negative everywhere.

        κ = Σ / Σ_crit, where both are positive.
        """
        lens_sys = LensSystem(z_lens=0.5, z_source=2.0)

        profiles = [
            PointMassProfile(mass=1e12, lens_system=lens_sys),
            NFWProfile(M_vir=1e12, concentration=10.0, lens_system=lens_sys),
        ]

        r_test = np.linspace(0.1, 10, 100)

        for profile in profiles:
            kappa = profile.convergence(r_test, np.zeros_like(r_test))

            assert np.all(kappa >= 0), (
                f"{profile.__class__.__name__}: Convergence must be non-negative"
            )
            assert np.all(np.isfinite(kappa)), (
                f"{profile.__class__.__name__}: Convergence must be finite"
            )

    def test_lensing_potential_monotonicity(self):
        """
        Lensing potential ψ should increase with radius.

        Since dψ/dr = α > 0 (deflection is outward), ψ should increase.
        """
        lens_sys = LensSystem(z_lens=0.5, z_source=2.0)
        pm = PointMassProfile(mass=1e12, lens_system=lens_sys)

        r_test = np.array([0.1, 0.5, 1.0, 2.0, 5.0])

        psi = pm.lensing_potential(r_test, np.zeros_like(r_test))

        # Potential should increase with radius
        assert np.all(np.diff(psi) > -0.01), (
            "Potential should increase monotonically with radius"
        )

    def test_poisson_equation_nfw(self):
        """
        Verify Poisson equation: ∇²ψ = 2κ

        For NFW profile, check consistency numerically.
        """
        lens_sys = LensSystem(z_lens=0.5, z_source=2.0)
        nfw = NFWProfile(M_vir=1e12, concentration=10.0, lens_system=lens_sys)

        # Test radius
        r = 1.0

        # Compute Laplacian numerically
        eps = 1e-4

        psi_center = nfw.lensing_potential(r, 0.0)
        psi_x_plus = nfw.lensing_potential(r + eps, 0.0)
        psi_x_minus = nfw.lensing_potential(r - eps, 0.0)
        psi_y_plus = nfw.lensing_potential(r, eps)
        psi_y_minus = nfw.lensing_potential(r, -eps)

        # Laplacian = d²ψ/dx² + d²ψ/dy²
        d2psi_dx2 = (psi_x_plus - 2 * psi_center + psi_x_minus) / eps**2
        d2psi_dy2 = (psi_y_plus - 2 * psi_center + psi_y_minus) / eps**2
        laplacian = d2psi_dx2 + d2psi_dy2

        # Poisson equation: ∇²ψ = 2κ
        kappa = nfw.convergence(r, 0.0)

        # Should be approximately equal (within 5%)
        ratio = laplacian / (2 * kappa)

        assert 0.95 < ratio < 1.05, f"Poisson equation violated: ∇²ψ/2κ = {ratio:.3f}"

        print(
            f"Poisson check: ∇²ψ = {laplacian:.4f}, 2κ = {2 * kappa:.4f}, ratio = {ratio:.3f}"
        )


class TestNumericalStability:
    """Test numerical stability at boundaries."""

    def test_origin_behavior(self):
        """
        Test behavior at r = 0.

        - Deflection should be 0
        - Convergence should be finite
        - Potential should be finite
        """
        lens_sys = LensSystem(z_lens=0.5, z_source=2.0)
        pm = PointMassProfile(mass=1e12, lens_system=lens_sys)
        nfw = NFWProfile(M_vir=1e12, concentration=10.0, lens_system=lens_sys)

        # Very close to origin
        r = 1e-6

        for profile in [pm, nfw]:
            alpha_x, alpha_y = profile.deflection_angle(r, 0.0)
            kappa = profile.convergence(r, 0.0)
            psi = profile.lensing_potential(r, 0.0)

            # Deflection should be near zero
            assert abs(alpha_x) < 1e-3, (
                f"{profile.__class__.__name__}: Deflection at origin should be ~0"
            )

            # Other quantities should be finite
            assert np.isfinite(kappa), (
                f"{profile.__class__.__name__}: Convergence at origin should be finite"
            )
            assert np.isfinite(psi), (
                f"{profile.__class__.__name__}: Potential at origin should be finite"
            )

    def test_large_radius_behavior(self):
        """
        Test behavior at very large radii.

        - Deflection should approach 0
        - Convergence should approach 0
        """
        lens_sys = LensSystem(z_lens=0.5, z_source=2.0)
        nfw = NFWProfile(M_vir=1e12, concentration=10.0, lens_system=lens_sys)

        # Very large radius
        r = 1000.0  # arcsec

        alpha_x, alpha_y = nfw.deflection_angle(r, 0.0)
        kappa = nfw.convergence(r, 0.0)

        # Should be small
        assert abs(alpha_x) < 1.0, "Deflection at large radius should be small"
        assert kappa < 0.1, "Convergence at large radius should be small"

    def test_numerical_derivatives(self):
        """
        Test that numerical derivatives are stable.

        Compute deflection from potential gradient and compare.
        """
        lens_sys = LensSystem(z_lens=0.5, z_source=2.0)
        pm = PointMassProfile(mass=1e12, lens_system=lens_sys)

        r = 1.0
        eps = 1e-6

        # Direct deflection
        alpha_x_direct, alpha_y_direct = pm.deflection_angle(r, 0.0)

        # From potential gradient
        psi_plus_x = pm.lensing_potential(r + eps, 0.0)
        psi_minus_x = pm.lensing_potential(r - eps, 0.0)
        alpha_x_grad = (psi_plus_x - psi_minus_x) / (2 * eps)

        # Should match
        diff = abs(alpha_x_direct - alpha_x_grad)
        assert diff < 1e-3, f"Numerical derivative mismatch: {diff:.6f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
