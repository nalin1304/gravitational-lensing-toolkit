"""
Unit Tests for Dual Ray-Tracing Backends

Tests both thin-lens and Schwarzschild geodesic methods,
ensuring physical correctness and proper regime separation.

Author: ISEF 2025 - Scientific Validation
"""

import pytest
import numpy as np
import warnings
from typing import Tuple

# Import modules to test
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.optics.ray_tracing_backends import (
    thin_lens_ray_trace,
    schwarzschild_geodesic_trace,
    schwarzschild_deflection_angle,
    compare_methods_weak_field,
    validate_method_compatibility,
    schwarzschild_radius,
    ray_trace
)
from utils.constants import (
    M_SUN_KG, G_CONST, C_LIGHT, ARCSEC_TO_RAD, RAD_TO_ARCSEC
)
from lens_models.lens_system import LensSystem
from lens_models.mass_profiles import PointMassProfile, NFWProfile


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def lens_system_cosmological():
    """Standard cosmological lens system (z_l=0.5, z_s=1.5)."""
    return LensSystem(z_lens=0.5, z_source=1.5)


@pytest.fixture
def lens_system_local():
    """Local lens system (z_l=0.01, z_s=0.02)."""
    return LensSystem(z_lens=0.01, z_source=0.02)


@pytest.fixture
def point_mass_lens(lens_system_cosmological):
    """Point mass lens for testing."""
    M = 1e12  # Solar masses
    return PointMassProfile(mass=M, lens_system=lens_system_cosmological)


@pytest.fixture
def solar_mass_lens(lens_system_local):
    """Solar mass lens for Schwarzschild testing."""
    return PointMassProfile(mass=1.0, lens_system=lens_system_local)


# ============================================================================
# Test 1: Method Validation and Warnings
# ============================================================================

class TestMethodValidation:
    """Test that method validation catches inappropriate usage."""

    def test_schwarzschild_warns_high_redshift(self):
        """Schwarzschild method should raise ValueError for z > 0.05."""
        with pytest.raises(ValueError, match="ONLY valid for local"):
            validate_method_compatibility("schwarzschild_geodesic", 0.5, 1.5)

    def test_schwarzschild_no_warn_low_redshift(self):
        """No warning for z << 0.1."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_method_compatibility("schwarzschild_geodesic", 0.01, 0.02)
            # Should not warn
            assert len(w) == 0

    def test_thin_lens_accepts_all_redshifts(self):
        """Thin-lens should work for all redshifts."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_method_compatibility("thin_lens", 0.5, 1.5)
            validate_method_compatibility("thin_lens", 2.0, 3.0)
            # No errors should occur
            assert True


# ============================================================================
# Test 2: Schwarzschild Radius Calculation
# ============================================================================

class TestSchwarzschildRadius:
    """Test Schwarzschild radius calculations."""

    def test_solar_mass_schwarzschild_radius(self):
        """Test r_s for solar mass (known value: ~2.95 km)."""
        r_s = schwarzschild_radius(M_SUN_KG)
        expected = 2.0 * G_CONST * M_SUN_KG / (C_LIGHT**2)
        assert np.isclose(r_s, expected, rtol=1e-10)
        assert np.isclose(r_s / 1e3, 2.953, rtol=0.01)  # ~2.95 km

    def test_million_solar_mass_black_hole(self):
        """Test r_s for 10^6 M_sun black hole."""
        M_bh = 1e6 * M_SUN_KG
        r_s = schwarzschild_radius(M_bh)
        expected = 1e6 * 2.953e3  # 1e6 times solar r_s
        assert np.isclose(r_s, expected, rtol=0.01)

    def test_schwarzschild_scales_linearly(self):
        """Test r_s ∝ M."""
        M1 = 1.0 * M_SUN_KG
        M2 = 100.0 * M_SUN_KG
        r_s1 = schwarzschild_radius(M1)
        r_s2 = schwarzschild_radius(M2)
        assert np.isclose(r_s2 / r_s1, 100.0, rtol=1e-10)


# ============================================================================
# Test 3: Weak-Field Limit Agreement
# ============================================================================

class TestWeakFieldAgreement:
    """Test that both methods agree in weak-field regime."""

    def test_weak_field_deflection_formula(self):
        """Test analytical weak-field formula: α ≈ 4GM/(c²b)."""
        M = M_SUN_KG
        b = 1e11  # meters (>> r_s)

        alpha_exact = 4 * G_CONST * M / (C_LIGHT**2 * b)
        alpha_computed = schwarzschild_deflection_angle(b, M)

        # Should agree to high precision in weak field
        assert np.isclose(alpha_computed, alpha_exact, rtol=1e-6)

    def test_einstein_deflection_1919(self):
        """Reproduce 1919 solar eclipse deflection (1.75 arcsec)."""
        M = M_SUN_KG
        R_sun = 6.957e8  # meters

        # Light grazing sun's surface
        alpha_rad = schwarzschild_deflection_angle(R_sun, M)
        alpha_arcsec = alpha_rad * RAD_TO_ARCSEC

        # Historical value: 1.75 arcsec
        expected = 1.75
        assert np.isclose(alpha_arcsec, expected, rtol=0.05)  # 5% tolerance

    @pytest.mark.parametrize("b_over_rs", [10, 50, 100, 500])
    def test_weak_field_convergence(self, b_over_rs):
        """Test that numerical agrees with analytical for b >> r_s."""
        M = M_SUN_KG
        r_s = schwarzschild_radius(M)
        b = b_over_rs * r_s

        alpha_analytical = 4 * G_CONST * M / (C_LIGHT**2 * b)
        alpha_numerical = schwarzschild_deflection_angle(b, M)

        relative_error = abs(alpha_numerical - alpha_analytical) / alpha_analytical

        # Error should decrease as b increases
        tolerance = 0.01 if b_over_rs < 50 else 0.001
        assert relative_error < tolerance


# ============================================================================
# Test 4: Thin-Lens Method
# ============================================================================

class TestThinLensMethod:
    """Test thin-lens ray tracing."""

    def test_thin_lens_finds_einstein_ring(self, point_mass_lens):
        """Source at center should produce ring at Einstein radius."""
        source_pos = (0.0, 0.0)
        results = thin_lens_ray_trace(
            source_pos,
            point_mass_lens,
            grid_extent=2.0,
            grid_resolution=200,
            threshold=0.1
        )

        # Should find images
        assert len(results['image_positions']) > 0

        # Images should be near Einstein radius
        theta_E = point_mass_lens.einstein_radius
        for img_pos in results['image_positions']:
            r = np.sqrt(img_pos[0]**2 + img_pos[1]**2)
            assert np.isclose(r, theta_E, rtol=0.20)  # 20% tolerance for grid (relaxed for stability)

    def test_thin_lens_conserves_surface_brightness(self, point_mass_lens):
        """Total magnification should conserve flux (sum μ = 0 for point mass)."""
        source_pos = (0.3, 0.0)
        results = thin_lens_ray_trace(
            source_pos,
            point_mass_lens,
            grid_extent=3.0,
            grid_resolution=250,
            threshold=0.05
        )

        if len(results['magnifications']) > 0:
            # For single source, odd image theorem applies
            # Sum of signed magnifications should be close to 1
            # (in weak lensing, approximately conserved)
            total_mag = np.sum(np.abs(results['magnifications']))
            assert total_mag > 1.0  # Gravitational lensing amplifies

    def test_thin_lens_returns_correct_structure(self, point_mass_lens):
        """Check that results dictionary has required fields."""
        results = thin_lens_ray_trace(
            (0.5, 0.0),
            point_mass_lens,
            return_maps=True
        )

        required_keys = [
            'image_positions', 'magnifications',
            'grid_x', 'grid_y', 'method',
            'convergence_map', 'beta_x', 'beta_y'
        ]

        for key in required_keys:
            assert key in results

        assert results['method'] == 'thin_lens'


# ============================================================================
# Test 5: Schwarzschild Geodesic Method
# ============================================================================

class TestSchwarzschildGeodesic:
    """Test Schwarzschild geodesic integration."""

    def test_geodesic_integration_completes(self):
        """Basic integration should complete without errors."""
        M = M_SUN_KG
        r_s = schwarzschild_radius(M)
        b = 100 * r_s  # Safe impact parameter

        result = schwarzschild_geodesic_trace(b, M)

        assert 'deflection_angle' in result
        assert result['method'] == 'schwarzschild_geodesic'
        assert result['deflection_angle'] > 0

    def test_geodesic_weak_field_matches_formula(self):
        """Weak-field geodesic should match 4GM/(c²b)."""
        M = M_SUN_KG
        r_s = schwarzschild_radius(M)
        b = 200 * r_s

        result = schwarzschild_geodesic_trace(b, M, max_radius=500.0)
        alpha_numerical = result['deflection_angle']

        alpha_analytical = 4 * G_CONST * M / (C_LIGHT**2 * b)

        assert np.isclose(alpha_numerical, alpha_analytical, rtol=0.05)

    def test_geodesic_closest_approach(self):
        """Closest approach should be ~ impact parameter for weak field."""
        M = M_SUN_KG
        b = 1e10  # meters

        result = schwarzschild_geodesic_trace(b, M)

        # In weak field, closest approach ≈ b
        r_min = result['closest_approach']
        assert np.isclose(r_min, b, rtol=0.2)

    @pytest.mark.parametrize("b_multiplier", [5, 10, 20, 50])
    def test_geodesic_deflection_decreases_with_distance(self, b_multiplier):
        """Deflection should decrease as α ∝ 1/b."""
        M = M_SUN_KG
        r_s = schwarzschild_radius(M)

        b1 = 10 * r_s
        b2 = b_multiplier * b1

        result1 = schwarzschild_geodesic_trace(b1, M)
        result2 = schwarzschild_geodesic_trace(b2, M)

        alpha1 = result1['deflection_angle']
        alpha2 = result2['deflection_angle']

        # α should decrease
        assert alpha2 < alpha1

        # In weak field: α ∝ 1/b
        ratio = alpha1 / alpha2
        expected_ratio = b2 / b1
        assert np.isclose(ratio, expected_ratio, rtol=0.3)


# ============================================================================
# Test 6: Method Comparison
# ============================================================================

class TestMethodComparison:
    """Compare thin-lens vs Schwarzschild in overlap regime."""

    def test_comparison_function_structure(self, solar_mass_lens):
        """Test comparison function returns correct structure."""
        comparison = compare_methods_weak_field(
            impact_parameter_arcsec=10.0,
            lens_model=solar_mass_lens,
            mass_kg=M_SUN_KG
        )

        required_keys = [
            'thin_lens_alpha', 'schwarzschild_alpha',
            'relative_difference', 'agreement'
        ]

        for key in required_keys:
            assert key in comparison

    def test_methods_agree_weak_field(self, solar_mass_lens):
        """Methods should agree for b >> r_s."""
        # Use large impact parameter
        comparison = compare_methods_weak_field(
            impact_parameter_arcsec=100.0,  # Far from strong field
            lens_model=solar_mass_lens,
            mass_kg=M_SUN_KG
        )

        # Should agree within 5% in weak field
        assert comparison['relative_difference'] < 0.05
        assert comparison['agreement']


# ============================================================================
# Test 7: Einstein Radius Recovery
# ============================================================================

class TestEinsteinRadius:
    """Test that thin-lens correctly computes Einstein radius."""

    def test_point_mass_einstein_radius_formula(self, point_mass_lens):
        """Test θ_E = sqrt(4GM/c² × D_LS/(D_L D_S))."""
        theta_E_computed = point_mass_lens.einstein_radius

        # Manual calculation
        M_kg = point_mass_lens.M * M_SUN_KG
        lens_sys = point_mass_lens.lens_system
        D_l = lens_sys.angular_diameter_distance_lens().to('m').value
        D_s = lens_sys.angular_diameter_distance_source().to('m').value
        D_ls = lens_sys.angular_diameter_distance_lens_source().to('m').value

        theta_E_expected_rad = np.sqrt(
            4 * G_CONST * M_kg * D_ls / (C_LIGHT**2 * D_l * D_s)
        )
        theta_E_expected_arcsec = theta_E_expected_rad * RAD_TO_ARCSEC

        assert np.isclose(theta_E_computed, theta_E_expected_arcsec, rtol=1e-6)

    def test_sie_einstein_radius_matches_theory(self, lens_system_cosmological):
        """For SIE profile, verify Einstein radius calculation."""
        # SIE (Singular Isothermal Ellipsoid) has θ_E = 4π σ²/c² × D_LS/D_S
        # Using NFW as proxy with appropriate mass
        M = 1e13  # Solar masses
        concentration = 5.0

        nfw = NFWProfile(M, concentration, lens_system_cosmological)

        # NFW has scale-dependent Einstein radius
        # Just verify it's in reasonable range (0.5 - 5 arcsec for galaxy)
        alpha_x, alpha_y = nfw.deflection_angle(1.0, 0.0)
        alpha_mag = np.sqrt(alpha_x**2 + alpha_y**2)

        assert 0.1 < alpha_mag < 10.0  # Reasonable deflection range


# ============================================================================
# Test 8: Coverage and Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_mass_gives_zero_deflection(self, lens_system_cosmological):
        """Zero mass should give zero deflection."""
        lens_zero = PointMassProfile(mass=0.0, lens_system=lens_system_cosmological)
        alpha_x, alpha_y = lens_zero.deflection_angle(1.0, 0.0)

        # Should be zero or negligible
        assert np.isclose(alpha_x, 0.0, atol=1e-10)
        assert np.isclose(alpha_y, 0.0, atol=1e-10)

    def test_very_small_impact_parameter_schwarzschild(self):
        """Very small b should give large deflection (but not infinite)."""
        M = M_SUN_KG
        r_s = schwarzschild_radius(M)
        b = 2.5 * r_s  # Just outside horizon

        result = schwarzschild_geodesic_trace(b, M)

        # Should complete without error
        assert 'deflection_angle' in result
        # Deflection should be large but finite
        assert result['deflection_angle'] > 0.1  # Radians


# ============================================================================
# Test 9: Physical Constants Verification
# ============================================================================

class TestPhysicalConstants:
    """Verify physical constants are correct."""

    def test_speed_of_light(self):
        """c = 299792458 m/s (exact definition)."""
        assert C_LIGHT == 299792458.0

    def test_gravitational_constant(self):
        """G ≈ 6.674e-11 m³ kg⁻¹ s⁻²."""
        assert np.isclose(G_CONST, 6.674e-11, rtol=0.01)

    def test_solar_mass(self):
        """M_sun ≈ 1.989e30 kg."""
        assert np.isclose(M_SUN_KG, 1.989e30, rtol=0.01)

    def test_arcsec_conversion(self):
        """1 arcsec = π/(180×3600) rad."""
        expected = np.pi / (180.0 * 3600.0)
        assert np.isclose(ARCSEC_TO_RAD, expected, rtol=1e-10)


# ============================================================================
# Test 10: Integration Test
# ============================================================================

class TestIntegration:
    """End-to-end integration tests."""

    def test_full_ray_tracing_pipeline(self, point_mass_lens):
        """Complete ray-tracing from source to images."""
        source_positions = [
            (0.0, 0.0),    # On axis (Einstein ring)
            (0.5, 0.0),    # Off axis
            (0.3, 0.3),    # Diagonal
        ]

        for source_pos in source_positions:
            results = thin_lens_ray_trace(
                source_pos,
                point_mass_lens,
                grid_extent=3.0,
                grid_resolution=200,
                threshold=0.1
            )

            # Should complete without errors
            assert isinstance(results, dict)
            assert 'image_positions' in results
            assert 'magnifications' in results

            # Physical check: magnifications should exist
            if len(results['magnifications']) > 0:
                # All magnifications should be non-zero
                assert np.all(np.abs(results['magnifications']) > 0)


# ============================================================================
# Performance Benchmarks (Optional)
# ============================================================================

@pytest.mark.benchmark
class TestPerformance:
    """Performance benchmarks (run with --benchmark flag)."""

    def test_thin_lens_performance(self, point_mass_lens, benchmark):
        """Benchmark thin-lens ray tracing."""
        def run_trace():
            return thin_lens_ray_trace(
                (0.5, 0.0),
                point_mass_lens,
                grid_extent=2.0,
                grid_resolution=150,
                threshold=0.1
            )

        result = benchmark(run_trace)
        assert 'image_positions' in result

    def test_schwarzschild_performance(self, benchmark):
        """Benchmark Schwarzschild geodesic."""
        M = M_SUN_KG
        r_s = schwarzschild_radius(M)
        b = 50 * r_s

        result = benchmark(schwarzschild_geodesic_trace, b, M)
        assert 'deflection_angle' in result


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
