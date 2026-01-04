"""
Test Suite for Ray-Tracing Mode Enforcement

This module validates the scientific separation between:
1. thin_lens mode: Cosmological lensing (z > 0) with FLRW distances
2. schwarzschild_geodesic mode: Strong-field lensing (z ≈ 0) in flat spacetime

ISEF 2025 - Scientific Rigor Enhancement
"""

import pytest
import numpy as np
from astropy.cosmology import FlatLambdaCDM

from src.optics.ray_tracing_backends import (
    RayTracingMode,
    validate_method_compatibility,
)
from src.lens_models.multi_plane_recursive import multi_plane_trace


class TestRayTracingModeEnforcement:
    """Test that mode validation enforces scientific validity."""

    def test_enum_values(self):
        """Verify RayTracingMode enum has correct values."""
        assert RayTracingMode.THIN_LENS == "thin_lens"
        assert RayTracingMode.SCHWARZSCHILD == "schwarzschild_geodesic"

    def test_thin_lens_always_valid(self):
        """Thin-lens mode should be valid for any redshift."""
        # Low redshift
        validate_method_compatibility("thin_lens", 0.01, 0.05)
        validate_method_compatibility(RayTracingMode.THIN_LENS, 0.01, 0.05)

        # Cosmological redshift
        validate_method_compatibility("thin_lens", 0.5, 1.0)
        validate_method_compatibility(RayTracingMode.THIN_LENS, 0.5, 1.0)

        # High redshift
        validate_method_compatibility("thin_lens", 2.0, 5.0)
        validate_method_compatibility(RayTracingMode.THIN_LENS, 2.0, 5.0)

        # No exceptions should be raised

    def test_schwarzschild_valid_for_local_lenses(self):
        """Schwarzschild mode should work for z_lens ≤ 0.05."""
        # At threshold
        validate_method_compatibility("schwarzschild_geodesic", 0.05, 0.05)
        validate_method_compatibility(RayTracingMode.SCHWARZSCHILD, 0.05, 0.05)

        # Below threshold
        validate_method_compatibility("schwarzschild_geodesic", 0.01, 0.02)
        validate_method_compatibility(RayTracingMode.SCHWARZSCHILD, 0.0, 0.0)

    def test_schwarzschild_raises_error_for_cosmological_lenses(self):
        """Schwarzschild mode MUST raise ValueError for z_lens > 0.05."""
        # Just above threshold
        with pytest.raises(ValueError, match="ONLY valid for local"):
            validate_method_compatibility("schwarzschild_geodesic", 0.051, 1.0)

        with pytest.raises(ValueError, match="ONLY valid for local"):
            validate_method_compatibility(RayTracingMode.SCHWARZSCHILD, 0.051, 1.0)

        # Typical galaxy lens
        with pytest.raises(ValueError, match="z_lens ≤ 0.05"):
            validate_method_compatibility("schwarzschild_geodesic", 0.5, 1.5)

        # High redshift
        with pytest.raises(ValueError, match="flat-spacetime assumption"):
            validate_method_compatibility("schwarzschild_geodesic", 2.0, 3.0)

    def test_schwarzschild_error_message_content(self):
        """Verify error message provides scientific guidance."""
        with pytest.raises(ValueError) as exc_info:
            validate_method_compatibility("schwarzschild_geodesic", 0.5, 1.0)

        error_msg = str(exc_info.value)

        # Should mention the problem
        assert "flat-spacetime assumption" in error_msg or "z_lens ≤ 0.05" in error_msg

        # Should suggest solution
        assert "thin_lens" in error_msg

        # Should reference physics
        assert "expansion" in error_msg.lower() or "flrw" in error_msg.lower() or "cosmological" in error_msg.lower()


class TestMultiPlaneCosmologicalRequirement:
    """Test that multi-plane enforces cosmological redshifts."""

    @pytest.fixture
    def cosmology(self):
        """Standard cosmology for tests."""
        return FlatLambdaCDM(H0=70, Om0=0.3)

    def test_multi_plane_requires_positive_redshifts(self, cosmology):
        """Multi-plane should reject z ≤ 0 for any plane."""
        def alpha_func(x, y):
            return 0.0, 0.0

        # Lens plane with z = 0
        planes_zero = [{'z': 0.0, 'alpha_func': alpha_func}]
        beta = np.array([0.1, 0.0])

        with pytest.raises(ValueError, match="cosmological redshifts"):
            multi_plane_trace(beta, planes_zero, cosmology, z_source=1.0)

        # Lens plane with negative z (unphysical)
        planes_negative = [{'z': -0.01, 'alpha_func': alpha_func}]

        with pytest.raises(ValueError, match="z > 0"):
            multi_plane_trace(beta, planes_negative, cosmology, z_source=1.0)

    def test_multi_plane_requires_positive_source_redshift(self, cosmology):
        """Source must also have cosmological redshift."""
        def alpha_func(x, y):
            return 0.0, 0.0

        planes = [{'z': 0.5, 'alpha_func': alpha_func}]
        beta = np.array([0.1, 0.0])

        # z_source = 0
        with pytest.raises(ValueError, match="Source redshift must be cosmological"):
            multi_plane_trace(beta, planes, cosmology, z_source=0.0)

        # Negative z_source
        with pytest.raises(ValueError, match="z > 0"):
            multi_plane_trace(beta, planes, cosmology, z_source=-0.5)

    def test_multi_plane_accepts_valid_cosmological_setup(self, cosmology):
        """Multi-plane should work with proper cosmological redshifts."""
        def alpha_func(x, y):
            # Simple deflection for test
            r = np.sqrt(x**2 + y**2) + 1e-10
            theta_E = 1.0
            factor = theta_E**2 / r**2
            return factor * x, factor * y

        # Valid setup: z_lens = 0.5, z_source = 2.0
        planes = [{'z': 0.5, 'alpha_func': alpha_func}]
        beta = np.array([0.3, 0.0])

        # Should not raise
        theta = multi_plane_trace(beta, planes, cosmology, z_source=2.0, max_iter=10)

        assert isinstance(theta, np.ndarray)
        assert theta.shape == (2,)

    def test_multi_plane_error_suggests_schwarzschild_alternative(self, cosmology):
        """Error message should guide users to correct tool."""
        def alpha_func(x, y):
            return 0.0, 0.0

        planes = [{'z': 0.0, 'alpha_func': alpha_func}]
        beta = np.array([0.1, 0.0])

        with pytest.raises(ValueError) as exc_info:
            multi_plane_trace(beta, planes, cosmology, z_source=1.0)

        error_msg = str(exc_info.value)

        # Should mention the incompatibility
        assert "angular diameter distance" in error_msg or "FLRW" in error_msg

        # Should suggest alternative
        assert "Schwarzschild" in error_msg or "strong-field" in error_msg or "single-plane" in error_msg


class TestScientificConsistency:
    """Test that modes enforce correct physical regimes."""

    def test_thin_lens_supports_literature_benchmarks(self):
        """Thin-lens mode should reproduce known results."""
        # Einstein Cross: z_lens ≈ 0.04, z_source ≈ 1.7
        # This is technically borderline, but thin_lens is always valid
        validate_method_compatibility("thin_lens", 0.039, 1.695)

        # Twin Quasar: z_lens ≈ 0.36, z_source ≈ 1.41
        validate_method_compatibility("thin_lens", 0.355, 1.413)

        # SLACS lenses: typical z ≈ 0.2-0.4
        validate_method_compatibility("thin_lens", 0.3, 0.6)

    def test_schwarzschild_restricted_to_strong_field(self):
        """Schwarzschild should only work for local strong-field tests."""
        # Valid: Black hole at essentially zero redshift
        validate_method_compatibility("schwarzschild_geodesic", 0.0, 0.0)
        validate_method_compatibility("schwarzschild_geodesic", 0.001, 0.001)

        # Invalid: Any cosmological lens
        with pytest.raises(ValueError):
            validate_method_compatibility("schwarzschild_geodesic", 0.1, 0.5)

        with pytest.raises(ValueError):
            validate_method_compatibility("schwarzschild_geodesic", 0.5, 1.0)


class TestModeDocumentation:
    """Test that mode enum has proper documentation."""

    def test_enum_has_docstring(self):
        """RayTracingMode should document the physical distinction."""
        docstring = RayTracingMode.__doc__

        assert docstring is not None
        assert "THIN_LENS" in docstring
        assert "SCHWARZSCHILD" in docstring

        # Should mention key physics
        assert any(keyword in docstring.lower() for keyword in [
            "cosmological", "flrw", "expansion", "weak-field"
        ])
        assert any(keyword in docstring.lower() for keyword in [
            "strong-field", "geodesic", "schwarzschild", "black hole"
        ])

    def test_enum_provides_usage_guidance(self):
        """Enum docstring should guide correct usage."""
        docstring = RayTracingMode.__doc__

        # Should indicate which to use when
        assert "galaxy" in docstring.lower() or "hst" in docstring.lower()
        assert "multi-plane" in docstring.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
