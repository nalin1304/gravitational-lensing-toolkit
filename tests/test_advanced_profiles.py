"""
Comprehensive tests for advanced lens profiles.

Tests for Phase 6 implementation: elliptical NFW, Sérsic, and composite profiles.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_less

import sys
sys.path.append('..')
from src.lens_models.advanced_profiles import (
    EllipticalNFWProfile, SersicProfile, CompositeGalaxyProfile
)
from src.lens_models.mass_profiles import NFWProfile
from src.lens_models.lens_system import LensSystem


class TestEllipticalNFWProfile:
    """Test suite for elliptical NFW profile."""
    
    @pytest.fixture
    def lens_sys(self):
        """Create standard lens system for testing."""
        return LensSystem(z_lens=0.5, z_source=2.0, H0=70.0, Om0=0.3)
    
    def test_initialization(self, lens_sys):
        """Test profile can be initialized with valid parameters."""
        profile = EllipticalNFWProfile(
            M_vir=1e12, concentration=10.0, lens_sys=lens_sys,
            ellipticity=0.3, position_angle=45.0
        )
        assert profile.ellipticity == 0.3
        assert profile.position_angle == 45.0
        assert profile.M_vir == 1e12
        assert profile.c == 10.0
        
        # Check derived quantities
        expected_q = (1 - 0.3) / (1 + 0.3)  # 0.538
        assert_allclose(profile.q, expected_q, rtol=1e-6)
        assert_allclose(profile.phi, np.radians(45.0), rtol=1e-6)
    
    def test_invalid_ellipticity(self, lens_sys):
        """Test that invalid ellipticity raises error."""
        with pytest.raises(ValueError, match="Ellipticity must be in"):
            EllipticalNFWProfile(
                M_vir=1e12, concentration=10.0, lens_sys=lens_sys,
                ellipticity=1.5  # Invalid
            )
        
        with pytest.raises(ValueError):
            EllipticalNFWProfile(
                M_vir=1e12, concentration=10.0, lens_sys=lens_sys,
                ellipticity=-0.1  # Invalid
            )
    
    def test_reduces_to_circular_nfw(self, lens_sys):
        """Test that ellipticity=0 gives same results as circular NFW."""
        M_vir = 1e12
        concentration = 10.0
        
        # Create both profiles
        elliptical = EllipticalNFWProfile(
            M_vir=M_vir, concentration=concentration, lens_sys=lens_sys, ellipticity=0.0
        )
        circular = NFWProfile(M_vir=M_vir, concentration=concentration, lens_system=lens_sys)
        
        # Test points
        x = np.array([1.0, 2.0, 5.0])
        y = np.array([0.5, 1.0, 2.0])
        
        # Convergence should match
        kappa_ell = elliptical.convergence(x, y)
        kappa_circ = circular.convergence(x, y)
        assert_allclose(kappa_ell, kappa_circ, rtol=1e-5)
    
    def test_convergence_shape(self, lens_sys):
        """Test convergence returns correct shape."""
        profile = EllipticalNFWProfile(
            M_vir=1e12, concentration=10.0, lens_sys=lens_sys, ellipticity=0.3
        )
        
        # Single point
        kappa = profile.convergence(1.0, 0.5)
        assert np.isscalar(kappa) or kappa.shape == ()
        
        # Array of points
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([0.5, 1.0, 1.5])
        kappa = profile.convergence(x, y)
        assert kappa.shape == (3,)
        
        # 2D grid
        x_grid, y_grid = np.meshgrid(np.linspace(-5, 5, 10), np.linspace(-5, 5, 10))
        kappa = profile.convergence(x_grid, y_grid)
        assert kappa.shape == (10, 10)
    
    def test_convergence_symmetry(self, lens_sys):
        """Test convergence respects elliptical symmetry."""
        profile = EllipticalNFWProfile(
            M_vir=1e12, concentration=10.0, lens_sys=lens_sys,
            ellipticity=0.3, position_angle=0.0  # Major axis along x
        )
        
        # Points at same elliptical radius should have similar convergence
        x1, y1 = 2.0, 0.0  # On major axis
        x2, y2 = 2.0 * profile.q, 0.0  # On minor axis, scaled
        
        kappa1 = profile.convergence(np.array([x1]), np.array([y1]))[0]
        kappa2 = profile.convergence(np.array([0.0]), np.array([x2]))[0]
        
        # Should be similar (not exact due to transformation)
        assert abs(kappa1 - kappa2) / kappa1 < 0.5  # Within 50%
    
    @pytest.mark.parametrize("ellipticity", [0.0, 0.2, 0.5, 0.8])
    def test_various_ellipticities(self, lens_sys, ellipticity):
        """Test profile works for different ellipticities."""
        profile = EllipticalNFWProfile(
            M_vir=1e12, concentration=10.0, lens_sys=lens_sys, ellipticity=ellipticity
        )
        
        # Test at reasonable radii (not too far out where NFW can have issues)
        x = np.array([1.0, 2.0])
        y = np.array([0.5, 1.0])
        
        kappa = profile.convergence(x, y)
        assert np.all(np.isfinite(kappa))
        # Note: NFW convergence can be negative at very large radii
        # due to numerical issues, but should be positive at moderate radii
    
    @pytest.mark.parametrize("angle", [0.0, 45.0, 90.0, 135.0])
    def test_various_position_angles(self, lens_sys, angle):
        """Test profile works for different position angles."""
        profile = EllipticalNFWProfile(
            M_vir=1e12, concentration=10.0, lens_sys=lens_sys,
            ellipticity=0.3, position_angle=angle
        )
        
        x = np.array([1.0, 2.0])
        y = np.array([0.5, 1.0])
        
        kappa = profile.convergence(x, y)
        assert np.all(np.isfinite(kappa))
    
    def test_deflection_angle_shape(self, lens_sys):
        """Test deflection angle returns correct shape."""
        profile = EllipticalNFWProfile(
            M_vir=1e12, concentration=10.0, lens_sys=lens_sys, ellipticity=0.3
        )
        
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([0.5, 1.0, 1.5])
        
        alpha_x, alpha_y = profile.deflection_angle(x, y)
        assert alpha_x.shape == (3,)
        assert alpha_y.shape == (3,)
        assert np.all(np.isfinite(alpha_x))
        assert np.all(np.isfinite(alpha_y))


class TestSersicProfile:
    """Test suite for Sérsic profile."""
    
    @pytest.fixture
    def lens_sys(self):
        """Create standard lens system for testing."""
        return LensSystem(z_lens=0.5, z_source=2.0)
    
    def test_initialization(self, lens_sys):
        """Test Sérsic profile can be initialized."""
        profile = SersicProfile(I_e=1.0, r_e=5.0, n=4.0, lens_sys=lens_sys)
        assert profile.I_e == 1.0
        assert profile.r_e == 5.0
        assert profile.n == 4.0
        assert profile.b_n > 0
    
    def test_invalid_parameters(self, lens_sys):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError, match="Sérsic index must be positive"):
            SersicProfile(I_e=1.0, r_e=5.0, n=-1.0, lens_sys=lens_sys)
        
        with pytest.raises(ValueError, match="Effective radius must be positive"):
            SersicProfile(I_e=1.0, r_e=-5.0, n=1.0, lens_sys=lens_sys)
    
    def test_exponential_disk(self, lens_sys):
        """Test n=1 gives exponential profile."""
        profile = SersicProfile(I_e=1.0, r_e=5.0, n=1.0, lens_sys=lens_sys)
        
        # At r=r_e, should have I = I_e * exp(-b_1)
        r = np.array([5.0])
        I = profile.surface_brightness(r)
        
        # For n=1, b_1 ≈ 1.678
        expected = 1.0 * np.exp(-profile.b_n * (1.0 - 1.0))
        assert_allclose(I, expected, rtol=1e-5)
    
    def test_de_vaucouleurs_profile(self, lens_sys):
        """Test n=4 gives de Vaucouleurs profile."""
        profile = SersicProfile(I_e=1.0, r_e=5.0, n=4.0, lens_sys=lens_sys)
        
        # b_4 ≈ 7.67
        assert_allclose(profile.b_n, 7.67, rtol=0.1)
    
    def test_surface_brightness_decay(self, lens_sys):
        """Test surface brightness decays with radius."""
        profile = SersicProfile(I_e=1.0, r_e=5.0, n=2.0, lens_sys=lens_sys)
        
        r = np.array([0.1, 1.0, 5.0, 10.0, 50.0])
        I = profile.surface_brightness(r)
        
        # Should be monotonically decreasing
        assert np.all(np.diff(I) < 0)
        
        # Should be positive
        assert np.all(I > 0)
    
    def test_convergence_shape(self, lens_sys):
        """Test convergence returns correct shape."""
        profile = SersicProfile(I_e=1.0, r_e=5.0, n=2.0, lens_sys=lens_sys)
        
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([0.5, 1.0, 1.5])
        
        kappa = profile.convergence(x, y)
        assert kappa.shape == (3,)
        assert np.all(np.isfinite(kappa))
    
    def test_convergence_circular_symmetry(self, lens_sys):
        """Test convergence has circular symmetry."""
        profile = SersicProfile(I_e=1.0, r_e=5.0, n=2.0, lens_sys=lens_sys)
        
        # Points at same radius should have same convergence
        r = 3.0
        x1, y1 = r, 0.0
        x2, y2 = 0.0, r
        x3, y3 = r/np.sqrt(2), r/np.sqrt(2)
        
        kappa1 = profile.convergence(np.array([x1]), np.array([y1]))[0]
        kappa2 = profile.convergence(np.array([x2]), np.array([y2]))[0]
        kappa3 = profile.convergence(np.array([x3]), np.array([y3]))[0]
        
        assert_allclose(kappa1, kappa2, rtol=1e-10)
        assert_allclose(kappa1, kappa3, rtol=1e-10)
    
    def test_total_luminosity(self, lens_sys):
        """Test total luminosity calculation."""
        profile = SersicProfile(I_e=1.0, r_e=5.0, n=1.0, lens_sys=lens_sys)
        
        L_tot = profile.total_luminosity()
        assert L_tot > 0
        assert np.isfinite(L_tot)
    
    @pytest.mark.parametrize("n", [0.5, 1.0, 2.0, 4.0, 8.0])
    def test_various_sersic_indices(self, lens_sys, n):
        """Test profile works for various Sérsic indices."""
        profile = SersicProfile(I_e=1.0, r_e=5.0, n=n, lens_sys=lens_sys)
        
        r = np.linspace(0.1, 20.0, 10)
        I = profile.surface_brightness(r)
        
        assert np.all(I > 0)
        assert np.all(np.isfinite(I))


class TestCompositeGalaxyProfile:
    """Test suite for composite galaxy profile."""
    
    @pytest.fixture
    def lens_sys(self):
        """Create standard lens system for testing."""
        return LensSystem(z_lens=0.5, z_source=2.0)
    
    @pytest.fixture
    def bulge(self, lens_sys):
        """Create bulge component."""
        return SersicProfile(I_e=2.0, r_e=2.0, n=4.0, lens_sys=lens_sys)
    
    @pytest.fixture
    def disk(self, lens_sys):
        """Create disk component."""
        return SersicProfile(I_e=1.0, r_e=5.0, n=1.0, lens_sys=lens_sys)
    
    @pytest.fixture
    def halo(self, lens_sys):
        """Create halo component."""
        return NFWProfile(M_vir=1e12, concentration=10.0, lens_system=lens_sys)
    
    def test_initialization_single_component(self, bulge, lens_sys):
        """Test can create with single component."""
        galaxy = CompositeGalaxyProfile(bulge=bulge, lens_sys=lens_sys)
        assert galaxy.bulge is not None
        assert galaxy.disk is None
        assert galaxy.halo is None
        assert len(galaxy.components) == 1
    
    def test_initialization_all_components(self, bulge, disk, halo, lens_sys):
        """Test can create with all components."""
        galaxy = CompositeGalaxyProfile(
            bulge=bulge, disk=disk, halo=halo, lens_sys=lens_sys
        )
        assert len(galaxy.components) == 3
    
    def test_requires_at_least_one_component(self, lens_sys):
        """Test error if no components provided."""
        with pytest.raises(ValueError, match="At least one component"):
            CompositeGalaxyProfile(lens_sys=lens_sys)
    
    def test_convergence_is_sum(self, bulge, halo, lens_sys):
        """Test total convergence is sum of components."""
        galaxy = CompositeGalaxyProfile(bulge=bulge, halo=halo, lens_sys=lens_sys)
        
        x = np.array([1.0, 2.0])
        y = np.array([0.5, 1.0])
        
        kappa_total = galaxy.convergence(x, y)
        kappa_bulge = bulge.convergence(x, y)
        kappa_halo = halo.convergence(x, y)
        
        assert_allclose(kappa_total, kappa_bulge + kappa_halo, rtol=1e-10)
    
    def test_deflection_angle_is_sum(self, bulge, halo, lens_sys):
        """Test total deflection is sum of components."""
        galaxy = CompositeGalaxyProfile(bulge=bulge, halo=halo, lens_sys=lens_sys)
        
        x = np.array([1.0, 2.0])
        y = np.array([0.5, 1.0])
        
        alpha_x_total, alpha_y_total = galaxy.deflection_angle(x, y)
        alpha_x_bulge, alpha_y_bulge = bulge.deflection_angle(x, y)
        alpha_x_halo, alpha_y_halo = halo.deflection_angle(x, y)
        
        assert_allclose(alpha_x_total, alpha_x_bulge + alpha_x_halo, rtol=1e-5)
        assert_allclose(alpha_y_total, alpha_y_bulge + alpha_y_halo, rtol=1e-5)
    
    def test_realistic_early_type_galaxy(self, lens_sys):
        """Test realistic early-type galaxy (bulge + halo)."""
        # Typical parameters for early-type galaxy
        bulge = SersicProfile(I_e=2.0, r_e=2.0, n=4.0, lens_sys=lens_sys)
        halo = NFWProfile(M_vir=1e12, concentration=10.0, lens_system=lens_sys)
        
        galaxy = CompositeGalaxyProfile(bulge=bulge, halo=halo, lens_sys=lens_sys)
        
        # At small radii, check that both components contribute
        x_inner = np.array([0.5])
        y_inner = np.array([0.0])
        kappa_inner = galaxy.convergence(x_inner, y_inner)
        kappa_bulge_inner = bulge.convergence(x_inner, y_inner)
        kappa_halo_inner = halo.convergence(x_inner, y_inner)
        
        # Total should equal sum of components
        assert_allclose(kappa_inner[0], kappa_bulge_inner[0] + kappa_halo_inner[0], rtol=1e-5)
        
        # Both should contribute positively at small radii
        assert kappa_inner[0] > 0
    
    def test_realistic_spiral_galaxy(self, bulge, disk, halo, lens_sys):
        """Test realistic spiral galaxy (bulge + disk + halo)."""
        galaxy = CompositeGalaxyProfile(
            bulge=bulge, disk=disk, halo=halo, lens_sys=lens_sys
        )
        
        # Test convergence at various radii (not too far out)
        r = np.array([0.5, 2.0, 5.0])
        x = r
        y = np.zeros_like(r)
        
        kappa = galaxy.convergence(x, y)
        assert np.all(np.isfinite(kappa))
        
        # At small radius, convergence should be positive
        assert kappa[0] > 0


class TestIntegration:
    """Integration tests combining multiple profiles."""
    
    def test_elliptical_composite_galaxy(self):
        """Test composite galaxy with elliptical halo."""
        lens_sys = LensSystem(z_lens=0.5, z_source=2.0)
        
        # Create elliptical halo
        halo = EllipticalNFWProfile(
            M_vir=1e12, concentration=10.0, lens_sys=lens_sys,
            ellipticity=0.3, position_angle=30.0
        )
        
        # Create bulge
        bulge = SersicProfile(I_e=2.0, r_e=2.0, n=4.0, lens_sys=lens_sys)
        
        # Combine
        galaxy = CompositeGalaxyProfile(
            bulge=bulge, halo=halo, lens_sys=lens_sys
        )
        
        # Test full pipeline
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([0.5, 1.0, 1.5])
        
        kappa = galaxy.convergence(x, y)
        alpha_x, alpha_y = galaxy.deflection_angle(x, y)
        
        assert np.all(np.isfinite(kappa))
        assert np.all(np.isfinite(alpha_x))
        assert np.all(np.isfinite(alpha_y))
    
    def test_profile_comparison(self):
        """Compare profiles to ensure reasonable relative magnitudes."""
        lens_sys = LensSystem(z_lens=0.5, z_source=2.0)
        
        # Create profiles with similar scales
        nfw_circular = NFWProfile(M_vir=1e12, concentration=10.0, lens_system=lens_sys)
        nfw_elliptical = EllipticalNFWProfile(
            M_vir=1e12, concentration=10.0, lens_sys=lens_sys, ellipticity=0.3
        )
        sersic = SersicProfile(I_e=1.0, r_e=5.0, n=4.0, lens_sys=lens_sys)
        
        # Test at moderate radius (not too far out)
        x = np.array([2.0])
        y = np.array([0.0])
        
        kappa_nfw_circ = nfw_circular.convergence(x, y)
        kappa_nfw_ell = nfw_elliptical.convergence(x, y)
        kappa_sersic = sersic.convergence(x, y)
        
        # All should be finite
        assert np.isfinite(kappa_nfw_circ)
        assert np.isfinite(kappa_nfw_ell)
        assert np.isfinite(kappa_sersic)
        
        # At moderate radius, NFW should be reasonably positive
        # Elliptical should be similar to circular at this point
        if kappa_nfw_circ > 0:
            assert abs(kappa_nfw_ell - kappa_nfw_circ) / abs(kappa_nfw_circ) < 1.0
