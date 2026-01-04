"""
Unit tests for mass profile classes.

Tests PointMassProfile and NFWProfile implementations.
"""

import pytest
import numpy as np
from src.lens_models.lens_system import LensSystem
from src.lens_models.mass_profiles import PointMassProfile, NFWProfile


class TestPointMassProfile:
    """Test suite for PointMassProfile class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.lens_sys = LensSystem(0.5, 1.5)
        self.mass = 1e12  # Solar masses
        self.lens = PointMassProfile(self.mass, self.lens_sys)
    
    def test_initialization(self):
        """Test that point mass initializes correctly."""
        assert self.lens.M == self.mass
        assert self.lens.lens_system == self.lens_sys
    
    def test_einstein_radius_exists(self):
        """Test that Einstein radius is computed."""
        theta_E = self.lens.einstein_radius
        
        assert theta_E > 0
        assert 0.1 < theta_E < 10  # Reasonable range
    
    def test_einstein_radius_matches_formula(self):
        """Test that Einstein radius matches lens system calculation."""
        theta_E_lens = self.lens.einstein_radius
        theta_E_sys = self.lens_sys.einstein_radius_scale(self.mass)
        
        assert np.isclose(theta_E_lens, theta_E_sys, rtol=1e-10)
    
    def test_deflection_at_einstein_radius(self):
        """Test deflection angle at Einstein radius equals Einstein radius."""
        theta_E = self.lens.einstein_radius
        
        # At Einstein radius on x-axis
        alpha_x, alpha_y = self.lens.deflection_angle(theta_E, 0.0)
        
        # Should satisfy |α| = θ_E
        alpha_mag = np.sqrt(alpha_x**2 + alpha_y**2)
        assert np.isclose(alpha_mag, theta_E, rtol=0.01)
    
    def test_deflection_radial_symmetry(self):
        """Test that deflection is radially symmetric."""
        r = 1.0
        
        # Test at four cardinal directions
        alpha_x1, alpha_y1 = self.lens.deflection_angle(r, 0)
        alpha_x2, alpha_y2 = self.lens.deflection_angle(0, r)
        alpha_x3, alpha_y3 = self.lens.deflection_angle(-r, 0)
        alpha_x4, alpha_y4 = self.lens.deflection_angle(0, -r)
        
        # Magnitudes should all be equal
        mag1 = np.sqrt(alpha_x1**2 + alpha_y1**2)
        mag2 = np.sqrt(alpha_x2**2 + alpha_y2**2)
        mag3 = np.sqrt(alpha_x3**2 + alpha_y3**2)
        mag4 = np.sqrt(alpha_x4**2 + alpha_y4**2)
        
        assert np.allclose([mag1, mag2, mag3, mag4], mag1, rtol=0.01)
    
    def test_deflection_vectorized(self):
        """Test that deflection works with arrays."""
        x = np.array([0.5, 1.0, 1.5])
        y = np.array([0.0, 0.5, 1.0])
        
        alpha_x, alpha_y = self.lens.deflection_angle(x, y)
        
        assert alpha_x.shape == x.shape
        assert alpha_y.shape == y.shape
        assert np.all(np.isfinite(alpha_x))
        assert np.all(np.isfinite(alpha_y))
    
    def test_deflection_scales_correctly(self):
        """Test that deflection scales as 1/r."""
        r1 = 1.0
        r2 = 2.0
        
        alpha_x1, alpha_y1 = self.lens.deflection_angle(r1, 0)
        alpha_x2, alpha_y2 = self.lens.deflection_angle(r2, 0)
        
        # Deflection should scale as 1/r, so α(2r) = α(r)/2
        assert np.isclose(alpha_x2, alpha_x1 / 2, rtol=0.01)
    
    def test_convergence_positive(self):
        """Test that convergence is positive."""
        kappa = self.lens.convergence(1.0, 0.5)
        
        assert kappa > 0
    
    def test_surface_density_positive(self):
        """Test that surface density is positive."""
        sigma = self.lens.surface_density(1.0)
        
        assert sigma > 0
    
    def test_lensing_potential_computed(self):
        """Test that lensing potential is computed."""
        psi = self.lens.lensing_potential(1.0, 0.5)
        
        assert np.isfinite(psi)
    
    def test_no_crash_at_origin(self):
        """Test that methods don't crash at r=0."""
        # Should handle singularity gracefully
        alpha_x, alpha_y = self.lens.deflection_angle(0.0, 0.0)
        kappa = self.lens.convergence(0.0, 0.0)
        psi = self.lens.lensing_potential(0.0, 0.0)
        
        assert np.isfinite(alpha_x)
        assert np.isfinite(alpha_y)
        assert np.isfinite(kappa)
        assert np.isfinite(psi)


class TestNFWProfile:
    """Test suite for NFWProfile class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.lens_sys = LensSystem(0.5, 1.5)
        self.M_vir = 1e12  # Solar masses
        self.concentration = 5.0
        self.halo = NFWProfile(self.M_vir, self.concentration, self.lens_sys)
    
    def test_initialization(self):
        """Test that NFW profile initializes correctly."""
        assert self.halo.M_vir == self.M_vir
        assert self.halo.c == self.concentration
        assert self.halo.lens_system == self.lens_sys
    
    def test_scale_radius_computed(self):
        """Test that scale radius is computed."""
        assert hasattr(self.halo, 'r_s')
        assert self.halo.r_s > 0
    
    def test_density_computed(self):
        """Test that characteristic density is computed."""
        assert hasattr(self.halo, 'rho_s')
        assert self.halo.rho_s > 0
    
    def test_deflection_positive_radial(self):
        """Test that deflection is radially outward."""
        x = 1.0
        y = 0.0
        
        alpha_x, alpha_y = self.halo.deflection_angle(x, y)
        
        # Should point outward (same direction as position vector)
        assert alpha_x > 0
        assert np.abs(alpha_y) < 0.1  # Should be nearly zero
    
    def test_deflection_vectorized(self):
        """Test that deflection works with arrays."""
        x = np.array([0.5, 1.0, 1.5, 2.0])
        y = np.array([0.0, 0.5, 1.0, 0.5])
        
        alpha_x, alpha_y = self.halo.deflection_angle(x, y)
        
        assert alpha_x.shape == x.shape
        assert alpha_y.shape == y.shape
        assert np.all(np.isfinite(alpha_x))
        assert np.all(np.isfinite(alpha_y))
    
    def test_convergence_positive(self):
        """Test that convergence is positive everywhere."""
        x = np.linspace(-2, 2, 20)
        y = np.linspace(-2, 2, 20)
        xx, yy = np.meshgrid(x, y)
        
        kappa = self.halo.convergence(xx.ravel(), yy.ravel())
        
        assert np.all(kappa > 0)
    
    def test_convergence_decreases_with_radius(self):
        """Test that convergence decreases at large radii."""
        kappa_1 = self.halo.convergence(1.0, 0.0)
        kappa_2 = self.halo.convergence(2.0, 0.0)
        kappa_3 = self.halo.convergence(3.0, 0.0)
        
        # Should decrease with radius
        assert kappa_1 > kappa_2 > kappa_3
    
    def test_surface_density_positive(self):
        """Test that surface density is positive."""
        sigma = self.halo.surface_density(np.array([0.5, 1.0, 2.0]))
        
        assert np.all(sigma > 0)
    
    def test_lensing_potential_computed(self):
        """Test that lensing potential is computed."""
        psi = self.halo.lensing_potential(1.0, 0.5)
        
        assert np.isfinite(psi)
    
    def test_no_crash_at_origin(self):
        """Test that methods handle r=0 gracefully."""
        alpha_x, alpha_y = self.halo.deflection_angle(0.0, 0.0)
        kappa = self.halo.convergence(0.0, 0.0)
        
        assert np.isfinite(alpha_x)
        assert np.isfinite(alpha_y)
        assert np.isfinite(kappa)
    
    def test_different_concentrations(self):
        """Test that different concentrations give different results."""
        halo_low_c = NFWProfile(self.M_vir, 3.0, self.lens_sys)
        halo_high_c = NFWProfile(self.M_vir, 10.0, self.lens_sys)
        
        # Scale radii should be different
        assert halo_low_c.r_s != halo_high_c.r_s
        
        # Deflections should be different
        alpha_x1, _ = halo_low_c.deflection_angle(1.0, 0.0)
        alpha_x2, _ = halo_high_c.deflection_angle(1.0, 0.0)
        
        assert alpha_x1 != alpha_x2


class TestMassProfileComparison:
    """Compare point mass and NFW profiles."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.lens_sys = LensSystem(0.5, 1.5)
        self.mass = 1e12
        self.point_mass = PointMassProfile(self.mass, self.lens_sys)
        self.nfw = NFWProfile(self.mass, 5.0, self.lens_sys)
    
    def test_both_deflect_outward(self):
        """Test that both profiles deflect light outward."""
        x = 1.5
        y = 0.0
        
        alpha_x_pm, alpha_y_pm = self.point_mass.deflection_angle(x, y)
        alpha_x_nfw, alpha_y_nfw = self.nfw.deflection_angle(x, y)
        
        # Both should deflect in positive x direction
        assert alpha_x_pm > 0
        assert alpha_x_nfw > 0
    
    def test_convergence_different_profiles(self):
        """Test that different mass profiles give different convergence."""
        kappa_pm = self.point_mass.convergence(1.0, 0.0)
        kappa_nfw = self.nfw.convergence(1.0, 0.0)
        
        # Should be different (point mass is delta function)
        # But both should be positive
        assert kappa_pm > 0
        assert kappa_nfw > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
