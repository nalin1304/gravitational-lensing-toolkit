"""
Unit tests for LensSystem class.

Tests cosmological distance calculations and unit conversions.
"""

import pytest
import numpy as np
from src.lens_models.lens_system import LensSystem


class TestLensSystem:
    """Test suite for LensSystem class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.z_lens = 0.5
        self.z_source = 1.5
        self.lens_sys = LensSystem(self.z_lens, self.z_source)
    
    def test_initialization(self):
        """Test that lens system initializes correctly."""
        assert self.lens_sys.z_l == self.z_lens
        assert self.lens_sys.z_s == self.z_source
        assert self.lens_sys.cosmology.H0.value == 70.0
        assert self.lens_sys.cosmology.Om0 == 0.3
    
    def test_invalid_redshifts(self):
        """Test that invalid redshifts raise errors."""
        # Negative lens redshift
        with pytest.raises(ValueError, match="Lens redshift must be positive"):
            LensSystem(-0.1, 1.5)
        
        # Source redshift less than lens redshift
        with pytest.raises(ValueError, match="Source redshift must be greater"):
            LensSystem(1.5, 0.5)
        
        # Equal redshifts
        with pytest.raises(ValueError, match="Source redshift must be greater"):
            LensSystem(0.5, 0.5)
    
    def test_distances_positive(self):
        """Test that all distances are positive."""
        D_l = self.lens_sys.angular_diameter_distance_lens()
        D_s = self.lens_sys.angular_diameter_distance_source()
        D_ls = self.lens_sys.angular_diameter_distance_lens_source()
        
        assert D_l.value > 0
        assert D_s.value > 0
        assert D_ls.value > 0
    
    def test_distance_ordering(self):
        """Test that D_ls < D_s (since lens is closer than source)."""
        D_s = self.lens_sys.angular_diameter_distance_source()
        D_ls = self.lens_sys.angular_diameter_distance_lens_source()
        
        # D_ls should be less than D_s
        assert D_ls.value < D_s.value
    
    def test_critical_surface_density(self):
        """Test that critical surface density is reasonable."""
        sigma_cr = self.lens_sys.critical_surface_density()
        
        # Should be positive
        assert sigma_cr > 0
        
        # Typical value should be ~10^3 Msun/pc² for cosmological lenses
        # (This is the correct value; surface density, not volume density)
        assert 1e3 < sigma_cr < 1e4, f"Got {sigma_cr:.2e} Msun/pc²"
    
    def test_critical_density_caching(self):
        """Test that critical density is cached correctly."""
        sigma_cr1 = self.lens_sys.critical_surface_density()
        sigma_cr2 = self.lens_sys.critical_surface_density()
        
        # Should return the same value (from cache)
        assert sigma_cr1 == sigma_cr2
    
    def test_arcsec_to_kpc_positive(self):
        """Test that angular to physical conversion gives positive result."""
        size_kpc = self.lens_sys.arcsec_to_kpc(1.0)
        
        assert size_kpc > 0
    
    def test_arcsec_to_kpc_scale(self):
        """Test that conversion scales linearly."""
        size_1 = self.lens_sys.arcsec_to_kpc(1.0)
        size_2 = self.lens_sys.arcsec_to_kpc(2.0)
        
        assert np.isclose(size_2, 2 * size_1, rtol=1e-10)
    
    def test_einstein_radius_positive(self):
        """Test that Einstein radius is positive."""
        M = 1e12  # Solar masses
        theta_E = self.lens_sys.einstein_radius_scale(M)
        
        assert theta_E > 0
    
    def test_einstein_radius_reasonable(self):
        """Test that Einstein radius is in reasonable range."""
        M = 1e12  # Typical galaxy mass
        theta_E = self.lens_sys.einstein_radius_scale(M)
        
        # Should be ~0.5-2 arcsec for typical lenses
        assert 0.1 < theta_E < 10, f"Got θ_E = {theta_E:.3f} arcsec"
    
    def test_einstein_radius_mass_scaling(self):
        """Test that Einstein radius scales as sqrt(M)."""
        M1 = 1e12
        M2 = 4e12  # 4 times larger
        
        theta_E1 = self.lens_sys.einstein_radius_scale(M1)
        theta_E2 = self.lens_sys.einstein_radius_scale(M2)
        
        # Should scale as sqrt(M), so ratio should be 2
        ratio = theta_E2 / theta_E1
        assert np.isclose(ratio, 2.0, rtol=0.01)
    
    def test_repr(self):
        """Test string representation."""
        repr_str = repr(self.lens_sys)
        
        assert 'LensSystem' in repr_str
        assert '0.500' in repr_str  # z_lens
        assert '1.500' in repr_str  # z_source
        assert '70.0' in repr_str   # H0


class TestLensSystemDifferentRedshifts:
    """Test lens system with different redshift configurations."""
    
    def test_nearby_lens(self):
        """Test lens at low redshift."""
        lens_sys = LensSystem(0.1, 1.0)
        D_l = lens_sys.angular_diameter_distance_lens()
        
        # Low redshift lens should have smaller distance
        assert D_l.value < 1000  # Less than 1 Gpc
    
    def test_distant_source(self):
        """Test source at high redshift."""
        lens_sys = LensSystem(0.5, 3.0)
        D_s = lens_sys.angular_diameter_distance_source()
        
        # High redshift source
        assert D_s.value > 0
    
    def test_close_redshifts(self):
        """Test lens and source at similar redshifts."""
        lens_sys = LensSystem(0.5, 0.6)
        D_ls = lens_sys.angular_diameter_distance_lens_source()
        
        # Close redshifts should give small D_ls
        assert D_ls.value > 0
        assert D_ls.value < 500  # Should be relatively small


class TestLensSystemCosmology:
    """Test lens system with different cosmological parameters."""
    
    def test_custom_h0(self):
        """Test with custom Hubble constant."""
        lens_sys = LensSystem(0.5, 1.5, H0=67.0)
        
        assert lens_sys.cosmology.H0.value == 67.0
    
    def test_custom_om0(self):
        """Test with custom matter density."""
        lens_sys = LensSystem(0.5, 1.5, Om0=0.27)
        
        assert lens_sys.cosmology.Om0 == 0.27
    
    def test_different_cosmology_affects_distances(self):
        """Test that different H0 affects distances."""
        lens_sys1 = LensSystem(0.5, 1.5, H0=70.0)
        lens_sys2 = LensSystem(0.5, 1.5, H0=67.0)
        
        sigma_cr1 = lens_sys1.critical_surface_density()
        sigma_cr2 = lens_sys2.critical_surface_density()
        
        # Different H0 should give different critical densities
        assert sigma_cr1 != sigma_cr2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
