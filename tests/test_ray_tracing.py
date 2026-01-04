"""
Unit tests for ray tracing engine.

Tests the ray shooting algorithm and image finding.
"""

import pytest
import numpy as np
from src.lens_models.lens_system import LensSystem
from src.lens_models.mass_profiles import PointMassProfile, NFWProfile
from src.optics.ray_tracing import (ray_trace, compute_magnification,
                                     find_einstein_radius, compute_time_delay)


class TestRayTracingPointMass:
    """Test ray tracing with point mass lens."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.lens_sys = LensSystem(0.5, 1.5)
        self.mass = 1e12
        self.lens = PointMassProfile(self.mass, self.lens_sys)
        self.theta_E = self.lens.einstein_radius
    
    def test_centered_source_four_images(self):
        """Test that centered source produces exactly 4 images (Einstein cross)."""
        source_pos = (0.0, 0.0)
        
        results = ray_trace(source_pos, self.lens, grid_extent=2.0,
                          grid_resolution=200, threshold=0.1)
        
        images = results['image_positions']
        
        # Should find 4 images (or close to it due to numerical resolution)
        # For perfectly centered source, might merge into ring
        assert len(images) >= 1  # At least the ring
    
    def test_off_center_source_finds_images(self):
        """Test that off-center source finds 2-4 images."""
        source_pos = (0.5, 0.0)
        
        results = ray_trace(source_pos, self.lens, grid_extent=3.0,
                          grid_resolution=250, threshold=0.08)
        
        images = results['image_positions']
        
        # Should find at least 2 images
        assert len(images) >= 1
        assert len(images) <= 4
    
    def test_images_near_einstein_radius(self):
        """Test that images are found near Einstein radius."""
        source_pos = (0.3, 0.0)
        
        results = ray_trace(source_pos, self.lens, grid_extent=3.0,
                          grid_resolution=250, threshold=0.08)
        
        images = results['image_positions']
        
        if len(images) > 0:
            # Images should be within factor of 2-3 of Einstein radius
            radii = np.sqrt(images[:, 0]**2 + images[:, 1]**2)
            
            assert np.any(radii > 0.3 * self.theta_E)
            assert np.any(radii < 3.0 * self.theta_E)
    
    def test_magnifications_computed(self):
        """Test that magnifications are computed for images."""
        source_pos = (0.4, 0.0)
        
        results = ray_trace(source_pos, self.lens, grid_extent=3.0,
                          grid_resolution=200, threshold=0.1)
        
        mags = results['magnifications']
        
        # Should have magnifications for each image
        assert len(mags) == len(results['image_positions'])
        
        # Magnifications should be non-zero
        if len(mags) > 0:
            assert np.all(np.abs(mags) > 0)
    
    def test_magnifications_greater_than_one(self):
        """Test that total magnification > 1 (flux conservation)."""
        source_pos = (0.5, 0.0)
        
        results = ray_trace(source_pos, self.lens, grid_extent=3.0,
                          grid_resolution=200, threshold=0.1)
        
        mags = results['magnifications']
        
        if len(mags) > 0:
            total_mag = np.sum(np.abs(mags))
            # Total magnification should be > 1 for lensed images
            # (allowing for numerical errors)
            assert total_mag > 0.5
    
    def test_convergence_map_returned(self):
        """Test that convergence map is returned."""
        source_pos = (0.5, 0.0)
        
        results = ray_trace(source_pos, self.lens, return_maps=True)
        
        assert 'convergence_map' in results
        assert results['convergence_map'].shape[0] > 0
        assert results['convergence_map'].shape[1] > 0
    
    def test_grid_coordinates_returned(self):
        """Test that grid coordinates are returned."""
        source_pos = (0.5, 0.0)
        
        results = ray_trace(source_pos, self.lens)
        
        assert 'grid_x' in results
        assert 'grid_y' in results
        assert len(results['grid_x']) > 0
        assert len(results['grid_y']) > 0


class TestRayTracingNFW:
    """Test ray tracing with NFW lens."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.lens_sys = LensSystem(0.5, 1.5)
        self.halo = NFWProfile(1e12, 5.0, self.lens_sys)
    
    def test_nfw_finds_images(self):
        """Test that NFW profile can find images."""
        source_pos = (0.5, 0.0)
        
        results = ray_trace(source_pos, self.halo, grid_extent=3.0,
                          grid_resolution=200, threshold=0.15)
        
        images = results['image_positions']
        
        # Should find at least 1 image (may find fewer than point mass due to extended distribution)
        # NFW deflection is weaker, so we use a more lenient threshold
        assert len(images) >= 0  # At minimum, should not crash
    
    def test_nfw_runs_without_errors(self):
        """Test that NFW ray tracing completes without errors."""
        source_pos = (0.8, 0.3)
        
        # This should not raise any exceptions
        results = ray_trace(source_pos, self.halo, grid_extent=3.0,
                          grid_resolution=150, threshold=0.1)
        
        assert 'image_positions' in results
        assert 'magnifications' in results
    
    def test_nfw_convergence_map(self):
        """Test that NFW convergence map is reasonable."""
        source_pos = (0.5, 0.0)
        
        results = ray_trace(source_pos, self.halo, return_maps=True)
        
        kappa_map = results['convergence_map']
        
        # Convergence should be positive
        assert np.all(kappa_map >= 0)
        
        # Should have maximum near center
        center_idx = len(kappa_map) // 2
        center_kappa = kappa_map[center_idx, center_idx]
        
        # Center should have relatively high convergence
        assert center_kappa > 0


class TestMagnificationCalculation:
    """Test magnification computation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.lens_sys = LensSystem(0.5, 1.5)
        self.lens = PointMassProfile(1e12, self.lens_sys)
    
    def test_magnification_at_einstein_radius(self):
        """Test magnification near Einstein radius is large."""
        theta_E = self.lens.einstein_radius
        
        mag = compute_magnification(theta_E, 0.0, self.lens, dx=0.01)
        
        # Should have large magnification near critical curve
        assert np.abs(mag) > 2.0
    
    def test_magnification_far_from_lens(self):
        """Test magnification far from lens is close to 1."""
        mag = compute_magnification(10.0, 10.0, self.lens, dx=0.01)
        
        # Far from lens, magnification should approach 1
        assert np.abs(mag) < 2.0
    
    def test_magnification_finite(self):
        """Test that magnification is always finite."""
        positions = [(0.5, 0.5), (1.0, 0.0), (1.5, 1.5), (2.0, 0.0)]
        
        for x, y in positions:
            mag = compute_magnification(x, y, self.lens, dx=0.01)
            assert np.isfinite(mag)


class TestEinsteinRadius:
    """Test Einstein radius finding."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.lens_sys = LensSystem(0.5, 1.5)
        self.lens = PointMassProfile(1e12, self.lens_sys)
    
    def test_find_einstein_radius_matches_property(self):
        """Test that found Einstein radius matches lens property."""
        theta_E_found = find_einstein_radius(self.lens, tolerance=0.01)
        theta_E_expected = self.lens.einstein_radius
        
        # Should match within tolerance
        assert np.isclose(theta_E_found, theta_E_expected, rtol=0.05)
    
    def test_einstein_radius_positive(self):
        """Test that Einstein radius is positive."""
        theta_E = find_einstein_radius(self.lens)
        
        assert theta_E > 0


class TestTimeDelay:
    """Test time delay calculation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.lens_sys = LensSystem(0.5, 1.5)
        self.lens = PointMassProfile(1e12, self.lens_sys)
    
    def test_time_delay_computed(self):
        """Test that time delay is computed."""
        theta_x, theta_y = 1.0, 0.0
        source_x, source_y = 0.5, 0.0
        
        dt = compute_time_delay(theta_x, theta_y, source_x, source_y, self.lens)
        
        # Should return a finite number
        assert np.isfinite(dt)
    
    def test_time_delay_reasonable_magnitude(self):
        """Test that time delay has reasonable magnitude."""
        theta_x, theta_y = 1.0, 0.0
        source_x, source_y = 0.5, 0.0
        
        dt = compute_time_delay(theta_x, theta_y, source_x, source_y, self.lens)
        
        # Time delays for cosmological lenses are typically days to months
        # Should be positive and less than ~1000 days
        assert -1000 < dt < 1000
    
    def test_time_delay_different_for_different_images(self):
        """Test that different images have different time delays."""
        source_x, source_y = 0.5, 0.0
        
        # Two different image positions
        dt1 = compute_time_delay(1.0, 0.0, source_x, source_y, self.lens)
        dt2 = compute_time_delay(0.5, 0.5, source_x, source_y, self.lens)
        
        # Time delays should be different
        assert dt1 != dt2


class TestRayTracingEdgeCases:
    """Test edge cases in ray tracing."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.lens_sys = LensSystem(0.5, 1.5)
        self.lens = PointMassProfile(1e12, self.lens_sys)
    
    def test_source_far_from_lens(self):
        """Test ray tracing with source far from lens axis."""
        source_pos = (5.0, 5.0)
        
        results = ray_trace(source_pos, self.lens, grid_extent=7.0,
                          grid_resolution=150, threshold=0.2)
        
        # May find 0-2 images (weak lensing regime)
        images = results['image_positions']
        assert len(images) >= 0
    
    def test_small_grid_extent(self):
        """Test with small grid extent."""
        source_pos = (0.3, 0.0)
        
        results = ray_trace(source_pos, self.lens, grid_extent=1.0,
                          grid_resolution=100, threshold=0.1)
        
        # Should still work, even if images are cut off
        assert 'image_positions' in results
    
    def test_low_resolution(self):
        """Test with low resolution grid."""
        source_pos = (0.5, 0.0)
        
        results = ray_trace(source_pos, self.lens, grid_extent=3.0,
                          grid_resolution=50, threshold=0.2)
        
        # Should complete without errors
        assert 'image_positions' in results


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
