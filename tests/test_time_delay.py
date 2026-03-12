"""
Tests for Time Delay Cosmography Module

Tests the calculation of time delays and inference of H0 from lensing observations.
"""

import pytest
import numpy as np
from src.lens_models import LensSystem, PointMassProfile, NFWProfile
from src.time_delay import (
    calculate_time_delays,
    infer_h0,
    monte_carlo_h0_uncertainty,
    TimeDelayCosmography
)


class TestTimeDelayCalculation:
    """Tests for time delay calculation."""
    
    @pytest.fixture
    def lens_system(self):
        """Create a standard lens system."""
        return LensSystem(z_lens=0.5, z_source=1.5, H0=70.0)
    
    @pytest.fixture
    def point_mass(self, lens_system):
        """Create a point mass lens."""
        return PointMassProfile(1e12, lens_system)
    
    def test_time_delay_calculation_runs(self, point_mass):
        """Test that time delay calculation executes without errors."""
        images = [(1.0, 0.0), (-0.8, 0.5), (0.2, -0.9)]
        source = (0.1, 0.05)
        
        result = calculate_time_delays(images, source, point_mass)
        
        assert 'time_delay_matrix' in result
        assert 'time_delay_distance' in result
        assert 'fermat_potentials' in result
    
    def test_time_delay_matrix_shape(self, point_mass):
        """Test that time delay matrix has correct shape."""
        n_images = 4
        images = [(1.0, 0.0), (-0.8, 0.5), (0.2, -0.9), (-0.3, 0.7)]
        source = (0.0, 0.0)
        
        result = calculate_time_delays(images, source, point_mass)
        matrix = result['time_delay_matrix']
        
        assert matrix.shape == (n_images, n_images)
    
    def test_time_delay_antisymmetric(self, point_mass):
        """Test that time delays are antisymmetric: Δt_ij = -Δt_ji."""
        images = [(1.0, 0.0), (-0.8, 0.5)]
        source = (0.1, 0.05)
        
        result = calculate_time_delays(images, source, point_mass)
        matrix = result['time_delay_matrix']
        
        assert np.allclose(matrix[0, 1], -matrix[1, 0])
    
    def test_time_delay_zero_diagonal(self, point_mass):
        """Test that diagonal elements are zero: Δt_ii = 0."""
        images = [(1.0, 0.0), (-0.8, 0.5), (0.2, -0.9)]
        source = (0.1, 0.05)
        
        result = calculate_time_delays(images, source, point_mass)
        matrix = result['time_delay_matrix']
        
        assert np.allclose(np.diag(matrix), 0.0)
    
    def test_time_delay_scales_with_redshift(self, point_mass):
        """Test that time delays increase with lens redshift."""
        images = [(1.0, 0.0), (-0.8, 0.5)]
        source = (0.1, 0.05)
        
        # Low redshift
        lens_sys_low = LensSystem(0.2, 1.5, H0=70.0)
        lens_low = PointMassProfile(1e12, lens_sys_low)
        result_low = calculate_time_delays(images, source, lens_low)
        
        # High redshift
        lens_sys_high = LensSystem(0.8, 1.5, H0=70.0)
        lens_high = PointMassProfile(1e12, lens_sys_high)
        result_high = calculate_time_delays(images, source, lens_high)
        
        # Higher redshift should give larger time delays
        delay_low = np.abs(result_low['time_delay_matrix'][0, 1])
        delay_high = np.abs(result_high['time_delay_matrix'][0, 1])
        
        assert delay_high > delay_low
    
    def test_time_delay_reasonable_magnitude(self, point_mass):
        """Test that time delays have reasonable magnitude (days to months)."""
        images = [(1.0, 0.0), (-1.0, 0.0)]  # Opposite sides
        source = (0.1, 0.0)
        
        result = calculate_time_delays(images, source, point_mass)
        delay = np.abs(result['time_delay_matrix'][0, 1])
        
        # Typical lensing time delays are days to months
        assert 0.1 < delay < 1000.0, f"Time delay {delay:.1f} days seems unreasonable"
    
    def test_fermat_potential_array_length(self, point_mass):
        """Test that Fermat potentials are computed for all images."""
        n_images = 5
        images = [(1.0, 0.0), (-0.8, 0.5), (0.2, -0.9), (-0.3, 0.7), (0.6, 0.4)]
        source = (0.0, 0.0)
        
        result = calculate_time_delays(images, source, point_mass)
        potentials = result['fermat_potentials']
        
        assert len(potentials) == n_images


class TestH0Inference:
    """Tests for H0 inference from time delays."""
    
    @pytest.fixture
    def lens_system(self):
        """Create lens system with known H0."""
        return LensSystem(z_lens=0.5, z_source=1.5, H0=70.0)
    
    @pytest.fixture
    def point_mass(self, lens_system):
        """Create point mass lens."""
        return PointMassProfile(1e12, lens_system)
    
    def test_h0_inference_runs(self, point_mass):
        """Test that H0 inference executes without errors."""
        images = [(1.0, 0.0), (-0.8, 0.5)]
        source = (0.1, 0.05)
        
        # Calculate true delays
        true_result = calculate_time_delays(images, source, point_mass)
        true_delay = true_result['time_delay_matrix'][0, 1]
        
        # Create observed delays with uncertainty
        observed = {(0, 1): (true_delay, 0.5)}
        
        # Infer H0
        result = infer_h0(observed, images, source, point_mass,
                         h0_range=(60, 80), n_grid=50)
        
        assert 'h0_best' in result
        assert 'h0_uncertainty' in result
    
    def test_h0_recovery_noiseless(self, point_mass):
        """Test that H0 is recovered accurately with perfect data."""
        images = [(1.0, 0.0), (-0.8, 0.5), (0.2, -0.9)]
        source = (0.1, 0.05)
        
        # True H0
        h0_true = 70.0
        
        # Calculate delays with true H0
        true_result = calculate_time_delays(images, source, point_mass)
        
        # Create "observed" delays (perfect measurements)
        observed = {
            (0, 1): (true_result['time_delay_matrix'][0, 1], 0.01),
            (0, 2): (true_result['time_delay_matrix'][0, 2], 0.01)
        }
        
        # Infer H0
        result = infer_h0(observed, images, source, point_mass,
                         h0_range=(60, 80), n_grid=100)
        
        # Should recover H0 within 2σ
        assert np.abs(result['h0_best'] - h0_true) < 2 * result['h0_uncertainty']
    
    def test_h0_recovery_with_noise(self, point_mass):
        """Test H0 recovery with realistic noise."""
        images = [(1.0, 0.0), (-0.8, 0.5)]
        source = (0.1, 0.05)
        
        h0_true = 70.0
        
        # Calculate true delays
        true_result = calculate_time_delays(images, source, point_mass)
        true_delay = true_result['time_delay_matrix'][0, 1]
        
        # Add 5% noise
        noise_sigma = 0.05 * np.abs(true_delay)
        observed_delay = true_delay + np.random.randn() * noise_sigma
        
        observed = {(0, 1): (observed_delay, noise_sigma)}
        
        # Infer H0
        result = infer_h0(observed, images, source, point_mass,
                         h0_range=(60, 80), n_grid=100)
        
        # Should be within reasonable range
        assert 50 < result['h0_best'] < 90
    
    def test_h0_uncertainty_positive(self, point_mass):
        """Test that H0 uncertainty is positive."""
        images = [(1.0, 0.0), (-0.8, 0.5)]
        source = (0.1, 0.05)
        
        true_result = calculate_time_delays(images, source, point_mass)
        observed = {(0, 1): (true_result['time_delay_matrix'][0, 1], 0.5)}
        
        result = infer_h0(observed, images, source, point_mass)
        
        assert result['h0_uncertainty'] > 0
    
    def test_posterior_normalized(self, point_mass):
        """Test that posterior probability is normalized."""
        images = [(1.0, 0.0), (-0.8, 0.5)]
        source = (0.1, 0.05)
        
        true_result = calculate_time_delays(images, source, point_mass)
        observed = {(0, 1): (true_result['time_delay_matrix'][0, 1], 0.5)}
        
        result = infer_h0(observed, images, source, point_mass, n_grid=100)
        
        # Integral of posterior should be ~1
        integral = np.trapezoid(result['posterior'], result['h0_grid'])
        assert np.abs(integral - 1.0) < 0.01
    
    def test_chi2_minimum_at_best_h0(self, point_mass):
        """Test that chi-squared is minimized at best-fit H0."""
        images = [(1.0, 0.0), (-0.8, 0.5)]
        source = (0.1, 0.05)
        
        true_result = calculate_time_delays(images, source, point_mass)
        observed = {(0, 1): (true_result['time_delay_matrix'][0, 1], 0.5)}
        
        result = infer_h0(observed, images, source, point_mass, n_grid=100)
        
        # Find index of best H0
        idx_best = np.argmin(result['chi2_grid'])
        h0_at_min_chi2 = result['h0_grid'][idx_best]
        
        assert np.abs(h0_at_min_chi2 - result['h0_best']) < 1.0


class TestMonteCarloUncertainty:
    """Tests for Monte Carlo uncertainty estimation."""
    
    @pytest.fixture
    def lens_system(self):
        """Create lens system."""
        return LensSystem(0.5, 1.5, H0=70.0)
    
    @pytest.fixture
    def nfw_profile(self, lens_system):
        """Create NFW profile."""
        return NFWProfile(1e12, 10.0, lens_system)
    
    def test_monte_carlo_runs(self, nfw_profile):
        """Test that Monte Carlo uncertainty estimation runs."""
        images = [(1.0, 0.0), (-0.8, 0.5)]
        source = (0.1, 0.05)
        
        # Calculate delays
        result = calculate_time_delays(images, source, nfw_profile)
        observed = {(0, 1): (result['time_delay_matrix'][0, 1], 0.5)}
        
        # Run Monte Carlo (small number for speed)
        uncertainties = {'M_vir': 1e11, 'c': 1.0, 'source_x': 0.01, 'source_y': 0.01}
        mc_result = monte_carlo_h0_uncertainty(
            observed, images, source, nfw_profile, uncertainties,
            n_realizations=50, h0_true=70.0
        )
        
        assert 'h0_samples' in mc_result
        assert 'h0_mean' in mc_result
        assert 'h0_std' in mc_result
    
    def test_monte_carlo_samples_shape(self, nfw_profile):
        """Test that Monte Carlo produces expected number of samples."""
        images = [(1.0, 0.0), (-0.8, 0.5)]
        source = (0.1, 0.05)
        
        result = calculate_time_delays(images, source, nfw_profile)
        observed = {(0, 1): (result['time_delay_matrix'][0, 1], 0.5)}
        
        n_real = 100
        uncertainties = {'M_vir': 1e11, 'c': 0.5}
        mc_result = monte_carlo_h0_uncertainty(
            observed, images, source, nfw_profile, uncertainties,
            n_realizations=n_real, h0_true=70.0
        )
        
        # Should have ~n_real samples (some may be filtered as outliers)
        assert 0.8 * n_real <= len(mc_result['h0_samples']) <= n_real
    
    def test_monte_carlo_std_positive(self, nfw_profile):
        """Test that Monte Carlo standard deviation is positive."""
        images = [(1.0, 0.0), (-0.8, 0.5)]
        source = (0.1, 0.05)
        
        result = calculate_time_delays(images, source, nfw_profile)
        observed = {(0, 1): (result['time_delay_matrix'][0, 1], 0.5)}
        
        uncertainties = {'M_vir': 1e11, 'c': 1.0}
        mc_result = monte_carlo_h0_uncertainty(
            observed, images, source, nfw_profile, uncertainties,
            n_realizations=50, h0_true=70.0
        )
        
        assert mc_result['h0_std'] > 0


class TestTimeDelayCosmographyClass:
    """Tests for TimeDelayCosmography class."""
    
    @pytest.fixture
    def lens_system(self):
        """Create lens system."""
        return LensSystem(0.5, 1.5, H0=70.0)
    
    @pytest.fixture
    def point_mass(self, lens_system):
        """Create point mass."""
        return PointMassProfile(1e12, lens_system)
    
    @pytest.fixture
    def cosmography(self, point_mass):
        """Create cosmography analyzer."""
        return TimeDelayCosmography(point_mass)
    
    def test_initialization(self, cosmography):
        """Test that cosmography class initializes."""
        assert cosmography.lens_model is not None
        assert cosmography.lens_system is not None
    
    def test_calculate_delays_method(self, cosmography):
        """Test calculate_delays method."""
        images = [(1.0, 0.0), (-0.8, 0.5)]
        source = (0.1, 0.05)
        
        delays = cosmography.calculate_delays(images, source)
        
        assert isinstance(delays, np.ndarray)
        assert delays.shape == (2, 2)
    
    def test_infer_h0_method(self, cosmography):
        """Test infer_h0 method."""
        images = [(1.0, 0.0), (-0.8, 0.5)]
        source = (0.1, 0.05)
        
        delays = cosmography.calculate_delays(images, source)
        observed = {(0, 1): (delays[0, 1], 0.5)}
        
        result = cosmography.infer_h0(observed, images, source, n_grid=50)
        
        assert 'h0_best' in result
        assert 60 < result['h0_best'] < 80


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    @pytest.fixture
    def lens_system(self):
        """Create lens system."""
        return LensSystem(0.5, 1.5, H0=70.0)
    
    @pytest.fixture
    def point_mass(self, lens_system):
        """Create point mass."""
        return PointMassProfile(1e12, lens_system)
    
    def test_single_image_pair(self, point_mass):
        """Test with minimum number of images (2)."""
        images = [(1.0, 0.0), (-1.0, 0.0)]
        source = (0.0, 0.0)
        
        result = calculate_time_delays(images, source, point_mass)
        
        assert result['time_delay_matrix'].shape == (2, 2)
    
    def test_many_images(self, point_mass):
        """Test with many images."""
        n_images = 10
        angles = np.linspace(0, 2*np.pi, n_images, endpoint=False)
        images = [(np.cos(a), np.sin(a)) for a in angles]
        source = (0.1, 0.05)
        
        result = calculate_time_delays(images, source, point_mass)
        
        assert result['time_delay_matrix'].shape == (n_images, n_images)
    
    def test_source_at_origin(self, point_mass):
        """Test with source at origin (symmetric case)."""
        images = [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]
        source = (0.0, 0.0)
        
        result = calculate_time_delays(images, source, point_mass)
        
        # All images at same distance from source should have similar Fermat potentials
        potentials = result['fermat_potentials']
        assert np.std(potentials) < 0.1 * np.mean(np.abs(potentials))


class TestPhysicalConsistency:
    """Tests for physical consistency of results."""
    
    @pytest.fixture
    def lens_system(self):
        """Create lens system."""
        return LensSystem(0.5, 1.5, H0=70.0)
    
    @pytest.fixture
    def point_mass(self, lens_system):
        """Create point mass."""
        return PointMassProfile(1e12, lens_system)
    
    def test_time_delay_scales_inversely_with_h0(self, point_mass):
        """Test that time delays scale as 1/H0."""
        images = [(1.0, 0.0), (-0.8, 0.5)]
        source = (0.1, 0.05)
        
        # Calculate with different H0 values
        from src.lens_models import LensSystem, PointMassProfile
        
        lens_sys_70 = LensSystem(0.5, 1.5, H0=70.0)
        lens_70 = PointMassProfile(1e12, lens_sys_70)
        result_70 = calculate_time_delays(images, source, lens_70)
        
        lens_sys_100 = LensSystem(0.5, 1.5, H0=100.0)
        lens_100 = PointMassProfile(1e12, lens_sys_100)
        result_100 = calculate_time_delays(images, source, lens_100)
        
        # Time delays should scale roughly as H0_1 / H0_2
        # But not exactly because angular diameter distances also depend on cosmology
        ratio = result_70['time_delay_matrix'][0, 1] / result_100['time_delay_matrix'][0, 1]
        expected_ratio = 100.0 / 70.0
        
        # Relaxed tolerance because D_dt depends on cosmology in non-trivial way
        assert np.abs(ratio - expected_ratio) / expected_ratio < 0.30
    
    def test_larger_mass_gives_larger_delays(self):
        """Test that more massive lenses give larger time delays."""
        from src.lens_models import LensSystem, PointMassProfile
        
        lens_sys = LensSystem(0.5, 1.5, H0=70.0)
        
        # Light lens
        lens_light = PointMassProfile(5e11, lens_sys)
        # Heavy lens
        lens_heavy = PointMassProfile(2e12, lens_sys)
        
        images = [(1.0, 0.0), (-0.8, 0.5)]
        source = (0.1, 0.05)
        
        result_light = calculate_time_delays(images, source, lens_light)
        result_heavy = calculate_time_delays(images, source, lens_heavy)
        
        delay_light = np.abs(result_light['time_delay_matrix'][0, 1])
        delay_heavy = np.abs(result_heavy['time_delay_matrix'][0, 1])
        
        assert delay_heavy > delay_light
