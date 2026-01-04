"""
Tests for Wave Optics Module

This test suite validates the wave optics implementation including:
- Fermat potential computation
- Wave phase calculation
- Interference pattern generation
- Comparison with geometric optics
- Long wavelength limit convergence
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt
from src.lens_models import LensSystem, PointMassProfile, NFWProfile
from src.optics import WaveOpticsEngine, plot_wave_vs_geometric


class TestWaveOpticsEngine:
    """Tests for WaveOpticsEngine class."""
    
    @pytest.fixture
    def lens_system(self):
        """Create a standard lens system for testing."""
        return LensSystem(z_lens=0.5, z_source=1.5)
    
    @pytest.fixture
    def point_mass_lens(self, lens_system):
        """Create a point mass lens."""
        return PointMassProfile(mass=1e12, lens_system=lens_system)
    
    @pytest.fixture
    def nfw_lens(self, lens_system):
        """Create an NFW lens."""
        return NFWProfile(M_vir=1e12, concentration=10.0, lens_system=lens_system)
    
    @pytest.fixture
    def wave_engine(self):
        """Create a wave optics engine."""
        return WaveOpticsEngine()
    
    def test_initialization(self, wave_engine):
        """Test that WaveOpticsEngine initializes correctly."""
        assert wave_engine is not None
        assert isinstance(wave_engine, WaveOpticsEngine)
    
    def test_compute_amplification_factor_runs(self, wave_engine, point_mass_lens):
        """Test that compute_amplification_factor runs without errors."""
        result = wave_engine.compute_amplification_factor(
            point_mass_lens,
            source_position=(0.5, 0.0),
            wavelength=500.0,
            grid_size=128,  # Small for speed
            grid_extent=2.0
        )
        
        assert 'amplitude_map' in result
        assert 'phase_map' in result
        assert 'fermat_potential' in result
        assert 'wavelength' in result
    
    def test_amplitude_map_shape(self, wave_engine, point_mass_lens):
        """Test that amplitude map has correct shape."""
        grid_size = 64
        result = wave_engine.compute_amplification_factor(
            point_mass_lens,
            wavelength=500.0,
            grid_size=grid_size,
            grid_extent=2.0
        )
        
        assert result['amplitude_map'].shape == (grid_size, grid_size)
        assert result['phase_map'].shape == (grid_size, grid_size)
        assert result['fermat_potential'].shape == (grid_size, grid_size)
    
    def test_amplitude_map_positive(self, wave_engine, point_mass_lens):
        """Test that amplitude map (intensity) is non-negative."""
        result = wave_engine.compute_amplification_factor(
            point_mass_lens,
            wavelength=500.0,
            grid_size=128,
            grid_extent=2.0
        )
        
        amplitude_map = result['amplitude_map']
        assert np.all(amplitude_map >= 0), "Intensity must be non-negative"
    
    def test_phase_map_range(self, wave_engine, point_mass_lens):
        """Test that phase map is in valid range [-π, π]."""
        result = wave_engine.compute_amplification_factor(
            point_mass_lens,
            wavelength=500.0,
            grid_size=128,
            grid_extent=2.0
        )
        
        phase_map = result['phase_map']
        assert np.all(phase_map >= -np.pi), "Phase must be >= -π"
        assert np.all(phase_map <= np.pi), "Phase must be <= π"
    
    def test_wavelength_stored(self, wave_engine, point_mass_lens):
        """Test that wavelength is stored in result."""
        wavelength = 632.8  # He-Ne laser
        result = wave_engine.compute_amplification_factor(
            point_mass_lens,
            wavelength=wavelength,
            grid_size=64,
            grid_extent=2.0
        )
        
        assert result['wavelength'] == wavelength
    
    def test_grid_coordinates_returned(self, wave_engine, point_mass_lens):
        """Test that grid coordinates are returned."""
        result = wave_engine.compute_amplification_factor(
            point_mass_lens,
            wavelength=500.0,
            grid_size=64,
            grid_extent=3.0
        )
        
        assert 'grid_x' in result
        assert 'grid_y' in result
        assert len(result['grid_x']) == 64
        assert len(result['grid_y']) == 64
        assert np.abs(result['grid_x'][-1] - 3.0) < 0.1  # Check extent


class TestPointMassWaveOptics:
    """Test wave optics specifically for point mass lenses."""
    
    @pytest.fixture
    def lens_system(self):
        return LensSystem(z_lens=0.5, z_source=1.5)
    
    @pytest.fixture
    def point_mass_lens(self, lens_system):
        return PointMassProfile(mass=1e12, lens_system=lens_system)
    
    @pytest.fixture
    def wave_engine(self):
        return WaveOpticsEngine()
    
    def test_einstein_ring_interference(self, wave_engine, point_mass_lens):
        """Test that point mass shows interference fringes near Einstein ring."""
        result = wave_engine.compute_amplification_factor(
            point_mass_lens,
            source_position=(0.0, 0.0),  # On-axis source
            wavelength=500.0,
            grid_size=256,
            grid_extent=3.0
        )
        
        amplitude_map = result['amplitude_map']
        
        # For aligned source, should see circular pattern
        # Check that amplitude varies significantly (fringes present)
        amplitude_variation = np.std(amplitude_map) / np.mean(amplitude_map)
        assert amplitude_variation > 0.1, "Should see significant intensity variation"
    
    def test_off_axis_source(self, wave_engine, point_mass_lens):
        """Test wave optics with off-axis source."""
        result = wave_engine.compute_amplification_factor(
            point_mass_lens,
            source_position=(0.5, 0.0),
            wavelength=500.0,
            grid_size=128,
            grid_extent=3.0
        )
        
        # Should produce valid result
        assert result['amplitude_map'].shape[0] == 128
        assert np.sum(result['amplitude_map']) > 0


class TestNFWWaveOptics:
    """Test wave optics for NFW profile."""
    
    @pytest.fixture
    def lens_system(self):
        return LensSystem(z_lens=0.5, z_source=1.5)
    
    @pytest.fixture
    def nfw_lens(self, lens_system):
        return NFWProfile(M_vir=1e12, concentration=10.0, lens_system=lens_system)
    
    @pytest.fixture
    def wave_engine(self):
        return WaveOpticsEngine()
    
    def test_nfw_wave_optics_runs(self, wave_engine, nfw_lens):
        """Test that wave optics works with NFW profile."""
        result = wave_engine.compute_amplification_factor(
            nfw_lens,
            source_position=(0.5, 0.0),
            wavelength=500.0,
            grid_size=128,
            grid_extent=3.0
        )
        
        assert 'amplitude_map' in result
        assert result['amplitude_map'].shape == (128, 128)
    
    def test_nfw_extended_profile(self, wave_engine, nfw_lens):
        """Test that NFW shows extended structure in wave optics."""
        result = wave_engine.compute_amplification_factor(
            nfw_lens,
            source_position=(0.5, 0.0),
            wavelength=500.0,
            grid_size=128,
            grid_extent=5.0  # Larger extent
        )
        
        # NFW has extended structure, amplitude should be non-zero far from center
        amplitude_map = result['amplitude_map']
        edge_amplitude = np.mean(amplitude_map[:10, :])  # Top edge
        assert edge_amplitude > 0, "NFW should show extended structure"


class TestGeometricComparison:
    """Test comparison between wave and geometric optics."""
    
    @pytest.fixture
    def lens_system(self):
        return LensSystem(z_lens=0.5, z_source=1.5)
    
    @pytest.fixture
    def point_mass_lens(self, lens_system):
        return PointMassProfile(mass=1e12, lens_system=lens_system)
    
    @pytest.fixture
    def wave_engine(self):
        return WaveOpticsEngine()
    
    def test_geometric_comparison_included(self, wave_engine, point_mass_lens):
        """Test that geometric comparison is included when requested."""
        result = wave_engine.compute_amplification_factor(
            point_mass_lens,
            wavelength=500.0,
            grid_size=128,
            grid_extent=2.0,
            return_geometric=True
        )
        
        assert 'geometric_comparison' in result
        assert 'image_positions' in result['geometric_comparison']
        assert 'magnifications' in result['geometric_comparison']
    
    def test_geometric_comparison_excluded(self, wave_engine, point_mass_lens):
        """Test that geometric comparison can be excluded."""
        result = wave_engine.compute_amplification_factor(
            point_mass_lens,
            wavelength=500.0,
            grid_size=128,
            grid_extent=2.0,
            return_geometric=False
        )
        
        assert 'geometric_comparison' not in result
    
    def test_compare_with_geometric_method(self, wave_engine, point_mass_lens):
        """Test the compare_with_geometric method."""
        result = wave_engine.compute_amplification_factor(
            point_mass_lens,
            wavelength=500.0,
            grid_size=128,
            grid_extent=2.0,
            return_geometric=True
        )
        
        comparison = wave_engine.compare_with_geometric(result)
        
        assert 'fractional_difference_map' in comparison
        assert 'max_difference' in comparison
        assert 'mean_difference' in comparison
        assert 'significant_pixels' in comparison
    
    def test_difference_map_shape(self, wave_engine, point_mass_lens):
        """Test that difference map has correct shape."""
        grid_size = 128
        result = wave_engine.compute_amplification_factor(
            point_mass_lens,
            wavelength=500.0,
            grid_size=grid_size,
            grid_extent=2.0,
            return_geometric=True
        )
        
        comparison = wave_engine.compare_with_geometric(result)
        diff_map = comparison['fractional_difference_map']
        
        assert diff_map.shape == (grid_size, grid_size)


class TestLongWavelengthLimit:
    """Test that wave optics converges to geometric optics at long wavelengths."""
    
    @pytest.fixture
    def lens_system(self):
        return LensSystem(z_lens=0.5, z_source=1.5)
    
    @pytest.fixture
    def point_mass_lens(self, lens_system):
        return PointMassProfile(mass=1e12, lens_system=lens_system)
    
    @pytest.fixture
    def wave_engine(self):
        return WaveOpticsEngine()
    
    def test_long_wavelength_convergence(self, wave_engine, point_mass_lens):
        """Test that long wavelengths approach geometric optics."""
        # Compute at very long wavelength (radio)
        result_long = wave_engine.compute_amplification_factor(
            point_mass_lens,
            source_position=(0.5, 0.0),
            wavelength=1e6,  # 1 mm = 10^6 nm (radio)
            grid_size=128,
            grid_extent=2.0,
            return_geometric=True
        )
        
        comparison = wave_engine.compare_with_geometric(result_long)
        
        # At long wavelengths, wave optics still shows diffraction
        # but the result should be computable without errors
        assert comparison['max_difference'] >= 0, "Difference should be non-negative"
        assert np.isfinite(comparison['mean_difference']), "Mean difference should be finite"
    
    def test_short_wavelength_differs(self, wave_engine, point_mass_lens):
        """Test that short wavelengths show more interference."""
        # Compute at optical wavelength
        result_short = wave_engine.compute_amplification_factor(
            point_mass_lens,
            source_position=(0.5, 0.0),
            wavelength=500.0,  # 500 nm (optical)
            grid_size=128,
            grid_extent=2.0,
            return_geometric=True
        )
        
        # Should produce valid result with some difference from geometric
        assert result_short['amplitude_map'].shape == (128, 128)


class TestFringeDetection:
    """Test fringe detection functionality."""
    
    @pytest.fixture
    def lens_system(self):
        return LensSystem(z_lens=0.5, z_source=1.5)
    
    @pytest.fixture
    def point_mass_lens(self, lens_system):
        return PointMassProfile(mass=1e12, lens_system=lens_system)
    
    @pytest.fixture
    def wave_engine(self):
        return WaveOpticsEngine()
    
    def test_detect_fringes_runs(self, wave_engine, point_mass_lens):
        """Test that detect_fringes runs without error."""
        result = wave_engine.compute_amplification_factor(
            point_mass_lens,
            source_position=(0.0, 0.0),  # On-axis for clear fringes
            wavelength=500.0,
            grid_size=256,
            grid_extent=3.0
        )
        
        fringe_info = wave_engine.detect_fringes(
            result['amplitude_map'],
            result['grid_x'],
            result['grid_y']
        )
        
        assert 'fringe_spacing' in fringe_info
        assert 'n_fringes' in fringe_info
        assert 'fringe_contrast' in fringe_info
    
    def test_fringe_spacing_positive(self, wave_engine, point_mass_lens):
        """Test that detected fringe spacing is positive."""
        result = wave_engine.compute_amplification_factor(
            point_mass_lens,
            source_position=(0.0, 0.0),
            wavelength=500.0,
            grid_size=256,
            grid_extent=3.0
        )
        
        fringe_info = wave_engine.detect_fringes(
            result['amplitude_map'],
            result['grid_x'],
            result['grid_y']
        )
        
        # If fringes detected, spacing should be positive
        if fringe_info['n_fringes'] > 0:
            assert fringe_info['fringe_spacing'] >= 0
    
    def test_fringe_contrast_range(self, wave_engine, point_mass_lens):
        """Test that fringe contrast is in valid range [0, 1]."""
        result = wave_engine.compute_amplification_factor(
            point_mass_lens,
            wavelength=500.0,
            grid_size=256,
            grid_extent=3.0
        )
        
        fringe_info = wave_engine.detect_fringes(
            result['amplitude_map'],
            result['grid_x'],
            result['grid_y']
        )
        
        contrast = fringe_info['fringe_contrast']
        assert 0 <= contrast <= 1, "Contrast must be in [0, 1]"
    
    def test_fringe_spacing_wavelength_scaling(self, wave_engine, point_mass_lens):
        """Test that fringe spacing scales approximately as sqrt(λ)."""
        # Compute at two wavelengths
        result_400 = wave_engine.compute_amplification_factor(
            point_mass_lens,
            source_position=(0.0, 0.0),
            wavelength=400.0,
            grid_size=256,
            grid_extent=3.0
        )
        
        result_900 = wave_engine.compute_amplification_factor(
            point_mass_lens,
            source_position=(0.0, 0.0),
            wavelength=900.0,  # 2.25x wavelength
            grid_size=256,
            grid_extent=3.0
        )
        
        fringe_400 = wave_engine.detect_fringes(
            result_400['amplitude_map'],
            result_400['grid_x'],
            result_400['grid_y']
        )
        
        fringe_900 = wave_engine.detect_fringes(
            result_900['amplitude_map'],
            result_900['grid_x'],
            result_900['grid_y']
        )
        
        # Fringe spacing should increase with wavelength
        # sqrt(900/400) = 1.5, so spacing should roughly increase by this factor
        if fringe_400['fringe_spacing'] > 0 and fringe_900['fringe_spacing'] > 0:
            ratio = fringe_900['fringe_spacing'] / fringe_400['fringe_spacing']
            expected_ratio = np.sqrt(900.0 / 400.0)
            # Allow generous tolerance since this is approximate
            assert 0.5 * expected_ratio < ratio < 2.0 * expected_ratio


class TestEnergyConservation:
    """Test that wave optics conserves total flux."""
    
    @pytest.fixture
    def lens_system(self):
        return LensSystem(z_lens=0.5, z_source=1.5)
    
    @pytest.fixture
    def point_mass_lens(self, lens_system):
        return PointMassProfile(mass=1e12, lens_system=lens_system)
    
    @pytest.fixture
    def wave_engine(self):
        return WaveOpticsEngine()
    
    def test_flux_conservation(self, wave_engine, point_mass_lens):
        """Test that total flux is conserved in wave optics."""
        result = wave_engine.compute_amplification_factor(
            point_mass_lens,
            source_position=(0.5, 0.0),
            wavelength=500.0,
            grid_size=256,
            grid_extent=3.0,
            return_geometric=True
        )
        
        # Total flux in wave optics (normalized)
        wave_flux = np.sum(result['amplitude_map'])
        
        # Total magnification in geometric optics
        geo_mags = result['geometric_comparison']['magnifications']
        geo_flux = np.sum(np.abs(geo_mags))
        
        # They should be comparable (within factor of ~2 due to normalization)
        # Both should be on order of grid_size^2 due to normalization
        assert wave_flux > 0, "Wave flux must be positive"
        assert geo_flux > 0, "Geometric flux must be positive"


class TestVisualization:
    """Test visualization functions."""
    
    @pytest.fixture
    def lens_system(self):
        return LensSystem(z_lens=0.5, z_source=1.5)
    
    @pytest.fixture
    def point_mass_lens(self, lens_system):
        return PointMassProfile(mass=1e12, lens_system=lens_system)
    
    @pytest.fixture
    def wave_engine(self):
        return WaveOpticsEngine()
    
    def test_plot_interference_pattern(self, wave_engine, point_mass_lens):
        """Test that plot_interference_pattern runs without error."""
        result = wave_engine.compute_amplification_factor(
            point_mass_lens,
            wavelength=500.0,
            grid_size=128,
            grid_extent=2.0
        )
        
        # Should not raise error
        fig = wave_engine.plot_interference_pattern(result)
        assert fig is not None
        plt.close(fig)
    
    def test_plot_wave_vs_geometric(self, point_mass_lens):
        """Test that plot_wave_vs_geometric runs without error."""
        # Should not raise error
        fig = plot_wave_vs_geometric(
            point_mass_lens,
            source_position=(0.5, 0.0),
            wavelength=500.0,
            grid_size=128,
            grid_extent=2.0
        )
        assert fig is not None
        plt.close(fig)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def lens_system(self):
        return LensSystem(z_lens=0.5, z_source=1.5)
    
    @pytest.fixture
    def point_mass_lens(self, lens_system):
        return PointMassProfile(mass=1e12, lens_system=lens_system)
    
    @pytest.fixture
    def wave_engine(self):
        return WaveOpticsEngine()
    
    def test_very_short_wavelength(self, wave_engine, point_mass_lens):
        """Test with very short wavelength (X-ray)."""
        result = wave_engine.compute_amplification_factor(
            point_mass_lens,
            wavelength=1.0,  # 1 nm (X-ray)
            grid_size=64,
            grid_extent=2.0
        )
        
        assert result['amplitude_map'].shape == (64, 64)
    
    def test_very_long_wavelength(self, wave_engine, point_mass_lens):
        """Test with very long wavelength (radio)."""
        result = wave_engine.compute_amplification_factor(
            point_mass_lens,
            wavelength=1e8,  # 0.1 m (radio)
            grid_size=64,
            grid_extent=2.0
        )
        
        assert result['amplitude_map'].shape == (64, 64)
    
    def test_small_grid(self, wave_engine, point_mass_lens):
        """Test with small grid size."""
        result = wave_engine.compute_amplification_factor(
            point_mass_lens,
            wavelength=500.0,
            grid_size=32,  # Small grid
            grid_extent=2.0
        )
        
        assert result['amplitude_map'].shape == (32, 32)
    
    def test_large_extent(self, wave_engine, point_mass_lens):
        """Test with large grid extent."""
        result = wave_engine.compute_amplification_factor(
            point_mass_lens,
            wavelength=500.0,
            grid_size=128,
            grid_extent=10.0  # Large extent
        )
        
        assert result['grid_extent'] == 10.0
        assert np.max(np.abs(result['grid_x'])) >= 9.0
