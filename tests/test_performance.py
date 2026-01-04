"""
Unit tests for performance optimization module (Phase 7).

Tests GPU acceleration, vectorization, benchmarking, and caching.
"""

import pytest
import numpy as np
import time
from src.ml.performance import (
    ArrayBackend, get_backend, set_backend,
    PerformanceMonitor, timer,
    benchmark_convergence_map, compare_cpu_gpu_performance,
    cached_convergence, clear_cache,
    GPU_AVAILABLE
)
from src.ml.generate_dataset import generate_convergence_map_vectorized
from src.lens_models import LensSystem, NFWProfile


class TestArrayBackend:
    """Test array backend selection."""
    
    def test_backend_initialization(self):
        """Test backend initializes correctly."""
        backend = ArrayBackend(use_gpu=False)
        assert backend.backend_name == "NumPy (CPU)"
        
    def test_array_creation(self):
        """Test array creation."""
        backend = ArrayBackend(use_gpu=False)
        arr = backend.array([1, 2, 3])
        assert arr.shape == (3,)
        assert isinstance(arr, np.ndarray)
    
    def test_zeros_creation(self):
        """Test zeros array creation."""
        backend = ArrayBackend(use_gpu=False)
        arr = backend.zeros((10, 10))
        assert arr.shape == (10, 10)
        assert np.all(arr == 0)
    
    def test_linspace(self):
        """Test linspace creation."""
        backend = ArrayBackend(use_gpu=False)
        arr = backend.linspace(0, 1, 11)
        assert len(arr) == 11
        assert arr[0] == 0.0
        assert arr[-1] == 1.0
    
    def test_meshgrid(self):
        """Test meshgrid creation."""
        backend = ArrayBackend(use_gpu=False)
        x = backend.linspace(0, 1, 5)
        y = backend.linspace(0, 1, 5)
        X, Y = backend.meshgrid(x, y)
        assert X.shape == (5, 5)
        assert Y.shape == (5, 5)
    
    def test_to_numpy(self):
        """Test conversion to numpy."""
        backend = ArrayBackend(use_gpu=False)
        arr = backend.array([1, 2, 3])
        np_arr = backend.to_numpy(arr)
        assert isinstance(np_arr, np.ndarray)


class TestPerformanceMonitor:
    """Test performance monitoring."""
    
    def test_initialization(self):
        """Test monitor initializes."""
        monitor = PerformanceMonitor()
        assert len(monitor.timings) == 0
    
    def test_time_block(self):
        """Test timing a code block."""
        monitor = PerformanceMonitor()
        
        with monitor.time_block("test_operation"):
            time.sleep(0.01)
        
        assert "test_operation" in monitor.timings
        assert len(monitor.timings["test_operation"]) == 1
        assert monitor.timings["test_operation"][0] >= 0.01
    
    def test_multiple_timings(self):
        """Test multiple timing measurements."""
        monitor = PerformanceMonitor()
        
        for _ in range(3):
            with monitor.time_block("repeated_op"):
                time.sleep(0.01)
        
        assert len(monitor.timings["repeated_op"]) == 3
    
    def test_get_stats(self):
        """Test statistics computation."""
        monitor = PerformanceMonitor()
        
        for _ in range(5):
            with monitor.time_block("test_op"):
                time.sleep(0.01)
        
        stats = monitor.get_stats("test_op")
        assert stats['count'] == 5
        assert stats['mean'] >= 0.01
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'total' in stats
    
    def test_clear(self):
        """Test clearing timings."""
        monitor = PerformanceMonitor()
        
        with monitor.time_block("test"):
            time.sleep(0.01)
        
        monitor.clear()
        assert len(monitor.timings) == 0


class TestTimerDecorator:
    """Test timer decorator."""
    
    def test_timer_decorator(self, capsys):
        """Test timer decorator prints timing."""
        
        @timer
        def slow_function():
            time.sleep(0.01)
            return 42
        
        result = slow_function()
        captured = capsys.readouterr()
        
        assert result == 42
        assert "slow_function took" in captured.out
        assert "seconds" in captured.out


class TestVectorizedConvergenceMap:
    """Test vectorized convergence map generation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.lens_sys = LensSystem(0.5, 1.5, H0=70, Om0=0.3)
        self.halo = NFWProfile(1e12, 10.0, self.lens_sys)
    
    def test_vectorized_produces_correct_shape(self):
        """Test vectorized version produces correct shape."""
        kappa_map = generate_convergence_map_vectorized(
            self.halo, grid_size=64, extent=3.0
        )
        assert kappa_map.shape == (64, 64)
    
    def test_vectorized_produces_valid_values(self):
        """Test vectorized version produces valid convergence values."""
        kappa_map = generate_convergence_map_vectorized(
            self.halo, grid_size=32, extent=2.0
        )
        # Convergence should be real and mostly positive
        assert np.all(np.isfinite(kappa_map))
        # Center should have higher convergence than edges
        center_val = kappa_map[16, 16]
        edge_val = kappa_map[0, 0]
        assert center_val > edge_val
    
    def test_vectorized_matches_single_point(self):
        """Test vectorized matches single-point evaluation."""
        grid_size = 16
        extent = 2.0
        
        # Vectorized
        kappa_map = generate_convergence_map_vectorized(
            self.halo, grid_size=grid_size, extent=extent
        )
        
        # Single point check
        x_test = np.linspace(-extent, extent, grid_size)[8]
        y_test = np.linspace(-extent, extent, grid_size)[8]
        kappa_single = self.halo.convergence(x_test, y_test)
        
        # Should match (allowing for small numerical differences)
        assert np.isclose(kappa_map[8, 8], kappa_single, rtol=1e-5)
    
    def test_different_grid_sizes(self):
        """Test vectorized works with different grid sizes."""
        for size in [16, 32, 64, 128]:
            kappa_map = generate_convergence_map_vectorized(
                self.halo, grid_size=size, extent=3.0
            )
            assert kappa_map.shape == (size, size)


class TestBenchmarking:
    """Test benchmarking utilities."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.lens_sys = LensSystem(0.5, 1.5, H0=70, Om0=0.3)
        self.halo = NFWProfile(1e12, 10.0, self.lens_sys)
    
    def test_benchmark_convergence_map(self):
        """Test convergence map benchmarking."""
        results = benchmark_convergence_map(
            self.halo,
            grid_sizes=[16, 32],
            use_gpu=False
        )
        
        assert 'grid_sizes' in results
        assert 'timings' in results
        assert 'backend' in results
        assert len(results['timings']) == 2
        assert all(t > 0 for t in results['timings'])
    
    def test_benchmark_scaling(self):
        """Test that timing scales with grid size."""
        results = benchmark_convergence_map(
            self.halo,
            grid_sizes=[16, 32, 64],
            use_gpu=False
        )
        
        timings = results['timings']
        # Larger grids should take more time (not strict due to overhead)
        assert timings[2] > timings[0]


class TestCaching:
    """Test convergence map caching."""
    
    def setup_method(self):
        """Set up test fixtures."""
        clear_cache()  # Start with clean cache
        self.lens_sys = LensSystem(0.5, 1.5, H0=70, Om0=0.3)
        self.halo = NFWProfile(1e12, 10.0, self.lens_sys)
    
    def test_cached_convergence(self):
        """Test cached convergence returns valid result."""
        kappa_map = cached_convergence(self.halo, grid_size=32, extent=2.0)
        assert kappa_map.shape == (32, 32)
        assert np.all(np.isfinite(kappa_map))
    
    def test_cache_reuse(self):
        """Test cache returns same result on second call."""
        kappa_map1 = cached_convergence(self.halo, grid_size=32, extent=2.0)
        kappa_map2 = cached_convergence(self.halo, grid_size=32, extent=2.0)
        
        # Should be identical (same memory location)
        assert np.array_equal(kappa_map1, kappa_map2)
    
    def test_cache_different_params(self):
        """Test cache distinguishes different parameters."""
        kappa_map1 = cached_convergence(self.halo, grid_size=32, extent=2.0)
        kappa_map2 = cached_convergence(self.halo, grid_size=64, extent=2.0)
        
        # Different grid sizes = different results
        assert kappa_map1.shape != kappa_map2.shape
    
    def test_clear_cache(self):
        """Test clearing cache."""
        _ = cached_convergence(self.halo, grid_size=32, extent=2.0)
        clear_cache()
        
        # Cache should be empty (no easy way to verify directly)
        # Just test it doesn't crash
        _ = cached_convergence(self.halo, grid_size=32, extent=2.0)


class TestGPUSupport:
    """Test GPU support (conditional on CuPy availability)."""
    
    def test_gpu_availability_flag(self):
        """Test GPU availability flag is set."""
        # Just check it's a boolean
        assert isinstance(GPU_AVAILABLE, bool)
    
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="CuPy not available")
    def test_gpu_backend(self):
        """Test GPU backend if available."""
        backend = ArrayBackend(use_gpu=True)
        assert "GPU" in backend.backend_name
    
    @pytest.mark.skipif(GPU_AVAILABLE, reason="CuPy is available")
    def test_fallback_to_cpu(self):
        """Test fallback to CPU when GPU not available."""
        backend = ArrayBackend(use_gpu=True)
        assert backend.backend_name == "NumPy (CPU)"


class TestBackendSwitching:
    """Test backend switching."""
    
    def test_set_backend_cpu(self):
        """Test setting CPU backend."""
        set_backend(use_gpu=False)
        backend = get_backend()
        assert "CPU" in backend.backend_name
    
    def test_get_backend(self):
        """Test getting current backend."""
        backend = get_backend()
        assert hasattr(backend, 'backend_name')
        assert hasattr(backend, 'xp')


class TestIntegrationWithProfiles:
    """Test performance optimizations with various lens profiles."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.lens_sys = LensSystem(0.5, 1.5, H0=70, Om0=0.3)
    
    def test_with_nfw_profile(self):
        """Test vectorized generation with NFW profile."""
        from src.lens_models.mass_profiles import NFWProfile
        
        halo = NFWProfile(1e12, 10.0, self.lens_sys)
        kappa_map = generate_convergence_map_vectorized(halo, grid_size=32)
        
        assert kappa_map.shape == (32, 32)
        assert np.all(np.isfinite(kappa_map))
    
    def test_with_point_mass(self):
        """Test vectorized generation with point mass."""
        from src.lens_models import PointMassProfile
        
        lens = PointMassProfile(1e11, self.lens_sys)
        kappa_map = generate_convergence_map_vectorized(lens, grid_size=32)
        
        assert kappa_map.shape == (32, 32)
        assert np.all(np.isfinite(kappa_map))
    
    def test_with_elliptical_nfw(self):
        """Test vectorized generation with elliptical NFW."""
        from src.lens_models.advanced_profiles import EllipticalNFWProfile
        
        halo = EllipticalNFWProfile(
            M_vir=1e12,
            concentration=10.0,
            lens_sys=self.lens_sys,
            ellipticity=0.3,
            position_angle=45.0
        )
        kappa_map = generate_convergence_map_vectorized(halo, grid_size=32)
        
        assert kappa_map.shape == (32, 32)
        assert np.all(np.isfinite(kappa_map))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
