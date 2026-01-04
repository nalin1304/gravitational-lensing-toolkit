"""
Test Suite for Phase 13: Scientific Validation & Benchmarking

Author: Phase 13 Implementation
Date: October 2025
"""

import pytest
import numpy as np
from pathlib import Path
import json
from unittest.mock import Mock, patch

# Import benchmark modules
from benchmarks.metrics import (
    calculate_relative_error,
    calculate_chi_squared,
    calculate_rmse,
    calculate_mae,
    calculate_structural_similarity,
    calculate_peak_signal_noise_ratio,
    calculate_pearson_correlation,
    calculate_fractional_bias,
    calculate_residuals,
    calculate_confidence_interval,
    calculate_normalized_cross_correlation,
    calculate_all_metrics,
)
from benchmarks.profiler import (
    ProfilerContext,
    time_profile,
    memory_profile,
    profile_function,
    profile_block,
    PerformanceBenchmark,
    compare_implementations,
)
from benchmarks.comparisons import (
    analytic_nfw_convergence,
    compare_with_analytic,
    benchmark_convergence_accuracy,
    benchmark_inference_speed,
)


# ============================================================================
# Metrics Tests
# ============================================================================

class TestMetrics:
    """Test scientific validation metrics"""
    
    def test_relative_error_perfect_match(self):
        """Test relative error with perfect match"""
        predicted = np.array([1.0, 2.0, 3.0])
        ground_truth = np.array([1.0, 2.0, 3.0])
        
        error = calculate_relative_error(predicted, ground_truth)
        
        assert error == 0.0, "Perfect match should give zero error"
    
    def test_relative_error_with_zeros(self):
        """Test relative error with zero ground truth (uses epsilon)"""
        predicted = np.array([0.1, 0.2, 0.3])
        ground_truth = np.array([0.0, 0.0, 0.0])
        
        error = calculate_relative_error(predicted, ground_truth)
        
        assert error > 0, "Should handle zeros with epsilon"
        assert np.isfinite(error), "Should not produce inf/nan"
    
    def test_chi_squared_perfect_fit(self):
        """Test chi-squared with perfect fit"""
        observed = np.array([10.0, 20.0, 30.0])
        expected = np.array([10.0, 20.0, 30.0])
        uncertainties = np.array([1.0, 2.0, 3.0])
        
        chi2, p_value = calculate_chi_squared(observed, expected, uncertainties)
        
        assert chi2 == 0.0, "Perfect fit should give chi2=0"
        assert p_value == 1.0, "Perfect fit should give p=1"
    
    def test_rmse(self):
        """Test RMSE calculation"""
        predicted = np.array([1.0, 2.0, 3.0])
        ground_truth = np.array([1.1, 2.1, 3.1])
        
        rmse = calculate_rmse(predicted, ground_truth)
        expected_rmse = 0.1
        
        assert np.isclose(rmse, expected_rmse, atol=1e-10)
    
    def test_mae(self):
        """Test MAE calculation"""
        predicted = np.array([1.0, 2.0, 3.0])
        ground_truth = np.array([1.1, 2.1, 2.9])
        
        mae = calculate_mae(predicted, ground_truth)
        expected_mae = 0.1
        
        assert np.isclose(mae, expected_mae, atol=1e-10)
    
    def test_ssim_identical_images(self):
        """Test SSIM with identical 2D images"""
        image = np.random.rand(64, 64)
        
        ssim = calculate_structural_similarity(image, image)
        
        assert np.isclose(ssim, 1.0, atol=1e-6), "Identical images should have SSIM=1"
    
    def test_psnr_identical_images(self):
        """Test PSNR with identical images"""
        image = np.random.rand(64, 64)
        
        psnr = calculate_peak_signal_noise_ratio(image, image)
        
        assert psnr > 100, "Identical images should have very high PSNR"
    
    def test_pearson_correlation_perfect(self):
        """Test Pearson correlation with perfect correlation"""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 2 * x + 1  # Perfect linear relationship
        
        corr, p_value = calculate_pearson_correlation(x, y)
        
        assert np.isclose(corr, 1.0, atol=1e-10), "Perfect correlation should give r=1"
        assert p_value < 0.05, "Should be statistically significant"
    
    def test_fractional_bias_zero(self):
        """Test fractional bias with perfect match"""
        predicted = np.array([1.0, 2.0, 3.0])
        observed = np.array([1.0, 2.0, 3.0])
        
        bias = calculate_fractional_bias(predicted, observed)
        
        assert bias == 0.0, "Perfect match should give zero bias"
    
    def test_residuals(self):
        """Test residual statistics"""
        predicted = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ground_truth = np.array([1.1, 2.0, 2.9, 4.1, 5.0])
        
        stats = calculate_residuals(predicted, ground_truth)
        
        assert 'mean' in stats
        assert 'std' in stats
        assert 'median' in stats
        assert 'q25' in stats
        assert 'q75' in stats
        assert 'iqr' in stats
        assert np.isclose(stats['mean'], 0.0, atol=0.1)
    
    def test_confidence_interval(self):
        """Test confidence interval calculation"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        lower, upper = calculate_confidence_interval(data)
        mean = np.mean(data)
        
        assert lower < mean < upper, "Mean should be within CI"
        assert upper - lower > 0, "CI should have positive width"
    
    def test_normalized_cross_correlation(self):
        """Test normalized cross-correlation"""
        template = np.random.rand(32, 32)
        
        ncc = calculate_normalized_cross_correlation(template, template)
        
        assert np.isclose(ncc, 1.0, atol=1e-6), "Identical images should have NCC=1"
    
    def test_all_metrics(self):
        """Test aggregate metrics calculation"""
        predicted = np.random.rand(64, 64)
        ground_truth = predicted + 0.01 * np.random.rand(64, 64)
        
        metrics = calculate_all_metrics(predicted, ground_truth)
        
        # Check all expected keys (note: residuals are prefixed with residual_)
        expected_keys = [
            'relative_error', 'rmse', 'mae', 'ssim', 'psnr',
            'pearson_correlation', 'pearson_pvalue', 'fractional_bias',
            'normalized_cross_correlation'
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing metric: {key}"
        
        # Check residual keys
        residual_keys = ['residual_mean', 'residual_std', 'residual_median']
        for key in residual_keys:
            assert key in metrics, f"Missing residual metric: {key}"
        
        # Check values are reasonable
        assert 0 <= metrics['ssim'] <= 1
        assert metrics['psnr'] > 0
        assert -1 <= metrics['pearson_correlation'] <= 1


# ============================================================================
# Profiler Tests
# ============================================================================

class TestProfiler:
    """Test performance profiling tools"""
    
    def test_profiler_context(self):
        """Test ProfilerContext manager"""
        with ProfilerContext() as prof:
            # Simulate some work
            _ = [i**2 for i in range(1000)]
        
        assert prof.duration > 0, "Should measure positive time"
        assert prof.memory_peak >= 0, "Should measure memory"
    
    def test_time_profile_decorator(self):
        """Test time profiling decorator"""
        
        @time_profile
        def slow_function():
            return sum(range(1000))
        
        result = slow_function()
        
        assert result == sum(range(1000)), "Should return correct result"
        # Decorator should print timing info (can't easily test print output)
    
    def test_memory_profile_decorator(self):
        """Test memory profiling decorator"""
        
        @memory_profile
        def memory_function():
            data = [0] * 10000
            return len(data)
        
        result = memory_function()
        
        assert result == 10000, "Should return correct result"
    
    def test_profile_function_decorator(self):
        """Test combined profiling decorator"""
        
        @profile_function
        def combined_function():
            return [i**2 for i in range(1000)]
        
        result = combined_function()
        
        assert len(result) == 1000, "Should return correct result"
    
    def test_profile_block(self):
        """Test profile_block context manager"""
        with profile_block("test operation"):
            _ = sum(range(10000))
        
        # Should complete without error
        assert True
    
    def test_performance_benchmark(self):
        """Test PerformanceBenchmark class"""
        
        def test_func():
            return sum(range(100))
        
        benchmark = PerformanceBenchmark("test_benchmark")
        
        # Run iterations using run_iterations method
        stats = benchmark.run_iterations(test_func, n_iterations=10)
        
        assert 'duration_mean' in stats
        assert 'duration_std' in stats
        assert 'duration_min' in stats
        assert 'duration_max' in stats
        assert 'duration_median' in stats
        assert 'n_iterations' in stats
        assert stats['n_iterations'] == 10
        assert stats['duration_mean'] > 0
    
    def test_compare_implementations(self):
        """Test implementation comparison"""
        
        def impl_a():
            return sum(range(100))
        
        def impl_b():
            return sum([i for i in range(100)])
        
        # Don't pass iterations as an arg to the functions
        comparison = compare_implementations(impl_a, impl_b, n_iterations=10)
        
        # Check for actual keys returned
        assert 'impl1_name' in comparison
        assert 'impl2_name' in comparison
        assert 'speedup' in comparison
        assert 'faster' in comparison
        assert comparison['speedup'] > 0


# ============================================================================
# Comparisons Tests
# ============================================================================

class TestComparisons:
    """Test benchmark comparisons"""
    
    def test_analytic_nfw_convergence(self):
        """Test analytic NFW convergence calculation"""
        x = np.linspace(-1, 1, 10)
        y = np.linspace(-1, 1, 10)
        X, Y = np.meshgrid(x, y)
        
        kappa = analytic_nfw_convergence(
            X, Y,
            mass=1e12,
            scale_radius=200.0,
            concentration=5.0
        )
        
        assert kappa.shape == (10, 10), "Should have correct shape"
        assert np.all(kappa >= 0), "Convergence should be non-negative"
        assert np.isclose(kappa[5, 5], np.max(kappa)), "Peak should be at center"
    
    @patch('benchmarks.comparisons.generate_synthetic_convergence')
    @patch('benchmarks.comparisons.load_pretrained_model')
    def test_compare_with_analytic(self, mock_model, mock_generate):
        """Test comparison with analytic solution"""
        # Mock the PINN prediction
        mock_kappa = np.random.rand(64, 64)
        x = np.linspace(-1, 1, 64)
        y = np.linspace(-1, 1, 64)
        X, Y = np.meshgrid(x, y)
        mock_generate.return_value = (mock_kappa, {'x': X, 'y': Y})
        
        # Mock the model
        mock_model.return_value = Mock()
        
        results = compare_with_analytic(
            grid_size=64,
            mass=1e12,
            scale_radius=200.0
        )
        
        assert 'our_map' in results
        assert 'analytic_map' in results
        assert 'metrics' in results
        assert 'our_time' in results
        assert 'analytic_time' in results
        assert 'speedup' in results
    
    @patch('benchmarks.comparisons.compare_with_analytic')
    def test_benchmark_convergence_accuracy(self, mock_compare):
        """Test convergence accuracy benchmark"""
        # Mock comparison results
        mock_compare.return_value = {
            'metrics': {
                'relative_error': 0.01,
                'rmse': 0.001
            },
            'our_time': 0.1,
            'analytic_time': 0.05
        }
        
        results = benchmark_convergence_accuracy(
            grid_sizes=[32, 64],
            masses=[1e12]
        )
        
        assert 'grid_size_tests' in results
        assert 'mass_tests' in results
        assert 'summary' in results
        assert len(results['grid_size_tests']) == 2
        assert len(results['mass_tests']) == 1
    
    @patch('benchmarks.comparisons.generate_synthetic_convergence')
    @patch('benchmarks.comparisons.load_pretrained_model')
    @patch('benchmarks.comparisons.prepare_model_input')
    def test_benchmark_inference_speed(self, mock_prepare, mock_model, mock_generate):
        """Test inference speed benchmark"""
        # Mock the generation function
        mock_kappa = np.random.rand(64, 64)
        mock_generate.return_value = (mock_kappa, {})
        
        # Mock model
        mock_model_instance = Mock()
        mock_model_instance.return_value = mock_kappa
        mock_model.return_value = mock_model_instance
        
        # Mock input prep
        mock_prepare.return_value = mock_kappa
        
        results = benchmark_inference_speed(
            n_runs=5,
            grid_size=64
        )
        
        assert 'mean_time' in results
        assert 'std_time' in results
        assert 'min_time' in results
        assert 'max_time' in results
        assert 'median_time' in results
        assert 'throughput' in results
        assert results['n_runs'] == 5
        assert results['grid_size'] == 64
        assert results['throughput'] > 0


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Test end-to-end benchmark workflows"""
    
    @patch('benchmarks.comparisons.compare_with_analytic')
    @patch('benchmarks.comparisons.benchmark_inference_speed')
    @patch('benchmarks.comparisons.benchmark_convergence_accuracy')
    def test_comprehensive_benchmark(self, mock_accuracy, mock_speed, mock_analytic):
        """Test comprehensive benchmark suite"""
        from benchmarks.comparisons import run_comprehensive_benchmark
        
        # Mock the component benchmarks
        mock_accuracy.return_value = {
            'grid_size_tests': [{'grid_size': 64, 'mean_error': 0.01}],
            'mass_tests': [{'mass': 1e12, 'mean_error': 0.01}],
            'summary': {
                'overall_mean_error': 0.01,
                'overall_std_error': 0.001,
                'best_grid_size': 64
            }
        }
        mock_speed.return_value = {
            'mean_time': 0.1,
            'throughput': 10.0,
            'n_runs': 10,
            'grid_size': 64
        }
        mock_analytic.return_value = {
            'our_map': np.random.rand(64, 64),
            'analytic_map': np.random.rand(64, 64),
            'metrics': {'relative_error': 0.01},
            'our_time': 0.1,
            'analytic_time': 0.05
        }
        
        results = run_comprehensive_benchmark()
        
        assert 'accuracy' in results
        assert 'speed' in results
        assert 'analytic' in results
        assert 'timestamp' in results
    
    def test_json_serialization(self):
        """Test that results can be serialized to JSON"""
        results = {
            'accuracy': {
                'grid_size_tests': [
                    {'grid_size': 32, 'mean_error': 0.01}
                ],
                'best_grid_size': 32
            },
            'speed': {
                'mean_time': 0.1,
                'throughput': 10.0
            }
        }
        
        # Should not raise exception
        json_str = json.dumps(results)
        parsed = json.loads(json_str)
        
        assert parsed['accuracy']['best_grid_size'] == 32


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_convergence_map():
    """Generate sample convergence map for testing"""
    x = np.linspace(-1, 1, 64)
    y = np.linspace(-1, 1, 64)
    X, Y = np.meshgrid(x, y)
    
    # Simple Gaussian
    kappa = np.exp(-(X**2 + Y**2) / 0.5)
    
    return kappa


@pytest.fixture
def sample_metrics():
    """Generate sample metrics for testing"""
    return {
        'relative_error': 0.01,
        'rmse': 0.001,
        'mae': 0.0008,
        'ssim': 0.95,
        'psnr': 40.0,
        'pearson_correlation': 0.99,
        'fractional_bias': 0.005
    }


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Test performance characteristics"""
    
    def test_metrics_performance(self, sample_convergence_map):
        """Test that metrics calculation is fast"""
        import time
        
        start = time.perf_counter()
        _ = calculate_all_metrics(
            sample_convergence_map,
            sample_convergence_map
        )
        elapsed = time.perf_counter() - start
        
        assert elapsed < 1.0, "Metrics should compute in <1 second"
    
    def test_analytic_nfw_performance(self):
        """Test analytic NFW calculation performance"""
        import time
        
        x = np.linspace(-2, 2, 128)
        y = np.linspace(-2, 2, 128)
        X, Y = np.meshgrid(x, y)
        
        start = time.perf_counter()
        _ = analytic_nfw_convergence(X, Y, mass=1e12, scale_radius=200.0, concentration=5.0)
        elapsed = time.perf_counter() - start
        
        assert elapsed < 0.5, "Analytic calculation should be fast"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
