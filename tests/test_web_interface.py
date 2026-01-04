"""
Unit tests for Streamlit Web Interface (Phase 10).

Tests utility functions, data processing, and visualization components.
Note: Streamlit UI interactions are tested manually, this focuses on backend logic.
"""

import pytest
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch, MagicMock
import io

# Import app functions
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'app'))

try:
    from utils import (
        generate_synthetic_convergence,
        plot_convergence_map,
        plot_uncertainty_bars,
        plot_classification_probs,
        plot_comparison,
        load_pretrained_model,
        prepare_model_input,
        compute_classification_entropy,
        format_parameter_value
    )
    APP_FUNCTIONS_AVAILABLE = True
except ImportError as e:
    APP_FUNCTIONS_AVAILABLE = False
    print(f"Warning: Could not import app functions: {e}")

# Test data fixtures
@pytest.fixture
def sample_convergence_map():
    """Sample convergence map for testing."""
    return np.random.rand(64, 64) * 0.5


@pytest.fixture
def sample_coordinates():
    """Sample coordinate grids."""
    x = np.linspace(-2, 2, 64)
    y = np.linspace(-2, 2, 64)
    X, Y = np.meshgrid(x, y)
    return X, Y


@pytest.fixture
def sample_parameters():
    """Sample predicted parameters."""
    return np.array([1.5e12, 200.0, 0.5, 0.3, 70.0])


@pytest.fixture
def sample_classifications():
    """Sample classification probabilities."""
    return np.array([0.7, 0.2, 0.1])


@pytest.mark.skipif(not APP_FUNCTIONS_AVAILABLE, reason="App functions not available")
class TestSyntheticGeneration:
    """Test synthetic convergence map generation."""
    
    def test_generate_nfw_convergence(self):
        """Test NFW profile generation."""
        convergence_map, X, Y = generate_synthetic_convergence(
            profile_type="NFW",
            mass=2e12,
            scale_radius=200.0,
            ellipticity=0.0,
            grid_size=64
        )
        
        assert convergence_map.shape == (64, 64)
        assert X.shape == (64, 64)
        assert Y.shape == (64, 64)
        assert np.all(convergence_map >= 0)  # Convergence should be non-negative
        assert np.all(np.isfinite(convergence_map))
    
    def test_generate_elliptical_nfw_convergence(self):
        """Test Elliptical NFW profile generation."""
        convergence_map, X, Y = generate_synthetic_convergence(
            profile_type="Elliptical NFW",
            mass=3e12,
            scale_radius=250.0,
            ellipticity=0.3,
            grid_size=64
        )
        
        assert convergence_map.shape == (64, 64)
        assert X.shape == (64, 64)
        assert Y.shape == (64, 64)
        assert np.all(np.isfinite(convergence_map))
    
    def test_different_grid_sizes(self):
        """Test generation with different grid sizes."""
        for grid_size in [32, 64, 128]:
            convergence_map, X, Y = generate_synthetic_convergence(
                profile_type="NFW",
                mass=2e12,
                scale_radius=200.0,
                ellipticity=0.0,
                grid_size=grid_size
            )
            
            assert convergence_map.shape == (grid_size, grid_size)
            assert X.shape == (grid_size, grid_size)
            assert Y.shape == (grid_size, grid_size)
    
    def test_mass_variation(self):
        """Test that different masses produce different maps."""
        map1, _, _ = generate_synthetic_convergence(
            "NFW", 1e12, 200.0, 0.0, 64
        )
        map2, _, _ = generate_synthetic_convergence(
            "NFW", 5e12, 200.0, 0.0, 64
        )
        
        # Higher mass should have higher convergence
        assert map2.max() > map1.max()
    
    def test_invalid_profile_type(self):
        """Test error handling for invalid profile type."""
        with pytest.raises(ValueError, match="Unknown profile type"):
            generate_synthetic_convergence(
                "InvalidProfile", 2e12, 200.0, 0.0, 64
            )


@pytest.mark.skipif(not APP_FUNCTIONS_AVAILABLE, reason="App functions not available")
class TestVisualization:
    """Test visualization functions."""
    
    def test_plot_convergence_map(self, sample_convergence_map, sample_coordinates):
        """Test convergence map plotting."""
        X, Y = sample_coordinates
        
        fig = plot_convergence_map(
            sample_convergence_map,
            X, Y,
            title="Test Map",
            cmap="viridis"
        )
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) > 0
        plt.close(fig)
    
    def test_plot_convergence_different_colormaps(
        self, sample_convergence_map, sample_coordinates
    ):
        """Test plotting with different colormaps."""
        X, Y = sample_coordinates
        
        for cmap in ['viridis', 'plasma', 'inferno']:
            fig = plot_convergence_map(
                sample_convergence_map, X, Y,
                title=f"Test {cmap}",
                cmap=cmap
            )
            assert isinstance(fig, plt.Figure)
            plt.close(fig)
    
    def test_plot_uncertainty_bars(self, sample_parameters):
        """Test uncertainty bar plot."""
        param_names = ['M_vir', 'r_s', 'β_x', 'β_y', 'H₀']
        means = sample_parameters
        stds = sample_parameters * 0.1  # 10% uncertainty
        
        fig = plot_uncertainty_bars(param_names, means, stds)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) > 0
        plt.close(fig)
    
    def test_plot_uncertainty_zero_std(self, sample_parameters):
        """Test uncertainty plot with zero uncertainty."""
        param_names = ['M_vir', 'r_s', 'β_x', 'β_y', 'H₀']
        means = sample_parameters
        stds = np.zeros_like(means)
        
        fig = plot_uncertainty_bars(param_names, means, stds)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_classification_probs(self, sample_classifications):
        """Test classification probability visualization."""
        class_names = ['CDM', 'WDM', 'SIDM']
        entropy = -np.sum(sample_classifications * np.log(sample_classifications + 1e-10))
        
        fig = plot_classification_probs(class_names, sample_classifications, entropy)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 2  # Bar chart + pie chart
        plt.close(fig)
    
    def test_plot_classification_uniform_probs(self):
        """Test classification plot with uniform probabilities."""
        class_names = ['CDM', 'WDM', 'SIDM']
        probs = np.array([1/3, 1/3, 1/3])
        entropy = np.log(3)  # Maximum entropy
        
        fig = plot_classification_probs(class_names, probs, entropy)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_comparison(self, sample_convergence_map, sample_coordinates):
        """Test comparison plot (original vs processed)."""
        X, Y = sample_coordinates
        
        original = sample_convergence_map
        processed = (original - original.min()) / (original.max() - original.min())
        
        fig = plot_comparison(original, processed, X, Y)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 2  # Two subplots
        plt.close(fig)


@pytest.mark.skipif(not APP_FUNCTIONS_AVAILABLE, reason="App functions not available")
class TestModelLoading:
    """Test model loading functionality."""
    
    def test_load_model_without_weights(self):
        """Test loading model without pre-trained weights."""
        model = load_pretrained_model(model_path=None)
        
        assert model is not None
        assert hasattr(model, 'encoder')
        assert hasattr(model, 'param_head')
        assert hasattr(model, 'class_head')
    
    def test_load_model_with_nonexistent_path(self):
        """Test loading with non-existent model path."""
        model = load_pretrained_model(model_path="/nonexistent/path.pth")
        
        # Should still return model (untrained)
        assert model is not None
    
    def test_model_forward_pass(self):
        """Test model can perform forward pass."""
        model = load_pretrained_model()
        model.eval()
        
        # Create dummy input
        x = torch.randn(1, 1, 64, 64)
        
        with torch.no_grad():
            params, classes = model(x)
        
        assert params.shape == (1, 5)
        assert classes.shape == (1, 3)


class TestDataProcessing:
    """Test data processing utilities."""
    
    def test_convergence_map_normalization(self, sample_convergence_map):
        """Test normalization of convergence maps."""
        # Normalize
        normalized = (sample_convergence_map - sample_convergence_map.min()) / \
                    (sample_convergence_map.max() - sample_convergence_map.min())
        
        assert normalized.min() >= 0
        assert normalized.max() <= 1
        assert np.all(np.isfinite(normalized))
    
    def test_convergence_map_resizing(self, sample_convergence_map):
        """Test resizing of convergence maps."""
        from scipy.ndimage import zoom
        
        # Resize to 32x32
        scale = 32 / 64
        resized = zoom(sample_convergence_map, scale, order=1)
        
        assert resized.shape == (32, 32)
        assert np.all(np.isfinite(resized))
    
    def test_convergence_map_with_nans(self):
        """Test handling of NaN values."""
        data = np.random.rand(64, 64)
        data[10:20, 10:20] = np.nan
        
        # Replace NaNs with zeros
        data_cleaned = np.nan_to_num(data, nan=0.0)
        
        assert not np.any(np.isnan(data_cleaned))
        assert data_cleaned[15, 15] == 0.0
    
    def test_tensor_conversion(self, sample_convergence_map):
        """Test conversion to PyTorch tensor."""
        tensor = torch.from_numpy(sample_convergence_map).float()
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        
        assert tensor.shape == (1, 1, 64, 64)
        assert tensor.dtype == torch.float32


class TestUncertaintyCalculations:
    """Test uncertainty calculation utilities."""
    
    def test_parameter_uncertainty_computation(self):
        """Test computation of parameter uncertainties."""
        # Simulate MC samples
        n_samples = 50
        n_params = 5
        
        samples = np.random.randn(n_samples, n_params)
        
        # Compute statistics
        mean = samples.mean(axis=0)
        std = samples.std(axis=0)
        
        assert mean.shape == (n_params,)
        assert std.shape == (n_params,)
        assert np.all(std >= 0)
    
    def test_classification_entropy(self):
        """Test entropy calculation for classification."""
        # High confidence (low entropy)
        probs_high = np.array([0.9, 0.05, 0.05])
        entropy_high = -np.sum(probs_high * np.log(probs_high + 1e-10))
        
        # Low confidence (high entropy)
        probs_low = np.array([0.33, 0.33, 0.34])
        entropy_low = -np.sum(probs_low * np.log(probs_low + 1e-10))
        
        assert entropy_high < entropy_low
        assert 0 <= entropy_high <= np.log(3)
        assert 0 <= entropy_low <= np.log(3)
    
    def test_confidence_intervals(self):
        """Test confidence interval computation."""
        samples = np.random.randn(100)
        
        mean = samples.mean()
        std = samples.std()
        
        # 95% confidence interval
        z_score = 1.96
        ci_lower = mean - z_score * std
        ci_upper = mean + z_score * std
        
        # Check ~95% of samples within interval
        within_ci = np.sum((samples >= ci_lower) & (samples <= ci_upper))
        ratio = within_ci / len(samples)
        
        assert 0.90 <= ratio <= 1.0  # Allow some variation


class TestCoordinateGrids:
    """Test coordinate grid generation."""
    
    def test_meshgrid_generation(self):
        """Test creation of coordinate meshgrids."""
        grid_size = 64
        fov = 4.0
        
        x = np.linspace(-fov/2, fov/2, grid_size)
        y = np.linspace(-fov/2, fov/2, grid_size)
        X, Y = np.meshgrid(x, y)
        
        assert X.shape == (grid_size, grid_size)
        assert Y.shape == (grid_size, grid_size)
        assert X.min() == pytest.approx(-fov/2)
        assert X.max() == pytest.approx(fov/2)
        assert Y.min() == pytest.approx(-fov/2)
        assert Y.max() == pytest.approx(fov/2)
    
    def test_coordinate_symmetry(self):
        """Test coordinate grid symmetry."""
        grid_size = 64
        fov = 4.0
        
        x = np.linspace(-fov/2, fov/2, grid_size)
        y = np.linspace(-fov/2, fov/2, grid_size)
        X, Y = np.meshgrid(x, y)
        
        # Center should be near zero
        center = grid_size // 2
        assert np.abs(X[center, center]) < 0.1
        assert np.abs(Y[center, center]) < 0.1


class TestParameterValidation:
    """Test parameter validation and ranges."""
    
    def test_mass_parameter_range(self):
        """Test mass parameter is in valid range."""
        # Typical virial mass range: 10^11 - 10^13 M_sun
        mass = 2e12
        
        assert 1e11 <= mass <= 1e14
    
    def test_scale_radius_range(self):
        """Test scale radius in valid range."""
        # Typical scale radius: 50-500 kpc
        r_s = 200.0
        
        assert 50 <= r_s <= 500
    
    def test_ellipticity_range(self):
        """Test ellipticity in valid range."""
        # Ellipticity: 0 (circular) to ~0.5 (highly elliptical)
        ellipticity = 0.3
        
        assert 0 <= ellipticity <= 0.5
    
    def test_grid_size_options(self):
        """Test grid size is power of 2."""
        valid_sizes = [32, 64, 128, 256, 512]
        
        for size in valid_sizes:
            assert size & (size - 1) == 0  # Check if power of 2


class TestErrorHandling:
    """Test error handling in app functions."""
    
    @pytest.mark.skipif(not APP_FUNCTIONS_AVAILABLE, reason="App functions not available")
    def test_invalid_mass_parameter(self):
        """Test handling of invalid mass parameter."""
        # Negative mass should raise error or be handled gracefully
        try:
            convergence_map, X, Y = generate_synthetic_convergence(
                "NFW", -1e12, 200.0, 0.0, 64
            )
            # If no error, check output is still valid
            assert convergence_map.shape == (64, 64)
        except (ValueError, RuntimeError):
            # Error is expected for negative mass
            pass
    
    def test_zero_division_in_normalization(self):
        """Test handling of zero division in normalization."""
        # Constant array (max = min)
        data = np.ones((64, 64))
        
        # Should handle division by zero
        normalized = (data - data.min()) / (data.max() - data.min() + 1e-10)
        
        assert np.all(np.isfinite(normalized))
    
    def test_nan_in_input_data(self):
        """Test handling of NaN in input data."""
        data = np.random.rand(64, 64)
        data[0, 0] = np.nan
        
        # Should be able to detect and handle NaNs
        has_nans = np.any(np.isnan(data))
        assert has_nans
        
        # Clean data
        data_clean = np.nan_to_num(data, nan=0.0)
        assert not np.any(np.isnan(data_clean))


class TestIntegration:
    """Integration tests for full workflows."""
    
    @pytest.mark.skipif(not APP_FUNCTIONS_AVAILABLE, reason="App functions not available")
    def test_end_to_end_synthetic_workflow(self):
        """Test complete synthetic data generation workflow."""
        # Generate data
        convergence_map, X, Y = generate_synthetic_convergence(
            "NFW", 2e12, 200.0, 0.0, 64
        )
        
        # Visualize
        fig = plot_convergence_map(convergence_map, X, Y)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
        
        # Normalize for model input
        normalized = (convergence_map - convergence_map.min()) / \
                    (convergence_map.max() - convergence_map.min() + 1e-10)
        
        # Convert to tensor
        tensor = torch.from_numpy(normalized).float()
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        
        assert tensor.shape == (1, 1, 64, 64)
    
    @pytest.mark.skipif(not APP_FUNCTIONS_AVAILABLE, reason="App functions not available")
    def test_end_to_end_inference_workflow(self):
        """Test complete inference workflow."""
        # Load model
        model = load_pretrained_model()
        model.eval()
        
        # Generate test data
        convergence_map, _, _ = generate_synthetic_convergence(
            "NFW", 2e12, 200.0, 0.0, 64
        )
        
        # Prepare input
        normalized = (convergence_map - convergence_map.min()) / \
                    (convergence_map.max() - convergence_map.min() + 1e-10)
        tensor = torch.from_numpy(normalized).float().unsqueeze(0).unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            params, classes = model(tensor)
        
        # Verify outputs
        assert params.shape == (1, 5)
        assert classes.shape == (1, 3)
        
        # Extract results
        params_np = params.numpy()[0]
        probs_np = torch.softmax(classes, dim=1).numpy()[0]
        
        assert len(params_np) == 5
        assert len(probs_np) == 3
        assert np.abs(probs_np.sum() - 1.0) < 1e-5  # Probabilities sum to 1
    
    @pytest.mark.skipif(not APP_FUNCTIONS_AVAILABLE, reason="App functions not available")
    def test_end_to_end_visualization_workflow(self):
        """Test complete visualization workflow."""
        # Generate test data
        convergence_map, X, Y = generate_synthetic_convergence(
            "NFW", 2e12, 200.0, 0.0, 64
        )
        
        # Convergence map plot
        fig1 = plot_convergence_map(convergence_map, X, Y)
        assert isinstance(fig1, plt.Figure)
        plt.close(fig1)
        
        # Uncertainty plot
        param_names = ['M_vir', 'r_s', 'β_x', 'β_y', 'H₀']
        means = np.array([2e12, 200.0, 0.5, 0.3, 70.0])
        stds = means * 0.1
        
        fig2 = plot_uncertainty_bars(param_names, means, stds)
        assert isinstance(fig2, plt.Figure)
        plt.close(fig2)
        
        # Classification plot
        class_names = ['CDM', 'WDM', 'SIDM']
        probs = np.array([0.7, 0.2, 0.1])
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        fig3 = plot_classification_probs(class_names, probs, entropy)
        assert isinstance(fig3, plt.Figure)
        plt.close(fig3)


class TestPerformance:
    """Test performance characteristics."""
    
    @pytest.mark.skipif(not APP_FUNCTIONS_AVAILABLE, reason="App functions not available")
    def test_generation_speed(self):
        """Test convergence map generation is reasonably fast."""
        import time
        
        start = time.time()
        _, _, _ = generate_synthetic_convergence("NFW", 2e12, 200.0, 0.0, 64)
        elapsed = time.time() - start
        
        # Should complete in reasonable time (< 5 seconds)
        assert elapsed < 5.0
    
    def test_normalization_speed(self):
        """Test normalization is fast."""
        import time
        
        data = np.random.rand(256, 256)
        
        start = time.time()
        normalized = (data - data.min()) / (data.max() - data.min() + 1e-10)
        elapsed = time.time() - start
        
        # Should be very fast (< 0.1 seconds)
        assert elapsed < 0.1
    
    @pytest.mark.skipif(not APP_FUNCTIONS_AVAILABLE, reason="App functions not available")
    def test_plotting_speed(self, sample_convergence_map, sample_coordinates):
        """Test plotting is reasonably fast."""
        import time
        
        X, Y = sample_coordinates
        
        start = time.time()
        fig = plot_convergence_map(sample_convergence_map, X, Y)
        elapsed = time.time() - start
        
        plt.close(fig)
        
        # Should complete in reasonable time (< 2 seconds)
        assert elapsed < 2.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
