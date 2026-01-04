"""
Unit tests for ML module (Phase 5: PINN Implementation)
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import tempfile
import h5py

from src.ml import (
    PhysicsInformedNN,
    physics_informed_loss,
    generate_training_data,
    evaluate_model,
    compute_metrics
)
from src.ml.generate_dataset import (
    generate_convergence_map,
    add_noise,
    generate_single_sample,
    LensDataset
)
from src.ml.pinn import train_step, validate_step


class TestPhysicsInformedNN:
    """Tests for PhysicsInformedNN architecture"""
    
    def test_model_initialization(self):
        """Test that model initializes correctly"""
        model = PhysicsInformedNN(input_size=64, dropout_rate=0.2)
        
        assert model is not None
        assert hasattr(model, 'encoder')
        assert hasattr(model, 'param_head')
        assert hasattr(model, 'class_head')
    
    def test_forward_pass_shapes(self):
        """Test forward pass produces correct output shapes"""
        model = PhysicsInformedNN(input_size=64, dropout_rate=0.2)
        batch_size = 8
        
        # Create dummy input
        x = torch.randn(batch_size, 1, 64, 64)
        
        # Forward pass
        params, class_logits = model(x)
        
        # Check shapes
        assert params.shape == (batch_size, 5), "Parameter output shape incorrect"
        assert class_logits.shape == (batch_size, 3), "Classification output shape incorrect"
    
    def test_predict_method(self):
        """Test predict method returns interpretable outputs"""
        model = PhysicsInformedNN(input_size=64, dropout_rate=0.2)
        model.eval()
        
        x = torch.randn(4, 1, 64, 64)
        
        with torch.no_grad():
            results = model.predict(x)
        
        # Check output structure
        assert 'params' in results
        assert 'M_vir' in results
        assert 'r_s' in results
        assert 'beta_x' in results
        assert 'beta_y' in results
        assert 'H0' in results
        assert 'class_probs' in results
        assert 'class_labels' in results
        assert 'dm_type' in results
        
        # Check shapes
        assert results['params'].shape == (4, 5)
        assert results['class_probs'].shape == (4, 3)
        assert results['class_labels'].shape == (4,)
        assert len(results['dm_type']) == 4
        
        # Check probabilities sum to 1
        prob_sums = results['class_probs'].sum(dim=1)
        assert torch.allclose(prob_sums, torch.ones(4), atol=1e-5)
    
    def test_model_on_different_devices(self):
        """Test model works on both CPU and CUDA (if available)"""
        model = PhysicsInformedNN(input_size=64, dropout_rate=0.2)
        x = torch.randn(2, 1, 64, 64)
        
        # Test on CPU
        model_cpu = model.to('cpu')
        x_cpu = x.to('cpu')
        params_cpu, logits_cpu = model_cpu(x_cpu)
        assert params_cpu.device.type == 'cpu'
        assert logits_cpu.device.type == 'cpu'
        
        # Test on CUDA if available
        if torch.cuda.is_available():
            model_cuda = model.to('cuda')
            x_cuda = x.to('cuda')
            params_cuda, logits_cuda = model_cuda(x_cuda)
            assert params_cuda.device.type == 'cuda'
            assert logits_cuda.device.type == 'cuda'


class TestPhysicsInformedLoss:
    """Tests for physics-informed loss function"""
    
    def test_loss_computation(self):
        """Test that loss computes without errors"""
        batch_size = 4
        
        pred_params = torch.randn(batch_size, 5)
        true_params = torch.randn(batch_size, 5)
        pred_classes = torch.randn(batch_size, 3)
        true_classes = torch.randint(0, 3, (batch_size,))
        images = torch.randn(batch_size, 1, 64, 64)
        
        device = torch.device('cpu')
        lambda_physics = 0.1
        
        losses = physics_informed_loss(
            pred_params, true_params, pred_classes, true_classes,
            images, lambda_physics, device
        )
        
        # Check all components are present
        assert 'total' in losses
        assert 'mse_params' in losses
        assert 'ce_class' in losses
        assert 'physics_residual' in losses
        
        # Check all losses are positive scalars
        assert losses['total'] >= 0
        assert losses['mse_params'] >= 0
        assert losses['ce_class'] >= 0
        assert losses['physics_residual'] >= 0
    
    def test_loss_components_contribution(self):
        """Test that total loss is sum of components"""
        batch_size = 4
        
        pred_params = torch.randn(batch_size, 5)
        true_params = torch.randn(batch_size, 5)
        pred_classes = torch.randn(batch_size, 3)
        true_classes = torch.randint(0, 3, (batch_size,))
        images = torch.randn(batch_size, 1, 64, 64)
        
        device = torch.device('cpu')
        lambda_physics = 0.1
        
        losses = physics_informed_loss(
            pred_params, true_params, pred_classes, true_classes,
            images, lambda_physics, device
        )
        
        # Check total is sum of components
        expected_total = (
            losses['mse_params'] + 
            losses['ce_class'] + 
            lambda_physics * losses['physics_residual']
        )
        
        assert abs(losses['total'] - expected_total) < 1e-5
    
    def test_perfect_prediction_low_loss(self):
        """Test that perfect predictions give near-zero loss (except physics)"""
        batch_size = 4
        
        # Perfect parameter predictions
        true_params = torch.randn(batch_size, 5)
        pred_params = true_params.clone()
        
        # Perfect class predictions
        true_classes = torch.randint(0, 3, (batch_size,))
        pred_classes = torch.zeros(batch_size, 3)
        for i in range(batch_size):
            pred_classes[i, true_classes[i]] = 100.0  # High logit for true class
        
        images = torch.randn(batch_size, 1, 64, 64)
        device = torch.device('cpu')
        
        losses = physics_informed_loss(
            pred_params, true_params, pred_classes, true_classes,
            images, 0.1, device
        )
        
        # MSE should be very small
        assert losses['mse_params'] < 1e-5
        # CE should be small (not zero due to softmax)
        assert losses['ce_class'] < 0.1


class TestDatasetGeneration:
    """Tests for dataset generation functions"""
    
    def test_generate_single_sample_cdm(self):
        """Test generating single CDM sample"""
        image, parameters, label = generate_single_sample(dm_type='CDM', grid_size=32)
        
        assert image.shape == (32, 32)
        assert parameters.shape == (5,)
        assert label == 0  # CDM label
        assert np.all(image >= 0) and np.all(image <= 1), "Image should be normalized"
    
    def test_generate_single_sample_wdm(self):
        """Test generating single WDM sample"""
        image, parameters, label = generate_single_sample(dm_type='WDM', grid_size=32)
        
        assert image.shape == (32, 32)
        assert parameters.shape == (5,)
        assert label == 1  # WDM label
    
    def test_generate_single_sample_sidm(self):
        """Test generating single SIDM sample"""
        image, parameters, label = generate_single_sample(dm_type='SIDM', grid_size=32)
        
        assert image.shape == (32, 32)
        assert parameters.shape == (5,)
        assert label == 2  # SIDM label
    
    def test_add_noise(self):
        """Test noise addition"""
        original = np.ones((32, 32)) * 0.5
        noisy = add_noise(original.copy(), gaussian_noise_std=0.01)
        
        # Noisy image should be different from original
        assert not np.allclose(original, noisy)
        
        # But mean should be similar
        assert abs(np.mean(original) - np.mean(noisy)) < 0.2
    
    def test_generate_training_data_small(self):
        """Test generating small training dataset"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / 'test_dataset.h5'
            
            n_samples = 30  # Small test dataset
            split_info = generate_training_data(
                n_samples=n_samples,
                output_file=str(output_file),
                grid_size=32,
                train_split=0.7,
                val_split=0.15,
                test_split=0.15,
                seed=42
            )
            
            # Check file was created
            assert output_file.exists()
            
            # Check split info (keys are 'train', 'val', 'test')
            assert 'train' in split_info
            assert 'val' in split_info
            assert 'test' in split_info
            total = split_info['train'] + split_info['val'] + split_info['test']
            assert total == n_samples
            
            # Check HDF5 contents
            with h5py.File(output_file, 'r') as f:
                assert 'train/images' in f
                assert 'train/parameters' in f
                assert 'train/labels' in f
                
                assert f['train/images'].shape[0] == split_info['train']
                assert f['train/parameters'].shape[0] == split_info['train']
                assert f['train/labels'].shape[0] == split_info['train']


class TestLensDataset:
    """Tests for PyTorch Dataset class"""
    
    @pytest.fixture
    def sample_dataset_file(self):
        """Create a temporary dataset file for testing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / 'test_dataset.h5'
            
            generate_training_data(
                n_samples=30,
                output_file=str(output_file),
                grid_size=32,
                seed=42
            )
            
            yield str(output_file)
    
    def test_dataset_loading(self, sample_dataset_file):
        """Test dataset loads correctly"""
        dataset = LensDataset(sample_dataset_file, split='train')
        
        assert len(dataset) > 0
    
    def test_dataset_getitem(self, sample_dataset_file):
        """Test dataset __getitem__ returns correct format"""
        dataset = LensDataset(sample_dataset_file, split='train')
        
        image, params, label = dataset[0]
        
        # Check types (can be numpy arrays or tensors)
        assert hasattr(image, 'shape')
        assert hasattr(params, 'shape')
        
        # Check shapes (grid_size was 32 in fixture)
        assert len(image.shape) == 3  # Should have channel dimension
        assert image.shape[0] == 1  # Single channel
        assert len(params.shape) == 1
        assert params.shape[0] == 5
    
    def test_dataset_splits(self, sample_dataset_file):
        """Test that different splits are accessible"""
        train_dataset = LensDataset(sample_dataset_file, split='train')
        val_dataset = LensDataset(sample_dataset_file, split='val')
        test_dataset = LensDataset(sample_dataset_file, split='test')
        
        # All splits should have data
        assert len(train_dataset) > 0
        assert len(val_dataset) > 0
        assert len(test_dataset) > 0
        
        # Train should be largest
        assert len(train_dataset) > len(val_dataset)
        assert len(train_dataset) > len(test_dataset)


class TestTrainingSteps:
    """Tests for training and validation steps"""
    
    def test_train_step(self):
        """Test training step executes without errors"""
        model = PhysicsInformedNN(input_size=64, dropout_rate=0.2)  # Use 64 to match expected input
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        images = torch.randn(4, 1, 64, 64)  # 64x64 images
        params = torch.randn(4, 5)
        labels = torch.randint(0, 3, (4,))
        
        device = torch.device('cpu')
        
        losses = train_step(model, images, params, labels, optimizer, 0.1, device)
        
        assert 'total' in losses
        assert isinstance(losses['total'], float)
        assert losses['total'] >= 0
    
    def test_validate_step(self):
        """Test validation step executes without errors"""
        model = PhysicsInformedNN(input_size=64, dropout_rate=0.2)  # Use 64 to match expected input
        model.eval()
        
        images = torch.randn(4, 1, 64, 64)  # 64x64 images
        params = torch.randn(4, 5)
        labels = torch.randint(0, 3, (4,))
        
        device = torch.device('cpu')
        
        losses = validate_step(model, images, params, labels, 0.1, device)
        
        assert 'total' in losses
        assert isinstance(losses['total'], float)
        assert losses['total'] >= 0


class TestEvaluationMetrics:
    """Tests for evaluation metrics computation"""
    
    def test_compute_metrics(self):
        """Test metrics computation"""
        n_samples = 50
        
        # Generate synthetic predictions
        pred_params = np.random.randn(n_samples, 5)
        true_params = pred_params + np.random.randn(n_samples, 5) * 0.1
        
        pred_classes = np.random.randint(0, 3, n_samples)
        true_classes = pred_classes.copy()
        # Add some errors
        error_indices = np.random.choice(n_samples, size=5, replace=False)
        pred_classes[error_indices] = (pred_classes[error_indices] + 1) % 3
        
        metrics = compute_metrics(pred_params, true_params, pred_classes, true_classes)
        
        # Check structure
        assert 'parameter_metrics' in metrics
        assert 'classification_accuracy' in metrics
        
        # Check parameter metrics
        assert 'overall_MAE' in metrics
        assert 'overall_RMSE' in metrics
        assert 'overall_MAPE' in metrics
        
        # Check classification metrics
        assert 'classification_accuracy' in metrics
        assert 'per_class_accuracy' in metrics
        assert 'confusion_matrix' in metrics
        assert metrics['classification_accuracy'] >= 0
        assert metrics['classification_accuracy'] <= 1
    
    def test_perfect_predictions_metrics(self):
        """Test metrics with perfect predictions"""
        n_samples = 20
        
        # Perfect predictions
        true_params = np.random.randn(n_samples, 5)
        pred_params = true_params.copy()
        
        true_classes = np.random.randint(0, 3, n_samples)
        pred_classes = true_classes.copy()
        
        metrics = compute_metrics(pred_params, true_params, pred_classes, true_classes)
        
        # All errors should be zero
        assert metrics['overall_MAE'] < 1e-10
        assert metrics['overall_RMSE'] < 1e-10
        
        # Accuracy should be 100%
        assert abs(metrics['classification_accuracy'] - 1.0) < 1e-5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
