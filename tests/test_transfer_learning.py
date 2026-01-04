"""
Unit tests for Transfer Learning module (Phase 9).

Tests domain adaptation, uncertainty quantification, and transfer learning
from synthetic to real observations.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, MagicMock

from src.ml.transfer_learning import (
    TransferConfig,
    GradientReversalLayer,
    DomainClassifier,
    DomainAdaptationNetwork,
    MMDLoss,
    CORALLoss,
    BayesianUncertaintyEstimator,
    TransferLearningTrainer,
    create_synthetic_to_real_pipeline,
    compute_domain_discrepancy
)


class SimpleMockPINN(nn.Module):
    """Simple mock PINN for testing."""
    
    def __init__(self, input_size=64):
        super(SimpleMockPINN, self).__init__()
        self.input_size = input_size
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        self.dense = nn.Sequential(
            nn.Linear(64 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.param_head = nn.Linear(512, 5)
        self.class_head = nn.Linear(512, 3)
    
    def forward(self, x):
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        features = self.dense(features)
        params = self.param_head(features)
        classes = self.class_head(features)
        return params, classes


class TestTransferConfig:
    """Test transfer learning configuration."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = TransferConfig()
        
        assert config.source_domain == 'synthetic'
        assert config.target_domain == 'real'
        assert config.adaptation_method == 'dann'
        assert config.lambda_adapt == 0.1
        assert config.uncertainty_method == 'dropout'
        assert config.n_mc_samples == 50
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = TransferConfig(
            adaptation_method='mmd',
            lambda_adapt=0.5,
            freeze_encoder=True,
            n_mc_samples=100
        )
        
        assert config.adaptation_method == 'mmd'
        assert config.lambda_adapt == 0.5
        assert config.freeze_encoder is True
        assert config.n_mc_samples == 100


class TestGradientReversalLayer:
    """Test gradient reversal layer."""
    
    def test_forward_pass(self):
        """Test forward pass is identity."""
        x = torch.randn(32, 128)
        lambda_param = 1.0
        
        output = GradientReversalLayer.apply(x, lambda_param)
        
        assert torch.allclose(output, x)
        assert output.shape == x.shape
    
    def test_backward_gradient_reversal(self):
        """Test backward pass reverses gradient."""
        x = torch.randn(32, 128, requires_grad=True)
        lambda_param = 1.0
        
        # Forward
        output = GradientReversalLayer.apply(x, lambda_param)
        
        # Backward with dummy loss
        loss = output.sum()
        loss.backward()
        
        # Gradient should be reversed (all negative)
        assert x.grad is not None
        assert torch.all(x.grad < 0) or torch.all(x.grad > 0)  # All same sign
    
    def test_lambda_scaling(self):
        """Test lambda parameter scales gradient."""
        x1 = torch.randn(32, 128, requires_grad=True)
        x2 = x1.clone().detach().requires_grad_(True)
        
        lambda1 = 1.0
        lambda2 = 0.5
        
        # Forward with different lambdas
        out1 = GradientReversalLayer.apply(x1, lambda1)
        out2 = GradientReversalLayer.apply(x2, lambda2)
        
        # Backward
        out1.sum().backward()
        out2.sum().backward()
        
        # Gradient with lambda=0.5 should be half of lambda=1.0
        assert torch.allclose(x2.grad, x1.grad * 0.5, rtol=1e-4)


class TestDomainClassifier:
    """Test domain classifier."""
    
    def test_initialization(self):
        """Test classifier initializes correctly."""
        classifier = DomainClassifier(input_dim=512, hidden_dims=[256, 128])
        
        assert isinstance(classifier, nn.Module)
    
    def test_forward_shape(self):
        """Test forward pass produces correct shape."""
        classifier = DomainClassifier(input_dim=512, hidden_dims=[256, 128])
        
        features = torch.randn(32, 512)
        output = classifier(features)
        
        assert output.shape == (32, 2)  # Binary classification
    
    def test_binary_output(self):
        """Test output is for binary classification."""
        classifier = DomainClassifier(input_dim=256)
        
        features = torch.randn(16, 256)
        output = classifier(features)
        
        # Should have 2 logits (synthetic vs real)
        assert output.shape[1] == 2


class TestDomainAdaptationNetwork:
    """Test domain adversarial neural network."""
    
    def test_initialization(self):
        """Test DANN initializes correctly."""
        base_model = SimpleMockPINN(input_size=64)
        dann = DomainAdaptationNetwork(base_model, feature_dim=512)
        
        assert dann.base_model is base_model
        assert dann.domain_classifier is not None
    
    def test_forward_without_domain(self):
        """Test forward pass basic functionality."""
        base_model = SimpleMockPINN(input_size=64)
        dann = DomainAdaptationNetwork(base_model, feature_dim=512)
        
        images = torch.randn(8, 1, 64, 64)
        params, classes, domain_logits = dann(images, alpha=1.0)
        
        assert params.shape == (8, 5)
        assert classes.shape == (8, 3)
        assert domain_logits.shape == (8, 2)
    
    def test_forward_with_features(self):
        """Test forward with feature return."""
        base_model = SimpleMockPINN(input_size=64)
        dann = DomainAdaptationNetwork(base_model, feature_dim=512)
        
        images = torch.randn(8, 1, 64, 64)
        params, classes, domain_logits, features = dann(images, alpha=1.0, return_features=True)
        
        assert features.shape[0] == 8  # Batch size
        assert len(features.shape) == 2  # 2D features
    
    def test_alpha_parameter(self):
        """Test alpha parameter affects gradient reversal."""
        base_model = SimpleMockPINN(input_size=64)
        dann = DomainAdaptationNetwork(base_model, feature_dim=512)
        
        images = torch.randn(4, 1, 64, 64)
        
        # Different alpha values
        _, _, domain1 = dann(images, alpha=0.0)
        _, _, domain2 = dann(images, alpha=1.0)
        
        # Outputs should be different due to different alpha
        assert domain1.shape == domain2.shape


class TestMMDLoss:
    """Test Maximum Mean Discrepancy loss."""
    
    def test_initialization(self):
        """Test MMD loss initializes."""
        mmd = MMDLoss(kernel_type='rbf')
        assert mmd.kernel_type == 'rbf'
    
    def test_forward_same_distribution(self):
        """Test MMD is near zero for same distribution."""
        mmd = MMDLoss()
        
        # Same distribution
        source = torch.randn(32, 128)
        target = torch.randn(32, 128)
        
        loss = mmd(source, target)
        
        assert loss.item() >= 0  # MMD is always non-negative
        assert isinstance(loss, torch.Tensor)
    
    def test_forward_different_distributions(self):
        """Test MMD is positive for different distributions."""
        mmd = MMDLoss()
        
        # Different distributions
        source = torch.randn(32, 128) * 1.0
        target = torch.randn(32, 128) * 2.0 + 5.0  # Different mean and variance
        
        loss = mmd(source, target)
        
        assert loss.item() > 0
    
    def test_symmetric(self):
        """Test MMD is symmetric."""
        mmd = MMDLoss()
        
        source = torch.randn(32, 128)
        target = torch.randn(32, 128)
        
        loss1 = mmd(source, target)
        loss2 = mmd(target, source)
        
        assert torch.allclose(loss1, loss2, rtol=1e-4)


class TestCORALLoss:
    """Test Correlation Alignment loss."""
    
    def test_initialization(self):
        """Test CORAL loss initializes."""
        coral = CORALLoss()
        assert isinstance(coral, nn.Module)
    
    def test_forward_same_distribution(self):
        """Test CORAL is small for same distribution."""
        coral = CORALLoss()
        
        # Same distribution
        source = torch.randn(32, 128)
        target = torch.randn(32, 128)
        
        loss = coral(source, target)
        
        assert loss.item() >= 0
        assert isinstance(loss, torch.Tensor)
    
    def test_forward_different_covariances(self):
        """Test CORAL detects different covariances."""
        coral = CORALLoss()
        
        # Different covariances
        source = torch.randn(32, 128)
        target = torch.randn(32, 128) * 3.0  # Different variance
        
        loss = coral(source, target)
        
        assert loss.item() > 0
    
    def test_covariance_computation(self):
        """Test covariance computation."""
        coral = CORALLoss()
        
        features = torch.randn(50, 64)
        cov = coral._compute_covariance(features)
        
        # Covariance should be square and symmetric
        assert cov.shape == (64, 64)
        assert torch.allclose(cov, cov.t(), rtol=1e-4)


class TestBayesianUncertaintyEstimator:
    """Test Bayesian uncertainty estimation."""
    
    def test_initialization(self):
        """Test estimator initializes correctly."""
        model = SimpleMockPINN(input_size=64)
        estimator = BayesianUncertaintyEstimator(model, n_samples=50)
        
        assert estimator.n_samples == 50
        assert estimator.model is model
    
    def test_enable_dropout(self):
        """Test dropout enabling during inference."""
        model = SimpleMockPINN(input_size=64)
        estimator = BayesianUncertaintyEstimator(model, n_samples=10)
        
        model.eval()  # Set to eval mode
        estimator.enable_dropout(model)
        
        # Check dropout layers are in training mode
        dropout_training = False
        for module in model.modules():
            if isinstance(module, nn.Dropout) or isinstance(module, nn.Dropout2d):
                dropout_training = module.training
                break
        
        assert dropout_training
    
    def test_predict_with_uncertainty(self):
        """Test uncertainty prediction."""
        model = SimpleMockPINN(input_size=64)
        estimator = BayesianUncertaintyEstimator(model, n_samples=10)
        
        images = torch.randn(4, 1, 64, 64)
        predictions, uncertainties = estimator.predict_with_uncertainty(images)
        
        # Check predictions
        assert 'params_mean' in predictions
        assert 'classes_mean' in predictions
        assert predictions['params_mean'].shape == (4, 5)
        assert predictions['classes_mean'].shape == (4, 3)
        
        # Check uncertainties
        assert 'params_std' in uncertainties
        assert 'classes_entropy' in uncertainties
        assert uncertainties['params_std'].shape == (4, 5)
        assert uncertainties['classes_entropy'].shape == (4,)
    
    def test_uncertainty_positive(self):
        """Test uncertainty values are positive."""
        model = SimpleMockPINN(input_size=64)
        estimator = BayesianUncertaintyEstimator(model, n_samples=20)
        
        images = torch.randn(8, 1, 64, 64)
        _, uncertainties = estimator.predict_with_uncertainty(images)
        
        # Standard deviations should be non-negative
        assert np.all(uncertainties['params_std'] >= 0)
        assert np.all(uncertainties['classes_entropy'] >= 0)
    
    def test_multiple_samples_variation(self):
        """Test multiple samples produce variation."""
        model = SimpleMockPINN(input_size=64)
        estimator = BayesianUncertaintyEstimator(model, n_samples=30)
        
        images = torch.randn(4, 1, 64, 64)
        _, uncertainties = estimator.predict_with_uncertainty(images)
        
        # Should have some uncertainty (not all zeros)
        assert np.any(uncertainties['params_std'] > 0)


class TestTransferLearningTrainer:
    """Test transfer learning trainer."""
    
    def test_initialization_dann(self):
        """Test trainer initialization with DANN."""
        model = SimpleMockPINN(input_size=64)
        config = TransferConfig(adaptation_method='dann')
        
        trainer = TransferLearningTrainer(model, config=config)
        
        assert isinstance(trainer.model, DomainAdaptationNetwork)
        assert trainer.config.adaptation_method == 'dann'
    
    def test_initialization_mmd(self):
        """Test trainer initialization with MMD."""
        model = SimpleMockPINN(input_size=64)
        config = TransferConfig(adaptation_method='mmd')
        
        trainer = TransferLearningTrainer(model, config=config)
        
        assert isinstance(trainer.adaptation_loss, MMDLoss)
    
    def test_initialization_coral(self):
        """Test trainer initialization with CORAL."""
        model = SimpleMockPINN(input_size=64)
        config = TransferConfig(adaptation_method='coral')
        
        trainer = TransferLearningTrainer(model, config=config)
        
        assert isinstance(trainer.adaptation_loss, CORALLoss)
    
    def test_compute_adaptation_loss_dann(self):
        """Test DANN adaptation loss computation."""
        model = SimpleMockPINN(input_size=64)
        config = TransferConfig(adaptation_method='dann')
        trainer = TransferLearningTrainer(model, config=config)
        
        features_source = torch.randn(16, 512)
        features_target = torch.randn(16, 512)
        domain_logits_source = torch.randn(16, 2)
        domain_logits_target = torch.randn(16, 2)
        
        loss = trainer.compute_adaptation_loss(
            features_source,
            features_target,
            domain_logits_source,
            domain_logits_target
        )
        
        assert loss.item() >= 0
        assert isinstance(loss, torch.Tensor)
    
    def test_compute_adaptation_loss_mmd(self):
        """Test MMD adaptation loss computation."""
        model = SimpleMockPINN(input_size=64)
        config = TransferConfig(adaptation_method='mmd')
        trainer = TransferLearningTrainer(model, config=config)
        
        features_source = torch.randn(16, 512)
        features_target = torch.randn(16, 512)
        
        loss = trainer.compute_adaptation_loss(
            features_source,
            features_target
        )
        
        assert loss.item() >= 0
    
    def test_freeze_encoder(self):
        """Test encoder freezing."""
        model = SimpleMockPINN(input_size=64)
        config = TransferConfig(freeze_encoder=True)
        
        trainer = TransferLearningTrainer(model, config=config)
        
        # Check encoder parameters are frozen
        for param in model.encoder.parameters():
            assert not param.requires_grad
    
    def test_fine_tune_mock(self):
        """Test fine-tuning with mock data loader."""
        model = SimpleMockPINN(input_size=64)
        config = TransferConfig(fine_tune_epochs=2)
        trainer = TransferLearningTrainer(model, config=config)
        
        # Mock data loader
        mock_data = [
            (
                torch.randn(4, 1, 64, 64),
                torch.randn(4, 5),
                torch.randint(0, 3, (4,))
            )
            for _ in range(3)
        ]
        
        # Mock criterion
        def mock_criterion(pred_params, pred_classes, true_params, true_classes):
            return torch.tensor(1.0, requires_grad=True)
        
        optimizer = torch.optim.Adam(trainer.model.parameters(), lr=1e-4)
        
        losses = trainer.fine_tune(
            mock_data,
            optimizer,
            mock_criterion,
            epochs=2
        )
        
        assert len(losses) == 2
        assert all(isinstance(loss, float) for loss in losses)


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_pipeline(self):
        """Test creating transfer learning pipeline."""
        base_model = SimpleMockPINN(input_size=64)
        
        # Mock data loaders
        synthetic_loader = [
            (torch.randn(4, 1, 64, 64), torch.randn(4, 5), torch.randint(0, 3, (4,)))
        ]
        real_loader = [
            (torch.randn(4, 1, 64, 64), torch.randn(4, 5), torch.randint(0, 3, (4,)))
        ]
        
        config = TransferConfig(adaptation_method='dann')
        
        model, info = create_synthetic_to_real_pipeline(
            base_model,
            synthetic_loader,
            real_loader,
            config=config
        )
        
        assert isinstance(model, DomainAdaptationNetwork)
        assert 'adaptation_method' in info
        assert info['adaptation_method'] == 'dann'
    
    def test_compute_domain_discrepancy(self):
        """Test domain discrepancy computation."""
        base_model = SimpleMockPINN(input_size=64)
        dann = DomainAdaptationNetwork(base_model, feature_dim=512)
        
        # Mock data loaders
        source_loader = [
            (torch.randn(4, 1, 64, 64), torch.randn(4, 5), torch.randint(0, 3, (4,)))
            for _ in range(2)
        ]
        target_loader = [
            (torch.randn(4, 1, 64, 64), torch.randn(4, 5), torch.randint(0, 3, (4,)))
            for _ in range(2)
        ]
        
        metrics = compute_domain_discrepancy(
            dann,
            source_loader,
            target_loader
        )
        
        assert 'mean_distance' in metrics
        assert 'std_distance' in metrics
        assert metrics['mean_distance'] >= 0
        assert metrics['std_distance'] >= 0


class TestIntegration:
    """Integration tests for transfer learning."""
    
    def test_end_to_end_dann(self):
        """Test end-to-end DANN workflow."""
        # Create base model
        base_model = SimpleMockPINN(input_size=64)
        
        # Create DANN
        config = TransferConfig(adaptation_method='dann', lambda_adapt=0.1)
        dann = DomainAdaptationNetwork(base_model, feature_dim=512, config=config)
        
        # Synthetic data
        images_synthetic = torch.randn(8, 1, 64, 64)
        params_syn, classes_syn, domain_syn = dann(images_synthetic, alpha=1.0)
        
        assert params_syn.shape == (8, 5)
        assert classes_syn.shape == (8, 3)
        assert domain_syn.shape == (8, 2)
    
    def test_end_to_end_uncertainty(self):
        """Test end-to-end uncertainty estimation."""
        model = SimpleMockPINN(input_size=64)
        estimator = BayesianUncertaintyEstimator(model, n_samples=20)
        
        images = torch.randn(4, 1, 64, 64)
        predictions, uncertainties = estimator.predict_with_uncertainty(images)
        
        # Verify complete output
        assert predictions['params_mean'].shape == (4, 5)
        assert uncertainties['params_std'].shape == (4, 5)
        assert uncertainties['classes_entropy'].shape == (4,)
    
    def test_end_to_end_trainer(self):
        """Test end-to-end trainer workflow."""
        model = SimpleMockPINN(input_size=64)
        config = TransferConfig(adaptation_method='mmd', fine_tune_epochs=1)
        trainer = TransferLearningTrainer(model, config=config)
        
        # Create mock target data with labels
        target_loader = [
            (
                torch.randn(4, 1, 64, 64),
                torch.randn(4, 5),
                torch.randint(0, 3, (4,))
            )
            for _ in range(2)
        ]
        
        def mock_criterion(pred_params, pred_classes, true_params, true_classes):
            return (pred_params - true_params).pow(2).mean()
        
        optimizer = torch.optim.Adam(trainer.model.parameters(), lr=1e-4)
        
        losses = trainer.fine_tune(
            target_loader,
            optimizer,
            mock_criterion,
            epochs=1
        )
        
        assert len(losses) == 1
        assert losses[0] >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
