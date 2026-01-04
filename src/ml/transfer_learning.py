"""
Transfer Learning and Domain Adaptation for Gravitational Lensing (Phase 9)

This module implements advanced ML techniques to bridge the gap between
synthetic simulations and real telescope observations:

1. Domain Adaptation Networks (DAN)
2. Adversarial Domain Adaptation (DANN)
3. Fine-tuning strategies for real data
4. Uncertainty quantification with Bayesian inference
5. Synthetic-to-Real transfer learning

Key Challenges:
- Domain shift: simulated data vs real observations
- Limited real labeled data
- Systematic differences (PSF, noise, backgrounds)
- Generalization to unseen lens configurations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List, Callable
from dataclasses import dataclass
import warnings


@dataclass
class TransferConfig:
    """Configuration for transfer learning."""
    source_domain: str = 'synthetic'  # Source domain (synthetic simulations)
    target_domain: str = 'real'       # Target domain (real observations)
    adaptation_method: str = 'dann'   # 'dann', 'mmd', 'coral', 'fine_tune'
    freeze_encoder: bool = False      # Freeze encoder weights during adaptation
    lambda_adapt: float = 0.1         # Adaptation loss weight
    uncertainty_method: str = 'dropout'  # 'dropout', 'ensemble', 'bayesian'
    n_mc_samples: int = 50            # Monte Carlo samples for uncertainty
    fine_tune_epochs: int = 10        # Epochs for fine-tuning
    learning_rate: float = 1e-4       # Learning rate for adaptation


class GradientReversalLayer(torch.autograd.Function):
    """
    Gradient Reversal Layer for adversarial domain adaptation.
    
    Forward pass: identity function
    Backward pass: reverses gradient and scales by lambda
    
    This layer enables adversarial training where the feature extractor
    learns domain-invariant representations by trying to confuse the
    domain classifier.
    """
    
    @staticmethod
    def forward(ctx, x, lambda_param):
        ctx.lambda_param = lambda_param
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_param * grad_output, None


class DomainClassifier(nn.Module):
    """
    Domain classifier for adversarial domain adaptation.
    
    Classifies whether features come from source (synthetic) or
    target (real) domain. Used in DANN to learn domain-invariant features.
    
    Parameters
    ----------
    input_dim : int
        Dimension of input features
    hidden_dims : list of int
        Hidden layer dimensions
    dropout_rate : float
        Dropout probability
    
    Examples
    --------
    >>> classifier = DomainClassifier(input_dim=512, hidden_dims=[256, 128])
    >>> features = torch.randn(32, 512)
    >>> domain_logits = classifier(features)
    >>> print(domain_logits.shape)  # (32, 2) - [p_synthetic, p_real]
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.3
    ):
        super(DomainClassifier, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Binary classification: synthetic (0) vs real (1)
        layers.append(nn.Linear(prev_dim, 2))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.classifier(x)


class DomainAdaptationNetwork(nn.Module):
    """
    Domain Adversarial Neural Network (DANN) for transfer learning.
    
    Extends the base PINN with domain adaptation capabilities:
    - Feature extractor learns domain-invariant representations
    - Domain classifier tries to distinguish domains
    - Gradient reversal creates adversarial training
    
    Architecture:
        Input → Encoder → Features
                            ├→ Task predictor (params, classes)
                            └→ Domain classifier (synthetic/real)
    
    Parameters
    ----------
    base_model : nn.Module
        Base PINN model
    feature_dim : int
        Dimension of features for domain adaptation
    config : TransferConfig
        Transfer learning configuration
    
    Examples
    --------
    >>> from src.ml.pinn import PhysicsInformedNN
    >>> base_model = PhysicsInformedNN(input_size=64)
    >>> dann_model = DomainAdaptationNetwork(base_model, feature_dim=512)
    >>> 
    >>> # Training with domain adaptation
    >>> images_synthetic = torch.randn(32, 1, 64, 64)
    >>> images_real = torch.randn(16, 1, 64, 64)
    >>> 
    >>> params_syn, classes_syn, domain_syn = dann_model(images_synthetic, alpha=1.0)
    >>> params_real, classes_real, domain_real = dann_model(images_real, alpha=1.0)
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        feature_dim: int = 512,
        config: Optional[TransferConfig] = None
    ):
        super(DomainAdaptationNetwork, self).__init__()
        
        self.base_model = base_model
        self.config = config or TransferConfig()
        
        # Domain classifier
        self.domain_classifier = DomainClassifier(
            input_dim=feature_dim,
            hidden_dims=None  # Use default
        )
        
        # For extracting features before final layers
        self.feature_extractor = None
        self._setup_feature_extraction()
    
    def _setup_feature_extraction(self):
        """Setup feature extraction from base model."""
        # Extract encoder and dense layers (before task heads)
        if hasattr(self.base_model, 'encoder') and hasattr(self.base_model, 'dense'):
            self.feature_extractor = nn.Sequential(
                self.base_model.encoder,
                nn.Flatten(),
                self.base_model.dense
            )
    
    def forward(
        self,
        x: torch.Tensor,
        alpha: float = 1.0,
        return_features: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with domain adaptation.
        
        Parameters
        ----------
        x : torch.Tensor
            Input images (batch, 1, H, W)
        alpha : float
            Gradient reversal strength (0 to 1)
        return_features : bool
            Whether to return intermediate features
        
        Returns
        -------
        params : torch.Tensor
            Predicted parameters (batch, 5)
        classes : torch.Tensor
            Predicted class logits (batch, 3)
        domain_logits : torch.Tensor
            Domain classification logits (batch, 2)
        features : torch.Tensor, optional
            Intermediate features if return_features=True
        """
        # Get task predictions from base model
        params, classes = self.base_model(x)
        
        # Extract features for domain adaptation
        if self.feature_extractor is not None:
            features = self.feature_extractor(x)
        else:
            # Fallback: use encoder output
            features = self.base_model.encoder(x)
            features = features.view(features.size(0), -1)
        
        # Apply gradient reversal layer
        reversed_features = GradientReversalLayer.apply(features, alpha)
        
        # Domain classification
        domain_logits = self.domain_classifier(reversed_features)
        
        if return_features:
            return params, classes, domain_logits, features
        
        return params, classes, domain_logits


class MMDLoss(nn.Module):
    """
    Maximum Mean Discrepancy (MMD) loss for domain adaptation.
    
    MMD measures the distance between two distributions in a reproducing
    kernel Hilbert space (RKHS). Used as an alternative to adversarial
    domain adaptation.
    
    References
    ----------
    Gretton et al. "A Kernel Two-Sample Test" (2012)
    """
    
    def __init__(self, kernel_type: str = 'rbf', kernel_mul: float = 2.0, kernel_num: int = 5):
        super(MMDLoss, self).__init__()
        self.kernel_type = kernel_type
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
    
    def gaussian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5):
        """Compute Gaussian (RBF) kernel matrix."""
        n_samples = int(source.size(0)) + int(target.size(0))
        total = torch.cat([source, target], dim=0)
        
        # Compute pairwise distances
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        
        # Multi-scale Gaussian kernel
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        
        return sum(kernel_val)
    
    def forward(self, source, target):
        """
        Compute MMD between source and target distributions.
        
        Parameters
        ----------
        source : torch.Tensor
            Source domain features (batch_src, feature_dim)
        target : torch.Tensor
            Target domain features (batch_tgt, feature_dim)
        
        Returns
        -------
        mmd : torch.Tensor
            MMD loss value
        """
        batch_size = int(source.size(0))
        kernels = self.gaussian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num)
        
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        
        mmd = torch.mean(XX) + torch.mean(YY) - torch.mean(XY) - torch.mean(YX)
        
        return mmd


class CORALLoss(nn.Module):
    """
    CORrelation ALignment (CORAL) loss for domain adaptation.
    
    Aligns second-order statistics (covariances) of source and target
    feature distributions. Simpler and more efficient than MMD.
    
    References
    ----------
    Sun & Saenko "Deep CORAL: Correlation Alignment for Deep Domain Adaptation" (2016)
    """
    
    def forward(self, source, target):
        """
        Compute CORAL loss between source and target distributions.
        
        Parameters
        ----------
        source : torch.Tensor
            Source domain features (batch_src, feature_dim)
        target : torch.Tensor
            Target domain features (batch_tgt, feature_dim)
        
        Returns
        -------
        coral_loss : torch.Tensor
            CORAL loss value
        """
        d = source.size(1)  # Feature dimension
        
        # Source covariance
        source_c = self._compute_covariance(source)
        
        # Target covariance
        target_c = self._compute_covariance(target)
        
        # Frobenius norm of difference
        loss = torch.sum((source_c - target_c) ** 2) / (4 * d * d)
        
        return loss
    
    def _compute_covariance(self, features):
        """Compute covariance matrix."""
        n = features.size(0)
        features = features - torch.mean(features, dim=0, keepdim=True)
        cov = (features.t() @ features) / (n - 1)
        return cov


class BayesianUncertaintyEstimator:
    """
    Bayesian uncertainty estimation using Monte Carlo Dropout.
    
    Performs multiple forward passes with dropout enabled to estimate
    predictive uncertainty. Useful for assessing model confidence on
    real observations.
    
    Parameters
    ----------
    model : nn.Module
        Trained model with dropout layers
    n_samples : int
        Number of MC samples
    device : str
        Device for computation
    
    Examples
    --------
    >>> model = PhysicsInformedNN(input_size=64)
    >>> estimator = BayesianUncertaintyEstimator(model, n_samples=50)
    >>> 
    >>> images = torch.randn(8, 1, 64, 64)
    >>> predictions, uncertainties = estimator.predict_with_uncertainty(images)
    >>> 
    >>> print(predictions['params_mean'].shape)  # (8, 5)
    >>> print(uncertainties['params_std'].shape)  # (8, 5)
    """
    
    def __init__(
        self,
        model: nn.Module,
        n_samples: int = 50,
        device: str = 'cpu'
    ):
        self.model = model
        self.n_samples = n_samples
        self.device = device
    
    def enable_dropout(self, model):
        """Enable dropout during inference for MC sampling."""
        for module in model.modules():
            if isinstance(module, nn.Dropout) or isinstance(module, nn.Dropout2d):
                module.train()
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Predict with uncertainty estimates.
        
        Parameters
        ----------
        x : torch.Tensor
            Input images (batch, 1, H, W)
        
        Returns
        -------
        predictions : dict
            Mean predictions for params and classes
        uncertainties : dict
            Uncertainty estimates (standard deviations)
        """
        self.model.eval()
        self.enable_dropout(self.model)
        
        # Collect predictions from multiple forward passes
        param_samples = []
        class_samples = []
        
        with torch.no_grad():
            for _ in range(self.n_samples):
                params, classes = self.model(x)
                param_samples.append(params.cpu().numpy())
                class_samples.append(F.softmax(classes, dim=1).cpu().numpy())
        
        # Stack samples
        param_samples = np.stack(param_samples, axis=0)  # (n_samples, batch, 5)
        class_samples = np.stack(class_samples, axis=0)  # (n_samples, batch, 3)
        
        # Compute statistics
        predictions = {
            'params_mean': np.mean(param_samples, axis=0),
            'classes_mean': np.mean(class_samples, axis=0)
        }
        
        uncertainties = {
            'params_std': np.std(param_samples, axis=0),
            'params_epistemic': np.std(param_samples, axis=0),  # Epistemic uncertainty
            'classes_entropy': self._compute_predictive_entropy(class_samples)
        }
        
        return predictions, uncertainties
    
    def _compute_predictive_entropy(self, class_samples):
        """Compute predictive entropy for classification uncertainty."""
        # Average probabilities across samples
        mean_probs = np.mean(class_samples, axis=0)  # (batch, 3)
        
        # Compute entropy
        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10), axis=1)
        
        return entropy


class TransferLearningTrainer:
    """
    Trainer for transfer learning from synthetic to real data.
    
    Implements multiple adaptation strategies:
    1. DANN: Domain Adversarial Neural Networks
    2. MMD: Maximum Mean Discrepancy
    3. CORAL: Correlation Alignment
    4. Fine-tuning: Simple fine-tuning on target domain
    
    Parameters
    ----------
    model : nn.Module
        Base model or domain adaptation model
    config : TransferConfig
        Transfer learning configuration
    device : str
        Device for training
    
    Examples
    --------
    >>> from src.ml.pinn import PhysicsInformedNN
    >>> base_model = PhysicsInformedNN(input_size=64)
    >>> 
    >>> config = TransferConfig(
    ...     adaptation_method='dann',
    ...     lambda_adapt=0.1,
    ...     fine_tune_epochs=10
    ... )
    >>> 
    >>> trainer = TransferLearningTrainer(base_model, config)
    >>> 
    >>> # Training loop
    >>> for epoch in range(config.fine_tune_epochs):
    ...     loss = trainer.train_epoch(
    ...         source_loader,  # Synthetic data
    ...         target_loader   # Real data (unlabeled)
    ...     )
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[TransferConfig] = None,
        device: str = 'cpu'
    ):
        self.config = config or TransferConfig()
        self.device = device
        
        # Setup model for domain adaptation
        if self.config.adaptation_method == 'dann':
            self.model = DomainAdaptationNetwork(model, feature_dim=512, config=config).to(device)
        else:
            self.model = model.to(device)
        
        # Setup adaptation loss
        if self.config.adaptation_method == 'mmd':
            self.adaptation_loss = MMDLoss()
        elif self.config.adaptation_method == 'coral':
            self.adaptation_loss = CORALLoss()
        else:
            self.adaptation_loss = None
        
        # Freeze encoder if specified
        if self.config.freeze_encoder and hasattr(model, 'encoder'):
            for param in model.encoder.parameters():
                param.requires_grad = False
    
    def compute_adaptation_loss(
        self,
        features_source: torch.Tensor,
        features_target: torch.Tensor,
        domain_logits_source: Optional[torch.Tensor] = None,
        domain_logits_target: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute domain adaptation loss.
        
        Parameters
        ----------
        features_source : torch.Tensor
            Source domain features
        features_target : torch.Tensor
            Target domain features
        domain_logits_source : torch.Tensor, optional
            Domain logits for source (DANN only)
        domain_logits_target : torch.Tensor, optional
            Domain logits for target (DANN only)
        
        Returns
        -------
        loss : torch.Tensor
            Adaptation loss
        """
        if self.config.adaptation_method == 'dann':
            # Domain classification loss
            domain_labels_source = torch.zeros(features_source.size(0), dtype=torch.long).to(self.device)
            domain_labels_target = torch.ones(features_target.size(0), dtype=torch.long).to(self.device)
            
            loss_source = F.cross_entropy(domain_logits_source, domain_labels_source)
            loss_target = F.cross_entropy(domain_logits_target, domain_labels_target)
            
            return loss_source + loss_target
        
        elif self.config.adaptation_method in ['mmd', 'coral']:
            return self.adaptation_loss(features_source, features_target)
        
        else:
            return torch.tensor(0.0).to(self.device)
    
    def fine_tune(
        self,
        target_loader,
        optimizer,
        criterion,
        epochs: int = None
    ) -> List[float]:
        """
        Fine-tune model on target domain with labeled data.
        
        Parameters
        ----------
        target_loader : DataLoader
            Target domain data with labels
        optimizer : torch.optim.Optimizer
            Optimizer
        criterion : callable
            Loss function
        epochs : int, optional
            Number of epochs (uses config if None)
        
        Returns
        -------
        losses : list of float
            Training losses per epoch
        """
        epochs = epochs or self.config.fine_tune_epochs
        losses = []
        
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            for batch in target_loader:
                images, labels_params, labels_classes = batch
                images = images.to(self.device)
                labels_params = labels_params.to(self.device)
                labels_classes = labels_classes.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                if isinstance(self.model, DomainAdaptationNetwork):
                    pred_params, pred_classes, _ = self.model(images, alpha=0.0)
                else:
                    pred_params, pred_classes = self.model(images)
                
                # Compute loss
                loss = criterion(pred_params, pred_classes, labels_params, labels_classes)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            losses.append(avg_loss)
        
        return losses


def create_synthetic_to_real_pipeline(
    base_model: nn.Module,
    synthetic_loader,
    real_loader,
    config: Optional[TransferConfig] = None,
    device: str = 'cpu'
) -> Tuple[nn.Module, Dict]:
    """
    Create complete transfer learning pipeline from synthetic to real data.
    
    This convenience function sets up the entire transfer learning process:
    1. Initialize domain adaptation model
    2. Setup trainer with specified method
    3. Configure optimizer and losses
    4. Return ready-to-train model
    
    Parameters
    ----------
    base_model : nn.Module
        Pre-trained model on synthetic data
    synthetic_loader : DataLoader
        Synthetic (source) domain data
    real_loader : DataLoader
        Real (target) domain data
    config : TransferConfig, optional
        Transfer learning configuration
    device : str
        Device for training
    
    Returns
    -------
    model : nn.Module
        Domain adaptation model
    trainer : TransferLearningTrainer
        Configured trainer
    
    Examples
    --------
    >>> from src.ml.pinn import PhysicsInformedNN
    >>> 
    >>> # Pre-trained model on synthetic data
    >>> base_model = PhysicsInformedNN(input_size=64)
    >>> # ... load weights ...
    >>> 
    >>> config = TransferConfig(
    ...     adaptation_method='dann',
    ...     lambda_adapt=0.1
    ... )
    >>> 
    >>> model, trainer = create_synthetic_to_real_pipeline(
    ...     base_model,
    ...     synthetic_loader,
    ...     real_loader,
    ...     config=config
    ... )
    """
    config = config or TransferConfig()
    
    # Initialize trainer
    trainer = TransferLearningTrainer(base_model, config=config, device=device)
    
    # Setup information
    info = {
        'adaptation_method': config.adaptation_method,
        'lambda_adapt': config.lambda_adapt,
        'uncertainty_method': config.uncertainty_method,
        'n_mc_samples': config.n_mc_samples
    }
    
    return trainer.model, info


# Utility functions for transfer learning evaluation

def compute_domain_discrepancy(
    model: nn.Module,
    source_loader,
    target_loader,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Compute domain discrepancy metrics between source and target.
    
    Measures how well the model has aligned the two domains by computing:
    - A-distance (proxy based on domain classification accuracy)
    - Feature distribution statistics
    
    Parameters
    ----------
    model : nn.Module
        Domain adaptation model
    source_loader : DataLoader
        Source domain data
    target_loader : DataLoader
        Target domain data
    device : str
        Device for computation
    
    Returns
    -------
    metrics : dict
        Domain discrepancy metrics
    """
    model.eval()
    
    source_features = []
    target_features = []
    
    # Extract features
    with torch.no_grad():
        for images, _, _ in source_loader:
            images = images.to(device)
            if isinstance(model, DomainAdaptationNetwork):
                _, _, _, features = model(images, return_features=True)
            else:
                features = model.encoder(images)
                features = features.view(features.size(0), -1)
            source_features.append(features.cpu().numpy())
        
        for images, _, _ in target_loader:
            images = images.to(device)
            if isinstance(model, DomainAdaptationNetwork):
                _, _, _, features = model(images, return_features=True)
            else:
                features = model.encoder(images)
                features = features.view(features.size(0), -1)
            target_features.append(features.cpu().numpy())
    
    source_features = np.concatenate(source_features, axis=0)
    target_features = np.concatenate(target_features, axis=0)
    
    # Compute statistics
    metrics = {
        'mean_distance': float(np.linalg.norm(
            source_features.mean(axis=0) - target_features.mean(axis=0)
        )),
        'std_distance': float(np.abs(
            source_features.std(axis=0).mean() - target_features.std(axis=0).mean()
        ))
    }
    
    return metrics
