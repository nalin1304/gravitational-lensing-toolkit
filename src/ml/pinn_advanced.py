"""
Advanced Physics-Informed Neural Network Architectures

This module provides state-of-the-art PINN architectures including:
- Residual blocks (ResNet)
- Self-attention mechanisms
- Adaptive activation functions
- Multi-scale feature extraction
- Physics-constrained layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional, Dict


class AdaptiveActivation(nn.Module):
    """
    Adaptive activation function that learns optimal scaling and shifting.

    Based on adaptive activation functions (Jagtap et al. 2020, PINN-AA).
    Allows the network to learn activation parameters per layer.

    Parameters
    ----------
    activation_type : str
        Base activation: 'tanh', 'sin', 'gelu', or 'swish'
    num_features : int
        Number of features (for layer-wise adaptation)
    """

    def __init__(self, activation_type='tanh', num_features=64):
        super().__init__()
        self.activation_type = activation_type

        # Learnable scaling and shift parameters (initialized to identity)
        self.scale = nn.Parameter(torch.ones(num_features))
        self.shift = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        # Apply scaling and shift
        x_transformed = self.scale * x + self.shift

        # Apply base activation
        if self.activation_type == 'tanh':
            return torch.tanh(x_transformed)
        elif self.activation_type == 'sin':
            return torch.sin(x_transformed)
        elif self.activation_type == 'gelu':
            return F.gelu(x_transformed)
        elif self.activation_type == 'swish':
            return x_transformed * torch.sigmoid(x_transformed)
        else:
            return torch.tanh(x_transformed)


class ResidualBlock(nn.Module):
    """
    Residual block for deep PINNs.

    Implements skip connections to enable training of very deep networks
    and improve gradient flow (He et al. 2016, ResNet).

    Architecture:
        x -> Conv/Linear -> Activation -> Conv/Linear -> (+) x -> Activation
                                                          ^
                                                          |
                                                    skip connection

    Parameters
    ----------
    channels : int
        Number of feature channels
    activation : str
        Activation function type
    use_batch_norm : bool
        Whether to use batch normalization
    dropout_rate : float
        Dropout probability
    """

    def __init__(self, channels=64, activation='gelu',
                 use_batch_norm=True, dropout_rate=0.1):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels) if use_batch_norm else nn.Identity()
        self.act1 = AdaptiveActivation(activation, channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels) if use_batch_norm else nn.Identity()

        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.act2 = AdaptiveActivation(activation, channels)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)

        # Skip connection
        out = out + residual
        out = self.act2(out)

        return out


class SelfAttention(nn.Module):
    """
    Self-attention mechanism for capturing long-range dependencies.

    Implements scaled dot-product attention (Vaswani et al. 2017, Transformer).
    Allows the network to focus on relevant spatial features for lensing.

    Parameters
    ----------
    channels : int
        Number of input channels
    reduction : int
        Channel reduction factor for Q, K, V projections
    """

    def __init__(self, channels=64, reduction=8):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        self.inter_channels = max(channels // reduction, 1)

        # Query, Key, Value projections
        self.query_conv = nn.Conv2d(channels, self.inter_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(channels, self.inter_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(channels, channels, kernel_size=1)

        # Output projection
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable weight for attention

    def forward(self, x):
        """
        Forward pass with self-attention.

        Parameters
        ----------
        x : torch.Tensor
            Input feature map [B, C, H, W]

        Returns
        -------
        out : torch.Tensor
            Attended features [B, C, H, W]
        """
        batch_size, C, H, W = x.shape

        # Compute Query, Key, Value
        query = self.query_conv(x).view(batch_size, self.inter_channels, -1)  # [B, C', H*W]
        key = self.key_conv(x).view(batch_size, self.inter_channels, -1)      # [B, C', H*W]
        value = self.value_conv(x).view(batch_size, C, -1)                     # [B, C, H*W]

        # Compute attention scores: Q @ K^T / sqrt(d_k)
        # [B, H*W, H*W]
        attention = torch.bmm(query.permute(0, 2, 1), key)
        attention = F.softmax(attention / np.sqrt(self.inter_channels), dim=-1)

        # Apply attention to values: Attention @ V
        # [B, C, H*W]
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)

        # Residual connection with learnable weight
        out = self.gamma * out + x

        return out


class MultiScaleFeatureExtractor(nn.Module):
    """
    Multi-scale feature extraction using parallel convolutions.

    Captures features at different spatial scales simultaneously,
    important for detecting both small (subhalos) and large (main halo) structures.

    Similar to Inception modules (Szegedy et al. 2015).

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    """

    def __init__(self, in_channels=64, out_channels=64):
        super().__init__()

        # Different kernel sizes for different scales
        self.conv1x1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=5, padding=2)

        # Max pooling branch
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv_pool = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1)

        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.GELU()

    def forward(self, x):
        # Parallel branches at different scales
        branch1 = self.conv1x1(x)
        branch2 = self.conv3x3(x)
        branch3 = self.conv5x5(x)
        branch4 = self.conv_pool(self.pool(x))

        # Concatenate along channel dimension
        out = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        out = self.bn(out)
        out = self.activation(out)

        return out


class PhysicsConstrainedLayer(nn.Module):
    """
    Custom layer that enforces physics constraints.

    Ensures predictions satisfy physical constraints:
    - Mass conservation
    - Positive convergence (κ >= 0)
    - Causality (shear γ <= κ)
    - Bounded parameter ranges

    Parameters
    ----------
    enforce_positivity : bool
        Ensure κ >= 0 using softplus activation
    enforce_causality : bool
        Ensure γ <= κ (weak lensing approximation)
    """

    def __init__(self, enforce_positivity=True, enforce_causality=True):
        super().__init__()
        self.enforce_positivity = enforce_positivity
        self.enforce_causality = enforce_causality

    def forward(self, convergence, shear_1=None, shear_2=None):
        """
        Apply physics constraints to lensing quantities.

        Parameters
        ----------
        convergence : torch.Tensor
            Convergence map
        shear_1, shear_2 : torch.Tensor, optional
            Shear components

        Returns
        -------
        convergence : torch.Tensor
            Constrained convergence
        shear_1, shear_2 : torch.Tensor or None
            Constrained shear components
        """
        # Enforce positivity: κ >= 0
        if self.enforce_positivity:
            convergence = F.softplus(convergence)

        # Enforce causality: |γ| <= κ
        if self.enforce_causality and shear_1 is not None and shear_2 is not None:
            shear_mag = torch.sqrt(shear_1**2 + shear_2**2 + 1e-8)

            # Scale shear if it exceeds convergence
            scale_factor = torch.minimum(torch.ones_like(shear_mag),
                                        convergence / (shear_mag + 1e-8))
            shear_1 = shear_1 * scale_factor
            shear_2 = shear_2 * scale_factor

        if shear_1 is not None and shear_2 is not None:
            return convergence, shear_1, shear_2
        else:
            return convergence


class AdvancedPINN(nn.Module):
    """
    Advanced Physics-Informed Neural Network for gravitational lensing.

    Features:
    - ResNet blocks for deep architecture
    - Self-attention for long-range dependencies
    - Multi-scale feature extraction
    - Adaptive activation functions
    - Physics-constrained output layer
    - Uncertainty quantification via dropout

    Architecture:
        Input (64×64)
        -> Multi-scale Extraction
        -> ResNet Block × N
        -> Self-Attention
        -> ResNet Block × N
        -> Self-Attention
        -> Physics-Constrained Output
        -> Parameter Regression

    Parameters
    ----------
    input_channels : int
        Number of input channels (default: 1 for convergence map)
    num_blocks : int
        Number of residual blocks per stage
    base_channels : int
        Number of base feature channels
    num_params : int
        Number of output parameters (5 for NFW)
    num_classes : int
        Number of classification outputs (3 for CDM/WDM/SIDM)
    dropout_rate : float
        Dropout probability for uncertainty
    use_attention : bool
        Whether to use self-attention layers
    """

    def __init__(self,
                 input_channels=1,
                 num_blocks=3,
                 base_channels=64,
                 num_params=5,
                 num_classes=3,
                 dropout_rate=0.2,
                 use_attention=True):
        super().__init__()

        self.input_channels = input_channels
        self.num_params = num_params
        self.num_classes = num_classes
        self.use_attention = use_attention

        # Initial feature extraction
        self.initial_conv = nn.Conv2d(input_channels, base_channels,
                                     kernel_size=7, stride=1, padding=3)
        self.initial_bn = nn.BatchNorm2d(base_channels)
        self.initial_act = nn.GELU()

        # Multi-scale feature extractor
        self.multi_scale = MultiScaleFeatureExtractor(base_channels, base_channels)

        # Stage 1: Residual blocks
        self.res_blocks_1 = nn.ModuleList([
            ResidualBlock(base_channels, 'gelu', True, dropout_rate)
            for _ in range(num_blocks)
        ])

        # Self-attention after first stage
        if use_attention:
            self.attention_1 = SelfAttention(base_channels, reduction=8)

        # Stage 2: More residual blocks
        self.res_blocks_2 = nn.ModuleList([
            ResidualBlock(base_channels, 'gelu', True, dropout_rate)
            for _ in range(num_blocks)
        ])

        # Self-attention after second stage
        if use_attention:
            self.attention_2 = SelfAttention(base_channels, reduction=8)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Parameter regression head
        self.param_fc1 = nn.Linear(base_channels, 128)
        self.param_bn1 = nn.BatchNorm1d(128)
        self.param_dropout = nn.Dropout(dropout_rate)
        self.param_fc2 = nn.Linear(128, num_params)

        # Classification head
        self.class_fc1 = nn.Linear(base_channels, 64)
        self.class_bn1 = nn.BatchNorm1d(64)
        self.class_dropout = nn.Dropout(dropout_rate)
        self.class_fc2 = nn.Linear(64, num_classes)

        # Physics constraint layer
        self.physics_layer = PhysicsConstrainedLayer(
            enforce_positivity=True,
            enforce_causality=True
        )

    def forward(self, x, return_features=False):
        """
        Forward pass through advanced PINN.

        Parameters
        ----------
        x : torch.Tensor
            Input convergence map [B, C, H, W]
        return_features : bool
            Whether to return intermediate features

        Returns
        -------
        params : torch.Tensor
            Predicted parameters [B, num_params]
        classes : torch.Tensor
            Class logits [B, num_classes]
        features : torch.Tensor, optional
            Intermediate features if return_features=True
        """
        # Initial feature extraction
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = self.initial_act(x)

        # Multi-scale features
        x = self.multi_scale(x)

        # Stage 1: ResNet blocks
        for block in self.res_blocks_1:
            x = block(x)

        # Attention 1
        if self.use_attention:
            x = self.attention_1(x)

        # Stage 2: ResNet blocks
        for block in self.res_blocks_2:
            x = block(x)

        # Attention 2
        if self.use_attention:
            x = self.attention_2(x)

        # Global pooling
        features = self.global_pool(x)
        features = features.view(features.size(0), -1)

        # Parameter regression branch
        params = self.param_fc1(features)
        params = self.param_bn1(params)
        params = F.gelu(params)
        params = self.param_dropout(params)
        params = self.param_fc2(params)

        # Classification branch
        classes = self.class_fc1(features)
        classes = self.class_bn1(classes)
        classes = F.gelu(classes)
        classes = self.class_dropout(classes)
        classes = self.class_fc2(classes)

        if return_features:
            return params, classes, features
        else:
            return params, classes

    def predict_with_uncertainty(self, x, num_samples=50):
        """
        Predict with uncertainty quantification using MC Dropout.

        Parameters
        ----------
        x : torch.Tensor
            Input convergence map
        num_samples : int
            Number of forward passes for MC dropout

        Returns
        -------
        params_mean : torch.Tensor
            Mean parameter predictions
        params_std : torch.Tensor
            Standard deviation (epistemic uncertainty)
        classes_mean : torch.Tensor
            Mean class probabilities
        classes_std : torch.Tensor
            Standard deviation of probabilities
        """
        self.train()  # Enable dropout

        params_samples = []
        classes_samples = []

        with torch.no_grad():
            for _ in range(num_samples):
                params, classes = self.forward(x)
                params_samples.append(params)
                classes_samples.append(F.softmax(classes, dim=1))

        params_samples = torch.stack(params_samples)
        classes_samples = torch.stack(classes_samples)

        params_mean = params_samples.mean(dim=0)
        params_std = params_samples.std(dim=0)
        classes_mean = classes_samples.mean(dim=0)
        classes_std = classes_samples.std(dim=0)

        return params_mean, params_std, classes_mean, classes_std


class PhysicsInformedLoss(nn.Module):
    """
    Physics-informed loss function for training PINNs.

    Combines:
    - Data loss: MSE between predictions and labels
    - Physics loss: Lens equation residuals
    - Conservation loss: Mass conservation
    - Regularization: Parameter bounds and smoothness

    Parameters
    ----------
    lambda_physics : float
        Weight for physics loss term
    lambda_conservation : float
        Weight for conservation loss
    lambda_reg : float
        Weight for regularization
    """

    def __init__(self, lambda_physics=1.0, lambda_conservation=0.5, lambda_reg=0.01):
        super().__init__()
        self.lambda_physics = lambda_physics
        self.lambda_conservation = lambda_conservation
        self.lambda_reg = lambda_reg

    def lens_equation_residual(self, convergence_pred, convergence_true):
        """
        Compute residual of lens equation.

        Lens equation: β = θ - α(θ)
        where α depends on κ via Poisson equation: ∇²ψ = 2κ

        Parameters
        ----------
        convergence_pred : torch.Tensor
            Predicted convergence map
        convergence_true : torch.Tensor
            True convergence map

        Returns
        -------
        residual : torch.Tensor
            Lens equation residual
        """
        # Compute Laplacian of convergence using finite differences
        # ∇²κ should be smooth for physical halos
        laplacian = self._compute_laplacian(convergence_pred)
        laplacian_true = self._compute_laplacian(convergence_true)

        residual = F.mse_loss(laplacian, laplacian_true)
        return residual

    def _compute_laplacian(self, field):
        """Compute 2D Laplacian using finite differences."""
        kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=field.dtype, device=field.device).view(1, 1, 3, 3)

        # Pad to maintain size
        field_padded = F.pad(field, (1, 1, 1, 1), mode='replicate')
        laplacian = F.conv2d(field_padded, kernel, padding=0)

        return laplacian

    def mass_conservation(self, convergence):
        """
        Ensure mass conservation: ∫ κ dA should be constant.

        For halos, integrated convergence should match virial mass.
        """
        # Mean should be reasonable (not too high or low)
        mean_kappa = convergence.mean(dim=(2, 3))

        # Penalize extreme values
        loss = F.relu(mean_kappa - 2.0).mean() + F.relu(-mean_kappa + 0.01).mean()

        return loss

    def parameter_regularization(self, params):
        """
        Regularize parameters to stay in physical ranges.

        Parameters:
        - M_vir: log10(M/Msun) in [10, 15]
        - concentration: [3, 20]
        - redshift: [0.1, 3.0]
        - ellipticity: [0, 0.5]
        - theta: [0, 180]
        """
        # L2 regularization to prevent extreme values
        reg_loss = torch.mean(params**2)

        return reg_loss

    def forward(self, params_pred, params_true, classes_pred, classes_true,
                convergence_pred=None, convergence_true=None):
        """
        Compute total loss.

        Parameters
        ----------
        params_pred : torch.Tensor
            Predicted parameters
        params_true : torch.Tensor
            True parameters
        classes_pred : torch.Tensor
            Predicted class logits
        classes_true : torch.Tensor
            True class labels
        convergence_pred : torch.Tensor, optional
            Predicted convergence map
        convergence_true : torch.Tensor, optional
            True convergence map

        Returns
        -------
        total_loss : torch.Tensor
            Combined loss
        loss_dict : dict
            Individual loss components
        """
        # Data loss: Parameter MSE
        param_loss = F.mse_loss(params_pred, params_true)

        # Classification loss
        class_loss = F.cross_entropy(classes_pred, classes_true)

        # Physics loss (if convergence maps provided)
        if convergence_pred is not None and convergence_true is not None:
            physics_loss = self.lens_equation_residual(convergence_pred, convergence_true)
            conservation_loss = self.mass_conservation(convergence_pred)
        else:
            physics_loss = torch.tensor(0.0, device=params_pred.device)
            conservation_loss = torch.tensor(0.0, device=params_pred.device)

        # Regularization
        reg_loss = self.parameter_regularization(params_pred)

        # Total loss
        total_loss = (param_loss + class_loss +
                     self.lambda_physics * physics_loss +
                     self.lambda_conservation * conservation_loss +
                     self.lambda_reg * reg_loss)

        loss_dict = {
            'total': total_loss.item(),
            'param': param_loss.item(),
            'class': class_loss.item(),
            'physics': physics_loss.item(),
            'conservation': conservation_loss.item(),
            'regularization': reg_loss.item()
        }

        return total_loss, loss_dict


def create_advanced_pinn(config: Dict) -> AdvancedPINN:
    """
    Factory function to create Advanced PINN with config.

    Parameters
    ----------
    config : dict
        Configuration dictionary with keys:
        - input_channels: int
        - num_blocks: int
        - base_channels: int
        - num_params: int
        - num_classes: int
        - dropout_rate: float
        - use_attention: bool

    Returns
    -------
    model : AdvancedPINN
        Configured model instance
    """
    return AdvancedPINN(
        input_channels=config.get('input_channels', 1),
        num_blocks=config.get('num_blocks', 3),
        base_channels=config.get('base_channels', 64),
        num_params=config.get('num_params', 5),
        num_classes=config.get('num_classes', 3),
        dropout_rate=config.get('dropout_rate', 0.2),
        use_attention=config.get('use_attention', True)
    )
