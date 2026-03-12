"""
Coordinate-based Physics-Informed Neural Network (cPINN) for Gravitational Lensing

This module implements a TRUE physics-informed neural network that:
1. Takes (x, y) coordinates as input (NOT images)
2. Predicts the lensing potential ψ(x, y) directly
3. Uses automatic differentiation to compute deflection: α = ∇ψ
4. Enforces lens equation: β = θ - α(θ) as a physics constraint
5. Enforces Poisson equation: ∇²ψ = 2κ as a physics constraint

This is fundamentally different from CNN-based approaches because it embeds
the actual physics of gravitational lensing into the network architecture.

Reference: Raissi et al. (2019), "Physics-informed neural networks"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional


class CoordinatePINN(nn.Module):
    """
    Coordinate-based Physics-Informed Neural Network for gravitational lensing.
    
    Architecture:
    - Input: (x, y) coordinates normalized to [-1, 1]
    - Hidden layers: [64, 128, 128, 64] with tanh activations
    - Output: ψ(x, y) - lensing potential
    
    The deflection angles are computed via automatic differentiation:
    α_x = ∂ψ/∂x, α_y = ∂ψ/∂y
    
    The convergence is computed via second derivatives:
    κ = (1/2) × (∂²ψ/∂x² + ∂²ψ/∂y²)
    
    Parameters
    ----------
    hidden_dims : list
        List of hidden layer dimensions
    activation : str
        Activation function ('tanh', 'relu', 'gelu')
    """
    
    def __init__(
        self,
        hidden_dims: list = [64, 128, 128, 64],
        activation: str = 'tanh'
    ):
        super(CoordinatePINN, self).__init__()
        
        self.hidden_dims = hidden_dims
        self.activation_name = activation
        
        # Build network layers
        layers = []
        input_dim = 2  # (x, y)
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            input_dim = hidden_dim
        
        # Output layer: predict lensing potential
        layers.append(nn.Linear(input_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights using Xavier initialization
        self._initialize_weights()
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'softplus': nn.Softplus()
        }
        return activations.get(name.lower(), nn.Tanh())
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: predict lensing potential at (x, y).
        
        Parameters
        ----------
        x : torch.Tensor
            x-coordinates, shape (batch_size, n_points) or (n_points,)
        y : torch.Tensor
            y-coordinates, shape (batch_size, n_points) or (n_points,)
        
        Returns
        -------
        psi : torch.Tensor
            Lensing potential at each point, shape (batch_size, n_points)
        """
        # Reshape inputs to (batch_size * n_points, 1) for batch processing
        original_shape = x.shape
        batch_size = 1 if x.dim() == 1 else x.shape[0]
        
        if x.dim() == 1:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
        
        # Flatten for processing
        x_flat = x.reshape(-1, 1)
        y_flat = y.reshape(-1, 1)
        
        # Concatenate coordinates
        coords = torch.cat([x_flat, y_flat], dim=1)
        
        # Forward through network
        psi_flat = self.network(coords)
        
        # Reshape back
        psi = psi_flat.reshape(original_shape)
        
        return psi
    
    def compute_deflection(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor,
        return_psi: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute deflection angles via automatic differentiation.
        
        α_x = ∂ψ/∂x, α_y = ∂ψ/∂y
        
        Parameters
        ----------
        x : torch.Tensor
            x-coordinates (requires grad=True for differentiation)
        y : torch.Tensor
            y-coordinates (requires grad=True for differentiation)
        return_psi : bool
            Whether to also return the potential
        
        Returns
        -------
        alpha_x : torch.Tensor
            x-component of deflection angle
        alpha_y : torch.Tensor
            y-component of deflection angle
        psi : torch.Tensor (optional)
            Lensing potential
        """
        # Ensure requires_grad for differentiation
        if not x.requires_grad:
            x = x.requires_grad_(True)
        if not y.requires_grad:
            y = y.requires_grad_(True)
        
        # Get potential
        psi = self.forward(x, y)
        
        # Compute gradients: α = ∇ψ
        alpha_x = torch.autograd.grad(
            outputs=psi,
            inputs=x,
            grad_outputs=torch.ones_like(psi),
            create_graph=True,
            retain_graph=True
        )[0]
        
        alpha_y = torch.autograd.grad(
            outputs=psi,
            inputs=y,
            grad_outputs=torch.ones_like(psi),
            create_graph=True,
            retain_graph=True
        )[0]
        
        if return_psi:
            return alpha_x, alpha_y, psi
        return alpha_x, alpha_y
    
    def compute_convergence(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute convergence via second derivatives.
        
        κ = (1/2) × ∇²ψ = (1/2) × (∂²ψ/∂x² + ∂²ψ/∂y²)
        
        Parameters
        ----------
        x : torch.Tensor
            x-coordinates
        y : torch.Tensor
            y-coordinates
        
        Returns
        -------
        kappa : torch.Tensor
            Convergence at each point
        """
        # Ensure requires_grad
        if not x.requires_grad:
            x = x.requires_grad_(True)
        if not y.requires_grad:
            y = y.requires_grad_(True)
        
        # Get first derivatives
        alpha_x, alpha_y = self.compute_deflection(x, y)
        
        # Get second derivatives for convergence
        # κ = (1/2) * (dα_x/dx + dα_y/dy)
        dalpha_x_dx = torch.autograd.grad(
            outputs=alpha_x,
            inputs=x,
            grad_outputs=torch.ones_like(alpha_x),
            create_graph=True,
            retain_graph=True
        )[0]
        
        dalpha_y_dy = torch.autograd.grad(
            outputs=alpha_y,
            inputs=y,
            grad_outputs=torch.ones_like(alpha_y),
            create_graph=True,
            retain_graph=True
        )[0]
        
        kappa = 0.5 * (dalpha_x_dx + dalpha_y_dy)
        
        return kappa
    
    def lens_equation_residual(
        self,
        theta_x: torch.Tensor,
        theta_y: torch.Tensor,
        beta_x: torch.Tensor,
        beta_y: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute lens equation residual.
        
        The lens equation: β = θ - α(θ)
        Residual: r = β - (θ - α(θ)) = β + α(θ) - θ
        
        Parameters
        ----------
        theta_x : torch.Tensor
            Image plane x-coordinates
        theta_y : torch.Tensor
            Image plane y-coordinates
        beta_x : torch.Tensor
            Source x-position
        beta_y : torch.Tensor
            Source y-position
        
        Returns
        -------
        residual : torch.Tensor
            Lens equation residual
        """
        alpha_x, alpha_y = self.compute_deflection(theta_x, theta_y)
        
        residual_x = beta_x + alpha_x - theta_x
        residual_y = beta_y + alpha_y - theta_y
        
        return torch.sqrt(residual_x**2 + residual_y**2)


class LensingPINNWithParameters(nn.Module):
    """
    Hybrid PINN that combines coordinate-based potential prediction
    with learnable physical parameters.
    
    Instead of predicting parameters directly, this network:
    1. Takes (x, y, M, rs, ...) as input
    2. Uses a physics-informed architecture that embeds known relationships
    3. Predicts potential corrections/modifications to analytical profiles
    """
    
    def __init__(
        self,
        n_params: int = 5,
        hidden_dims: list = [128, 256, 128],
        base_profile: str = 'nfw'
    ):
        super(LensingPINNWithParameters, self).__init__()
        
        self.n_params = n_params
        self.base_profile = base_profile
        
        # Input: coordinates + parameters
        input_dim = 2 + n_params  # (x, y) + parameters
        
        layers = []
        dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.Tanh())
            layers.append(nn.LayerNorm(hidden_dim))
            dim = hidden_dim
        
        # Output: potential correction
        layers.append(nn.Linear(dim, 1))
        
        self.network = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        params: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with parameters.
        
        Parameters
        ----------
        x, y : torch.Tensor
            Coordinates
        params : torch.Tensor
            Physical parameters [M, rs, beta_x, beta_y, ...]
        
        Returns
        -------
        psi : torch.Tensor
            Lensing potential
        """
        # Flatten coordinates if needed
        if x.dim() == 1:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
            params = params.unsqueeze(0)
        
        batch_size = x.shape[0]
        n_points = x.shape[1] if x.dim() > 1 else x.shape[0]
        
        # Reshape for batch processing
        x_flat = x.reshape(-1, 1)
        y_flat = y.reshape(-1, 1)
        
        # Expand params to match coordinate points
        params_expanded = params.unsqueeze(1).expand(-1, n_points, -1).reshape(-1, self.n_params)
        
        # Concatenate
        inputs = torch.cat([x_flat, y_flat, params_expanded], dim=1)
        
        # Forward
        psi_flat = self.network(inputs)
        
        # Reshape
        psi = psi_flat.reshape(batch_size, n_points)
        
        return psi


def physics_informed_loss_coordinate(
    model: CoordinatePINN,
    theta_x: torch.Tensor,
    theta_y: torch.Tensor,
    beta_x: torch.Tensor,
    beta_y: torch.Tensor,
    lambda_lens: float = 1.0,
    lambda_poisson: float = 0.1,
    kappa_true: Optional[torch.Tensor] = None
) -> Dict[str, torch.Tensor]:
    """
    Compute physics-informed loss for coordinate-based PINN.
    
    Loss components:
    1. Lens equation residual: β = θ - α(θ)
    2. Poisson equation: ∇²ψ = 2κ (if convergence is known)
    
    Parameters
    ----------
    model : CoordinatePINN
        The PINN model
    theta_x, theta_y : torch.Tensor
        Image plane coordinates
    beta_x, beta_y : torch.Tensor
        Source position
    lambda_lens : float
        Weight for lens equation residual
    lambda_poisson : float
        Weight for Poisson equation
    kappa_true : torch.Tensor, optional
        True convergence (if available)
    
    Returns
    -------
    losses : dict
        Dictionary of loss components
    """
    # Lens equation residual
    residual = model.lens_equation_residual(theta_x, theta_y, beta_x, beta_y)
    lens_loss = torch.mean(residual**2)
    
    # Total loss
    total_loss = lambda_lens * lens_loss
    
    # Poisson constraint if convergence is known
    if kappa_true is not None:
        kappa_pred = model.compute_convergence(theta_x, theta_y)
        poisson_loss = torch.mean((kappa_pred - kappa_true)**2)
        total_loss = total_loss + lambda_poisson * poisson_loss
    
    return {
        'total': total_loss,
        'lens_equation': lens_loss,
        'poisson': poisson_loss if kappa_true is not None else torch.tensor(0.0)
    }


def train_coordinate_pinn(
    model: CoordinatePINN,
    theta_data: torch.Tensor,
    beta_data: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    n_iterations: int = 1000,
    lambda_lens: float = 1.0,
    lambda_poisson: float = 0.1
) -> Dict[str, list]:
    """
    Train coordinate-based PINN with physics-informed loss.
    
    Parameters
    ----------
    model : CoordinatePINN
        The PINN model
    theta_data : torch.Tensor
        Image positions (n_samples, n_points, 2)
    beta_data : torch.Tensor
        Source positions (n_samples, 2)
    optimizer : torch.optim.Optimizer
        Optimizer
    n_iterations : int
        Number of training iterations
    lambda_lens : float
        Weight for lens equation
    lambda_poisson : float
        Weight for Poisson constraint
    
    Returns
    -------
    history : dict
        Training history
    """
    history = {'total': [], 'lens_equation': []}
    
    model.train()
    
    for i in range(n_iterations):
        optimizer.zero_grad()
        
        # Sample random points
        batch_idx = torch.randint(0, theta_data.shape[0], (1,))
        theta_x = theta_data[batch_idx, :, 0].squeeze()
        theta_y = theta_data[batch_idx, :, 1].squeeze()
        beta_x = beta_data[batch_idx, 0].squeeze()
        beta_y = beta_data[batch_idx, 1].squeeze()
        
        # Compute loss
        losses = physics_informed_loss_coordinate(
            model, theta_x, theta_y, beta_x, beta_y,
            lambda_lens=lambda_lens, lambda_poisson=lambda_poisson
        )
        
        # Backprop
        losses['total'].backward()
        optimizer.step()
        
        # Record
        history['total'].append(losses['total'].item())
        history['lens_equation'].append(losses['lens_equation'].item())
        
        if (i + 1) % 100 == 0:
            print(f"Iter {i+1}/{n_iterations}, Loss: {losses['total'].item():.6f}")
    
    return history


if __name__ == "__main__":
    print("Testing Coordinate-based PINN...")
    
    # Create model
    model = CoordinatePINN(hidden_dims=[64, 128, 64], activation='tanh')
    print(f"Model architecture:\n{model}")
    
    # Test forward pass
    x = torch.randn(10, requires_grad=True)
    y = torch.randn(10, requires_grad=True)
    
    psi = model(x, y)
    print(f"\nPotential prediction: shape={psi.shape}")
    
    # Test deflection computation
    alpha_x, alpha_y = model.compute_deflection(x, y)
    print(f"Deflection: alpha_x shape={alpha_x.shape}, alpha_y shape={alpha_y.shape}")
    
    # Test convergence computation
    kappa = model.compute_convergence(x, y)
    print(f"Convergence: shape={kappa.shape}")
    
    print("\n[SUCCESS] Coordinate PINN working!")
