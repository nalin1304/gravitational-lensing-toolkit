"""
Physics-Informed Neural Networks (PINNs) for Gravitational Lensing

Implements PINNs that learn to solve the lensing equation while respecting
physical constraints (Poisson equation, symmetries, boundary conditions).

References:
- Raissi et al. (2019): Physics-informed neural networks
- Keeton (2001): A Catalog of Mass Models for Gravitational Lensing
- Kochanek et al. (2001): Gravitational Lensing

Author: Phase 14 Implementation  
Date: October 2025
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Base PINN Architecture
# ============================================================================

class LensingPINN(nn.Module):
    """
    Physics-Informed Neural Network for gravitational lensing
    
    Architecture:
        - Input: (x, y, lens_params) where (x, y) are sky coordinates
        - Output: (convergence κ, potential ψ, deflection angles α_x, α_y)
        - Hidden: Multiple layers with skip connections
        
    Physical Constraints:
        - Poisson equation: ∇²ψ = 2κ
        - Deflection: α = ∇ψ
        - Symmetry: Respects lens symmetry (e.g., axisymmetric for NFW)
        - Boundary: κ → 0 as r → ∞
    
    Parameters:
        input_dim (int): Dimension of input (2 for x,y + N for lens params)
        hidden_dims (List[int]): Dimensions of hidden layers
        output_dim (int): Dimension of output (4 for κ, ψ, α_x, α_y)
        activation (str): Activation function ('tanh', 'sin', 'relu')
    """
    
    def __init__(
        self,
        input_dim: int = 2,
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 64, 64]
            
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.use_skip_connections = use_skip_connections
        
        # Activation function
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sin':
            self.activation = lambda x: torch.sin(x)
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.Tanh()
        
        # Build network layers
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        # Skip connection layers (if enabled)
        if use_skip_connections:
            self.skip_layers = nn.ModuleList([
                nn.Linear(input_dim, hidden_dims[i]) 
                for i in range(len(hidden_dims))
            ])
        
        # Initialize weights with Xavier initialization
        self.apply(self._init_weights)
        
        logger.info(f"Initialized LensingPINN: {input_dim}→{hidden_dims}→{output_dim}")
        logger.info(f"Total parameters: {self.count_parameters():,}")
    
    def _init_weights(self, module):
        """Xavier initialization for better convergence"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network
        
        Args:
            x: Input tensor of shape (batch, input_dim)
               First 2 dimensions are (x, y) coordinates
               Remaining dimensions are lens parameters
        
        Returns:
            Output tensor of shape (batch, output_dim)
            [κ, ψ, α_x, α_y] for each point
        """
        # Save input for skip connections
        x_input = x
        
        # Input layer
        x = self.activation(self.layers[0](x))
        
        # Hidden layers with optional skip connections
        for i in range(1, len(self.layers) - 1):
            if self.use_skip_connections and i <= len(self.skip_layers):
                # Add skip connection from input
                skip = self.skip_layers[i - 1](x_input)
                x = self.activation(self.layers[i](x) + skip)
            else:
                x = self.activation(self.layers[i](x))
        
        # Output layer (no activation for direct physical quantities)
        output = self.layers[-1](x)
        
        return output
    
    def predict_convergence(self, x: torch.Tensor, y: torch.Tensor, 
                           lens_params: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predict convergence κ at given coordinates
        
        Args:
            x: x-coordinates, shape (N,) or (batch, N)
            y: y-coordinates, shape (N,) or (batch, N)
            lens_params: Optional lens parameters, shape (batch, P)
        
        Returns:
            Convergence κ, shape (batch, N) or (N,)
        """
        # Prepare input
        if lens_params is not None:
            # Expand lens_params to match coordinate grid
            batch_size = lens_params.shape[0]
            n_points = x.shape[-1] if len(x.shape) > 1 else x.shape[0]
            
            # Reshape coordinates
            x_flat = x.view(batch_size, -1)
            y_flat = y.view(batch_size, -1)
            
            # Stack coordinates with lens params
            coords = torch.stack([x_flat, y_flat], dim=-1)  # (batch, N, 2)
            params_expanded = lens_params.unsqueeze(1).expand(-1, coords.shape[1], -1)
            inputs = torch.cat([coords, params_expanded], dim=-1)  # (batch, N, 2+P)
        else:
            # Simple case: just coordinates
            inputs = torch.stack([x, y], dim=-1)
        
        # Forward pass
        outputs = self(inputs)
        
        # Extract convergence (first output)
        convergence = outputs[..., 0]
        
        return convergence


# ============================================================================
# NFW-Specific PINN
# ============================================================================

class NFW_PINN(LensingPINN):
    """
    PINN specialized for NFW (Navarro-Frenk-White) mass profiles
    
    Incorporates physical priors:
    - Axisymmetry: κ(r) depends only on radius r = √(x² + y²)
    - Scale radius: Characteristic radius where profile changes behavior
    - Asymptotic behavior: κ ∝ r⁻² at large radii
    
    This specialization improves convergence and accuracy for NFW halos.
    """
    
    def __init__(
        self,
        hidden_dims: Optional[List[int]] = None,
        activation: str = 'tanh'
    ):
        # NFW input: (r_norm, log(1+r_norm), 1/(1+r_norm), log_mass, conc)
        # NFW output: (κ, ψ, α_r, dα_dr)
        if hidden_dims is None:
            hidden_dims = [64, 64, 64]
            
        super().__init__(
            input_dim=5,  # Physics-informed features
            hidden_dims=hidden_dims,
            output_dim=4,
            activation=activation,
            use_skip_connections=True
        )
        
        logger.info("Initialized NFW-specific PINN")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with NFW-specific structure
        
        Args:
            x: Input tensor (batch, 3) = [r, log_mass, concentration]
        
        Returns:
            Output tensor (batch, 4) = [κ, ψ, α_r, dα_dr]
        """
        # Extract inputs
        r = x[:, 0:1]  # Radius
        log_mass = x[:, 1:2]  # Log of halo mass
        conc = x[:, 2:3]  # Concentration parameter
        
        # Scale radius from concentration
        # r_s = r_vir / c, where r_vir ∝ M^(1/3)
        r_vir = torch.exp(log_mass / 3.0)  # Simplified relation
        r_s = r_vir / conc
        
        # Normalized radius
        r_norm = r / r_s
        
        # Physics-informed features
        # These features encode known NFW behavior
        features = torch.cat([
            r_norm,  # Normalized radius
            torch.log(1 + r_norm),  # Log term (appears in NFW formula)
            1 / (1 + r_norm),  # Denominator term
            log_mass,  # Mass
            conc  # Concentration
        ], dim=1)
        
        # Pass through base network
        output = super().forward(features)
        
        # Apply physical constraints
        # 1. Convergence must be positive
        kappa = torch.relu(output[:, 0:1])
        
        # 2. Potential increases with mass
        psi = torch.exp(log_mass / 10.0) * output[:, 1:2]
        
        # 3. Deflection angle
        alpha_r = output[:, 2:3]
        
        # 4. Derivative of deflection
        dalpha_dr = output[:, 3:4]
        
        return torch.cat([kappa, psi, alpha_r, dalpha_dr], dim=1)


# ============================================================================
# Physics Loss Functions
# ============================================================================

class PhysicsLoss:
    """
    Physics-informed loss function for gravitational lensing
    
    Combines:
    1. Data loss: MSE between prediction and ground truth
    2. Physics loss: Violation of Poisson equation ∇²ψ = 2κ
    3. Boundary loss: κ → 0 as r → ∞
    4. Symmetry loss: Axisymmetry constraint for NFW
    """
    
    def __init__(
        self,
        lambda_physics: float = 1.0,
        lambda_boundary: float = 0.1,
        lambda_symmetry: float = 0.1
    ):
        self.lambda_physics = lambda_physics
        self.lambda_boundary = lambda_boundary
        self.lambda_symmetry = lambda_symmetry
    
    def data_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """MSE between prediction and ground truth"""
        return torch.mean((pred - target) ** 2)
    
    def poisson_residual(
        self,
        kappa: torch.Tensor,
        psi: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Poisson equation residual: |∇²ψ - 2κ|
        
        Uses automatic differentiation to compute Laplacian
        """
        # Enable gradient computation
        x.requires_grad_(True)
        y.requires_grad_(True)
        
        # First derivatives
        d_psi_dx = torch.autograd.grad(
            psi, x, grad_outputs=torch.ones_like(psi),
            create_graph=True, retain_graph=True
        )[0]
        
        d_psi_dy = torch.autograd.grad(
            psi, y, grad_outputs=torch.ones_like(psi),
            create_graph=True, retain_graph=True
        )[0]
        
        # Second derivatives (Laplacian)
        d2_psi_dx2 = torch.autograd.grad(
            d_psi_dx, x, grad_outputs=torch.ones_like(d_psi_dx),
            create_graph=True, retain_graph=True
        )[0]
        
        d2_psi_dy2 = torch.autograd.grad(
            d_psi_dy, y, grad_outputs=torch.ones_like(d_psi_dy),
            create_graph=True, retain_graph=True
        )[0]
        
        laplacian_psi = d2_psi_dx2 + d2_psi_dy2
        
        # Poisson equation: ∇²ψ = 2κ
        residual = laplacian_psi - 2 * kappa
        
        return torch.mean(residual ** 2)
    
    def boundary_loss(self, kappa: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """
        Enforce boundary condition: κ → 0 as r → ∞
        
        Penalize non-zero convergence at large radii
        """
        # Find points at large radii (top 10% of radii)
        r_threshold = torch.quantile(r, 0.9)
        mask = r > r_threshold
        
        if mask.sum() > 0:
            boundary_kappa = kappa[mask]
            return torch.mean(boundary_kappa ** 2)
        else:
            return torch.tensor(0.0, device=kappa.device)
    
    def symmetry_loss(
        self,
        kappa_1: torch.Tensor,
        kappa_2: torch.Tensor
    ) -> torch.Tensor:
        """
        Enforce axisymmetry: κ(r, θ) = κ(r, θ')
        
        For points at same radius but different angles,
        convergence should be the same
        """
        return torch.mean((kappa_1 - kappa_2) ** 2)
    
    def total_loss(
        self,
        pred_kappa: torch.Tensor,
        true_kappa: torch.Tensor,
        pred_psi: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total physics-informed loss
        
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual loss components
        """
        # Data loss
        loss_data = self.data_loss(pred_kappa, true_kappa)
        
        # Physics loss (Poisson equation)
        loss_physics = self.poisson_residual(pred_kappa, pred_psi, x, y)
        
        # Boundary loss
        r = torch.sqrt(x ** 2 + y ** 2)
        loss_boundary = self.boundary_loss(pred_kappa, r)
        
        # Total loss
        total = (
            loss_data +
            self.lambda_physics * loss_physics +
            self.lambda_boundary * loss_boundary
        )
        
        # Return total and breakdown
        loss_dict = {
            'total': total.item(),
            'data': loss_data.item(),
            'physics': loss_physics.item(),
            'boundary': loss_boundary.item()
        }
        
        return total, loss_dict


# ============================================================================
# Training Utilities
# ============================================================================

class PINNTrainer:
    """Trainer for physics-informed neural networks"""
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.physics_loss = PhysicsLoss()
        self.history = {'loss': [], 'data_loss': [], 'physics_loss': []}
    
    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        target_kappa: torch.Tensor
    ) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        inputs = torch.stack([x, y], dim=-1)
        outputs = self.model(inputs)
        pred_kappa = outputs[:, 0]
        pred_psi = outputs[:, 1]
        
        # Compute loss
        loss, loss_dict = self.physics_loss.total_loss(
            pred_kappa, target_kappa, pred_psi, x, y
        )
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss_dict
    
    def train(
        self,
        train_data: Dict[str, torch.Tensor],
        n_epochs: int = 1000,
        verbose: int = 100
    ):
        """Train the PINN"""
        logger.info(f"Starting training for {n_epochs} epochs...")
        
        for epoch in range(n_epochs):
            loss_dict = self.train_step(
                train_data['x'],
                train_data['y'],
                train_data['kappa']
            )
            
            # Record history
            self.history['loss'].append(loss_dict['total'])
            self.history['data_loss'].append(loss_dict['data'])
            self.history['physics_loss'].append(loss_dict['physics'])
            
            # Print progress
            if (epoch + 1) % verbose == 0:
                logger.info(
                    f"Epoch {epoch+1}/{n_epochs} | "
                    f"Loss: {loss_dict['total']:.6f} | "
                    f"Data: {loss_dict['data']:.6f} | "
                    f"Physics: {loss_dict['physics']:.6f}"
                )


# ============================================================================
# Model Factory
# ============================================================================

def create_lensing_pinn(
    model_type: str = 'general',
    **kwargs
) -> LensingPINN:
    """
    Factory function to create lensing PINNs
    
    Args:
        model_type: Type of model ('general', 'nfw')
        **kwargs: Additional arguments passed to model constructor
    
    Returns:
        Initialized PINN model
    """
    if model_type == 'general':
        return LensingPINN(**kwargs)
    elif model_type == 'nfw':
        return NFW_PINN(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == '__main__':
    # Test the PINN
    logger.info("Testing LensingPINN...")
    
    # Create model
    model = create_lensing_pinn(model_type='nfw')
    
    # Generate test data
    batch_size = 100
    r = torch.rand(batch_size, 1) * 2.0  # Radii from 0 to 2
    log_mass = torch.ones(batch_size, 1) * 12.0  # 10^12 solar masses
    conc = torch.ones(batch_size, 1) * 5.0  # Concentration = 5
    
    inputs = torch.cat([r, log_mass, conc], dim=1)
    
    # Forward pass
    outputs = model(inputs)
    
    logger.info(f"Input shape: {inputs.shape}")
    logger.info(f"Output shape: {outputs.shape}")
    logger.info(f"Convergence range: [{outputs[:, 0].min():.6f}, {outputs[:, 0].max():.6f}]")
    logger.info("✅ PINN test successful!")
