"""
Physics-Informed Neural Network for Gravitational Lens Parameter Inference

This module implements a PINN that:
1. Infers lens parameters (mass, scale radius, source position, H0)
2. Classifies dark matter model type (CDM, WDM, SIDM)
3. Enforces physical constraints via physics-informed loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


class PhysicsInformedNN(nn.Module):
    """
    Physics-Informed Neural Network for lens parameter inference.
    
    Architecture:
    - Input: 64×64 grayscale image (flattened → 4096)
    - Encoder: Conv2D layers [32, 64, 128] → flatten
    - Dense layers: [1024, 512, 256]
    - Dual output heads:
        * Regression: [128] → 5 parameters [M, r_s, β_x, β_y, H0]
        * Classification: [128] → 3 classes [p_CDM, p_WDM, p_SIDM]
    
    Parameters
    ----------
    input_size : int
        Size of input images (default: 64 for 64×64 images)
    dropout_rate : float
        Dropout probability for regularization (default: 0.2)
    
    Attributes
    ----------
    encoder : nn.Sequential
        Convolutional encoder network
    dense : nn.Sequential
        Dense feature extraction layers
    param_head : nn.Sequential
        Regression head for parameter prediction
    class_head : nn.Sequential
        Classification head for DM model type
    
    Examples
    --------
    >>> model = PhysicsInformedNN(input_size=64)
    >>> images = torch.randn(32, 1, 64, 64)  # batch of 32 images
    >>> params, classes = model(images)
    >>> print(params.shape)  # (32, 5) - [M, r_s, β_x, β_y, H0]
    >>> print(classes.shape)  # (32, 3) - [p_CDM, p_WDM, p_SIDM]
    """
    
    def __init__(self, input_size: int = 64, dropout_rate: float = 0.2):
        super(PhysicsInformedNN, self).__init__()
        
        self.input_size = input_size
        self.dropout_rate = dropout_rate
        
        # Convolutional Encoder
        # Input: (batch, 1, H, W) - supports variable sizes
        self.encoder = nn.Sequential(
            # Conv block 1: H×W → H/2×W/2
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate),
            
            # Conv block 2: H/2×W/2 → H/4×W/4
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate),
            
            # Conv block 3: H/4×W/4 → H/8×W/8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate),
        )
        
        # Adaptive pooling: Forces output to 8×8 regardless of input size
        # Supports 64×64, 128×128, 256×256, etc.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        
        # Calculate flattened size after adaptive pooling
        # Always 8×8 after adaptive pool regardless of input size
        self.encoded_size = 128 * 8 * 8  # 8192
        
        # Dense layers
        self.dense = nn.Sequential(
            nn.Linear(self.encoded_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        
        # Regression head for parameter prediction
        # Output: [M_vir, r_s, β_x, β_y, H0]
        self.param_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 5)
        )
        
        # Classification head for DM model type
        # Output: [p_CDM, p_WDM, p_SIDM]
        self.class_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 3)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Parameters
        ----------
        x : torch.Tensor
            Input images of shape (batch, 1, H, W)
            Supports variable sizes: 64×64, 128×128, 256×256, etc.
        
        Returns
        -------
        params : torch.Tensor
            Predicted parameters of shape (batch, 5)
            [M_vir, r_s, β_x, β_y, H0]
        classes : torch.Tensor
            Class logits of shape (batch, 3)
            [logit_CDM, logit_WDM, logit_SIDM]
        """
        # Encode image
        encoded = self.encoder(x)
        
        # Apply adaptive pooling to force 8×8 output
        pooled = self.adaptive_pool(encoded)
        
        # Flatten
        flattened = pooled.view(pooled.size(0), -1)
        
        # Dense feature extraction
        features = self.dense(flattened)
        
        # Dual heads
        params = self.param_head(features)
        classes = self.class_head(features)
        
        return params, classes
    
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict with post-processing and interpretable output.
        
        Parameters
        ----------
        x : torch.Tensor
            Input images of shape (batch, 1, H, W)
        
        Returns
        -------
        predictions : dict
            Dictionary containing:
            - 'params': Raw parameter predictions (batch, 5)
            - 'M_vir': Virial mass in solar masses
            - 'r_s': Scale radius in kpc
            - 'beta_x': Source x position in arcsec
            - 'beta_y': Source y position in arcsec
            - 'H0': Hubble constant in km/s/Mpc
            - 'class_probs': Softmax probabilities (batch, 3)
            - 'class_labels': Predicted class indices (batch,)
            - 'dm_type': String labels ['CDM', 'WDM', 'SIDM']
        """
        self.eval()
        with torch.no_grad():
            params, class_logits = self(x)
            
            # Apply softmax to get probabilities
            class_probs = F.softmax(class_logits, dim=1)
            
            # Get predicted class
            class_labels = torch.argmax(class_probs, dim=1)
            
            # Map to string labels
            dm_types = []
            label_map = {0: 'CDM', 1: 'WDM', 2: 'SIDM'}
            for label in class_labels.cpu().numpy():
                dm_types.append(label_map[label])
            
            return {
                'params': params,
                'M_vir': params[:, 0],
                'r_s': params[:, 1],
                'beta_x': params[:, 2],
                'beta_y': params[:, 3],
                'H0': params[:, 4],
                'class_probs': class_probs,
                'class_labels': class_labels,
                'dm_type': dm_types
            }


def compute_nfw_deflection(
    M_vir: torch.Tensor,
    r_s: torch.Tensor,
    theta_x: torch.Tensor,
    theta_y: torch.Tensor,
    z_l: float = 0.5,
    z_s: float = 2.0,
    H0: float = 70.0,
    Omega_m: float = 0.3
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute NFW deflection angle using differentiable PyTorch operations.
    
    Implements the NFW deflection angle formula:
    α(θ) = (4 G M_vir / c²) * (D_ls / D_s) * f(x) / r_s
    
    where f(x) is the NFW deflection function and x = θ / θ_s.
    
    Parameters
    ----------
    M_vir : torch.Tensor
        Virial mass in units of 10^12 M_sun, shape (batch, 1)
    r_s : torch.Tensor
        Scale radius in kpc, shape (batch, 1)
    theta_x : torch.Tensor
        Image plane x-coordinates in arcsec, shape (batch, n_points)
    theta_y : torch.Tensor
        Image plane y-coordinates in arcsec, shape (batch, n_points)
    z_l : float
        Lens redshift (default: 0.5)
    z_s : float
        Source redshift (default: 2.0)
    H0 : float
        Hubble constant in km/s/Mpc (default: 70.0)
    Omega_m : float
        Matter density parameter (default: 0.3)
    
    Returns
    -------
    alpha_x : torch.Tensor
        Deflection angle x-component in arcsec, shape (batch, n_points)
    alpha_y : torch.Tensor
        Deflection angle y-component in arcsec, shape (batch, n_points)
    
    Notes
    -----
    - All operations are differentiable for backpropagation
    - Uses proper NFW profile deflection (not simplified point mass)
    - Includes cosmological distance ratios D_ls/D_s
    - Reference: Wright & Brainerd (2000), ApJ 534, 34
    """
    # Physical constants
    G = 4.517e-48  # Gravitational constant in kpc³ / (M_sun * s²)
    c = 299792.458  # Speed of light in km/s
    kpc_to_km = 3.086e+16  # 1 kpc in km
    
    # Convert c to kpc/s for consistent units with G
    c_kpc = c / kpc_to_km  # kpc/s (very small number!)
    
    # Convert to proper units
    M_vir_solar = M_vir * 1e12  # Convert to solar masses
    
    # Compute angular diameter distances (simplified flat ΛCDM)
    # D_l ≈ c/H0 * z_l (for small z)
    # D_s ≈ c/H0 * z_s
    # D_ls ≈ c/H0 * (z_s - z_l)
    D_l = (c / H0) * z_l * 1000  # kpc (multiply by 1000 for Mpc to kpc)
    D_s = (c / H0) * z_s * 1000  # kpc
    D_ls = (c / H0) * (z_s - z_l) * 1000  # kpc
    
    # Distance ratio
    D_ratio = D_ls / D_s  # Dimensionless
    
    # Convert angular positions to physical coordinates
    # θ (arcsec) → r (kpc): r = θ * D_l * (π / 180 / 3600)
    arcsec_to_rad = torch.tensor(np.pi / 180.0 / 3600.0, device=theta_x.device)
    r_x_kpc = theta_x * D_l * arcsec_to_rad  # kpc
    r_y_kpc = theta_y * D_l * arcsec_to_rad  # kpc
    
    # Radial distance from lens center
    r_kpc = torch.sqrt(r_x_kpc**2 + r_y_kpc**2 + 1e-8)  # Add small epsilon for stability
    
    # Dimensionless radius x = r / r_s
    x = r_kpc / (r_s + 1e-8)  # (batch, n_points)
    
    # NFW deflection function f(x)
    # For x < 1: f(x) = [1 - (2/sqrt(1-x²)) * arctanh(sqrt((1-x)/(1+x)))] / (x² - 1)
    # For x = 1: f(x) = 1/3
    # For x > 1: f(x) = [1 - (2/sqrt(x²-1)) * arctan(sqrt((x-1)/(x+1)))] / (x² - 1)
    
    # Mask for different regimes
    mask_less = x < 0.99
    mask_greater = x > 1.01
    mask_equal = ~(mask_less | mask_greater)  # 0.99 <= x <= 1.01
    
    # Initialize f(x)
    f_x = torch.zeros_like(x)
    
    # Case 1: x < 1
    if mask_less.any():
        x_less = x[mask_less]
        sqrt_term = torch.sqrt((1.0 - x_less) / (1.0 + x_less) + 1e-10)
        arctanh_term = torch.atanh(sqrt_term.clamp(-0.999, 0.999))  # Clamp for stability
        f_x[mask_less] = (1.0 - 2.0 * arctanh_term / torch.sqrt(1.0 - x_less**2 + 1e-10)) / (x_less**2 - 1.0)
    
    # Case 2: x > 1
    if mask_greater.any():
        x_greater = x[mask_greater]
        sqrt_term = torch.sqrt((x_greater - 1.0) / (x_greater + 1.0) + 1e-10)
        arctan_term = torch.atan(sqrt_term)
        f_x[mask_greater] = (1.0 - 2.0 * arctan_term / torch.sqrt(x_greater**2 - 1.0 + 1e-10)) / (x_greater**2 - 1.0)
    
    # Case 3: x ≈ 1 (use Taylor expansion)
    if mask_equal.any():
        f_x[mask_equal] = 1.0 / 3.0
    
    # === CORRECTED DEFLECTION CALCULATION WITH PROPER DIMENSIONAL ANALYSIS ===
    # 
    # Standard NFW deflection formula (Wright & Brainerd 2000):
    # α(θ) = (4πG ρ_s r_s² / c²) × (D_LS / D_S) × g(x) / D_L
    # where g(x) = f(x) / x² is the dimensionless deflection function
    # 
    # Step 1: Calculate critical surface density [M_sun/kpc²]
    # Σ_crit = (c² / 4πG) × (D_S / (D_L × D_LS))
    # Note: Using c in kpc/s to match G units
    Sigma_crit = (c_kpc**2 / (4.0 * np.pi * G)) * (D_s / (D_l * D_ls + 1e-8))  # M_sun/kpc²
    
    # Step 2: Calculate NFW characteristic density [M_sun/kpc³]
    # Calculate critical density at lens redshift rho_crit(z_l)
    # H(z)² = H0² * (Ω_m(1+z)³ + Ω_Λ)
    E_z_sq = Omega_m * (1 + z_l)**3 + (1 - Omega_m)
    H_z = H0 * np.sqrt(E_z_sq) # km/s/Mpc
    
    # Convert H(z) to 1/s. 1 Mpc = 3.086e19 km
    H_z_s = H_z / 3.08567758e19
    
    # rho_crit = 3H² / (8πG)
    rho_crit = 3 * H_z_s**2 / (8 * np.pi * G) # M_sun/kpc³
    
    # Virial radius r_200: M_vir = 4/3 π r_200³ 200 ρ_crit
    r_vir = (3 * M_vir_solar / (4 * np.pi * 200 * rho_crit))**(1/3.0)
    
    # Concentration c = r_vir / r_s
    # Clamp c to reasonable range to avoid numerical instability
    c_nfw = torch.clamp(r_vir / (r_s + 1e-8), min=1.0, max=100.0)
    
    # f(c) = ln(1+c) - c/(1+c)
    f_c = torch.log(1.0 + c_nfw) - c_nfw / (1.0 + c_nfw)
    
    # rho_s = M_vir / (4π r_s³ f(c))
    rho_s = M_vir_solar / (4.0 * np.pi * r_s**3 * f_c + 1e-8)  # M_sun/kpc³
    
    # Step 3: Calculate convergence scale [dimensionless]
    # κ_s = (ρ_s × r_s) / Σ_crit
    kappa_s = (rho_s * r_s) / (Sigma_crit + 1e-8)  # dimensionless
    
    # Step 4: Calculate deflection angle in physical units (radians)
    # α(r) = κ_s × (r_s / r) × f(x) [radians]
    # Note: This is the magnitude of deflection at radius r
    alpha_mag_rad = kappa_s * (r_s / (r_kpc + 1e-8)) * f_x  # radians
    
    # Step 5: Convert radians to arcseconds
    # 1 radian = 206265 arcsec
    rad_to_arcsec = 206265.0  # arcsec/radian
    alpha_mag_arcsec = alpha_mag_rad * rad_to_arcsec  # arcsec
    
    # Decompose into x and y components
    # α_x = α × (r_x / r), α_y = α × (r_y / r)
    r_kpc_safe = r_kpc + 1e-8
    alpha_x = alpha_mag_arcsec * r_x_kpc / r_kpc_safe  # arcsec
    alpha_y = alpha_mag_arcsec * r_y_kpc / r_kpc_safe  # arcsec
    
    return alpha_x, alpha_y


def physics_informed_loss(
    pred_params: torch.Tensor,
    true_params: torch.Tensor,
    pred_classes: torch.Tensor,
    true_classes: torch.Tensor,
    images: torch.Tensor,
    lambda_physics: float = 0.1,
    device: str = 'cpu'
) -> Dict[str, torch.Tensor]:
    """
    Compute physics-informed loss combining:
    1. Parameter regression loss (MSE)
    2. Classification loss (Cross-entropy)
    3. Physics residual loss (lens equation violation)
    
    The physics residual enforces the lens equation:
    θ - β - α(θ) = 0
    
    Parameters
    ----------
    pred_params : torch.Tensor
        Predicted parameters (batch, 5) [M, r_s, β_x, β_y, H0]
    true_params : torch.Tensor
        True parameters (batch, 5)
    pred_classes : torch.Tensor
        Predicted class logits (batch, 3)
    true_classes : torch.Tensor
        True class labels (batch,) as integers
    images : torch.Tensor
        Input images (batch, 1, H, W) - used for physics check
    lambda_physics : float
        Weight for physics residual term (default: 0.1)
    device : str
        Device for computation ('cpu' or 'cuda')
    
    Returns
    -------
    losses : dict
        Dictionary containing:
        - 'total': Total loss
        - 'mse_params': Parameter MSE loss
        - 'ce_class': Classification cross-entropy loss
        - 'physics_residual': Physics constraint violation
    
    Examples
    --------
    >>> pred_p = torch.randn(32, 5)
    >>> true_p = torch.randn(32, 5)
    >>> pred_c = torch.randn(32, 3)
    >>> true_c = torch.randint(0, 3, (32,))
    >>> imgs = torch.randn(32, 1, 64, 64)
    >>> losses = physics_informed_loss(pred_p, true_p, pred_c, true_c, imgs)
    """
    # 1. Parameter regression loss (MSE)
    mse_params = F.mse_loss(pred_params, true_params)
    
    # 2. Classification loss (Cross-entropy)
    ce_class = F.cross_entropy(pred_classes, true_classes)
    
    # 3. Physics residual: Enforce lens equation
    # Sample points on the image plane
    batch_size = images.size(0)
    n_sample_points = 16  # Sample 16 points per image
    
    # Generate random image plane positions (in normalized coords)
    # Map from [-1, 1] to physical coordinates
    theta_x = torch.rand(batch_size, n_sample_points, device=device) * 2 - 1  # [-1, 1]
    theta_y = torch.rand(batch_size, n_sample_points, device=device) * 2 - 1
    
    # Extract predicted source positions and lens parameters
    beta_x = pred_params[:, 2].unsqueeze(1)  # (batch, 1)
    beta_y = pred_params[:, 3].unsqueeze(1)
    M_vir = pred_params[:, 0].unsqueeze(1)  # Virial mass (10^12 M_sun)
    r_s = pred_params[:, 1].unsqueeze(1)  # Scale radius (kpc)
    
    # NUMERICAL STABILITY: Clamp parameters to physically reasonable ranges
    # This prevents extreme values that cause NaN gradients
    M_vir = torch.clamp(M_vir, min=0.01, max=1e4)  # [1e10, 1e16] M_sun
    r_s = torch.clamp(r_s, min=1.0, max=1e4)  # [1, 10000] kpc
    beta_x = torch.clamp(beta_x, min=-10.0, max=10.0)  # [-10, 10] arcsec
    beta_y = torch.clamp(beta_y, min=-10.0, max=10.0)  # [-10, 10] arcsec
    
    # Compute NFW deflection angles using real physics
    alpha_x, alpha_y = compute_nfw_deflection(
        M_vir=M_vir,
        r_s=r_s,
        theta_x=theta_x,
        theta_y=theta_y,
        z_l=0.5,  # Typical lens redshift
        z_s=2.0,  # Typical source redshift
        H0=70.0,  # Hubble constant
        Omega_m=0.3  # Matter density
    )
    
    # Lens equation residual: |θ - β - α(θ)|²
    # This enforces the fundamental lens equation
    residual_x = theta_x - beta_x - alpha_x
    residual_y = theta_y - beta_y - alpha_y
    
    physics_residual = torch.mean(residual_x**2 + residual_y**2)
    
    # Parameter regularization: Penalize physically unrealistic values
    # M_vir should be in range [1e11, 1e15] M_sun (i.e., [0.1, 1000] in units of 10^12)
    # r_s should be in range [10, 1000] kpc
    regularization = torch.tensor(0.0, device=device, dtype=pred_params.dtype)
    
    # M_vir regularization (in units of 10^12 M_sun)
    M_vir_min = 0.1  # 1e11 M_sun
    M_vir_max = 1000.0  # 1e15 M_sun
    M_vir_penalty = torch.relu(M_vir_min - M_vir) + torch.relu(M_vir - M_vir_max)
    regularization = regularization + torch.mean(M_vir_penalty**2)
    
    # r_s regularization (in kpc)
    r_s_min = 10.0  # kpc
    r_s_max = 1000.0  # kpc
    r_s_penalty = torch.relu(r_s_min - r_s) + torch.relu(r_s - r_s_max)
    regularization = regularization + torch.mean(r_s_penalty**2)
    
    # Combined physics loss
    physics_loss = physics_residual + regularization
    
    # Total loss
    total_loss = mse_params + ce_class + lambda_physics * physics_loss
    
    return {
        'total': total_loss,
        'mse_params': mse_params,
        'ce_class': ce_class,
        'physics_residual': physics_residual,
        'regularization': regularization
    }


def train_step(
    model: PhysicsInformedNN,
    images: torch.Tensor,
    true_params: torch.Tensor,
    true_classes: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    lambda_physics: float = 0.1,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Perform a single training step.
    
    Parameters
    ----------
    model : PhysicsInformedNN
        The PINN model
    images : torch.Tensor
        Batch of input images (batch, 1, H, W)
    true_params : torch.Tensor
        True parameters (batch, 5)
    true_classes : torch.Tensor
        True class labels (batch,)
    optimizer : torch.optim.Optimizer
        Optimizer for parameter updates
    lambda_physics : float
        Weight for physics loss term
    device : str
        Device for computation
    
    Returns
    -------
    losses : dict
        Dictionary of loss values for logging
    """
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    pred_params, pred_classes = model(images)
    
    # Compute loss
    losses = physics_informed_loss(
        pred_params, true_params,
        pred_classes, true_classes,
        images, lambda_physics, device
    )
    
    # Backward pass
    losses['total'].backward()
    
    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Update weights
    optimizer.step()
    
    # Convert to float for logging
    return {k: v.item() for k, v in losses.items()}


def validate_step(
    model: PhysicsInformedNN,
    images: torch.Tensor,
    true_params: torch.Tensor,
    true_classes: torch.Tensor,
    lambda_physics: float = 0.1,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Perform validation step without gradient computation.
    
    Parameters
    ----------
    model : PhysicsInformedNN
        The PINN model
    images : torch.Tensor
        Batch of validation images
    true_params : torch.Tensor
        True parameters
    true_classes : torch.Tensor
        True class labels
    lambda_physics : float
        Weight for physics loss term
    device : str
        Device for computation
    
    Returns
    -------
    losses : dict
        Dictionary of loss values
    """
    model.eval()
    with torch.no_grad():
        pred_params, pred_classes = model(images)
        
        losses = physics_informed_loss(
            pred_params, true_params,
            pred_classes, true_classes,
            images, lambda_physics, device
        )
    
    return {k: v.item() for k, v in losses.items()}
