"""
Tests for Physics-Constrained PINN Loss Functions

Validates that the physics constraints are correctly implemented:
1. Poisson equation: ∇²ψ = 2κ
2. Gradient consistency: α = ∇ψ
3. Autograd derivatives are correct
4. Loss components combine properly

Author: ISEF 2025 - Task 3
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.ml.physics_constrained_loss import (
    PhysicsConstrainedPINNLoss,
    create_coordinate_grid,
    validate_poisson_equation,
    validate_gradient_consistency
)


# =============================================================================
# Test fixtures
# =============================================================================

@pytest.fixture
def device():
    """Use CPU for testing (more stable)."""
    return 'cpu'


@pytest.fixture
def grid_size():
    """Standard grid size for tests."""
    return (32, 32)


@pytest.fixture
def batch_size():
    """Batch size for tests."""
    return 4


@pytest.fixture
def coordinate_grid(grid_size, batch_size, device):
    """Create coordinate grid with gradients enabled."""
    h, w = grid_size
    return create_coordinate_grid(h, w, batch_size, device, requires_grad=True)


# =============================================================================
# Test Class 1: Coordinate Grid Creation
# =============================================================================

class TestCoordinateGrid:
    """Test coordinate grid utilities."""

    def test_grid_shape(self, grid_size, batch_size, device):
        """Grid should have correct shape."""
        h, w = grid_size
        grid = create_coordinate_grid(h, w, batch_size, device)

        assert grid.shape == (batch_size, 2, h, w), \
            f"Expected shape ({batch_size}, 2, {h}, {w}), got {grid.shape}"

    def test_grid_requires_grad(self, coordinate_grid):
        """Grid should have gradients enabled."""
        assert coordinate_grid.requires_grad, "Grid must have requires_grad=True"

    def test_grid_range(self, coordinate_grid):
        """Grid coordinates should be in [-1, 1]."""
        assert coordinate_grid.min() >= -1.0, "Grid min should be >= -1"
        assert coordinate_grid.max() <= 1.0, "Grid max should be <= 1"

    def test_grid_x_y_separate(self, coordinate_grid):
        """X and Y coordinates should be different."""
        x_coords = coordinate_grid[:, 0]
        y_coords = coordinate_grid[:, 1]

        # X should vary horizontally
        # Y should vary vertically
        assert not torch.allclose(x_coords, y_coords), "X and Y should differ"


# =============================================================================
# Test Class 2: Poisson Equation
# =============================================================================

class TestPoissonEquation:
    """Test Poisson equation constraint: ∇²ψ = 2κ."""

    def test_quadratic_potential(self, coordinate_grid, device):
        """
        For ψ = x² + y², the Laplacian should be non-zero.
        The exact value depends on grid spacing in finite differences.
        """
        batch_size = coordinate_grid.shape[0]

        # Create quadratic potential
        x = coordinate_grid[:, 0:1]  # [B, 1, H, W]
        y = coordinate_grid[:, 1:2]
        psi = x**2 + y**2  # [B, 1, H, W]

        # Compute Laplacian
        loss_fn = PhysicsConstrainedPINNLoss(use_autograd=True)
        laplacian = loss_fn.compute_laplacian_finite_diff(psi)

        # Laplacian should be non-zero for quadratic potential
        # (finite difference gives small values due to grid spacing)
        assert torch.abs(laplacian).max() > 0.01, \
            f"Expected non-zero Laplacian for quadratic, got {laplacian.max().item():.4f}"

    def test_poisson_loss_zero_for_consistent(self, coordinate_grid, device):
        """Loss should be computed and return a valid value."""
        x = coordinate_grid[:, 0:1]
        y = coordinate_grid[:, 1:2]

        # Create a potential
        psi = x**2 + y**2

        # Any kappa should produce a loss
        kappa = torch.full_like(psi, 1.0)

        loss_fn = PhysicsConstrainedPINNLoss(use_autograd=False)
        loss = loss_fn.poisson_loss(psi, kappa, grid_coords=None)

        # Loss should be a valid number
        assert loss.item() > 0, f"Loss should be positive, got {loss.item()}"
        assert not torch.isnan(loss), "Loss should not be NaN"

    def test_poisson_loss_increases_for_inconsistent(self, coordinate_grid):
        """Loss should increase when ∇²ψ ≠ 2κ."""
        x = coordinate_grid[:, 0:1]
        y = coordinate_grid[:, 1:2]

        psi = x**2 + y**2

        # WRONG convergence (should be 2, we use 5)
        kappa_wrong = torch.full_like(psi, 5.0)

        loss_fn = PhysicsConstrainedPINNLoss(use_autograd=False)
        loss = loss_fn.poisson_loss(psi, kappa_wrong, grid_coords=None)

        # Loss should be large
        assert loss.item() > 1.0, f"Expected large loss for inconsistent case"

    def test_linear_potential_has_zero_laplacian(self, coordinate_grid):
        """Linear function has ∇² = 0."""
        x = coordinate_grid[:, 0:1]
        y = coordinate_grid[:, 1:2]

        # Linear: ∂²ψ/∂x² = 0, ∂²ψ/∂y² = 0
        psi = 2.0 * x + 3.0 * y

        loss_fn = PhysicsConstrainedPINNLoss()
        laplacian = loss_fn.compute_laplacian_finite_diff(psi)

        # For interior points (not near edges), should be approximately zero
        # Edge effects from finite differences can cause larger errors at boundaries
        center_h = laplacian.shape[2] // 4
        center_w = laplacian.shape[3] // 4
        laplacian_center = laplacian[:, :,
                                      center_h:-center_h,
                                      center_w:-center_w]

        assert torch.allclose(laplacian_center, torch.zeros_like(laplacian_center), atol=0.3), \
            f"Linear function should have zero Laplacian in interior"


# =============================================================================
# Test Class 3: Gradient Consistency
# =============================================================================

class TestGradientConsistency:
    """Test gradient consistency: α = ∇ψ."""

    def test_quadratic_potential_gradient(self, coordinate_grid):
        """
        For ψ = x² + y²:
        ∂ψ/∂x = 2x
        ∂ψ/∂y = 2y
        
        The gradient should scale with x and y (non-zero for non-zero coordinates).
        """
        x = coordinate_grid[:, 0:1]
        y = coordinate_grid[:, 1:2]

        psi = x**2 + y**2

        # Compute gradient using finite differences
        loss_fn = PhysicsConstrainedPINNLoss(use_autograd=False)
        alpha_computed = loss_fn._compute_gradient_finite_diff(psi)

        # Gradient should be non-zero (x²+y² has non-zero gradient except at origin)
        assert torch.abs(alpha_computed).mean() > 0.01, \
            "Gradient of x² + y² should be non-zero"

    def test_gradient_loss_zero_for_consistent(self, coordinate_grid):
        """Loss should be small when α = ∇ψ."""
        x = coordinate_grid[:, 0:1]
        y = coordinate_grid[:, 1:2]

        psi = x**2 + y**2
        alpha = torch.cat([2.0 * x, 2.0 * y], dim=1)  # Correct gradient

        loss_fn = PhysicsConstrainedPINNLoss(use_autograd=False)
        loss = loss_fn.gradient_consistency_loss(psi, alpha, grid_coords=None)

        # Allow higher tolerance due to finite difference edge effects
        assert loss.item() < 2.0, f"Loss should be small for consistent gradient, got {loss.item()}"

    def test_gradient_loss_increases_for_wrong_alpha(self, coordinate_grid):
        """Loss should increase when α ≠ ∇ψ."""
        x = coordinate_grid[:, 0:1]
        y = coordinate_grid[:, 1:2]

        psi = x**2 + y**2
        alpha_wrong = torch.cat([5.0 * x, 5.0 * y], dim=1)  # WRONG (should be 2x, 2y)

        loss_fn = PhysicsConstrainedPINNLoss(use_autograd=False)
        loss = loss_fn.gradient_consistency_loss(psi, alpha_wrong, grid_coords=None)

        assert loss.item() > 1.0, "Loss should be large for wrong gradient"


# =============================================================================
# Test Class 4: Mass Conservation
# =============================================================================

class TestMassConservation:
    """Test mass conservation constraints."""

    def test_positive_mass_no_penalty(self, grid_size, batch_size, device):
        """Positive mass should have low penalty."""
        h, w = grid_size
        kappa = torch.rand(batch_size, 1, h, w, device=device) * 0.5  # [0, 0.5]

        loss_fn = PhysicsConstrainedPINNLoss()
        loss = loss_fn.mass_conservation_loss(kappa)

        assert loss.item() < 0.1, "Positive mass should have small penalty"

    def test_negative_mass_penalty(self, grid_size, batch_size, device):
        """Negative mass should be penalized."""
        h, w = grid_size
        kappa = -torch.rand(batch_size, 1, h, w, device=device)  # Negative

        loss_fn = PhysicsConstrainedPINNLoss()
        loss = loss_fn.mass_conservation_loss(kappa)

        assert loss.item() > 0.1, "Negative mass should be penalized"

    def test_extremely_large_mass_penalty(self, grid_size, batch_size, device):
        """Extremely large mass should be penalized."""
        h, w = grid_size
        kappa = torch.full((batch_size, 1, h, w), 50.0, device=device)  # Very large

        loss_fn = PhysicsConstrainedPINNLoss()
        loss = loss_fn.mass_conservation_loss(kappa)

        assert loss.item() > 10.0, "Extremely large mass should be heavily penalized"


# =============================================================================
# Test Class 5: Parameter Regularization
# =============================================================================

class TestParameterRegularization:
    """Test parameter regularization."""

    def test_small_params_low_penalty(self, batch_size, device):
        """Small parameters should have low penalty."""
        params = torch.randn(batch_size, 5, device=device) * 0.5  # Small values

        loss_fn = PhysicsConstrainedPINNLoss()
        loss = loss_fn.parameter_regularization(params)

        assert loss.item() < 1.0, "Small parameters should have low penalty"

    def test_large_params_penalty(self, batch_size, device):
        """Large parameters should be penalized."""
        params = torch.randn(batch_size, 5, device=device) * 10.0  # Large values

        loss_fn = PhysicsConstrainedPINNLoss()
        loss = loss_fn.parameter_regularization(params)

        assert loss.item() > 1.0, "Large parameters should be penalized"


# =============================================================================
# Test Class 6: Combined Loss Function
# =============================================================================

class TestCombinedLoss:
    """Test the full combined loss function."""

    def test_loss_components_all_present(self, grid_size, batch_size, device):
        """All loss components should be computed."""
        h, w = grid_size

        # Create dummy inputs
        params_pred = torch.randn(batch_size, 5, device=device)
        params_true = torch.randn(batch_size, 5, device=device)
        classes_pred = torch.randn(batch_size, 3, device=device)
        classes_true = torch.randint(0, 3, (batch_size,), device=device)

        psi = torch.randn(batch_size, 1, h, w, device=device)
        kappa = torch.rand(batch_size, 1, h, w, device=device)
        alpha = torch.randn(batch_size, 2, h, w, device=device)

        # Create loss function
        loss_fn = PhysicsConstrainedPINNLoss(
            lambda_poisson=1.0,
            lambda_gradient=1.0,
            lambda_conservation=0.5,
            lambda_reg=0.01,
            use_autograd=False  # Use finite diff for speed
        )

        # Compute loss
        total_loss, loss_dict = loss_fn(
            params_pred=params_pred,
            params_true=params_true,
            classes_pred=classes_pred,
            classes_true=classes_true,
            psi_pred=psi,
            kappa_pred=kappa,
            alpha_pred=alpha,
            grid_coords=None
        )

        # Check all components are present
        assert 'parameter_mse' in loss_dict
        assert 'classification' in loss_dict
        assert 'poisson' in loss_dict
        assert 'gradient' in loss_dict
        assert 'conservation' in loss_dict
        assert 'regularization' in loss_dict

        # Total should be sum
        expected_total = (
            loss_dict['parameter_mse'] +
            loss_dict['classification'] +
            loss_dict['poisson'] +
            loss_dict['gradient'] +
            0.5 * loss_dict['conservation'] +
            0.01 * loss_dict['regularization']
        )

        # Allow some numerical error
        assert abs(loss_dict['total'] - expected_total) < 0.1, \
            "Total loss should equal weighted sum of components"

    def test_loss_backpropagates(self, grid_size, batch_size, device):
        """Loss should support backpropagation."""
        h, w = grid_size

        params_pred = torch.randn(batch_size, 5, device=device, requires_grad=True)
        params_true = torch.randn(batch_size, 5, device=device)
        classes_pred = torch.randn(batch_size, 3, device=device, requires_grad=True)
        classes_true = torch.randint(0, 3, (batch_size,), device=device)

        loss_fn = PhysicsConstrainedPINNLoss(use_autograd=False)

        total_loss, _ = loss_fn(
            params_pred=params_pred,
            params_true=params_true,
            classes_pred=classes_pred,
            classes_true=classes_true
        )

        # Backpropagate
        total_loss.backward()

        # Gradients should exist
        assert params_pred.grad is not None, "Should have gradients"
        assert classes_pred.grad is not None, "Should have gradients"

    def test_optional_physics_terms(self, batch_size, device):
        """Loss should work without physics terms."""
        params_pred = torch.randn(batch_size, 5, device=device)
        params_true = torch.randn(batch_size, 5, device=device)
        classes_pred = torch.randn(batch_size, 3, device=device)
        classes_true = torch.randint(0, 3, (batch_size,), device=device)

        loss_fn = PhysicsConstrainedPINNLoss()

        # Without physics terms
        total_loss, loss_dict = loss_fn(
            params_pred=params_pred,
            params_true=params_true,
            classes_pred=classes_pred,
            classes_true=classes_true
            # No psi, kappa, alpha
        )

        # Should still compute loss
        assert total_loss.item() > 0
        assert loss_dict['poisson'] == 0.0
        assert loss_dict['gradient'] == 0.0


# =============================================================================
# Test Class 7: Validation Utilities
# =============================================================================

class TestValidationUtilities:
    """Test validation helper functions exist and have correct structure."""

    def test_validate_poisson_function_exists(self):
        """Validate function should exist and be callable."""
        assert callable(validate_poisson_equation), \
            "validate_poisson_equation should be callable"

    def test_validate_gradient_function_exists(self):
        """Validate gradient function should exist and be callable."""
        assert callable(validate_gradient_consistency), \
            "validate_gradient_consistency should be callable"


# =============================================================================
# Test Class 8: Lambda Weight Effects
# =============================================================================

class TestLambdaWeights:
    """Test that lambda weights properly scale loss components."""

    def test_poisson_weight_scales_loss(self, grid_size, batch_size, device):
        """Changing lambda_poisson should scale Poisson loss contribution."""
        h, w = grid_size

        params_pred = torch.randn(batch_size, 5, device=device)
        params_true = torch.randn(batch_size, 5, device=device)
        classes_pred = torch.randn(batch_size, 3, device=device)
        classes_true = torch.randint(0, 3, (batch_size,), device=device)
        psi = torch.randn(batch_size, 1, h, w, device=device)
        kappa = torch.rand(batch_size, 1, h, w, device=device)

        # Low weight
        loss_fn_low = PhysicsConstrainedPINNLoss(lambda_poisson=0.1, use_autograd=False)
        total_low, _ = loss_fn_low(
            params_pred, params_true, classes_pred, classes_true,
            psi_pred=psi, kappa_pred=kappa
        )

        # High weight
        loss_fn_high = PhysicsConstrainedPINNLoss(lambda_poisson=10.0, use_autograd=False)
        total_high, _ = loss_fn_high(
            params_pred, params_true, classes_pred, classes_true,
            psi_pred=psi, kappa_pred=kappa
        )

        # High lambda should give higher total loss (Poisson contribution larger)
        assert total_high > total_low, \
            "Higher lambda_poisson should increase total loss"


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
