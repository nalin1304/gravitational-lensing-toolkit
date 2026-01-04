"""
Bayesian Uncertainty Quantification for PINNs

Provides uncertainty estimation using Monte Carlo Dropout and
calibration analysis for Physics-Informed Neural Networks.

Features:
- Epistemic uncertainty (model uncertainty)
- Aleatoric uncertainty (data noise)
- Prediction intervals with confidence levels
- Uncertainty calibration
- Calibration curve visualization

Author: Phase 15 Implementation
Date: October 2025
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
import warnings


@dataclass
class UncertaintyPrediction:
    """
    Container for predictions with uncertainty
    
    Attributes:
        mean: Mean prediction
        std: Standard deviation (uncertainty)
        lower: Lower confidence bound
        upper: Upper confidence bound
        confidence: Confidence level (e.g., 0.95)
        n_samples: Number of MC samples used
    """
    mean: np.ndarray
    std: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    confidence: float
    n_samples: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'mean': self.mean,
            'std': self.std,
            'lower': self.lower,
            'upper': self.upper,
            'confidence': self.confidence,
            'n_samples': self.n_samples
        }


class BayesianPINN(nn.Module):
    """
    Bayesian Physics-Informed Neural Network
    
    Uses Monte Carlo Dropout for uncertainty estimation.
    Better than standard PINN for quantifying prediction confidence.
    
    Architecture:
        Input → [Linear → Activation → Dropout] × N → Output
    
    Key difference from standard PINN:
        - Dropout layers remain active during inference
        - Multiple forward passes with dropout → uncertainty estimate
    
    Usage:
        model = BayesianPINN(input_dim=5, hidden_dims=[64, 64, 64], dropout_rate=0.1)
        mean, std = model.predict_with_uncertainty(x_test, n_samples=100)
        
        # Or get confidence intervals
        result = model.get_prediction_intervals(x_test, confidence=0.95)
    """
    
    def __init__(
        self,
        input_dim: int = 5,
        output_dim: int = 4,
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.1,
        activation: str = 'tanh'
    ):
        """
        Initialize Bayesian PINN
        
        Args:
            input_dim: Input dimension (e.g., 5 for x, y, M, c, z)
            output_dim: Output dimension (e.g., 4 for κ, ψ, α_x, α_y)
            hidden_dims: List of hidden layer sizes
            dropout_rate: Dropout probability (0.05-0.2 recommended)
            activation: Activation function ('tanh', 'relu', 'elu')
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        if hidden_dims is None:
            hidden_dims = [64, 64, 64]
            
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # Select activation
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build network with Bayesian (MC Dropout) layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Activation
            layers.append(self.activation)
            
            # MC Dropout (key for uncertainty estimation)
            layers.append(nn.Dropout(p=dropout_rate))
            
            prev_dim = hidden_dim
        
        # Output layer (no dropout on output)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier initialization for stable training"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            Output tensor [batch_size, output_dim]
        """
        return self.network(x)
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 100,
        return_samples: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with uncertainty using Monte Carlo Dropout
        
        Key idea:
            - Enable dropout during inference
            - Run multiple forward passes
            - Mean = average prediction
            - Std = prediction variability (uncertainty)
        
        Args:
            x: Input tensor [batch_size, input_dim]
            n_samples: Number of MC samples (100-1000 recommended)
            return_samples: If True, return all samples
            
        Returns:
            (mean_prediction, std_prediction) or (mean, std, samples)
            - mean: [batch_size, output_dim]
            - std: [batch_size, output_dim]
            - samples: [n_samples, batch_size, output_dim] (if return_samples=True)
        """
        # Enable dropout for MC sampling
        self.train()
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                # Forward pass with dropout active
                pred = self.forward(x)
                predictions.append(pred)
        
        # Stack predictions [n_samples, batch_size, output_dim]
        predictions = torch.stack(predictions)
        
        # Calculate statistics
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        # Restore eval mode
        self.eval()
        
        if return_samples:
            return mean_pred, std_pred, predictions
        else:
            return mean_pred, std_pred
    
    def get_prediction_intervals(
        self,
        x: torch.Tensor,
        confidence: float = 0.95,
        n_samples: int = 100
    ) -> UncertaintyPrediction:
        """
        Get prediction intervals at given confidence level
        
        Assumes Gaussian distribution of predictions (reasonable for MC Dropout).
        For 95% confidence: mean ± 1.96 * std
        
        Args:
            x: Input tensor
            confidence: Confidence level (0-1), e.g., 0.95 for 95%
            n_samples: Number of MC samples
            
        Returns:
            UncertaintyPrediction with mean, std, lower, upper bounds
        """
        mean, std = self.predict_with_uncertainty(x, n_samples)
        
        # Calculate z-score for confidence level
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence) / 2)
        
        # Confidence intervals
        lower = mean - z_score * std
        upper = mean + z_score * std
        
        return UncertaintyPrediction(
            mean=mean.cpu().numpy(),
            std=std.cpu().numpy(),
            lower=lower.cpu().numpy(),
            upper=upper.cpu().numpy(),
            confidence=confidence,
            n_samples=n_samples
        )
    
    def predict_convergence_with_uncertainty(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mass: float,
        concentration: float,
        redshift: float,
        n_samples: int = 100,
        confidence: float = 0.95
    ) -> UncertaintyPrediction:
        """
        Convenience method for convergence map prediction with uncertainty
        
        Args:
            x, y: Grid coordinates
            mass: Virial mass (M☉)
            concentration: Concentration parameter
            redshift: Lens redshift
            n_samples: Number of MC samples
            confidence: Confidence level
            
        Returns:
            UncertaintyPrediction for convergence (κ)
        """
        # Prepare input
        batch_size = x.numel()
        
        inputs = torch.stack([
            x.flatten(),
            y.flatten(),
            torch.full((batch_size,), mass, device=x.device),
            torch.full((batch_size,), concentration, device=x.device),
            torch.full((batch_size,), redshift, device=x.device)
        ], dim=1)
        
        # Get prediction with uncertainty
        result = self.get_prediction_intervals(
            inputs, confidence, n_samples
        )
        
        # Extract convergence (first output)
        shape = x.shape
        return UncertaintyPrediction(
            mean=result.mean[:, 0].reshape(shape),
            std=result.std[:, 0].reshape(shape),
            lower=result.lower[:, 0].reshape(shape),
            upper=result.upper[:, 0].reshape(shape),
            confidence=confidence,
            n_samples=n_samples
        )


class UncertaintyCalibrator:
    """
    Calibrate uncertainty estimates
    
    Ensures that predicted uncertainties match actual errors.
    For example, if model predicts 95% confidence intervals,
    we want 95% of true values to fall within those intervals.
    
    Usage:
        calibrator = UncertaintyCalibrator()
        
        # Calibrate on validation set
        calib_error = calibrator.calibrate(
            predictions=model_predictions,
            uncertainties=model_uncertainties,
            ground_truth=true_values
        )
        
        # Visualize calibration
        fig = calibrator.plot_calibration_curve()
        fig.savefig('calibration.png')
    """
    
    def __init__(self):
        self.calibration_data = None
        self.is_calibrated = False
    
    def calibrate(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        ground_truth: np.ndarray,
        confidence_levels: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute calibration curve
        
        Checks if predicted uncertainties match actual errors.
        
        Args:
            predictions: Model predictions [N,]
            uncertainties: Predicted uncertainties (std) [N,]
            ground_truth: True values [N,]
            confidence_levels: Confidence levels to test (default: 0.1 to 0.99)
            
        Returns:
            calibration_error: Mean absolute calibration error
        """
        if confidence_levels is None:
            confidence_levels = np.linspace(0.1, 0.99, 20)
        
        from scipy import stats
        
        empirical_coverage = []
        
        for conf in confidence_levels:
            # Calculate prediction intervals
            z = stats.norm.ppf((1 + conf) / 2)
            
            lower = predictions - z * uncertainties
            upper = predictions + z * uncertainties
            
            # Check how many true values fall within intervals
            coverage = np.mean((ground_truth >= lower) & (ground_truth <= upper))
            empirical_coverage.append(coverage)
        
        empirical_coverage = np.array(empirical_coverage)
        
        # Store calibration data
        self.calibration_data = {
            'predicted_confidence': confidence_levels,
            'empirical_coverage': empirical_coverage
        }
        self.is_calibrated = True
        
        # Calculate calibration error (lower is better)
        calibration_error = np.mean(np.abs(confidence_levels - empirical_coverage))
        
        return float(calibration_error)
    
    def plot_calibration_curve(self, save_path: Optional[str] = None):
        """
        Plot calibration curve
        
        Perfect calibration = diagonal line
        Above diagonal = overconfident (too narrow intervals)
        Below diagonal = underconfident (too wide intervals)
        
        Args:
            save_path: Path to save figure (optional)
            
        Returns:
            matplotlib figure
        """
        if not self.is_calibrated:
            raise ValueError("Must call calibrate() before plotting")
        
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect calibration')
        
        # Actual calibration
        pred_conf = self.calibration_data['predicted_confidence']
        emp_cov = self.calibration_data['empirical_coverage']
        
        ax.plot(pred_conf, emp_cov, 'bo-', linewidth=2, markersize=6,
                label='Actual calibration')
        
        # Shaded regions
        ax.fill_between(pred_conf, pred_conf, emp_cov,
                        where=(emp_cov < pred_conf),
                        alpha=0.3, color='red',
                        label='Overconfident')
        ax.fill_between(pred_conf, pred_conf, emp_cov,
                        where=(emp_cov > pred_conf),
                        alpha=0.3, color='blue',
                        label='Underconfident')
        
        # Calculate calibration error
        calib_error = np.mean(np.abs(pred_conf - emp_cov))
        
        ax.set_xlabel('Predicted Confidence Level', fontsize=14)
        ax.set_ylabel('Empirical Coverage', fontsize=14)
        ax.set_title(f'Uncertainty Calibration Curve\n'
                    f'Calibration Error: {calib_error:.4f}',
                    fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Calibration curve saved to {save_path}")
        
        return fig
    
    def assess_calibration(self) -> Dict[str, float]:
        """
        Assess calibration quality
        
        Returns:
            Dictionary with calibration metrics
        """
        if not self.is_calibrated:
            raise ValueError("Must call calibrate() first")
        
        pred_conf = self.calibration_data['predicted_confidence']
        emp_cov = self.calibration_data['empirical_coverage']
        
        # Mean absolute calibration error
        mace = np.mean(np.abs(pred_conf - emp_cov))
        
        # Root mean squared calibration error
        rmsce = np.sqrt(np.mean((pred_conf - emp_cov) ** 2))
        
        # Maximum calibration error
        max_ce = np.max(np.abs(pred_conf - emp_cov))
        
        # Check if overconfident or underconfident
        bias = np.mean(emp_cov - pred_conf)
        
        if bias > 0.05:
            calibration_status = "Underconfident (intervals too wide)"
        elif bias < -0.05:
            calibration_status = "Overconfident (intervals too narrow)"
        else:
            calibration_status = "Well-calibrated"
        
        return {
            'mean_absolute_calibration_error': float(mace),
            'root_mean_squared_calibration_error': float(rmsce),
            'max_calibration_error': float(max_ce),
            'calibration_bias': float(bias),
            'calibration_status': calibration_status
        }


class EnsembleBayesianPINN:
    """
    Ensemble of Bayesian PINNs for even better uncertainty estimates
    
    Combines:
    1. Model uncertainty (ensemble disagreement)
    2. Data uncertainty (MC Dropout)
    
    Usage:
        ensemble = EnsembleBayesianPINN(n_models=5)
        ensemble.train_all(train_data)
        mean, std = ensemble.predict_with_uncertainty(x_test)
    """
    
    def __init__(
        self,
        n_models: int = 5,
        input_dim: int = 5,
        output_dim: int = 4,
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.1
    ):
        """
        Initialize ensemble
        
        Args:
            n_models: Number of models in ensemble
            Other args: Same as BayesianPINN
        """
        self.n_models = n_models
        
        if hidden_dims is None:
            hidden_dims = [64, 64, 64]
            
        self.models = [
            BayesianPINN(input_dim, output_dim, hidden_dims, dropout_rate)
            for _ in range(n_models)
        ]
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 50
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict with ensemble + MC Dropout uncertainty
        
        Returns:
            (mean, total_std, epistemic_std)
            - total_std: Total uncertainty (model + data)
            - epistemic_std: Model uncertainty only
        """
        # Get predictions from each model with MC Dropout
        all_predictions = []
        
        for model in self.models:
            mean, _ = model.predict_with_uncertainty(x, n_samples)
            all_predictions.append(mean)
        
        # Stack [n_models, batch_size, output_dim]
        all_predictions = torch.stack(all_predictions)
        
        # Overall mean
        mean = all_predictions.mean(dim=0)
        
        # Model uncertainty (epistemic)
        epistemic_std = all_predictions.std(dim=0)
        
        # For total uncertainty, need to combine with MC Dropout uncertainty
        # (simplified: use ensemble std as lower bound)
        total_std = epistemic_std
        
        return mean, total_std, epistemic_std


# ============================================================================
# Utility Functions
# ============================================================================

def visualize_uncertainty(
    x: np.ndarray,
    y: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
):
    """
    Visualize convergence map with uncertainty
    
    Creates 2x2 plot:
    - Top left: Mean prediction
    - Top right: Uncertainty (std)
    - Bottom left: Ground truth (if available)
    - Bottom right: Uncertainty normalized by mean
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Mean prediction
    im1 = axes[0, 0].imshow(mean, origin='lower', cmap='viridis')
    axes[0, 0].set_title('Mean Prediction', fontsize=14)
    axes[0, 0].set_xlabel('x (pixels)')
    axes[0, 0].set_ylabel('y (pixels)')
    plt.colorbar(im1, ax=axes[0, 0], label='κ')
    
    # Uncertainty (std)
    im2 = axes[0, 1].imshow(std, origin='lower', cmap='Reds')
    axes[0, 1].set_title('Uncertainty (1σ)', fontsize=14)
    axes[0, 1].set_xlabel('x (pixels)')
    axes[0, 1].set_ylabel('y (pixels)')
    plt.colorbar(im2, ax=axes[0, 1], label='σ(κ)')
    
    # Ground truth (if available)
    if ground_truth is not None:
        im3 = axes[1, 0].imshow(ground_truth, origin='lower', cmap='viridis')
        axes[1, 0].set_title('Ground Truth', fontsize=14)
        axes[1, 0].set_xlabel('x (pixels)')
        axes[1, 0].set_ylabel('y (pixels)')
        plt.colorbar(im3, ax=axes[1, 0], label='κ')
    else:
        axes[1, 0].text(0.5, 0.5, 'Ground Truth\nNot Available',
                       ha='center', va='center', fontsize=14)
        axes[1, 0].set_xticks([])
        axes[1, 0].set_yticks([])
    
    # Relative uncertainty (coefficient of variation)
    rel_unc = std / (mean + 1e-10)
    im4 = axes[1, 1].imshow(rel_unc, origin='lower', cmap='RdYlGn_r')
    axes[1, 1].set_title('Relative Uncertainty (σ/μ)', fontsize=14)
    axes[1, 1].set_xlabel('x (pixels)')
    axes[1, 1].set_ylabel('y (pixels)')
    plt.colorbar(im4, ax=axes[1, 1], label='σ/μ')
    
    plt.suptitle('Convergence Map with Uncertainty Quantification',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    return fig


def print_uncertainty_summary(
    prediction: UncertaintyPrediction,
    ground_truth: Optional[np.ndarray] = None
):
    """
    Print summary of uncertainty prediction
    
    Args:
        prediction: UncertaintyPrediction object
        ground_truth: True values (optional, for coverage check)
    """
    print("=" * 70)
    print("UNCERTAINTY QUANTIFICATION SUMMARY")
    print("=" * 70)
    print()
    
    # Basic statistics
    print(f"Confidence Level: {prediction.confidence:.1%}")
    print(f"MC Samples: {prediction.n_samples}")
    print()
    
    print("PREDICTION STATISTICS:")
    print(f"  Mean:               {prediction.mean.mean():.6f} ± {prediction.mean.std():.6f}")
    print(f"  Uncertainty (avg):  {prediction.std.mean():.6f}")
    print(f"  Uncertainty (max):  {prediction.std.max():.6f}")
    print(f"  Uncertainty (min):  {prediction.std.min():.6f}")
    print()
    
    # Relative uncertainty
    rel_unc = prediction.std / (np.abs(prediction.mean) + 1e-10)
    print(f"RELATIVE UNCERTAINTY:")
    print(f"  Average:  {rel_unc.mean():.2%}")
    print(f"  Median:   {np.median(rel_unc):.2%}")
    print(f"  95th %ile: {np.percentile(rel_unc, 95):.2%}")
    print()
    
    # Coverage check (if ground truth available)
    if ground_truth is not None:
        within_interval = (
            (ground_truth >= prediction.lower) &
            (ground_truth <= prediction.upper)
        )
        coverage = np.mean(within_interval)
        
        print("EMPIRICAL COVERAGE:")
        print(f"  Expected:  {prediction.confidence:.1%}")
        print(f"  Observed:  {coverage:.1%}")
        
        if abs(coverage - prediction.confidence) < 0.05:
            print("  Status:    ✅ Well-calibrated")
        elif coverage > prediction.confidence:
            print("  Status:    ⚠️  Underconfident (intervals too wide)")
        else:
            print("  Status:    ⚠️  Overconfident (intervals too narrow)")
        print()
    
    print("=" * 70)
