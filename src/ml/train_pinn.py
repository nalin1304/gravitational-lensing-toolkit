"""
PINN Training Script with Benchmark Integration

Train physics-informed neural networks for gravitational lensing
and benchmark them against analytic solutions.

Usage:
    python src/ml/train_pinn.py --model nfw --epochs 5000 --benchmark

Author: Phase 14 Implementation
Date: October 2025
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import logging

# Import PINN models
from src.ml.pinn_models import create_lensing_pinn, PINNTrainer, PhysicsLoss

# Import benchmarks
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from benchmarks import (
    calculate_all_metrics,
    print_metrics_report,
    time_profile,
    compare_with_analytic
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Generation
# ============================================================================

def generate_nfw_training_data(
    n_samples: int = 1000,
    grid_size: int = 64,
    mass_range: Tuple[float, float] = (1e11, 1e13),
    conc_range: Tuple[float, float] = (3.0, 10.0),
    device: str = 'cpu'
) -> Dict[str, torch.Tensor]:
    """
    Generate synthetic NFW training data
    
    Args:
        n_samples: Number of training samples
        grid_size: Grid resolution
        mass_range: Range of halo masses (solar masses)
        conc_range: Range of concentration parameters
        device: Device to place tensors
    
    Returns:
        Dictionary with training data
    """
    logger.info(f"Generating {n_samples} training samples...")
    
    # Generate random lens parameters
    masses = torch.rand(n_samples) * (mass_range[1] - mass_range[0]) + mass_range[0]
    concentrations = torch.rand(n_samples) * (conc_range[1] - conc_range[0]) + conc_range[0]
    
    # Generate coordinate grid
    x_1d = torch.linspace(-2, 2, grid_size)
    y_1d = torch.linspace(-2, 2, grid_size)
    X, Y = torch.meshgrid(x_1d, y_1d, indexing='ij')
    
    # Calculate radius
    R = torch.sqrt(X**2 + Y**2)
    
    # Generate convergence maps for each lens
    all_x = []
    all_y = []
    all_r = []
    all_kappa = []
    all_mass = []
    all_conc = []
    
    for i in range(n_samples):
        mass = masses[i].item()
        conc = concentrations[i].item()
        
        # NFW scale radius (simplified)
        r_vir = (mass / 1e12) ** (1/3) * 200.0  # kpc
        r_s = r_vir / conc
        
        # NFW convergence (simplified analytic formula)
        x = R / r_s
        kappa_s = mass / (1e12 * r_s**2)  # Simplified normalization
        
        # NFW profile: κ(x) = κ_s / [x(1+x)²]
        kappa = kappa_s / (x * (1 + x)**2)
        kappa = torch.clamp(kappa, 0, 10)  # Clamp extreme values
        
        # Flatten and store
        x_flat = X.flatten()
        y_flat = Y.flatten()
        r_flat = R.flatten()
        kappa_flat = kappa.flatten()
        
        all_x.append(x_flat)
        all_y.append(y_flat)
        all_r.append(r_flat)
        all_kappa.append(kappa_flat)
        all_mass.append(torch.full_like(x_flat, np.log10(mass)))
        all_conc.append(torch.full_like(x_flat, conc))
    
    # Stack all data
    data = {
        'x': torch.cat(all_x).to(device),
        'y': torch.cat(all_y).to(device),
        'r': torch.cat(all_r).to(device),
        'kappa': torch.cat(all_kappa).to(device),
        'log_mass': torch.cat(all_mass).to(device),
        'concentration': torch.cat(all_conc).to(device),
        'n_samples': n_samples,
        'grid_size': grid_size
    }
    
    logger.info(f"Generated {data['x'].shape[0]} total data points")
    logger.info(f"Convergence range: [{data['kappa'].min():.6f}, {data['kappa'].max():.6f}]")
    
    return data


# ============================================================================
# Training Loop
# ============================================================================

def train_pinn(
    model_type: str = 'nfw',
    n_epochs: int = 5000,
    learning_rate: float = 1e-3,
    batch_size: int = 1024,
    n_samples: int = 100,
    device: str = 'cpu',
    save_path: Optional[Path] = None
) -> Tuple[nn.Module, Dict]:
    """
    Train a PINN model
    
    Args:
        model_type: Type of model ('general', 'nfw')
        n_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training
        n_samples: Number of training samples
        device: Device to train on
        save_path: Path to save trained model
    
    Returns:
        Trained model and training history
    """
    # Create model
    logger.info(f"Creating {model_type} PINN...")
    model = create_lensing_pinn(model_type=model_type)
    model = model.to(device)
    
    # Generate training data
    train_data = generate_nfw_training_data(
        n_samples=n_samples,
        device=device
    )
    
    # Create trainer
    trainer = PINNTrainer(
        model=model,
        learning_rate=learning_rate,
        device=device
    )
    
    # Training loop
    logger.info(f"Starting training for {n_epochs} epochs...")
    
    n_batches = len(train_data['x']) // batch_size
    
    for epoch in range(n_epochs):
        # Shuffle data
        perm = torch.randperm(len(train_data['x']))
        
        epoch_losses = []
        
        for batch_idx in range(n_batches):
            # Get batch
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(train_data['x']))
            
            batch_indices = perm[start_idx:end_idx]
            
            batch_x = train_data['x'][batch_indices]
            batch_y = train_data['y'][batch_indices]
            batch_r = train_data['r'][batch_indices]
            batch_kappa = train_data['kappa'][batch_indices]
            batch_log_mass = train_data['log_mass'][batch_indices]
            batch_conc = train_data['concentration'][batch_indices]
            
            # Prepare input for NFW model
            batch_input = torch.stack([
                batch_r,
                batch_log_mass,
                batch_conc
            ], dim=1)
            
            # Forward pass
            model.train()
            trainer.optimizer.zero_grad()
            
            outputs = model(batch_input)
            pred_kappa = outputs[:, 0]
            
            # Simple MSE loss for now
            loss = torch.mean((pred_kappa - batch_kappa) ** 2)
            
            # Backward pass
            loss.backward()
            trainer.optimizer.step()
            
            epoch_losses.append(loss.item())
        
        # Record epoch loss
        epoch_loss = np.mean(epoch_losses)
        trainer.history['loss'].append(epoch_loss)
        
        # Print progress
        if (epoch + 1) % 500 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1}/{n_epochs} | Loss: {epoch_loss:.6f}")
    
    # Save model
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_type': model_type,
            'history': trainer.history
        }, save_path)
        logger.info(f"Model saved to {save_path}")
    
    return model, trainer.history


# ============================================================================
# Benchmarking
# ============================================================================

@time_profile
def benchmark_trained_pinn(
    model: nn.Module,
    device: str = 'cpu',
    grid_size: int = 64
) -> Dict:
    """
    Benchmark a trained PINN against analytic solution
    
    Args:
        model: Trained PINN model
        device: Device for inference
        grid_size: Grid size for evaluation
    
    Returns:
        Benchmark results dictionary
    """
    logger.info("Benchmarking trained PINN...")
    
    model.eval()
    
    # Generate test case
    test_mass = 1e12  # Solar masses
    test_conc = 5.0
    
    # Create coordinate grid
    x_1d = torch.linspace(-2, 2, grid_size)
    y_1d = torch.linspace(-2, 2, grid_size)
    X, Y = torch.meshgrid(x_1d, y_1d, indexing='ij')
    R = torch.sqrt(X**2 + Y**2)
    
    # PINN prediction
    with torch.no_grad():
        r_flat = R.flatten().to(device)
        log_mass_flat = torch.full_like(r_flat, np.log10(test_mass))
        conc_flat = torch.full_like(r_flat, test_conc)
        
        inputs = torch.stack([r_flat, log_mass_flat, conc_flat], dim=1)
        outputs = model(inputs)
        pred_kappa = outputs[:, 0].cpu().numpy().reshape(grid_size, grid_size)
    
    # Analytic solution
    r_vir = (test_mass / 1e12) ** (1/3) * 200.0
    r_s = r_vir / test_conc
    x = R / r_s
    kappa_s = test_mass / (1e12 * r_s**2)
    true_kappa = (kappa_s / (x * (1 + x)**2)).numpy()
    true_kappa = np.clip(true_kappa, 0, 10)
    
    # Calculate metrics
    metrics = calculate_all_metrics(pred_kappa, true_kappa)
    
    # Print report
    print("\n" + "="*80)
    print("PINN BENCHMARK RESULTS")
    print("="*80)
    print_metrics_report(metrics)
    
    return {
        'pred_kappa': pred_kappa,
        'true_kappa': true_kappa,
        'metrics': metrics,
        'test_mass': test_mass,
        'test_conc': test_conc
    }


# ============================================================================
# Visualization
# ============================================================================

def plot_training_history(history: Dict, save_path: Optional[Path] = None):
    """Plot training loss history"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(history['loss'], linewidth=2, color='steelblue')
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('PINN Training Loss', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training plot saved to {save_path}")
    
    plt.close()


def plot_benchmark_comparison(
    results: Dict,
    save_path: Optional[Path] = None
):
    """Plot PINN vs analytic comparison"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # PINN prediction
    im1 = axes[0].imshow(results['pred_kappa'], cmap='viridis', origin='lower')
    axes[0].set_title('PINN Prediction')
    axes[0].set_xlabel('x [pixels]')
    axes[0].set_ylabel('y [pixels]')
    plt.colorbar(im1, ax=axes[0], label='κ')
    
    # Analytic solution
    im2 = axes[1].imshow(results['true_kappa'], cmap='viridis', origin='lower')
    axes[1].set_title('Analytic Solution')
    axes[1].set_xlabel('x [pixels]')
    axes[1].set_ylabel('y [pixels]')
    plt.colorbar(im2, ax=axes[1], label='κ')
    
    # Residual
    residual = results['pred_kappa'] - results['true_kappa']
    im3 = axes[2].imshow(residual, cmap='RdBu_r', origin='lower')
    axes[2].set_title('Residual (PINN - Analytic)')
    axes[2].set_xlabel('x [pixels]')
    axes[2].set_ylabel('y [pixels]')
    plt.colorbar(im3, ax=axes[2], label='Δκ')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comparison plot saved to {save_path}")
    
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train and benchmark PINN for gravitational lensing')
    
    parser.add_argument('--model', type=str, default='nfw',
                       choices=['general', 'nfw'],
                       help='Type of PINN model')
    parser.add_argument('--epochs', type=int, default=5000,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=1024,
                       help='Batch size')
    parser.add_argument('--n-samples', type=int, default=100,
                       help='Number of training samples')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to train on')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmark after training')
    parser.add_argument('--output-dir', type=str, default='results/pinn',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train model
    model, history = train_pinn(
        model_type=args.model,
        n_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
        device=args.device,
        save_path=output_dir / f'{args.model}_pinn.pth'
    )
    
    # Plot training history
    plot_training_history(
        history,
        save_path=output_dir / 'training_loss.png'
    )
    
    # Benchmark if requested
    if args.benchmark:
        benchmark_results = benchmark_trained_pinn(
            model,
            device=args.device
        )
        
        # Plot comparison
        plot_benchmark_comparison(
            benchmark_results,
            save_path=output_dir / 'benchmark_comparison.png'
        )
        
        # Save benchmark results (convert numpy types to Python types)
        def convert_to_python_types(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_python_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_python_types(item) for item in obj]
            else:
                return obj
        
        results_dict = {
            'metrics': convert_to_python_types(benchmark_results['metrics']),
            'test_mass': float(benchmark_results['test_mass']),
            'test_conc': float(benchmark_results['test_conc'])
        }
        
        with open(output_dir / 'benchmark_results.json', 'w') as f:
            json.dump(results_dict, f, indent=2)
    
    logger.info(f"\n✅ All results saved to {output_dir}")


if __name__ == '__main__':
    main()
