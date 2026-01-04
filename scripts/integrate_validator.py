"""
Integration Example: Using Scientific Validator with Trained PINN

This script demonstrates how to validate a trained PINN model from Phase 14
using the new scientific validation framework from Phase 15.
"""

import numpy as np
import torch
import sys
from pathlib import Path


# Removed sys.path hack


# Try to import PINN models
try:
    from src.ml.pinn_models import create_lensing_pinn
    PINN_AVAILABLE = True
except ImportError:
    PINN_AVAILABLE = False

from src.validation import rigorous_validate, ScientificValidator, ValidationLevel


def load_trained_model(model_path: str = "results/pinn_demo/nfw_pinn_final.pt"):
    """Load a trained PINN model"""
    if not PINN_AVAILABLE:
        print("⚠️  PINN models not available (this is OK for demonstration)")
        return None
    
    print(f"📦 Loading trained PINN model from {model_path}...")
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"⚠️  Model not found at {model_path}")
        return None
    
    # Load model
    model = create_lensing_pinn(profile_type='nfw')
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    print("✅ Model loaded successfully")
    return model


def generate_test_case(grid_size: int = 128):
    """Generate NFW test case with known solution"""
    print(f"\n📊 Generating NFW test case ({grid_size}x{grid_size} grid)...")
    
    # Parameters
    M_vir = 1e14  # Solar masses
    c = 5.0       # Concentration
    z_l = 0.5     # Lens redshift
    
    # Create grid
    x = torch.linspace(-5, 5, grid_size)
    y = torch.linspace(-5, 5, grid_size)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Model inputs (x, y, M, c, z)
    inputs = torch.stack([
        X.flatten(),
        Y.flatten(),
        torch.full((grid_size**2,), M_vir),
        torch.full((grid_size**2,), c),
        torch.full((grid_size**2,), z_l)
    ], dim=1)
    
    # Analytic NFW solution
    R = torch.sqrt(X**2 + Y**2)
    kappa_analytic = 1.0 / (1 + R/c)**2
    
    print(f"   Parameters: M={M_vir:.2e} M☉, c={c}, z={z_l}")
    print(f"   Grid: {grid_size}x{grid_size} = {grid_size**2} pixels")
    
    return inputs, kappa_analytic.numpy()


def validate_pinn_model(model, inputs, ground_truth):
    """Validate PINN model predictions"""
    print("\n🔬 Running PINN prediction...")
    
    # Predict with PINN
    with torch.no_grad():
        outputs = model(inputs)
        kappa_pred = outputs[:, 0].reshape(ground_truth.shape).numpy()
    
    print(f"   Prediction shape: {kappa_pred.shape}")
    print(f"   Prediction range: [{kappa_pred.min():.4f}, {kappa_pred.max():.4f}]")
    print(f"   Analytic range: [{ground_truth.min():.4f}, {ground_truth.max():.4f}]")
    
    # Run validation
    print("\n" + "="*70)
    print("SCIENTIFIC VALIDATION")
    print("="*70)
    
    result = rigorous_validate(
        predicted=kappa_pred,
        ground_truth=ground_truth,
        profile_type="NFW"
    )
    
    return result


def print_validation_summary(result):
    """Print concise summary of validation results"""
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    # Status
    if result.passed:
        print("✅ STATUS: PASSED")
    else:
        print("❌ STATUS: FAILED")
    
    print(f"   Confidence: {result.confidence_level:.1%}")
    print()
    
    # Key metrics
    print("📊 KEY METRICS:")
    print(f"   RMSE:               {result.metrics['rmse']:.6f}")
    print(f"   MAE:                {result.metrics['mae']:.6f}")
    print(f"   SSIM:               {result.metrics['ssim']:.4f}")
    print(f"   PSNR:               {result.metrics['psnr']:.2f} dB")
    print(f"   Mass conservation:  {result.metrics['mass_conservation_ratio']:.4f}")
    print()
    
    # NFW profile analysis
    if 'nfw_overall_fit_quality' in result.metrics:
        print("📐 NFW PROFILE:")
        print(f"   Inner slope:        {result.metrics['nfw_inner_slope_pred']:.3f} (expect: -1.0)")
        print(f"   Outer slope:        {result.metrics['nfw_outer_slope_pred']:.3f} (expect: -2.0)")
        print(f"   Fit quality:        {result.metrics['nfw_overall_fit_quality']:.1%}")
        print()
    
    # Warnings
    if result.warnings:
        print(f"⚠️  WARNINGS ({len(result.warnings)}):")
        for warning in result.warnings:
            print(f"   • {warning}")
        print()
    
    # Recommendations
    if result.recommendations:
        print(f"💡 RECOMMENDATIONS ({len(result.recommendations)}):")
        for rec in result.recommendations:
            print(f"   • {rec}")
        print()
    
    print("="*70)


def compare_validation_levels():
    """Compare different validation levels"""
    print("\n" + "="*70)
    print("COMPARISON: VALIDATION LEVELS")
    print("="*70)
    
    # Generate small test case
    inputs, ground_truth = generate_test_case(grid_size=64)
    
    # For demo, use synthetic prediction
    predicted = ground_truth + np.random.normal(0, 0.01, ground_truth.shape)
    predicted = np.maximum(predicted, 0)
    
    import time
    
    print(f"\n{'Level':<15} {'Time':<10} {'Metrics':<10} {'Confidence':<12} {'Status'}")
    print("-" * 65)
    
    for level in [ValidationLevel.QUICK, ValidationLevel.STANDARD, ValidationLevel.RIGOROUS]:
        validator = ScientificValidator(level=level)
        
        start = time.time()
        result = validator.validate_convergence_map(
            predicted, ground_truth, "NFW", verbose=False
        )
        elapsed = time.time() - start
        
        status = "✅ PASS" if result.passed else "❌ FAIL"
        conf = f"{result.confidence_level:.1%}"
        
        print(f"{level.value:<15} {elapsed:.3f}s    {len(result.metrics):<10} {conf:<12} {status}")


def main():
    """Main integration example"""
    print("\n" + "="*70)
    print("PHASE 15: SCIENTIFIC VALIDATION INTEGRATION EXAMPLE")
    print("="*70)
    print("Demonstrating validation of trained PINN models from Phase 14")
    print()
    
    # Try to load trained model
    model_path = "results/pinn_demo/nfw_pinn_final.pt"
    model = load_trained_model(model_path)
    
    if model is not None:
        # Validate real trained model
        inputs, ground_truth = generate_test_case(grid_size=128)
        result = validate_pinn_model(model, inputs, ground_truth)
        
        # Print full scientific notes
        print("\n" + result.scientific_notes)
        
        # Print summary
        print_validation_summary(result)
        
    else:
        print("\n⚠️  No trained model found, using synthetic data for demonstration...")
        
        # Use synthetic data
        _, ground_truth = generate_test_case(grid_size=128)
        predicted = ground_truth + np.random.normal(0, 0.005, ground_truth.shape)
        predicted = np.maximum(predicted, 0)
        
        print("\n🔬 Running validation on synthetic data...")
        
        validator = ScientificValidator(level=ValidationLevel.RIGOROUS)
        result = validator.validate_convergence_map(
            predicted=predicted,
            ground_truth=ground_truth,
            profile_type="NFW",
            verbose=True
        )
        
        # Print results
        print("\n" + result.scientific_notes)
        print_validation_summary(result)
    
    # Compare validation levels
    compare_validation_levels()
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Train PINN model: python src/ml/train_pinn.py --model nfw")
    print("2. Validate results: python scripts/integrate_validator.py")
    print("3. Use in Streamlit: streamlit run app/main.py")
    print()
    print("📚 Documentation: docs/Phase15_Part1_Complete.md")
    print("🧪 Test suite: python scripts/test_validator.py")
    print("="*70)


if __name__ == "__main__":
    main()
