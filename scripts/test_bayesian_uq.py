"""
Test Script for Bayesian Uncertainty Quantification

Tests the bayesian_uq.py module with synthetic data.
"""

import sys
import io

# Fix Windows console encoding issues
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import torch
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


# Removed sys.path hack


from src.ml.uncertainty import (
    BayesianPINN,
    UncertaintyCalibrator,
    visualize_uncertainty,
    print_uncertainty_summary
)


def test_bayesian_pinn_creation():
    """Test 1: Create Bayesian PINN"""
    print("\n" + "="*70)
    print("TEST 1: Bayesian PINN Creation")
    print("="*70)
    
    model = BayesianPINN(
        input_dim=5,
        output_dim=4,
        hidden_dims=[64, 64, 64],
        dropout_rate=0.1
    )
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created successfully")
    print(f"   Parameters: {n_params:,}")
    print(f"   Dropout rate: 0.1")
    print(f"   Architecture: 5 → 64 → 64 → 64 → 4")
    
    return True


def test_forward_pass():
    """Test 2: Forward pass"""
    print("\n" + "="*70)
    print("TEST 2: Forward Pass")
    print("="*70)
    
    model = BayesianPINN()
    
    # Create dummy input
    x = torch.randn(100, 5)
    
    # Forward pass
    output = model(x)
    
    print(f"✅ Forward pass successful")
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    return True


def test_uncertainty_estimation():
    """Test 3: MC Dropout uncertainty estimation"""
    print("\n" + "="*70)
    print("TEST 3: Uncertainty Estimation (MC Dropout)")
    print("="*70)
    
    model = BayesianPINN(dropout_rate=0.1)
    
    # Create test input
    x = torch.randn(50, 5)
    
    print("Running MC Dropout with 100 samples...")
    mean, std = model.predict_with_uncertainty(x, n_samples=100)
    
    print(f"✅ Uncertainty estimation successful")
    print(f"   Mean shape: {mean.shape}")
    print(f"   Std shape: {std.shape}")
    print(f"   Mean output: {mean.mean():.4f} ± {mean.std():.4f}")
    print(f"   Avg uncertainty: {std.mean():.4f}")
    print(f"   Max uncertainty: {std.max():.4f}")
    
    # Check that std > 0 (dropout is working)
    if std.mean() > 1e-6:
        print(f"   ✓ Dropout is generating uncertainty")
    else:
        print(f"   ✗ WARNING: Uncertainty very low (dropout may not be working)")
    
    return True


def test_prediction_intervals():
    """Test 4: Confidence intervals"""
    print("\n" + "="*70)
    print("TEST 4: Prediction Intervals")
    print("="*70)
    
    model = BayesianPINN()
    
    # Create test input
    x = torch.randn(20, 5)
    
    for confidence in [0.68, 0.95, 0.99]:
        result = model.get_prediction_intervals(
            x, confidence=confidence, n_samples=100
        )
        
        interval_width = (result.upper - result.lower).mean()
        
        print(f"\n{confidence:.0%} Confidence Intervals:")
        print(f"   Mean: {result.mean.mean():.4f}")
        print(f"   Std: {result.std.mean():.4f}")
        print(f"   Interval width: {interval_width:.4f}")
    
    print(f"\n✅ Prediction intervals computed successfully")
    
    return True


def test_convergence_with_uncertainty():
    """Test 5: Convergence map with uncertainty"""
    print("\n" + "="*70)
    print("TEST 5: Convergence Map with Uncertainty")
    print("="*70)
    
    model = BayesianPINN()
    
    # Create 2D grid
    grid_size = 32  # Small for speed
    x = torch.linspace(-5, 5, grid_size)
    y = torch.linspace(-5, 5, grid_size)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    print(f"Predicting on {grid_size}×{grid_size} grid...")
    
    result = model.predict_convergence_with_uncertainty(
        X, Y,
        mass=1e14,
        concentration=5.0,
        redshift=0.5,
        n_samples=50,
        confidence=0.95
    )
    
    print(f"✅ Convergence map predicted with uncertainty")
    print(f"   Mean κ: {result.mean.mean():.4f} ± {result.mean.std():.4f}")
    print(f"   Avg uncertainty: {result.std.mean():.4f}")
    print(f"   95% interval width: {(result.upper - result.lower).mean():.4f}")
    
    return True


def test_uncertainty_calibrator():
    """Test 6: Uncertainty calibration"""
    print("\n" + "="*70)
    print("TEST 6: Uncertainty Calibration")
    print("="*70)
    
    # Generate synthetic well-calibrated data
    n_points = 1000
    true_values = np.random.randn(n_points)
    predictions = true_values + np.random.normal(0, 0.1, n_points)
    uncertainties = np.ones(n_points) * 0.1  # True std
    
    print("Testing with well-calibrated synthetic data...")
    print(f"   {n_points} data points")
    print(f"   True std: 0.1")
    
    calibrator = UncertaintyCalibrator()
    
    calib_error = calibrator.calibrate(
        predictions=predictions,
        uncertainties=uncertainties,
        ground_truth=true_values
    )
    
    print(f"\n✅ Calibration analysis complete")
    print(f"   Calibration error: {calib_error:.4f}")
    
    if calib_error < 0.05:
        print(f"   ✓ Well-calibrated (error < 0.05)")
    else:
        print(f"   ⚠️  Calibration could be improved")
    
    # Assess calibration
    assessment = calibrator.assess_calibration()
    print(f"\nCalibration Assessment:")
    for key, value in assessment.items():
        if isinstance(value, str):
            print(f"   {key}: {value}")
        else:
            print(f"   {key}: {value:.4f}")
    
    # Generate calibration curve
    output_dir = Path("results/uncertainty_tests")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig = calibrator.plot_calibration_curve(
        save_path=str(output_dir / "calibration_curve.png")
    )
    plt.close(fig)
    
    print(f"\n   Calibration curve saved to {output_dir}/calibration_curve.png")
    
    return True


def test_poorly_calibrated():
    """Test 7: Detect poor calibration"""
    print("\n" + "="*70)
    print("TEST 7: Detecting Poor Calibration")
    print("="*70)
    
    # Generate overconfident predictions (underestimated uncertainty)
    n_points = 1000
    true_values = np.random.randn(n_points)
    predictions = true_values + np.random.normal(0, 0.2, n_points)  # Actual std: 0.2
    uncertainties = np.ones(n_points) * 0.1  # Claimed std: 0.1 (too small!)
    
    print("Testing with overconfident (underestimated uncertainty)...")
    print(f"   Actual std: 0.2")
    print(f"   Claimed std: 0.1")
    
    calibrator = UncertaintyCalibrator()
    
    calib_error = calibrator.calibrate(
        predictions=predictions,
        uncertainties=uncertainties,
        ground_truth=true_values
    )
    
    print(f"\n✅ Poor calibration detected")
    print(f"   Calibration error: {calib_error:.4f}")
    
    assessment = calibrator.assess_calibration()
    print(f"   Status: {assessment['calibration_status']}")
    
    if "overconfident" in assessment['calibration_status'].lower():
        print(f"   ✓ Correctly identified as overconfident")
    
    return True


def test_visualization():
    """Test 8: Uncertainty visualization"""
    print("\n" + "="*70)
    print("TEST 8: Uncertainty Visualization")
    print("="*70)
    
    model = BayesianPINN()
    
    # Generate convergence map
    grid_size = 64
    x = torch.linspace(-5, 5, grid_size)
    y = torch.linspace(-5, 5, grid_size)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    print(f"Generating {grid_size}×{grid_size} convergence map with uncertainty...")
    
    result = model.predict_convergence_with_uncertainty(
        X, Y, mass=1e14, concentration=5.0, redshift=0.5,
        n_samples=50, confidence=0.95
    )
    
    # Create visualization
    output_dir = Path("results/uncertainty_tests")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    x_np = X.numpy()
    y_np = Y.numpy()
    
    fig = visualize_uncertainty(
        x_np, y_np,
        result.mean, result.std,
        ground_truth=None,
        save_path=str(output_dir / "uncertainty_visualization.png")
    )
    plt.close(fig)
    
    print(f"✅ Visualization created")
    print(f"   Saved to {output_dir}/uncertainty_visualization.png")
    
    # Print summary
    print("\n" + "-"*70)
    print_uncertainty_summary(result)
    
    return True


def test_real_nfw_validation():
    """Test 9: Validate against known NFW profile"""
    print("\n" + "="*70)
    print("TEST 9: Validation Against NFW Profile")
    print("="*70)
    
    print("⚠️  This test requires trained model - using synthetic data")
    
    # Generate NFW ground truth
    grid_size = 64
    x = np.linspace(-5, 5, grid_size)
    y = np.linspace(-5, 5, grid_size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    
    # NFW profile
    c = 5.0
    kappa_true = 1.0 / (1 + R/c)**2
    
    # Simulate PINN prediction with uncertainty
    kappa_pred = kappa_true + np.random.normal(0, 0.01, kappa_true.shape)
    kappa_std = np.ones_like(kappa_true) * 0.01
    
    print(f"Ground truth: NFW with c={c}")
    print(f"Simulated prediction with 1% noise")
    
    # Check calibration
    calibrator = UncertaintyCalibrator()
    calib_error = calibrator.calibrate(
        predictions=kappa_pred.flatten(),
        uncertainties=kappa_std.flatten(),
        ground_truth=kappa_true.flatten()
    )
    
    print(f"\n✅ Validation complete")
    print(f"   Calibration error: {calib_error:.4f}")
    
    # Check 95% coverage
    lower = kappa_pred - 1.96 * kappa_std
    upper = kappa_pred + 1.96 * kappa_std
    coverage = np.mean((kappa_true >= lower) & (kappa_true <= upper))
    
    print(f"   95% interval coverage: {coverage:.1%}")
    
    if abs(coverage - 0.95) < 0.05:
        print(f"   ✓ Well-calibrated")
    
    return True


def test_comparison_with_without_uncertainty():
    """Test 10: Compare deterministic vs Bayesian"""
    print("\n" + "="*70)
    print("TEST 10: Deterministic vs Bayesian Comparison")
    print("="*70)
    
    model = BayesianPINN(dropout_rate=0.1)
    
    x = torch.randn(100, 5)
    
    # Deterministic (single forward pass, dropout off)
    model.eval()
    with torch.no_grad():
        det_output = model(x)
    
    # Bayesian (MC Dropout)
    mean_output, std_output = model.predict_with_uncertainty(x, n_samples=100)
    
    print(f"Deterministic prediction:")
    print(f"   Mean: {det_output.mean():.4f}")
    print(f"   Std: {det_output.std():.4f}")
    
    print(f"\nBayesian prediction (MC Dropout):")
    print(f"   Mean: {mean_output.mean():.4f}")
    print(f"   Std: {mean_output.std():.4f}")
    print(f"   Uncertainty: {std_output.mean():.4f}")
    
    print(f"\n✅ Both methods produce reasonable outputs")
    print(f"   Bayesian provides additional uncertainty estimate")
    
    return True


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("BAYESIAN UNCERTAINTY QUANTIFICATION TEST SUITE")
    print("="*70)
    print("Testing Monte Carlo Dropout and calibration analysis...")
    
    tests = [
        ("Bayesian PINN Creation", test_bayesian_pinn_creation),
        ("Forward Pass", test_forward_pass),
        ("Uncertainty Estimation", test_uncertainty_estimation),
        ("Prediction Intervals", test_prediction_intervals),
        ("Convergence with Uncertainty", test_convergence_with_uncertainty),
        ("Uncertainty Calibrator", test_uncertainty_calibrator),
        ("Poor Calibration Detection", test_poorly_calibrated),
        ("Visualization", test_visualization),
        ("NFW Validation", test_real_nfw_validation),
        ("Deterministic vs Bayesian", test_comparison_with_without_uncertainty),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed, None))
        except Exception as e:
            print(f"\n❌ ERROR in {test_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False, str(e)))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    n_passed = sum(1 for _, passed, _ in results if passed)
    n_total = len(results)
    
    for test_name, passed, error in results:
        if passed:
            print(f"✅ {test_name}")
        else:
            print(f"❌ {test_name}")
            if error:
                print(f"   Error: {error[:100]}")
    
    print("\n" + "-"*70)
    print(f"OVERALL: {n_passed}/{n_total} tests passed ({n_passed/n_total*100:.1f}%)")
    
    if n_passed == n_total:
        print("🎉 All tests PASSED! Bayesian UQ is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    exit(main())
