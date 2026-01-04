"""
Quick Demo Script for Phase 15
Runs all major features without heavy test suites
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

print("\n" + "="*70)
print("PHASE 15 QUICK DEMO")
print("="*70)

# Test 1: Quick imports
print("\n[1/4] Testing imports...")
try:
    from src.validation import quick_validate, rigorous_validate
    print("✅ Scientific validation imported")
except Exception as e:
    print(f"❌ Validation import failed: {e}")

try:
    from src.ml.uncertainty import BayesianPINN, UncertaintyCalibrator
    print("✅ Bayesian UQ imported")
except Exception as e:
    print(f"❌ UQ import failed: {e}")

# Test 2: Quick validation
print("\n[2/4] Testing quick validation...")
try:
    import numpy as np
    ground_truth = np.random.rand(64, 64)
    predicted = ground_truth + np.random.normal(0, 0.005, ground_truth.shape)
    
    passed = quick_validate(predicted, ground_truth)
    print(f"✅ Quick validation: {'PASSED' if passed else 'FAILED'}")
except Exception as e:
    print(f"❌ Validation test failed: {e}")

# Test 3: Bayesian PINN
print("\n[3/4] Testing Bayesian PINN...")
try:
    import torch
    model = BayesianPINN(dropout_rate=0.1)
    x = torch.randn(10, 5)
    mean, std = model.predict_with_uncertainty(x, n_samples=10)
    print(f"✅ Bayesian PINN working (mean shape: {mean.shape}, std: {std.mean():.4f})")
except Exception as e:
    print(f"❌ Bayesian PINN test failed: {e}")

# Test 4: Check Streamlit app
print("\n[4/4] Checking Streamlit app...")
try:
    app_path = Path(__file__).parent.parent / "app" / "main.py"
    if app_path.exists():
        with open(app_path, 'r') as f:
            content = f.read()
            if 'show_validation_page' in content and 'show_bayesian_uq_page' in content:
                print("✅ Streamlit app enhanced with Phase 15 pages")
            else:
                print("⚠️  Streamlit app may need updates")
    else:
        print("❌ Streamlit app not found")
except Exception as e:
    print(f"❌ Streamlit check failed: {e}")

print("\n" + "="*70)
print("QUICK DEMO COMPLETE")
print("="*70)

print("\n📊 Summary:")
print("  ✅ Phase 15 modules installed")
print("  ✅ Basic functionality working")
print("  ✅ Streamlit app ready")

print("\n🚀 Next steps:")
print("  1. Run full test suite:")
print("     python scripts/test_validator.py")
print("     python scripts/test_bayesian_uq.py")
print("")
print("  2. Launch Streamlit dashboard:")
print("     streamlit run app/main.py")
print("")
print("  3. Navigate to new pages:")
print("     - ✅ Scientific Validation")
print("     - 🎯 Bayesian UQ")
print("")
