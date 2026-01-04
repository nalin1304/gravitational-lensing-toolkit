
import sys
import traceback
from src.ml.generate_dataset import generate_synthetic_convergence

def test_generation():
    print("Testing NFW generation...")
    try:
        generate_synthetic_convergence(
            profile_type="NFW",
            mass=2e12,
            scale_radius=200.0,
            ellipticity=0.0,
            grid_size=64
        )
        print("NFW Success")
    except Exception:
        traceback.print_exc()

    print("\nTesting Elliptical NFW generation...")
    try:
        generate_synthetic_convergence(
            profile_type="Elliptical NFW",
            mass=1e12,
            scale_radius=150.0,
            ellipticity=0.3,
            grid_size=32
        )
        print("Elliptical NFW Success")
    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    test_generation()
