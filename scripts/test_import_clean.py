
import sys
# NO sys.path.insert here
try:
    from src.lens_models import NFWProfile
    from src.ml.pinn import PhysicsInformedNN
    print("✅ Import successful without sys.path hacks!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)
