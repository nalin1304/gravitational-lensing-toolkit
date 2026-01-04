"""
Benchmark for ray tracing performance.
"""
import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.lens_models.mass_profiles import NFWProfile
from src.lens_models.lens_system import LensSystem
from benchmarks.profiler import PerformanceBenchmark

def run_ray_tracing_benchmark():
    """Benchmark the NFW deflection angle calculation."""
    
    # Setup
    lens_sys = LensSystem(z_lens=0.5, z_source=2.0)
    profile = NFWProfile(M_vir=1e12, concentration=10.0, lens_system=lens_sys)
    
    # Create large grid
    grid_size = 1000
    x = np.linspace(-5, 5, grid_size)
    y = np.linspace(-5, 5, grid_size)
    X, Y = np.meshgrid(x, y)
    
    def calculation_task():
        """Task to benchmark"""
        return profile.deflection_angle(X, Y)

    # Run benchmark
    benchmark = PerformanceBenchmark("NFW Ray Tracing (1M points)")
    stats = benchmark.run_iterations(calculation_task, n_iterations=5)
    benchmark.print_report(stats)

if __name__ == "__main__":
    run_ray_tracing_benchmark()
