"""
Performance Optimization Module for Phase 7

This module provides GPU acceleration, vectorization, and performance utilities
for the gravitational lensing toolkit.

Features:
- GPU acceleration via CuPy (optional)
- Vectorized operations
- Performance benchmarking
- Automatic fallback to NumPy
"""

import numpy as np
import time
from typing import Callable, Dict, Any, Optional, Tuple
from functools import wraps
import sys
import logging

logger = logging.getLogger(__name__)

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
    logger.info("CuPy detected: GPU acceleration available")
except ImportError:
    cp = None
    GPU_AVAILABLE = False
    logger.info("CuPy not found: Using CPU (NumPy) only")


class ArrayBackend:
    """
    Unified array backend that automatically selects NumPy or CuPy.
    
    This provides a consistent interface regardless of whether GPU is available.
    """
    
    def __init__(self, use_gpu: bool = True):
        """
        Initialize array backend.
        
        Parameters
        ----------
        use_gpu : bool
            Whether to use GPU if available (default True)
        """
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np
        
    @property
    def backend_name(self) -> str:
        """Get name of active backend."""
        return "CuPy (GPU)" if self.use_gpu else "NumPy (CPU)"
    
    def array(self, data, dtype=None):
        """Create array on appropriate device."""
        return self.xp.array(data, dtype=dtype)
    
    def zeros(self, shape, dtype=float):
        """Create zeros array."""
        return self.xp.zeros(shape, dtype=dtype)
    
    def ones(self, shape, dtype=float):
        """Create ones array."""
        return self.xp.ones(shape, dtype=dtype)
    
    def linspace(self, start, stop, num):
        """Create linearly spaced array."""
        return self.xp.linspace(start, stop, num)
    
    def meshgrid(self, *arrays, indexing='xy'):
        """Create coordinate grids."""
        return self.xp.meshgrid(*arrays, indexing=indexing)
    
    def to_numpy(self, array):
        """Convert array to NumPy (no-op if already NumPy)."""
        if self.use_gpu and isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
        return np.asarray(array)
    
    def to_device(self, array):
        """Move array to device (GPU or CPU)."""
        if self.use_gpu:
            if isinstance(array, np.ndarray):
                return cp.asarray(array)
            return array
        else:
            if cp is not None and isinstance(array, cp.ndarray):
                return cp.asnumpy(array)
            return np.asarray(array)


# Global backend instance
_global_backend = ArrayBackend(use_gpu=True)


def get_backend() -> ArrayBackend:
    """Get global array backend."""
    return _global_backend


def set_backend(use_gpu: bool = True):
    """
    Set global array backend.
    
    Parameters
    ----------
    use_gpu : bool
        Whether to use GPU if available
    """
    global _global_backend
    _global_backend = ArrayBackend(use_gpu=use_gpu)


def timer(func: Callable) -> Callable:
    """
    Decorator to time function execution.
    
    Examples
    --------
    >>> @timer
    ... def slow_function():
    ...     time.sleep(1)
    >>> slow_function()  # Prints execution time
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper


class PerformanceMonitor:
    """
    Monitor and compare performance of operations.
    
    Examples
    --------
    >>> monitor = PerformanceMonitor()
    >>> with monitor.time_block("operation1"):
    ...     # Do work
    ...     pass
    >>> monitor.print_summary()
    """
    
    def __init__(self):
        self.timings: Dict[str, list] = {}
        
    def time_block(self, name: str):
        """Context manager to time a block of code."""
        return _TimedBlock(self, name)
    
    def record(self, name: str, duration: float):
        """Record a timing measurement."""
        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(duration)
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a named operation."""
        if name not in self.timings or len(self.timings[name]) == 0:
            return {}
        
        times = self.timings[name]
        return {
            'count': len(times),
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'total': np.sum(times)
        }
    
    def print_summary(self):
        """Print summary of all timings."""
        print("\n=== Performance Summary ===")
        for name in sorted(self.timings.keys()):
            stats = self.get_stats(name)
            print(f"\n{name}:")
            print(f"  Count:    {stats['count']}")
            print(f"  Mean:     {stats['mean']:.4f}s")
            print(f"  Std:      {stats['std']:.4f}s")
            print(f"  Min:      {stats['min']:.4f}s")
            print(f"  Max:      {stats['max']:.4f}s")
            print(f"  Total:    {stats['total']:.4f}s")
        print("=" * 30)
    
    def clear(self):
        """Clear all recorded timings."""
        self.timings.clear()


class _TimedBlock:
    """Context manager for timing blocks of code."""
    
    def __init__(self, monitor: PerformanceMonitor, name: str):
        self.monitor = monitor
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.perf_counter() - self.start_time
        self.monitor.record(self.name, duration)


def benchmark_convergence_map(
    lens_model,
    grid_sizes: list = None,
    use_gpu: bool = None
) -> Dict[str, Any]:
    """
    Benchmark convergence map generation at different resolutions.
    
    Parameters
    ----------
    lens_model : MassProfile
        Lens model to benchmark
    grid_sizes : list
        List of grid sizes to test (default: [32, 64, 128, 256])
    use_gpu : bool, optional
        Force GPU usage (None = auto-detect)
    
    Returns
    -------
    results : dict
        Benchmark results with timings and speedups
    """
    from .generate_dataset import generate_convergence_map_vectorized
    
    if grid_sizes is None:
        grid_sizes = [32, 64, 128, 256]
    
    results = {
        'grid_sizes': grid_sizes,
        'timings': [],
        'backend': get_backend().backend_name
    }
    
    print(f"\nBenchmarking convergence map generation ({results['backend']})...")
    
    for size in grid_sizes:
        start = time.perf_counter()
        _ = generate_convergence_map_vectorized(lens_model, grid_size=size)
        duration = time.perf_counter() - start
        
        results['timings'].append(duration)
        print(f"  Grid {size}x{size}: {duration:.4f}s ({size*size/duration:.0f} pts/s)")
    
    return results


def compare_cpu_gpu_performance(lens_model, grid_size: int = 128) -> Dict[str, float]:
    """
    Compare CPU vs GPU performance for convergence map generation.
    
    Parameters
    ----------
    lens_model : MassProfile
        Lens model to benchmark
    grid_size : int
        Grid resolution to test
    
    Returns
    -------
    results : dict
        Timing comparison and speedup factor
    """
    from .generate_dataset import generate_convergence_map_vectorized
    
    results = {}
    
    # CPU benchmark
    set_backend(use_gpu=False)
    start = time.perf_counter()
    _ = generate_convergence_map_vectorized(lens_model, grid_size=grid_size)
    cpu_time = time.perf_counter() - start
    results['cpu_time'] = cpu_time
    
    # GPU benchmark (if available)
    if GPU_AVAILABLE:
        set_backend(use_gpu=True)
        start = time.perf_counter()
        _ = generate_convergence_map_vectorized(lens_model, grid_size=grid_size)
        gpu_time = time.perf_counter() - start
        results['gpu_time'] = gpu_time
        results['speedup'] = cpu_time / gpu_time
        
        print(f"\n=== CPU vs GPU Comparison (Grid {grid_size}x{grid_size}) ===")
        print(f"CPU Time:  {cpu_time:.4f}s")
        print(f"GPU Time:  {gpu_time:.4f}s")
        print(f"Speedup:   {results['speedup']:.2f}x")
        print("=" * 50)
    else:
        print("GPU not available for comparison")
        results['gpu_time'] = None
        results['speedup'] = 1.0
    
    return results


# Cache for repeated calculations
_convergence_cache = {}


def cached_convergence(lens_model, grid_size: int = 64, extent: float = 3.0):
    """
    Generate convergence map with caching.
    
    Parameters
    ----------
    lens_model : MassProfile
        Lens model
    grid_size : int
        Grid size
    extent : float
        Physical extent
    
    Returns
    -------
    convergence_map : np.ndarray
        Cached or newly computed convergence map
    """
    from .generate_dataset import generate_convergence_map_vectorized
    
    # Create cache key
    key = (
        type(lens_model).__name__,
        str(lens_model),
        grid_size,
        extent
    )
    
    if key not in _convergence_cache:
        _convergence_cache[key] = generate_convergence_map_vectorized(
            lens_model, grid_size=grid_size, extent=extent
        )
    
    return _convergence_cache[key]


def clear_cache():
    """Clear convergence map cache."""
    global _convergence_cache
    _convergence_cache.clear()
