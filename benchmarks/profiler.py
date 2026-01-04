"""
Performance Profiling Tools

Provides decorators and context managers for profiling code performance

Author: Phase 13 Implementation
Date: October 2025
"""

import time
import functools
import tracemalloc
from typing import Callable, Any, Dict
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


class ProfilerContext:
    """Context manager for profiling code blocks"""
    
    def __init__(self, name: str = "Code block", log_results: bool = True):
        self.name = name
        self.log_results = log_results
        self.start_time = None
        self.end_time = None
        self.memory_start = None
        self.memory_peak = None
        
    def __enter__(self):
        """Start profiling"""
        self.start_time = time.perf_counter()
        tracemalloc.start()
        self.memory_start = tracemalloc.get_traced_memory()[0]
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop profiling and log results"""
        self.end_time = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        self.memory_peak = peak
        self.duration = self.end_time - self.start_time
        self.memory_used = (peak - self.memory_start) / 1024 / 1024  # MB
        
        if self.log_results:
            logger.info(
                f"{self.name}: {self.duration:.4f}s, "
                f"Memory: {self.memory_used:.2f} MB"
            )
    
    def get_stats(self) -> Dict[str, float]:
        """Get profiling statistics"""
        return {
            'duration_seconds': self.duration,
            'memory_mb': self.memory_used,
            'memory_peak_mb': self.memory_peak / 1024 / 1024
        }


def time_profile(func: Callable) -> Callable:
    """
    Decorator to profile function execution time
    
    Usage:
        @time_profile
        def my_function():
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        duration = end_time - start_time
        logger.info(f"{func.__name__} took {duration:.4f} seconds")
        
        return result
    
    return wrapper


def memory_profile(func: Callable) -> Callable:
    """
    Decorator to profile function memory usage
    
    Usage:
        @memory_profile
        def my_function():
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        
        result = func(*args, **kwargs)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        peak_mb = peak / 1024 / 1024
        logger.info(f"{func.__name__} peak memory: {peak_mb:.2f} MB")
        
        return result
    
    return wrapper


def profile_function(func: Callable) -> Callable:
    """
    Decorator to profile both time and memory
    
    Usage:
        @profile_function
        def my_function():
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        tracemalloc.start()
        
        result = func(*args, **kwargs)
        
        end_time = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        duration = end_time - start_time
        peak_mb = peak / 1024 / 1024
        
        logger.info(
            f"{func.__name__}: {duration:.4f}s, "
            f"Peak memory: {peak_mb:.2f} MB"
        )
        
        return result
    
    return wrapper


@contextmanager
def profile_block(name: str = "Code block"):
    """
    Context manager for profiling code blocks
    
    Usage:
        with profile_block("My operation"):
            # code to profile
            pass
    """
    profiler = ProfilerContext(name, log_results=True)
    with profiler:
        yield profiler


class PerformanceBenchmark:
    """Class for running performance benchmarks"""
    
    def __init__(self, name: str):
        self.name = name
        self.results = []
    
    def run_iteration(self, func: Callable, *args, **kwargs) -> Dict[str, float]:
        """Run one benchmark iteration"""
        start_time = time.perf_counter()
        tracemalloc.start()
        
        result = func(*args, **kwargs)
        
        end_time = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        stats = {
            'duration': end_time - start_time,
            'memory_peak_mb': peak / 1024 / 1024,
            'result': result
        }
        
        self.results.append(stats)
        return stats
    
    def run_iterations(
        self,
        func: Callable,
        n_iterations: int = 10,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """Run multiple benchmark iterations"""
        import numpy as np
        
        self.results = []
        
        for i in range(n_iterations):
            self.run_iteration(func, *args, **kwargs)
        
        # Calculate statistics
        durations = [r['duration'] for r in self.results]
        memories = [r['memory_peak_mb'] for r in self.results]
        
        stats = {
            'n_iterations': n_iterations,
            'duration_mean': np.mean(durations),
            'duration_std': np.std(durations),
            'duration_min': np.min(durations),
            'duration_max': np.max(durations),
            'duration_median': np.median(durations),
            'memory_mean': np.mean(memories),
            'memory_std': np.std(memories),
            'memory_max': np.max(memories)
        }
        
        return stats
    
    def print_report(self, stats: Dict[str, Any]):
        """Print benchmark report"""
        print("=" * 60)
        print(f"{self.name} Benchmark Results")
        print("=" * 60)
        print(f"Iterations: {stats['n_iterations']}")
        print("\nExecution Time:")
        print(f"  Mean:   {stats['duration_mean']:.4f} ± {stats['duration_std']:.4f} s")
        print(f"  Median: {stats['duration_median']:.4f} s")
        print(f"  Min:    {stats['duration_min']:.4f} s")
        print(f"  Max:    {stats['duration_max']:.4f} s")
        print("\nMemory Usage:")
        print(f"  Mean:   {stats['memory_mean']:.2f} ± {stats['memory_std']:.2f} MB")
        print(f"  Peak:   {stats['memory_max']:.2f} MB")
        print("=" * 60)


def compare_implementations(
    impl1: Callable,
    impl2: Callable,
    n_iterations: int = 10,
    *args,
    **kwargs
) -> Dict[str, Any]:
    """
    Compare two implementations
    
    Args:
        impl1: First implementation
        impl2: Second implementation
        n_iterations: Number of iterations to run
        *args, **kwargs: Arguments to pass to both implementations
        
    Returns:
        dict: Comparison results
    """
    bench1 = PerformanceBenchmark(impl1.__name__)
    bench2 = PerformanceBenchmark(impl2.__name__)
    
    stats1 = bench1.run_iterations(impl1, n_iterations, *args, **kwargs)
    stats2 = bench2.run_iterations(impl2, n_iterations, *args, **kwargs)
    
    speedup = stats2['duration_mean'] / stats1['duration_mean']
    memory_ratio = stats1['memory_mean'] / stats2['memory_mean']
    
    comparison = {
        'impl1_name': impl1.__name__,
        'impl2_name': impl2.__name__,
        'impl1_stats': stats1,
        'impl2_stats': stats2,
        'speedup': speedup,
        'memory_ratio': memory_ratio,
        'faster': impl1.__name__ if speedup > 1 else impl2.__name__,
        'more_efficient': impl1.__name__ if memory_ratio < 1 else impl2.__name__
    }
    
    return comparison


def print_comparison_report(comparison: Dict[str, Any]):
    """Print implementation comparison report"""
    print("=" * 60)
    print("Implementation Comparison")
    print("=" * 60)
    print(f"\n{comparison['impl1_name']} vs {comparison['impl2_name']}\n")
    
    print("Execution Time:")
    print(f"  {comparison['impl1_name']:20s}: {comparison['impl1_stats']['duration_mean']:.4f} s")
    print(f"  {comparison['impl2_name']:20s}: {comparison['impl2_stats']['duration_mean']:.4f} s")
    print(f"  Speedup: {comparison['speedup']:.2f}x")
    print(f"  Faster: {comparison['faster']}")
    
    print("\nMemory Usage:")
    print(f"  {comparison['impl1_name']:20s}: {comparison['impl1_stats']['memory_mean']:.2f} MB")
    print(f"  {comparison['impl2_name']:20s}: {comparison['impl2_stats']['memory_mean']:.2f} MB")
    print(f"  Ratio: {comparison['memory_ratio']:.2f}x")
    print(f"  More efficient: {comparison['more_efficient']}")
    
    print("=" * 60)
