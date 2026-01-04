"""
Benchmark Package for Scientific Validation

Provides tools for:
- Scientific validation metrics
- Performance profiling
- Comparison with analytic solutions and established codes
- Publication-ready visualizations
- CLI benchmark runner

Author: Phase 13 Implementation
Date: October 2025
"""

from .metrics import (
    calculate_relative_error,
    calculate_chi_squared,
    calculate_rmse,
    calculate_mae,
    calculate_structural_similarity,
    calculate_peak_signal_noise_ratio,
    calculate_pearson_correlation,
    calculate_fractional_bias,
    calculate_residuals,
    calculate_confidence_interval,
    calculate_normalized_cross_correlation,
    calculate_all_metrics,
    print_metrics_report,
)

from .comparisons import (
    analytic_nfw_convergence,
    compare_with_analytic,
    compare_with_lenstool,
    compare_with_glafic,
    benchmark_convergence_accuracy,
    benchmark_inference_speed,
    run_comprehensive_benchmark,
    print_benchmark_report,
)

from .profiler import (
    ProfilerContext,
    time_profile,
    memory_profile,
    profile_function,
    profile_block,
    PerformanceBenchmark,
    compare_implementations,
    print_comparison_report,
)

from .visualization import (
    plot_convergence_comparison,
    plot_accuracy_vs_grid_size,
    plot_speed_benchmark,
    plot_metrics_comparison,
    plot_comprehensive_results,
    create_publication_figure,
)

__version__ = '1.0.0'

__all__ = [
    # Metrics
    'calculate_relative_error',
    'calculate_chi_squared',
    'calculate_rmse',
    'calculate_mae',
    'calculate_structural_similarity',
    'calculate_peak_signal_noise_ratio',
    'calculate_pearson_correlation',
    'calculate_fractional_bias',
    'calculate_residuals',
    'calculate_confidence_interval',
    'calculate_normalized_cross_correlation',
    'calculate_all_metrics',
    'print_metrics_report',
    # Comparisons
    'analytic_nfw_convergence',
    'compare_with_analytic',
    'compare_with_lenstool',
    'compare_with_glafic',
    'benchmark_convergence_accuracy',
    'benchmark_inference_speed',
    'run_comprehensive_benchmark',
    'print_benchmark_report',
    # Profiler
    'ProfilerContext',
    'time_profile',
    'memory_profile',
    'profile_function',
    'profile_block',
    'PerformanceBenchmark',
    'compare_implementations',
    'print_comparison_report',
    # Visualization
    'plot_convergence_comparison',
    'plot_accuracy_vs_grid_size',
    'plot_speed_benchmark',
    'plot_metrics_comparison',
    'plot_comprehensive_results',
    'create_publication_figure',
]
