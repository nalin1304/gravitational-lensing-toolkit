"""
Benchmark Runner - CLI Tool for Running Scientific Benchmarks

Usage:
    python -m benchmarks.runner --all
    python -m benchmarks.runner --accuracy
    python -m benchmarks.runner --speed
    python -m benchmarks.runner --analytic
    python -m benchmarks.runner --visualize results/benchmark_results.json

Author: Phase 13 Implementation
Date: October 2025
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
import sys

from benchmarks.comparisons import (
    run_comprehensive_benchmark,
    benchmark_convergence_accuracy,
    benchmark_inference_speed,
    compare_with_analytic,
    print_benchmark_report,
)
from benchmarks.visualization import (
    plot_comprehensive_results,
    create_publication_figure,
)


def save_results(results: dict, output_path: Path):
    """
    Save benchmark results to JSON file
    
    Args:
        results: Benchmark results dictionary
        output_path: Path to save JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_for_json(obj):
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return obj
    
    results_json = convert_for_json(results)
    
    with open(output_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\n✅ Results saved to {output_path}")


def load_results(input_path: Path) -> dict:
    """
    Load benchmark results from JSON file
    
    Args:
        input_path: Path to JSON file
        
    Returns:
        Benchmark results dictionary
    """
    with open(input_path, 'r') as f:
        results = json.load(f)
    
    print(f"✅ Results loaded from {input_path}")
    return results


def run_accuracy_benchmark(args):
    """Run accuracy benchmark"""
    print("\n" + "="*80)
    print("RUNNING ACCURACY BENCHMARK")
    print("="*80 + "\n")
    
    grid_sizes = args.grid_sizes or [32, 64, 128]
    masses = args.masses or [5e11, 1e12, 5e12, 1e13]
    
    print(f"Grid sizes: {grid_sizes}")
    print(f"Masses: {[f'{m:.2e}' for m in masses]}")
    print()
    
    results = benchmark_convergence_accuracy(
        grid_sizes=grid_sizes,
        masses=masses
    )
    
    print("\n" + "="*80)
    print("ACCURACY BENCHMARK COMPLETE")
    print("="*80)
    
    return {'accuracy': results}


def run_speed_benchmark(args):
    """Run speed benchmark"""
    print("\n" + "="*80)
    print("RUNNING SPEED BENCHMARK")
    print("="*80 + "\n")
    
    grid_size = args.grid_size or 64
    n_runs = args.n_runs or 100
    warm_up = args.warm_up or 10
    
    print(f"Grid size: {grid_size}")
    print(f"Number of runs: {n_runs}")
    print(f"Warm-up runs: {warm_up}")
    print()
    
    results = benchmark_inference_speed(
        grid_size=grid_size,
        n_runs=n_runs,
        warm_up=warm_up
    )
    
    print("\n" + "="*80)
    print("SPEED BENCHMARK COMPLETE")
    print("="*80)
    print(f"\nMean inference time: {results['mean_time']*1000:.2f} ms")
    print(f"Throughput: {results['throughput']:.2f} inferences/second")
    
    return {'speed': results}


def run_analytic_benchmark(args):
    """Run analytic comparison benchmark"""
    print("\n" + "="*80)
    print("RUNNING ANALYTIC COMPARISON BENCHMARK")
    print("="*80 + "\n")
    
    grid_size = args.grid_size or 64
    mass = args.mass or 1e12
    concentration = args.concentration or 5.0
    
    print(f"Grid size: {grid_size}")
    print(f"Mass: {mass:.2e} M☉")
    print(f"Concentration: {concentration}")
    print()
    
    results = compare_with_analytic(
        grid_size=grid_size,
        mass=mass,
        concentration=concentration
    )
    
    print("\n" + "="*80)
    print("ANALYTIC COMPARISON COMPLETE")
    print("="*80)
    print(f"\nRelative error: {results['metrics']['relative_error']:.6e}")
    print(f"SSIM: {results['metrics']['ssim']:.4f}")
    print(f"Pearson correlation: {results['metrics']['pearson_correlation']:.6f}")
    
    return {'analytic': results}


def run_all_benchmarks(args):
    """Run all benchmarks"""
    print("\n" + "="*80)
    print("RUNNING COMPREHENSIVE BENCHMARK SUITE")
    print("="*80 + "\n")
    
    results = run_comprehensive_benchmark()
    
    # Print detailed report
    print_benchmark_report(results)
    
    return results


def visualize_results(args):
    """Create visualizations from saved results"""
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80 + "\n")
    
    results = load_results(args.input)
    
    output_dir = args.output_dir or Path('results/figures')
    output_dir = Path(output_dir)
    
    # Generate all plots
    plot_comprehensive_results(results, output_dir)
    
    # Generate publication figure
    pub_fig_path = output_dir / 'publication_figure.png'
    create_publication_figure(results, pub_fig_path)
    
    print(f"\n✅ All visualizations saved to {output_dir}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Run scientific benchmarks for gravitational lensing PINN',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all benchmarks
  python -m benchmarks.runner --all -o results/benchmark_results.json
  
  # Run only accuracy benchmark
  python -m benchmarks.runner --accuracy --grid-sizes 32 64 128
  
  # Run only speed benchmark
  python -m benchmarks.runner --speed --n-runs 100
  
  # Run analytic comparison
  python -m benchmarks.runner --analytic --mass 1e12
  
  # Generate visualizations from saved results
  python -m benchmarks.runner --visualize results/benchmark_results.json
        """
    )
    
    # Benchmark selection
    benchmark_group = parser.add_argument_group('Benchmark Selection')
    benchmark_group.add_argument('--all', action='store_true',
                                 help='Run all benchmarks')
    benchmark_group.add_argument('--accuracy', action='store_true',
                                 help='Run accuracy benchmark')
    benchmark_group.add_argument('--speed', action='store_true',
                                 help='Run speed benchmark')
    benchmark_group.add_argument('--analytic', action='store_true',
                                 help='Run analytic comparison')
    benchmark_group.add_argument('--visualize', type=str, metavar='INPUT_JSON',
                                 help='Generate visualizations from saved results')
    
    # Accuracy benchmark options
    accuracy_group = parser.add_argument_group('Accuracy Benchmark Options')
    accuracy_group.add_argument('--grid-sizes', type=int, nargs='+',
                               metavar='SIZE',
                               help='Grid sizes to test (default: 32 64 128)')
    accuracy_group.add_argument('--masses', type=float, nargs='+',
                               metavar='MASS',
                               help='Masses to test in solar masses (default: 5e11 1e12 5e12 1e13)')
    
    # Speed benchmark options
    speed_group = parser.add_argument_group('Speed Benchmark Options')
    speed_group.add_argument('--n-runs', type=int, metavar='N',
                            help='Number of benchmark runs (default: 100)')
    speed_group.add_argument('--warm-up', type=int, metavar='N',
                            help='Number of warm-up runs (default: 10)')
    
    # Analytic benchmark options
    analytic_group = parser.add_argument_group('Analytic Benchmark Options')
    analytic_group.add_argument('--mass', type=float, metavar='MASS',
                               help='Halo mass in solar masses (default: 1e12)')
    analytic_group.add_argument('--concentration', type=float, metavar='C',
                               help='NFW concentration parameter (default: 5.0)')
    
    # Common options
    common_group = parser.add_argument_group('Common Options')
    common_group.add_argument('--grid-size', type=int, metavar='SIZE',
                             help='Grid size (default: 64)')
    common_group.add_argument('-o', '--output', type=str,
                             metavar='FILE',
                             help='Output JSON file for results')
    common_group.add_argument('--output-dir', type=str,
                             metavar='DIR',
                             help='Output directory for visualizations (default: results/figures)')
    common_group.add_argument('--no-save', action='store_true',
                             help='Do not save results to file')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.all, args.accuracy, args.speed, args.analytic, args.visualize]):
        parser.error('Must specify at least one benchmark: --all, --accuracy, --speed, --analytic, or --visualize')
    
    try:
        results = {}
        
        # Run benchmarks
        if args.visualize:
            visualize_results(args)
            return 0
        
        if args.all:
            results = run_all_benchmarks(args)
        else:
            if args.accuracy:
                results.update(run_accuracy_benchmark(args))
            
            if args.speed:
                results.update(run_speed_benchmark(args))
            
            if args.analytic:
                results.update(run_analytic_benchmark(args))
        
        # Save results
        if not args.no_save:
            output_path = args.output or f'results/benchmark_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            save_results(results, output_path)
        
        print("\n" + "="*80)
        print("✅ BENCHMARK COMPLETE")
        print("="*80)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Benchmark interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
