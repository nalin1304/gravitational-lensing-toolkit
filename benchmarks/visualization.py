"""
Visualization Tools for Benchmarks

Creates publication-ready plots for benchmark results

Author: Phase 13 Implementation
Date: October 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

sns.set_palette("husl")


def plot_convergence_comparison(
    our_map: np.ndarray,
    reference_map: np.ndarray,
    title: str = "Convergence Map Comparison",
    save_path: Optional[Path] = None
):
    """
    Plot side-by-side comparison of convergence maps
    
    Args:
        our_map: Our predicted convergence map
        reference_map: Reference convergence map
        title: Plot title
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Our map
    im1 = axes[0, 0].imshow(our_map, cmap='viridis', origin='lower')
    axes[0, 0].set_title("Our Method")
    axes[0, 0].set_xlabel("x [pixels]")
    axes[0, 0].set_ylabel("y [pixels]")
    plt.colorbar(im1, ax=axes[0, 0], label="κ")
    
    # Reference map
    im2 = axes[0, 1].imshow(reference_map, cmap='viridis', origin='lower')
    axes[0, 1].set_title("Reference")
    axes[0, 1].set_xlabel("x [pixels]")
    axes[0, 1].set_ylabel("y [pixels]")
    plt.colorbar(im2, ax=axes[0, 1], label="κ")
    
    # Difference map
    diff = our_map - reference_map
    im3 = axes[1, 0].imshow(diff, cmap='RdBu_r', origin='lower')
    axes[1, 0].set_title("Difference (Our - Reference)")
    axes[1, 0].set_xlabel("x [pixels]")
    axes[1, 0].set_ylabel("y [pixels]")
    plt.colorbar(im3, ax=axes[1, 0], label="Δκ")
    
    # Residual histogram
    axes[1, 1].hist(diff.flatten(), bins=50, edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel("Residual")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title("Residual Distribution")
    axes[1, 1].axvline(0, color='red', linestyle='--', label='Zero')
    axes[1, 1].legend()
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved to {save_path}")
    
    plt.show()


def plot_accuracy_vs_grid_size(
    results: Dict[str, List[Dict]],
    save_path: Optional[Path] = None
):
    """
    Plot accuracy vs grid size
    
    Args:
        results: Benchmark results from benchmark_convergence_accuracy
        save_path: Path to save figure
    """
    grid_tests = results['grid_size_tests']
    
    grid_sizes = [t['grid_size'] for t in grid_tests]
    mean_errors = [t['mean_error'] for t in grid_tests]
    std_errors = [t['std_error'] for t in grid_tests]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.errorbar(grid_sizes, mean_errors, yerr=std_errors, 
                marker='o', markersize=8, capsize=5, capthick=2,
                linewidth=2, label='Mean ± Std')
    
    ax.set_xlabel("Grid Size", fontweight='bold')
    ax.set_ylabel("Relative Error", fontweight='bold')
    ax.set_title("Convergence Accuracy vs Grid Size", fontsize=14, fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved to {save_path}")
    
    plt.show()


def plot_speed_benchmark(
    results: Dict[str, float],
    save_path: Optional[Path] = None
):
    """
    Plot inference speed benchmark results
    
    Args:
        results: Speed benchmark results
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Timing statistics
    times = ['Mean', 'Median', 'Min', 'Max']
    values = [results['mean_time'], results['median_time'], 
              results['min_time'], results['max_time']]
    colors = ['steelblue', 'green', 'orange', 'red']
    
    bars = ax1.bar(times, values, color=colors, edgecolor='black', alpha=0.7)
    ax1.set_ylabel("Time [seconds]", fontweight='bold')
    ax1.set_title("Inference Time Statistics", fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}s', ha='center', va='bottom', fontsize=9)
    
    # Throughput
    throughput = results['throughput']
    ax2.bar(['Throughput'], [throughput], color='purple', edgecolor='black', alpha=0.7, width=0.4)
    ax2.set_ylabel("Inferences per Second", fontweight='bold')
    ax2.set_title("Inference Throughput", fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.text(0, throughput, f'{throughput:.2f}', ha='center', va='bottom', fontsize=10)
    
    fig.suptitle(f"Speed Benchmark (Grid Size: {results['grid_size']}, N={results['n_runs']})",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved to {save_path}")
    
    plt.show()


def plot_metrics_comparison(
    metrics: Dict[str, float],
    save_path: Optional[Path] = None
):
    """
    Plot comparison metrics
    
    Args:
        metrics: Dictionary of metrics
        save_path: Path to save figure
    """
    # Select key metrics for visualization
    key_metrics = {
        'Relative Error': metrics.get('relative_error', 0),
        'RMSE': metrics.get('rmse', 0),
        'MAE': metrics.get('mae', 0),
        'Pearson Corr.': metrics.get('pearson_correlation', 0),
        'SSIM': metrics.get('ssim', 0),
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Error metrics (log scale)
    error_metrics = {k: v for k, v in key_metrics.items() 
                     if k in ['Relative Error', 'RMSE', 'MAE']}
    
    bars1 = ax1.bar(error_metrics.keys(), error_metrics.values(),
                    color=['red', 'orange', 'yellow'], edgecolor='black', alpha=0.7)
    ax1.set_ylabel("Error Value", fontweight='bold')
    ax1.set_title("Error Metrics", fontsize=12, fontweight='bold')
    ax1.set_yscale('log')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.tick_params(axis='x', rotation=15)
    
    for bar, (name, value) in zip(bars1, error_metrics.items()):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2e}', ha='center', va='bottom', fontsize=8)
    
    # Quality metrics (0-1 scale)
    quality_metrics = {k: v for k, v in key_metrics.items() 
                       if k in ['Pearson Corr.', 'SSIM']}
    
    bars2 = ax2.bar(quality_metrics.keys(), quality_metrics.values(),
                    color=['green', 'blue'], edgecolor='black', alpha=0.7)
    ax2.set_ylabel("Score", fontweight='bold')
    ax2.set_title("Quality Metrics (Higher is Better)", fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 1.1])
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect Score')
    ax2.legend()
    
    for bar, (name, value) in zip(bars2, quality_metrics.items()):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}', ha='center', va='bottom', fontsize=9)
    
    fig.suptitle("Validation Metrics", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved to {save_path}")
    
    plt.show()


def plot_comprehensive_results(
    results: Dict[str, any],
    output_dir: Path
):
    """
    Create all benchmark plots
    
    Args:
        results: Complete benchmark results
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating benchmark visualizations...")
    
    # 1. Accuracy vs grid size
    if 'accuracy' in results:
        print("  - Accuracy vs Grid Size")
        plot_accuracy_vs_grid_size(
            results['accuracy'],
            save_path=output_dir / 'accuracy_vs_grid_size.png'
        )
    
    # 2. Speed benchmark
    if 'speed' in results:
        print("  - Speed Benchmark")
        plot_speed_benchmark(
            results['speed'],
            save_path=output_dir / 'speed_benchmark.png'
        )
    
    # 3. Analytic comparison
    if 'analytic' in results:
        print("  - Analytic Comparison")
        
        # Convergence maps
        plot_convergence_comparison(
            results['analytic']['our_map'],
            results['analytic']['analytic_map'],
            title="Comparison with Analytic NFW Solution",
            save_path=output_dir / 'analytic_comparison.png'
        )
        
        # Metrics
        plot_metrics_comparison(
            results['analytic']['metrics'],
            save_path=output_dir / 'metrics_comparison.png'
        )
    
    print(f"\nAll plots saved to {output_dir}")


def create_publication_figure(
    results: Dict[str, any],
    save_path: Path
):
    """
    Create single publication-ready figure with all key results
    
    Args:
        results: Complete benchmark results
        save_path: Path to save figure
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Convergence maps (top row)
    if 'analytic' in results:
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(results['analytic']['our_map'], cmap='viridis', origin='lower')
        ax1.set_title("(a) PINN Prediction", fontweight='bold')
        ax1.set_xlabel("x [pixels]")
        ax1.set_ylabel("y [pixels]")
        plt.colorbar(im1, ax=ax1, label="κ", fraction=0.046)
        
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(results['analytic']['analytic_map'], cmap='viridis', origin='lower')
        ax2.set_title("(b) Analytic Solution", fontweight='bold')
        ax2.set_xlabel("x [pixels]")
        ax2.set_ylabel("y [pixels]")
        plt.colorbar(im2, ax=ax2, label="κ", fraction=0.046)
        
        ax3 = fig.add_subplot(gs[0, 2])
        diff = results['analytic']['our_map'] - results['analytic']['analytic_map']
        im3 = ax3.imshow(diff, cmap='RdBu_r', origin='lower')
        ax3.set_title("(c) Residuals", fontweight='bold')
        ax3.set_xlabel("x [pixels]")
        ax3.set_ylabel("y [pixels]")
        plt.colorbar(im3, ax=ax3, label="Δκ", fraction=0.046)
    
    # 2. Accuracy vs grid size (middle left)
    if 'accuracy' in results:
        ax4 = fig.add_subplot(gs[1, 0:2])
        grid_tests = results['accuracy']['grid_size_tests']
        grid_sizes = [t['grid_size'] for t in grid_tests]
        mean_errors = [t['mean_error'] for t in grid_tests]
        std_errors = [t['std_error'] for t in grid_tests]
        
        ax4.errorbar(grid_sizes, mean_errors, yerr=std_errors,
                    marker='o', markersize=8, capsize=5, linewidth=2)
        ax4.set_xlabel("Grid Size", fontweight='bold')
        ax4.set_ylabel("Relative Error", fontweight='bold')
        ax4.set_title("(d) Accuracy vs Grid Size", fontweight='bold')
        ax4.set_xscale('log', base=2)
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3, linestyle='--')
    
    # 3. Speed metrics (middle right)
    if 'speed' in results:
        ax5 = fig.add_subplot(gs[1, 2])
        throughput = results['speed']['throughput']
        mean_time = results['speed']['mean_time']
        
        metrics_names = ['Throughput\n[inf/s]', 'Inference\nTime [ms]']
        metrics_values = [throughput, mean_time * 1000]
        colors = ['purple', 'steelblue']
        
        bars = ax5.bar(metrics_names, metrics_values, color=colors, edgecolor='black', alpha=0.7)
        ax5.set_title("(e) Performance Metrics", fontweight='bold')
        ax5.grid(axis='y', alpha=0.3, linestyle='--')
        
        for bar, value in zip(bars, metrics_values):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 4. Metrics table (bottom)
    if 'analytic' in results:
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')
        
        metrics = results['analytic']['metrics']
        table_data = [
            ['Metric', 'Value'],
            ['Relative Error', f"{metrics.get('relative_error', 0):.6e}"],
            ['RMSE', f"{metrics.get('rmse', 0):.6e}"],
            ['SSIM', f"{metrics.get('ssim', 0):.4f}"],
            ['Pearson Correlation', f"{metrics.get('pearson_correlation', 0):.6f}"],
            ['PSNR [dB]', f"{metrics.get('psnr', 0):.2f}"],
        ]
        
        table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                         colWidths=[0.4, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(2):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(table_data)):
            for j in range(2):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        ax6.set_title("(f) Validation Metrics Summary", fontweight='bold', pad=20)
    
    fig.suptitle("Gravitational Lensing PINN - Comprehensive Benchmark Results",
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"\nPublication figure saved to {save_path}")
    plt.show()
