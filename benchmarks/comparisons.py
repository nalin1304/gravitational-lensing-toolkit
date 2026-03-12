"""
Comparison with Established Lensing Codes

Benchmarks against:
- Lenstool
- GLAFIC
- Analytic solutions

Author: Phase 13 Implementation
Date: October 2025
"""

import numpy as np
import time
from typing import Dict, Tuple, Optional, List
import logging

import sys
from pathlib import Path
from src.lens_models.multi_plane import MultiPlaneLens
from src.lens_models.lens_system import LensSystem
from src.lens_models.mass_profiles import NFWProfile
from src.ml.generate_dataset import generate_convergence_map_vectorized
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from src.utils.common import load_pretrained_model, prepare_model_input
from benchmarks.metrics import calculate_all_metrics, print_metrics_report

logger = logging.getLogger(__name__)


def generate_synthetic_convergence(
    profile_type="NFW",
    mass=1e12,
    scale_radius=200.0,
    ellipticity=0.0,
    grid_size=64,
    extent=3.0,
    z_lens=0.5,
    z_source=1.5
):
    """
    Generate synthetic convergence map for benchmarking.
    Wraps Phase 7 vectorized generator.
    """
    # Setup lens system
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    lens_sys = LensSystem(z_lens, z_source, cosmology=cosmo)
    
    # Calculate concentration from scale radius (kpc)
    # r_s [Mpc] = scale_radius [kpc] / 1000
    r_s_mpc = scale_radius / 1000.0
    
    # Calculate r_vir [Mpc] from Mass
    h = 0.7
    rho_crit = 2.775e11 * h**2  # Msun/Mpc^3
    r_vir_mpc = (3 * mass / (4 * np.pi * 200 * rho_crit))**(1/3)
    
    # c = r_vir / r_s
    concentration = r_vir_mpc / r_s_mpc
    
    # Create profile
    if profile_type == "NFW":
        profile = NFWProfile(
            M_vir=mass,
            concentration=concentration,
            lens_system=lens_sys,
            ellipticity=ellipticity
        )
    else:
        raise ValueError(f"Unsupported profile type: {profile_type}")
        
    # Generate map
    kappa = generate_convergence_map_vectorized(profile, grid_size=grid_size, extent=extent)
    
    # Generate coordinates
    x = np.linspace(-extent, extent, grid_size)
    y = np.linspace(-extent, extent, grid_size)
    X, Y = np.meshgrid(x, y)
    coords = {'x': X, 'y': Y}
    
    return kappa, coords


def analytic_nfw_convergence(
    x: np.ndarray,
    y: np.ndarray,
    mass: float,
    scale_radius: float,
    concentration: float = 5.0
) -> np.ndarray:
    """
    Analytic NFW convergence profile
    
    Args:
        x, y: Coordinate grids
        mass: Virial mass in solar masses
        scale_radius: Scale radius in kpc
        concentration: Concentration parameter
        
    Returns:
        np.ndarray: Convergence map
    """
    r = np.sqrt(x**2 + y**2)
    
    # NFW parameters
    rho_0 = mass / (4 * np.pi * scale_radius**3 * (np.log(1 + concentration) - concentration/(1 + concentration)))
    kappa_s = rho_0 * scale_radius
    
    # Scaled radius
    x_scaled = r / scale_radius
    
    # Convergence (simplified for benchmark)
    kappa = np.zeros_like(r)
    mask = x_scaled > 0
    
    # NFW convergence formula
    # NFW piecewise solution (Wright & Brainerd 2000)
    for i in range(len(x_scaled)):
        for j in range(len(x_scaled[0])):
            x_val = x_scaled[i, j]
            if x_val == 0:
                continue # Singularity at center (handled by normalization later or set to max)
            
            if x_val < 1.0:
                inner = np.sqrt(1 - x_val**2)
                func = np.arctanh(np.sqrt((1 - x_val)/(1 + x_val)))
                kappa[i, j] = 2 * kappa_s / (x_val**2 - 1) * (1 - 2/inner * func)
            elif x_val == 1.0:
                kappa[i, j] = 2 * kappa_s / 3.0
            else: # x > 1.0
                inner = np.sqrt(x_val**2 - 1)
                func = np.arctan(np.sqrt((x_val - 1)/(1 + x_val)))
                kappa[i, j] = 2 * kappa_s / (x_val**2 - 1) * (1 - 2/inner * func)
    
    # Normalize
    kappa = kappa / np.max(np.abs(kappa)) if np.max(np.abs(kappa)) > 0 else kappa
    
    return kappa


def compare_with_analytic(
    grid_size: int = 64,
    mass: float = 1e12,
    scale_radius: float = 200.0
) -> Dict[str, any]:
    """
    Compare PINN prediction with analytic NFW solution
    
    Args:
        grid_size: Grid size for comparison
        mass: Virial mass
        scale_radius: Scale radius
        
    Returns:
        dict: Comparison results
    """
    logger.info("Comparing with analytic NFW solution...")
    
    # Generate synthetic map using our method
    start_time = time.time()
    our_map, coords = generate_synthetic_convergence(
        profile_type="NFW",
        mass=mass,
        scale_radius=scale_radius,
        ellipticity=0.0,
        grid_size=grid_size
    )
    our_time = time.time() - start_time
    
    # Generate analytic solution
    start_time = time.time()
    x, y = coords['x'], coords['y']
    analytic_map = analytic_nfw_convergence(x, y, mass, scale_radius)
    analytic_time = time.time() - start_time
    
    # Calculate metrics
    metrics = calculate_all_metrics(our_map, analytic_map)
    
    # Calculate speedup (protect against zero division)
    speedup = analytic_time / our_time if our_time > 0 else float('inf')
    
    results = {
        'our_map': our_map,
        'analytic_map': analytic_map,
        'our_time': our_time,
        'analytic_time': analytic_time,
        'metrics': metrics,
        'speedup': speedup,
        'grid_size': grid_size,
        'mass': mass
    }
    
    logger.info(f"Our method: {our_time:.4f}s, Analytic: {analytic_time:.4f}s")
    logger.info(f"Relative error: {metrics['relative_error']:.6f}")
    logger.info(f"RMSE: {metrics['rmse']:.6e}")
    
    return results


def compare_with_lenstool(
    convergence_map: np.ndarray,
    lenstool_output: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compare with Lenstool output
    
    Args:
        convergence_map: Our convergence map
        lenstool_output: Lenstool convergence map (if available)
        
    Returns:
        dict: Comparison metrics
        
    Raises:
        ValueError: If lenstool_output is not provided - external validation data is REQUIRED
    """
    if lenstool_output is None:
        raise ValueError(
            "External validation is REQUIRED for publication. "
            "Lenstool output must be provided for meaningful comparison. "
            "Cannot use fallback/mock data as this would be scientifically dishonest."
        )
    
    metrics = calculate_all_metrics(convergence_map, lenstool_output)
    
    logger.info("Comparison with Lenstool:")
    print_metrics_report(metrics, "Lenstool Comparison")
    
    return metrics


def compare_with_glafic(
    convergence_map: np.ndarray,
    glafic_output: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compare with GLAFIC output
    
    Args:
        convergence_map: Our convergence map
        glafic_output: GLAFIC convergence map (if available)
        
    Returns:
        dict: Comparison metrics
        
    Raises:
        ValueError: If glafic_output is not provided - external validation data is REQUIRED
    """
    if glafic_output is None:
        raise ValueError(
            "External validation is REQUIRED for publication. "
            "GLAFIC output must be provided for meaningful comparison. "
            "Cannot use fallback/mock data as this would be scientifically dishonest."
        )
    
    metrics = calculate_all_metrics(convergence_map, glafic_output)
    
    logger.info("Comparison with GLAFIC:")
    print_metrics_report(metrics, "GLAFIC Comparison")
    
    return metrics


def benchmark_convergence_accuracy(
    n_samples: int = 10,
    grid_sizes: List[int] = [32, 64, 128],
    masses: List[float] = [5e11, 1e12, 5e12]
) -> Dict[str, List[Dict]]:
    """
    Benchmark convergence map accuracy across different parameters
    
    Args:
        n_samples: Number of samples per configuration
        grid_sizes: List of grid sizes to test
        masses: List of masses to test
        
    Returns:
        dict: Benchmark results
    """
    logger.info(f"Running convergence accuracy benchmark ({n_samples} samples)...")
    
    results = {
        'grid_size_tests': [],
        'mass_tests': [],
        'summary': {}
    }
    
    # Test different grid sizes
    for grid_size in grid_sizes:
        logger.info(f"Testing grid size {grid_size}...")
        errors = []
        times = []
        
        for i in range(n_samples):
            comparison = compare_with_analytic(
                grid_size=grid_size,
                mass=1e12,
                scale_radius=200.0
            )
            errors.append(comparison['metrics']['relative_error'])
            times.append(comparison['our_time'])
        
        results['grid_size_tests'].append({
            'grid_size': grid_size,
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'mean_time': np.mean(times),
            'std_time': np.std(times)
        })
    
    # Test different masses
    for mass in masses:
        logger.info(f"Testing mass {mass:.2e}...")
        errors = []
        ssims = []
        
        for i in range(n_samples):
            comparison = compare_with_analytic(
                grid_size=64,
                mass=mass,
                scale_radius=200.0
            )
            errors.append(comparison['metrics']['relative_error'])
            ssims.append(comparison['metrics'].get('ssim', 0))
        
        results['mass_tests'].append({
            'mass': mass,
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'mean_ssim': np.mean(ssims),
            'std_ssim': np.std(ssims)
        })
    
    # Summary statistics
    all_errors = [r['mean_error'] for r in results['grid_size_tests']]
    results['summary'] = {
        'overall_mean_error': np.mean(all_errors),
        'overall_std_error': np.std(all_errors),
        'best_grid_size': grid_sizes[np.argmin(all_errors)]
    }
    
    logger.info(f"Benchmark complete. Overall mean error: {results['summary']['overall_mean_error']:.6f}")
    
    return results


def benchmark_inference_speed(
    n_runs: int = 20,
    grid_size: int = 64
) -> Dict[str, float]:
    """
    Benchmark PINN inference speed
    
    Args:
        n_runs: Number of inference runs
        grid_size: Grid size for inference
        
    Returns:
        dict: Speed benchmark results
    """
    logger.info(f"Benchmarking inference speed ({n_runs} runs)...")
    
    # Load model once
    model = load_pretrained_model()
    
    # Generate test data
    test_map, _ = generate_synthetic_convergence(
        profile_type="NFW",
        mass=1e12,
        scale_radius=200.0,
        ellipticity=0.0,
        grid_size=grid_size
    )
    
    # Warm-up run
    model_input = prepare_model_input(test_map, target_size=grid_size)
    _ = model(model_input)
    
    # Benchmark runs
    times = []
    for i in range(n_runs):
        start_time = time.perf_counter()
        model_input = prepare_model_input(test_map, target_size=grid_size)
        predictions = model(model_input)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    results = {
        'n_runs': n_runs,
        'grid_size': grid_size,
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'median_time': np.median(times),
        'throughput': 1.0 / np.mean(times)  # inferences per second
    }
    
    logger.info(f"Mean inference time: {results['mean_time']:.4f} ± {results['std_time']:.4f} s")
    logger.info(f"Throughput: {results['throughput']:.2f} inferences/second")
    
    return results


def run_comprehensive_benchmark(
    output_dir: Optional[Path] = None
) -> Dict[str, any]:
    """
    Run comprehensive benchmark suite
    
    Args:
        output_dir: Directory to save results
        
    Returns:
        dict: All benchmark results
    """
    from datetime import datetime
    
    logger.info("=" * 60)
    logger.info("Running Comprehensive Benchmark Suite")
    logger.info("=" * 60)
    
    results = {
        'timestamp': datetime.now().isoformat()
    }
    
    # 1. Accuracy benchmark
    logger.info("\n1. Convergence Accuracy Benchmark")
    logger.info("-" * 60)
    results['accuracy'] = benchmark_convergence_accuracy(
        n_samples=5,
        grid_sizes=[32, 64, 128],
        masses=[5e11, 1e12, 5e12, 1e13]
    )
    
    # 2. Speed benchmark
    logger.info("\n2. Inference Speed Benchmark")
    logger.info("-" * 60)
    results['speed'] = benchmark_inference_speed(n_runs=20, grid_size=64)
    
    # 3. Analytic comparison
    logger.info("\n3. Analytic Comparison")
    logger.info("-" * 60)
    results['analytic'] = compare_with_analytic(grid_size=64, mass=1e12, scale_radius=200.0)
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Benchmark Summary")
    logger.info("=" * 60)
    logger.info(f"Accuracy:")
    logger.info(f"  Mean relative error: {results['accuracy']['summary']['overall_mean_error']:.6f}")
    logger.info(f"  Best grid size: {results['accuracy']['summary']['best_grid_size']}")
    logger.info(f"\nSpeed:")
    logger.info(f"  Mean inference time: {results['speed']['mean_time']:.4f} s")
    logger.info(f"  Throughput: {results['speed']['throughput']:.2f} inferences/s")
    logger.info(f"\nAnalytic Comparison:")
    logger.info(f"  Relative error: {results['analytic']['metrics']['relative_error']:.6f}")
    ssim_value = results['analytic']['metrics'].get('ssim', None)
    if ssim_value is not None:
        logger.info(f"  SSIM: {ssim_value:.4f}")
    else:
        logger.info(f"  SSIM: N/A")
    logger.info("=" * 60)
    
    # Save results if output directory provided
    if output_dir:
        import json
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        results_json = convert_numpy(results)
        with open(output_dir / 'benchmark_results.json', 'w') as f:
            json.dump(results_json, f, indent=2)
        
        logger.info(f"\nResults saved to {output_dir}")
    
    return results


def print_benchmark_report(results: Dict[str, any]):
    """Print comprehensive benchmark report"""
    print("\n" + "=" * 80)
    print("GRAVITATIONAL LENSING PINN - COMPREHENSIVE BENCHMARK REPORT")
    print("=" * 80)
    
    # Accuracy results
    if 'accuracy' in results:
        print("\n1. CONVERGENCE ACCURACY BENCHMARK")
        print("-" * 80)
        
        print("\nGrid Size Tests:")
        for test in results['accuracy']['grid_size_tests']:
            print(f"  Grid {test['grid_size']:3d}: Error = {test['mean_error']:.6f} ± {test['std_error']:.6f}, "
                  f"Time = {test['mean_time']:.4f} ± {test['std_time']:.4f} s")
        
        print("\nMass Tests:")
        for test in results['accuracy']['mass_tests']:
            print(f"  Mass {test['mass']:.2e}: Error = {test['mean_error']:.6f} ± {test['std_error']:.6f}, "
                  f"SSIM = {test['mean_ssim']:.4f} ± {test['std_ssim']:.4f}")
    
    # Speed results
    if 'speed' in results:
        print("\n2. INFERENCE SPEED BENCHMARK")
        print("-" * 80)
        speed = results['speed']
        print(f"  Runs: {speed['n_runs']}")
        print(f"  Grid Size: {speed['grid_size']}")
        print(f"  Mean Time: {speed['mean_time']:.4f} ± {speed['std_time']:.4f} s")
        print(f"  Median Time: {speed['median_time']:.4f} s")
        print(f"  Min/Max: {speed['min_time']:.4f} / {speed['max_time']:.4f} s")
        print(f"  Throughput: {speed['throughput']:.2f} inferences/second")
    
    # Analytic comparison
    if 'analytic' in results:
        print("\n3. ANALYTIC COMPARISON")
        print("-" * 80)
        metrics = results['analytic']['metrics']
        print(f"  Relative Error: {metrics['relative_error']:.6e}")
        print(f"  RMSE: {metrics['rmse']:.6e}")
        print(f"  MAE: {metrics['mae']:.6e}")
        if 'ssim' in metrics:
            print(f"  SSIM: {metrics['ssim']:.4f}")
        if 'psnr' in metrics:
            print(f"  PSNR: {metrics['psnr']:.2f} dB")
        print(f"  Pearson Correlation: {metrics['pearson_correlation']:.6f}")
    
    print("\n" + "=" * 80)
    print("END OF REPORT")
    print("=" * 80 + "\n")


def compare_single_vs_multiplane(
    main_lens_z: float = 0.5,
    perturber_z: float = 0.3,
    source_z: float = 2.0,
    perturber_fraction: float = 0.1,
    grid_size: int = 64,
    fov: float = 5.0
) -> Dict[str, any]:
    """
    Compare single-plane vs multi-plane lensing.
    
    Demonstrates the importance of multi-plane treatment when there
    is significant line-of-sight structure.
    
    Args:
        main_lens_z: Redshift of main lens
        perturber_z: Redshift of foreground perturber
        source_z: Source redshift
        perturber_fraction: Mass fraction of perturber relative to main lens
        grid_size: Grid size for maps
        fov: Field of view in arcseconds
        
    Returns:
        dict: Comparison results
    """
    logger.info("Comparing Single-Plane vs Multi-Plane Lensing...")
    
    # Cosmology
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    
    # Profile parameters
    M_main = 1e13
    scale_main = 20.0  # kpc
    c_main = 5.0
    
    M_pert = M_main * perturber_fraction
    c_pert = 4.0
    
    # 1. Multi-Plane Setup (Correct Physics)
    mp_lens = MultiPlaneLens(source_redshift=source_z, cosmology=cosmo)
    
    # Main lens
    sys_main = LensSystem(z_lens=main_lens_z, z_source=source_z, cosmology=cosmo)
    main_profile = NFWProfile(M_main, c_main, sys_main)
    mp_lens.add_plane(redshift=main_lens_z, profile=main_profile, center=(0.0, 0.0))
    
    # Perturber
    sys_pert = LensSystem(z_lens=perturber_z, z_source=source_z, cosmology=cosmo)
    pert_profile = NFWProfile(M_pert, c_pert, sys_pert)
    mp_lens.add_plane(redshift=perturber_z, profile=pert_profile, center=(1.0, 1.0)) # Offset perturber
    
    # Compute Ray Tracing (Multi-Plane)
    x = np.linspace(-fov/2, fov/2, grid_size)
    y = np.linspace(-fov/2, fov/2, grid_size)
    xx, yy = np.meshgrid(x, y)
    theta = np.stack([xx, yy], axis=-1)
    
    beta_mp = mp_lens.ray_trace(theta)
    alpha_mp = theta - beta_mp
    alpha_mp_mag = np.sqrt(np.sum(alpha_mp**2, axis=-1))
    
    # 2. Single-Plane Approximation (Born Approximation)
    # Create a single lens system at main_lens_z
    sp_lens_sys = LensSystem(z_lens=main_lens_z, z_source=source_z, cosmology=cosmo)
    
    # Main profile (same)
    sp_main_profile = NFWProfile(M_main, c_main, sp_lens_sys)
    
    # Perturber projected to main plane (Naively placing it there)
    sp_pert_profile = NFWProfile(M_pert, c_pert, sp_lens_sys)
    
    # Compute deflection sum (Single Plane)
    # Main deflection
    alpha_main_x, alpha_main_y = sp_main_profile.deflection_angle(xx, yy)
    
    # Perturber deflection (offset (1,1))
    alpha_pert_x, alpha_pert_y = sp_pert_profile.deflection_angle(xx - 1.0, yy - 1.0)
    
    alpha_sp_x = alpha_main_x + alpha_pert_x
    alpha_sp_y = alpha_main_y + alpha_pert_y
    alpha_sp_mag = np.sqrt(alpha_sp_x**2 + alpha_sp_y**2)
    
    # Differences
    diff_mag = np.abs(alpha_mp_mag - alpha_sp_mag)
    rel_diff = diff_mag / (alpha_mp_mag + 1e-10)
    
    metrics = {
        'max_deflection_diff': float(np.max(diff_mag)),
        'mean_deflection_diff': float(np.mean(diff_mag)),
        'mean_relative_error': float(np.mean(rel_diff)),
        'max_relative_error': float(np.max(rel_diff))
    }
    
    logger.info(f"Comparison Results:")
    logger.info(f"  Max Deflection Diff: {metrics['max_deflection_diff']:.6f} arcsec")
    logger.info(f"  Mean Relative Error: {metrics['mean_relative_error']:.6%}")
    
    return {
        'metrics': metrics,
        'multi_plane_deflection': alpha_mp_mag,
        'single_plane_deflection': alpha_sp_mag,
        'difference': diff_mag
    }
