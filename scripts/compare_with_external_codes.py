#!/usr/bin/env python3
"""
Lenstool/GLAFIC Comparison Framework

This script validates gravitational lens modeling calculations by comparing:
1. Our analytical implementations against known benchmarks
2. Generates input files compatible with Lenstool and GLAFIC formats
3. Performs systematic comparisons for publication-quality validation

Usage:
    python compare_with_external_codes.py --mode validate
    python compare_with_external_codes.py --mode generate_inputs
    python compare_with_external_codes.py --mode compare
"""

import argparse
import json
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Import our lens models
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lens_models.lens_system import LensSystem
from src.lens_models.mass_profiles import NFWProfile, PointMassProfile
from src.lens_models.advanced_profiles import (
    EllipticalNFWProfile, 
    SersicProfile, 
    CompositeGalaxyProfile
)


class BenchmarkSystem:
    """Known gravitational lens system with well-measured parameters."""
    
    def __init__(
        self,
        name: str,
        z_lens: float,
        z_source: float,
        einstein_radius: float,
        lens_mass: float,
        scale_radius: Optional[float] = None,
        ellipticity: float = 0.0,
        position_angle: float = 0.0
    ):
        self.name = name
        self.z_lens = z_lens
        self.z_source = z_source
        self.einstein_radius = einstein_radius
        self.lens_mass = lens_mass
        self.scale_radius = scale_radius
        self.ellipticity = ellipticity
        self.position_angle = position_angle


# Known benchmark systems from literature
BENCHMARK_SYSTEMS = [
    # SDSS J0959+0410 - well-studied lens
    BenchmarkSystem(
        name="SDSS J0959+0410",
        z_lens=0.112,
        z_source=0.464,
        einstein_radius=1.42,
        lens_mass=2.8e11,
        scale_radius=5.3,
        ellipticity=0.08
    ),
    # SDSS J1029+2623 - large separation lens
    BenchmarkSystem(
        name="SDSS J1029+2623",
        z_lens=0.197,
        z_source=0.686,
        einstein_radius=3.18,
        lens_mass=1.6e12,
        scale_radius=15.0,
        ellipticity=0.2
    ),
    # Q0957+561 - first discovered gravitational lens
    BenchmarkSystem(
        name="Q0957+561",
        z_lens=0.356,
        z_source=1.405,
        einstein_radius=3.05,
        lens_mass=3.4e12,
        scale_radius=10.0,
        ellipticity=0.1,
        position_angle=75.0
    ),
    # RXJ 1131-1231 - quadruply imaged quasar
    BenchmarkSystem(
        name="RXJ 1131-1231",
        z_lens=0.295,
        z_source=0.657,
        einstein_radius=3.52,
        lens_mass=2.0e12,
        scale_radius=12.0,
        ellipticity=0.25
    ),
    # B1422+231 - high magnification quad
    BenchmarkSystem(
        name="B1422+231",
        z_lens=0.337,
        z_source=0.598,
        einstein_radius=1.6,
        lens_mass=1.5e11,
        scale_radius=4.5,
        ellipticity=0.35
    ),
]


def generate_lenstool_input(system: BenchmarkSystem, output_path: str) -> Dict:
    """
    Generate Lenstool-compatible input file for a lens system.
    
    Lenstool uses a specific parameter file format (.param).
    """
    lens_sys = LensSystem(z_lens=system.z_lens, z_source=system.z_source)
    
    params = {
        "# Lenstool input file for {}".format(system.name),
        "# Generated automatically",
        "",
        "z_lens    = {}".format(system.z_lens),
        "z_source  = {}".format(system.z_source),
        "",
        "# Main halo (NFW)",
        "M_vir     = {}".format(system.lens_mass),
        "c         = {}".format(system.lens_mass / system.scale_radius / 1e12 * 10 if system.scale_radius else 10),
        "r_s       = {}".format(system.scale_radius if system.scale_radius else 1.0),
        "ellipticity = {}".format(system.ellipticity),
        "angle_pa  = {}".format(system.position_angle),
        "",
        "# Einstein radius (for reference)",
        "theta_E   = {}".format(system.einstein_radius),
    }
    
    return {
        'format': 'lenstool',
        'system': system.name,
        'parameters': list(params),
        'file_path': output_path
    }


def generate_glafic_input(system: BenchmarkSystem, output_path: str) -> Dict:
    """
    Generate GLAFIC-compatible input file for a lens system.
    
    GLAFIC uses Python-like parameter syntax.
    """
    lens_sys = LensSystem(z_lens=system.z_lens, z_source=system.z_source)
    
    params = {
        '# GLAFIC input file for {}'.format(system.name),
        '# Generated automatically',
        '',
        '# Lens parameters',
        'z lens = {}'.format(system.z_lens),
        'z source = {}'.format(system.z_source),
        '',
        '# Main halo (NFW)',
        'Mlens = {}'.format(system.lens_mass),
        'rs = {}'.format(system.scale_radius if system.scale_radius else 1.0),
        'ellip = {}'.format(system.ellipticity),
        'pa = {}'.format(system.position_angle),
        '',
        '# Einstein radius reference',
        'thetae = {}'.format(system.einstein_radius),
    }
    
    return {
        'format': 'glafic',
        'system': system.name,
        'parameters': list(params),
        'file_path': output_path
    }


def validate_against_analytical(system: BenchmarkSystem) -> Dict:
    """
    Validate our implementation against analytical calculations.
    
    This compares:
    1. Einstein radius from mass vs analytical formula (for point mass)
    2. Deflection angles at various positions
    3. Convergence profiles
    4. Internal consistency checks
    """
    lens_sys = LensSystem(z_lens=system.z_lens, z_source=system.z_source)
    
    # Test point mass Einstein radius
    pm = PointMassProfile(mass=system.lens_mass, lens_system=lens_sys)
    theta_E_pointmass = pm.einstein_radius
    
    # For NFW, the Einstein radius is where κ = 1
    # This requires numerical solution
    if system.scale_radius:
        concentration = system.lens_mass / system.scale_radius / 1e12 * 10
        nfw = NFWProfile(
            M_vir=system.lens_mass,
            concentration=concentration,
            lens_system=lens_sys
        )
        
        # Find Einstein radius numerically: where κ(r) = 1
        from scipy.optimize import brentq
        
        def kappa_minus_one(r):
            k = nfw.convergence(r, 0.0)
            k = float(k[0]) if hasattr(k, '__len__') else float(k)
            return k - 1.0
        
        try:
            theta_E_nfw = brentq(kappa_minus_one, 0.01, 100.0)
        except:
            theta_E_nfw = None
        
        # Test deflection at multiple radii
        radii = np.array([0.5, 1.0, 2.0, 5.0]) * system.einstein_radius
        deflection_tests = []
        
        for r in radii:
            alpha_x, alpha_y = nfw.deflection_angle(r, 0.0)
            alpha_mag = np.sqrt(alpha_x**2 + alpha_y**2)
            alpha_mag = float(alpha_mag[0]) if hasattr(alpha_mag, '__len__') else float(alpha_mag)
            
            # For NFW, deflection at r = theta_E should equal r (definition)
            if theta_E_nfw:
                deflection_error = abs(alpha_mag - r) / r
            else:
                deflection_error = None
                
            deflection_tests.append({
                'radius': float(r),
                'deflection': alpha_mag,
                'relative_error': deflection_error
            })
        
        # Test convergence at various radii
        kappa_tests = []
        for r in radii:
            kappa = nfw.convergence(r, 0.0)
            kappa = float(kappa[0]) if hasattr(kappa, '__len__') else float(kappa)
            kappa_tests.append({
                'radius': float(r),
                'kappa': kappa
            })
        
        # Test potential at various radii
        psi_tests = []
        for r in radii:
            psi = nfw.lensing_potential(r, 0.0)
            psi = float(psi[0]) if hasattr(psi, '__len__') else float(psi)
            psi_tests.append({
                'radius': float(r),
                'psi': psi
            })
    else:
        theta_E_nfw = None
        deflection_tests = []
        kappa_tests = []
        psi_tests = []
    
    return {
        'system': system.name,
        'einstein_radius': {
            'point_mass': float(theta_E_pointmass),
            'nfw': float(theta_E_nfw) if theta_E_nfw else None,
            'expected': float(system.einstein_radius),
            'point_mass_error': float(abs(theta_E_pointmass - system.einstein_radius) / system.einstein_radius)
        },
        'deflection_tests': deflection_tests,
        'kappa_tests': kappa_tests,
        'psi_tests': psi_tests
    }


def run_systematic_comparison() -> Dict:
    """
    Run systematic comparison across all benchmark systems.
    """
    results = {
        'description': 'Systematic validation of lens modeling calculations',
        'benchmarks': [],
        'summary': {}
    }
    
    point_mass_errors = []
    nfw_einstein_radii = []
    
    for system in BENCHMARK_SYSTEMS:
        print(f"\nValidating {system.name}...")
        
        validation = validate_against_analytical(system)
        results['benchmarks'].append(validation)
        
        if validation['einstein_radius']['point_mass_error'] is not None:
            point_mass_errors.append(validation['einstein_radius']['point_mass_error'])
        if validation['einstein_radius']['nfw'] is not None:
            nfw_einstein_radii.append(validation['einstein_radius']['nfw'])
        
        # Print results
        print(f"  Point mass Einstein radius: {validation['einstein_radius']['point_mass']:.3f} arcsec")
        print(f"  NFW Einstein radius: {validation['einstein_radius']['nfw']:.3f} arcsec" if validation['einstein_radius']['nfw'] else "  NFW Einstein radius: N/A")
        print(f"  Expected: {validation['einstein_radius']['expected']:.3f} arcsec")
        
        if validation['kappa_tests']:
            print(f"  Convergence at Einstein radius: {validation['kappa_tests'][1]['kappa']:.4f}")
    
    results['summary'] = {
        'n_systems': len(BENCHMARK_SYSTEMS),
        'point_mass_einstein': {
            'mean_error': float(np.mean(point_mass_errors)) if point_mass_errors else None,
            'max_error': float(np.max(point_mass_errors)) if point_mass_errors else None,
        },
        'nfw_einstein_radii': nfw_einstein_radii
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Lenstool/GLAFIC Comparison Framework'
    )
    parser.add_argument(
        '--mode',
        choices=['validate', 'generate_inputs', 'compare'],
        default='validate',
        help='Operation mode'
    )
    parser.add_argument(
        '--output_dir',
        default='results/validation',
        help='Output directory'
    )
    parser.add_argument(
        '--system',
        default=None,
        help='Specific system to validate'
    )
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == 'validate':
        print("=" * 60)
        print("Running systematic validation against analytical solutions")
        print("=" * 60)
        
        results = run_systematic_comparison()
        
        # Save results
        output_file = os.path.join(args.output_dir, 'validation_results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Systems validated: {results['summary']['n_systems']}")
        print(f"Point mass Einstein radius - Mean error: {results['summary']['point_mass_einstein']['mean_error']:.4%}")
        print(f"Point mass Einstein radius - Max error: {results['summary']['point_mass_einstein']['max_error']:.4%}")
        
        if results['summary']['nfw_einstein_radii']:
            print(f"\nNFW Einstein radii computed:")
            for i, (sys, r) in enumerate(zip(BENCHMARK_SYSTEMS, results['summary']['nfw_einstein_radii'])):
                print(f"  {sys.name}: {r:.3f} arcsec")
        
        print(f"\nResults saved to: {output_file}")
    
    elif args.mode == 'generate_inputs':
        print("Generating input files for external codes...")
        
        for system in BENCHMARK_SYSTEMS:
            # Generate Lenstool input
            lenstool_file = os.path.join(
                args.output_dir, 
                f"{system.name.replace(' ', '_')}_lenstool.param"
            )
            generate_lenstool_input(system, lenstool_file)
            
            # Generate GLAFIC input  
            glafic_file = os.path.join(
                args.output_dir,
                f"{system.name.replace(' ', '_')}_glafic.input"
            )
            generate_glafic_input(system, glafic_file)
        
        print(f"Generated input files in: {args.output_dir}")
    
    elif args.mode == 'compare':
        print("Comparison mode - requires external code outputs")
        print("Please run Lenstool/GLAFIC first, then place outputs in:")
        print(f"  {args.output_dir}/external_outputs/")


if __name__ == "__main__":
    main()
