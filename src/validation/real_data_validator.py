"""
Real Data Validation Pipeline

Uses real gravitational lens observational data from MAST and published literature
for validation. This is REQUIRED for scientific publication.

Data sources:
- HST Archive (MAST): Real observations
- Published lens parameters from peer-reviewed papers
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

# Data directory
DATA_DIR = Path("data/real_data")

class RealDataValidator:
    """
    Validates PINN predictions against REAL observational data.
    
    This is the scientifically correct approach - no mocks, no fallbacks.
    """
    
    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir
        self.lens_catalog = self._load_lens_catalog()
        
    def _load_lens_catalog(self) -> Dict:
        """Load lens catalog from real data."""
        catalog_file = self.data_dir / "lens_catalog.json"
        if not catalog_file.exists():
            raise FileNotFoundError(
                f"Lens catalog not found at {catalog_file}. "
                "Run scripts/get_real_data.py first."
            )
        with open(catalog_file) as f:
            return json.load(f)
    
    def get_lens_data(self, lens_name: str) -> Dict:
        """Get real lens data by name."""
        # Try exact match first
        if lens_name in self.lens_catalog:
            return self._load_lens_file(lens_name)
        
        # Try fuzzy match
        for key in self.lens_catalog.keys():
            if lens_name.lower().replace("_", " ") in key.lower().replace("_", " "):
                return self._load_lens_file(key)
        
        raise ValueError(f"Lens {lens_name} not found in catalog")
    
    def _load_lens_file(self, lens_key: str) -> Dict:
        """Load individual lens data file."""
        # Convert lens name to filename
        filename = lens_key.lower().replace("+", "p").replace("-", "m").replace(".", "d")
        lens_file = self.data_dir / f"{filename}.json"
        
        if not lens_file.exists():
            raise FileNotFoundError(f"Lens data file not found: {lens_file}")
        
        with open(lens_file) as f:
            return json.load(f)
    
    def validate_against_real_data(
        self,
        predicted_params: Dict,
        lens_name: str
    ) -> Dict:
        """
        Validate predictions against real observational data.
        
        Parameters
        ----------
        predicted_params : dict
            Predicted parameters from PINN:
            - 'einstein_radius': predicted Einstein radius (arcsec)
            - 'mass': predicted mass (M_sun)
            - 'convergence_map': predicted convergence map
        lens_name : str
            Name of lens system to validate against
            
        Returns
        -------
        validation : dict
            Validation metrics comparing predictions to real data
        """
        # Get real lens data
        real_data = self.get_lens_data(lens_name)
        
        # Extract true values
        true_theta_e = real_data['einstein_radius_arcsec']
        true_mass = real_data['mass_msun']
        
        # Get predictions
        pred_theta_e = predicted_params.get('einstein_radius', None)
        pred_mass = predicted_params.get('mass', None)
        
        if pred_theta_e is None:
            raise ValueError("predicted_params must include 'einstein_radius'")
        if pred_mass is None:
            raise ValueError("predicted_params must include 'mass'")
        
        # Calculate errors
        theta_e_error = abs(pred_theta_e - true_theta_e) / true_theta_e * 100
        mass_error = abs(pred_mass - true_mass) / true_mass * 100
        
        # Get additional metadata
        z_lens = real_data['z_lens']
        z_source = real_data['z_source']
        
        return {
            'lens_name': lens_name,
            'z_lens': z_lens,
            'z_source': z_source,
            'true_einstein_radius': true_theta_e,
            'predicted_einstein_radius': pred_theta_e,
            'einstein_radius_error_percent': theta_e_error,
            'true_mass_msun': true_mass,
            'predicted_mass_msun': pred_mass,
            'mass_error_percent': mass_error,
            'source': real_data.get('source', 'Unknown'),
            'paper_reference': real_data.get('paper_reference', 'Unknown'),
            'passed': theta_e_error < 10 and mass_error < 20
        }
    
    def get_all_lenses(self) -> List[str]:
        """Get list of all available lens systems."""
        return list(self.lens_catalog.keys())
    
    def print_validation_report(self, results: Dict):
        """Print formatted validation report."""
        print("\n" + "=" * 70)
        print("REAL DATA VALIDATION REPORT")
        print("=" * 70)
        print(f"\nLens: {results['lens_name']}")
        print(f"Redshifts: z_l = {results['z_lens']}, z_s = {results['z_source']}")
        print(f"Data Source: {results['source']}")
        print(f"Reference: {results['paper_reference']}")
        print("\n" + "-" * 70)
        print("EINSTEIN RADIUS:")
        print(f"  True:      {results['true_einstein_radius']:.3f}\"")
        print(f"  Predicted: {results['predicted_einstein_radius']:.3f}\"")
        print(f"  Error:     {results['einstein_radius_error_percent']:.2f}%")
        print("\nMASS:")
        print(f"  True:      {results['true_mass_msun']:.2e} M_sun")
        print(f"  Predicted: {results['predicted_mass_msun']:.2e} M_sun")
        print(f"  Error:     {results['mass_error_percent']:.2f}%")
        print("\n" + "-" * 70)
        print(f"VALIDATION: {'PASSED ✓' if results['passed'] else 'FAILED ✗'}")
        print("=" * 70)


def load_real_lens_for_testing(lens_name: str = "SDSS_J1004+4112") -> Dict:
    """
    Load real lens data for testing/validation.
    
    This function is used by the validation pipeline.
    """
    validator = RealDataValidator()
    return validator.get_lens_data(lens_name)


if __name__ == "__main__":
    # Demo: validate against all real lenses
    print("\n" + "=" * 70)
    print("REAL DATA VALIDATION DEMO")
    print("=" * 70)
    
    validator = RealDataValidator()
    lenses = validator.get_all_lenses()
    
    print(f"\nAvailable lenses ({len(lenses)}):")
    for lens in lenses:
        data = validator.lens_catalog[lens]
        print(f"  - {lens}: z_l={data['z_lens']}, θ_E={data['einstein_radius_arcsec']}\"")
    
    print("\n" + "=" * 70)
    print("To validate PINN predictions:")
    print("1. Load real lens: validator = RealDataValidator()")
    print("2. Get lens data: lens_data = validator.get_lens_data('SDSS_J1004+4112')")
    print("3. Run PINN prediction")
    print("4. Compare: validator.validate_against_real_data(predictions, 'SDSS_J1004+4112')")
    print("=" * 70)
