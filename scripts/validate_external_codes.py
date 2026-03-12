#!/usr/bin/env python3
"""
External Code Validation Pipeline

Compares PINN predictions with external lens modeling codes:
- Lenstool
- GLAFIC
- LensModel
- PIEMD

Usage:
    python validate_with_external_codes.py --our_predictions <file> --code <lenstool|glafic>
"""

import argparse
import json
import numpy as np
from pathlib import Path

def load_our_predictions(pred_file):
    """Load our PINN predictions."""
    with open(pred_file) as f:
        return json.load(f)

def load_external_predictions(code_name, pred_file):
    """Load external code predictions."""
    # Implementation depends on file format
    pass

def compare_predictions(our_pred, ext_pred, metric="rmse"):
    """Compare predictions using specified metric."""
    if metric == "rmse":
        return np.sqrt(np.mean((our_pred - ext_pred)**2))
    elif metric == "mae":
        return np.mean(np.abs(our_pred - ext_pred))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def main():
    parser = argparse.ArgumentParser(description="External code validation")
    parser.add_argument("--our_predictions", required=True, help="Our predictions JSON")
    parser.add_argument("--code", required=True, choices=["lenstool", "glafic", "lensmodel", "piemd"])
    parser.add_argument("--external_predictions", required=True, help="External code output")
    parser.add_argument("--metric", default="rmse", choices=["rmse", "mae", "correlation"])
    
    args = parser.parse_args()
    
    # Load predictions
    our_pred = load_our_predictions(args.our_predictions)
    ext_pred = load_external_predictions(args.code, args.external_predictions)
    
    # Compare
    error = compare_predictions(our_pred, ext_pred, args.metric)
    
    print(f"Comparison with {args.code}:")
    print(f"  {args.metric.upper()}: {error:.4f}")
    
    return error

if __name__ == "__main__":
    main()
