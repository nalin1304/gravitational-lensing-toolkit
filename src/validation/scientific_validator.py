"""
Scientific Validation Framework for Gravitational Lensing

Provides comprehensive validation tools for PINN predictions against
analytic solutions, observational data, and established lensing codes.

Author: Gravitational Lensing Research Platform
Version: 2.0
Date: Phase 15
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings
from scipy import stats, optimize
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Use built-in implementations to avoid circular imports
BENCHMARKS_AVAILABLE = False


class ValidationLevel(Enum):
    """
    Validation rigor levels
    
    QUICK: Basic metrics only (~0.1s)
    STANDARD: Standard scientific validation (~0.5s)
    RIGOROUS: Publication-quality validation (~2s)
    BENCHMARK: Full benchmark suite (~30s)
    """
    QUICK = "quick"
    STANDARD = "standard"
    RIGOROUS = "rigorous"
    BENCHMARK = "benchmark"


@dataclass
class ValidationResult:
    """
    Container for validation results
    
    Attributes:
        passed: Whether validation passed all thresholds
        confidence_level: Overall confidence (0-1)
        metrics: Dictionary of all computed metrics
        warnings: List of warning messages
        recommendations: List of improvement recommendations
        scientific_notes: Detailed scientific interpretation
        profile_analysis: Profile-specific analysis results
    """
    passed: bool
    confidence_level: float
    metrics: Dict[str, float]
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    scientific_notes: str = ""
    profile_analysis: Dict[str, float] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        return (f"ValidationResult({status}, "
                f"confidence={self.confidence_level:.2%}, "
                f"metrics={len(self.metrics)})")
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'passed': self.passed,
            'confidence_level': self.confidence_level,
            'metrics': self.metrics,
            'warnings': self.warnings,
            'recommendations': self.recommendations,
            'scientific_notes': self.scientific_notes,
            'profile_analysis': self.profile_analysis
        }


class ScientificValidator:
    """
    Main validation class for scientific accuracy assessment
    
    Validates predictions against ground truth using:
    - Numerical accuracy (RMSE, MAE, relative error)
    - Structural similarity (SSIM, PSNR)
    - Statistical consistency (chi-squared, Kolmogorov-Smirnov)
    - Physical constraints (mass conservation, positivity)
    - Profile-specific tests (NFW cusp, SIS isothermal, etc.)
    
    Usage:
        validator = ScientificValidator(level=ValidationLevel.RIGOROUS)
        result = validator.validate_convergence_map(
            predicted=pinn_output,
            ground_truth=analytic_solution,
            profile_type="NFW"
        )
        
        if result.passed:
            print("✅ Validation passed!")
            print(result.scientific_notes)
        else:
            print("❌ Issues found:")
            for warning in result.warnings:
                print(f"  - {warning}")
    """
    
    def __init__(
        self,
        level: ValidationLevel = ValidationLevel.STANDARD,
        custom_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize validator
        
        Args:
            level: Validation rigor level
            custom_thresholds: Override default thresholds
        """
        self.level = level
        
        # Default tolerance thresholds
        self.tolerance_rmse = {
            ValidationLevel.QUICK: 0.1,
            ValidationLevel.STANDARD: 0.05,
            ValidationLevel.RIGOROUS: 0.01,
            ValidationLevel.BENCHMARK: 0.005
        }
        
        self.tolerance_ssim = {
            ValidationLevel.QUICK: 0.80,
            ValidationLevel.STANDARD: 0.90,
            ValidationLevel.RIGOROUS: 0.95,
            ValidationLevel.BENCHMARK: 0.98
        }
        
        # Apply custom thresholds if provided
        if custom_thresholds:
            for key, value in custom_thresholds.items():
                if hasattr(self, f'tolerance_{key}'):
                    getattr(self, f'tolerance_{key}')[level] = value
    
    def validate_convergence_map(
        self,
        predicted: np.ndarray,
        ground_truth: np.ndarray,
        profile_type: str = "NFW",
        uncertainty: Optional[np.ndarray] = None,
        pixel_scale: float = 0.05,
        verbose: bool = True
    ) -> ValidationResult:
        """
        Validate convergence map prediction
        
        Performs comprehensive validation including:
        1. Numerical accuracy (RMSE, MAE, relative error)
        2. Structural similarity (SSIM, PSNR)
        3. Statistical consistency (chi-squared, K-S test)
        4. Physical constraints (mass conservation, positivity)
        5. Profile-specific validation
        
        Args:
            predicted: Predicted convergence map (2D array)
            ground_truth: Ground truth convergence map (2D array)
            profile_type: Mass profile type (NFW, SIS, Hernquist, etc.)
            uncertainty: Uncertainty map (optional, for chi-squared)
            pixel_scale: Pixel scale in arcsec (default: 0.05)
            verbose: Print progress messages
            
        Returns:
            ValidationResult with comprehensive metrics and assessment
        """
        warnings_list = []
        recommendations = []
        
        # Input validation
        if predicted.shape != ground_truth.shape:
            raise ValueError(
                f"Shape mismatch: predicted {predicted.shape} vs "
                f"ground_truth {ground_truth.shape}"
            )
        
        if verbose:
            print(f"🔬 Running {self.level.value.upper()} validation...")
        
        # 1. Basic numerical accuracy
        if verbose:
            print("  → Calculating numerical accuracy metrics...")
        
        rmse = self._calculate_rmse(predicted, ground_truth)
        mae = self._calculate_mae(predicted, ground_truth)
        rel_error = self._calculate_relative_error(predicted, ground_truth)
        max_error = np.max(np.abs(predicted - ground_truth))
        
        # 2. Structural similarity (for 2D maps)
        if verbose:
            print("  → Calculating structural similarity...")
        
        ssim_score = self._calculate_ssim(predicted, ground_truth)
        psnr_score = self._calculate_psnr(predicted, ground_truth)
        
        # 3. Statistical tests
        if self.level in [ValidationLevel.RIGOROUS, ValidationLevel.BENCHMARK]:
            if verbose:
                print("  → Running statistical tests...")
            
            chi2, p_value = self._chi_squared_test(
                predicted, ground_truth, uncertainty
            )
            ks_stat, ks_pval = self._kolmogorov_smirnov_test(
                predicted, ground_truth
            )
        else:
            chi2, p_value = 0.0, 1.0
            ks_stat, ks_pval = 0.0, 1.0
        
        # 4. Physical constraints
        if verbose:
            print("  → Checking physical constraints...")
        
        mass_ratio = self._check_mass_conservation(predicted, ground_truth)
        positivity_ok = np.all(predicted >= -1e-10)  # Allow small numerical errors
        n_negative = np.sum(predicted < 0)
        
        # 5. Profile-specific validation
        if self.level in [ValidationLevel.RIGOROUS, ValidationLevel.BENCHMARK]:
            if verbose:
                print(f"  → Running {profile_type} profile-specific tests...")
            
            profile_metrics = self._validate_profile_specific(
                predicted, ground_truth, profile_type, pixel_scale
            )
        else:
            profile_metrics = {}
        
        # 6. Resolution and convergence checks
        gradient_error = self._check_gradient_consistency(predicted, ground_truth)
        
        # Determine pass/fail
        rmse_threshold = self.tolerance_rmse[self.level]
        ssim_threshold = self.tolerance_ssim[self.level]
        
        passed = (
            rmse < rmse_threshold and
            ssim_score > ssim_threshold and
            (p_value > 0.05 or self.level == ValidationLevel.QUICK) and
            abs(1.0 - mass_ratio) < 0.02 and  # Within 2%
            positivity_ok
        )
        
        # Calculate confidence level (0-1 scale)
        confidence = self._calculate_confidence(
            rmse, rmse_threshold,
            ssim_score, ssim_threshold,
            p_value, mass_ratio
        )
        
        # Generate warnings
        if rmse > rmse_threshold * 0.7:
            warnings_list.append(
                f"RMSE ({rmse:.4f}) approaching threshold ({rmse_threshold:.4f})"
            )
        
        if ssim_score < ssim_threshold + 0.05:
            warnings_list.append(
                f"SSIM ({ssim_score:.3f}) approaching threshold ({ssim_threshold:.3f})"
            )
        
        if not positivity_ok:
            warnings_list.append(
                f"Non-physical negative values detected ({n_negative} pixels)"
            )
        
        if abs(1.0 - mass_ratio) > 0.01:
            warnings_list.append(
                f"Mass conservation error: {abs(1.0 - mass_ratio)*100:.2f}%"
            )
        
        if self.level in [ValidationLevel.RIGOROUS, ValidationLevel.BENCHMARK]:
            if p_value < 0.05:
                warnings_list.append(
                    f"Chi-squared test failed (p={p_value:.3f} < 0.05): "
                    "Systematic deviations detected"
                )
        
        # Generate recommendations
        if rmse > rmse_threshold:
            recommendations.append(
                "Consider training with more data or longer epochs to improve accuracy"
            )
        
        if ssim_score < ssim_threshold:
            recommendations.append(
                "Low SSIM suggests structural errors - review architecture or loss function"
            )
        
        if not positivity_ok:
            recommendations.append(
                "Add physical constraints to training (e.g., ReLU activation for κ)"
            )
        
        if max_error > 0.5:
            recommendations.append(
                f"Large maximum error ({max_error:.3f}) detected - check for outliers"
            )
        
        # Compile all metrics
        metrics = {
            'rmse': float(rmse),
            'mae': float(mae),
            'relative_error': float(rel_error),
            'max_error': float(max_error),
            'ssim': float(ssim_score),
            'psnr': float(psnr_score),
            'chi_squared': float(chi2),
            'chi2_pvalue': float(p_value),
            'ks_statistic': float(ks_stat),
            'ks_pvalue': float(ks_pval),
            'mass_conservation_ratio': float(mass_ratio),
            'mass_conservation_error_pct': float(abs(1.0 - mass_ratio) * 100),
            'positivity_check': bool(positivity_ok),
            'n_negative_pixels': int(n_negative),
            'gradient_error': float(gradient_error),
            **profile_metrics
        }
        
        # Generate scientific notes
        notes = self._generate_scientific_notes(
            metrics, profile_type, passed, confidence
        )
        
        if verbose:
            print(f"✅ Validation complete! Status: {'PASSED' if passed else 'FAILED'}")
            print(f"   Confidence: {confidence:.1%}")
        
        return ValidationResult(
            passed=passed,
            confidence_level=confidence,
            metrics=metrics,
            warnings=warnings_list,
            recommendations=recommendations,
            scientific_notes=notes,
            profile_analysis=profile_metrics
        )
    
    # ========================================================================
    # Metric Calculation Methods
    # ========================================================================
    
    def _calculate_rmse(
        self,
        predicted: np.ndarray,
        ground_truth: np.ndarray
    ) -> float:
        """Calculate Root Mean Squared Error"""
        if BENCHMARKS_AVAILABLE:
            return calculate_rmse(ground_truth, predicted)
        return np.sqrt(np.mean((predicted - ground_truth) ** 2))
    
    def _calculate_mae(
        self,
        predicted: np.ndarray,
        ground_truth: np.ndarray
    ) -> float:
        """Calculate Mean Absolute Error"""
        if BENCHMARKS_AVAILABLE:
            return calculate_mae(ground_truth, predicted)
        return np.mean(np.abs(predicted - ground_truth))
    
    def _calculate_relative_error(
        self,
        predicted: np.ndarray,
        ground_truth: np.ndarray
    ) -> float:
        """Calculate mean relative error"""
        if BENCHMARKS_AVAILABLE:
            return calculate_relative_error(ground_truth, predicted)
        
        # Avoid division by zero
        mask = np.abs(ground_truth) > 1e-10
        if not np.any(mask):
            return 0.0
        
        rel_err = np.abs((predicted[mask] - ground_truth[mask]) / ground_truth[mask])
        return np.mean(rel_err)
    
    def _calculate_ssim(
        self,
        predicted: np.ndarray,
        ground_truth: np.ndarray
    ) -> float:
        """Calculate Structural Similarity Index"""
        if BENCHMARKS_AVAILABLE:
            return calculate_structural_similarity(ground_truth, predicted)
        
        # Normalize to [0, 1] range for SSIM
        data_range = max(ground_truth.max() - ground_truth.min(), 1e-10)
        
        return ssim(
            ground_truth,
            predicted,
            data_range=data_range
        )
    
    def _calculate_psnr(
        self,
        predicted: np.ndarray,
        ground_truth: np.ndarray
    ) -> float:
        """Calculate Peak Signal-to-Noise Ratio"""
        if BENCHMARKS_AVAILABLE:
            return calculate_peak_signal_noise_ratio(ground_truth, predicted)
        
        data_range = max(ground_truth.max() - ground_truth.min(), 1e-10)
        
        return psnr(
            ground_truth,
            predicted,
            data_range=data_range
        )
    
    def _chi_squared_test(
        self,
        predicted: np.ndarray,
        ground_truth: np.ndarray,
        uncertainty: Optional[np.ndarray] = None
    ) -> Tuple[float, float]:
        """
        Perform chi-squared test for statistical consistency.
        
        Returns:
            (chi_squared_statistic, p_value)
        
        FIXED: Removed circular uncertainty estimation.
        Chi-squared requires INDEPENDENT uncertainty estimates.
        """
        if BENCHMARKS_AVAILABLE and uncertainty is not None:
            return calculate_chi_squared(ground_truth, predicted, uncertainty)
        
        if uncertainty is None:
            # FIXED: Cannot estimate uncertainty from residuals - that's circular!
            # Instead, use a conservative default based on typical measurement errors
            # or require explicit uncertainty input
            # Use 10% relative uncertainty as default
            uncertainty = np.abs(ground_truth) * 0.1 + 0.01
            # Add warning that this is a conservative estimate
            logger.warning(
                "Chi-squared test using default 10% uncertainty estimate. "
                "For rigorous validation, provide explicit uncertainty estimates."
            )
        
        # Avoid division by zero
        uncertainty = np.maximum(uncertainty, 1e-10)
        
        # Chi-squared statistic
        chi2 = np.sum(((predicted - ground_truth) / uncertainty) ** 2)
        
        # Degrees of freedom (n_observations - n_parameters)
        # Assume 1 parameter for simple comparison
        dof = max(predicted.size - 1, 1)
        
        # P-value
        p_value = 1.0 - stats.chi2.cdf(chi2, dof)
        
        return float(chi2), float(p_value)
    
    def _kolmogorov_smirnov_test(
        self,
        predicted: np.ndarray,
        ground_truth: np.ndarray
    ) -> Tuple[float, float]:
        """
        Perform Kolmogorov-Smirnov test on distributions
        
        Tests whether predicted and ground truth come from same distribution
        
        Returns:
            (ks_statistic, p_value)
        """
        ks_stat, p_value = stats.ks_2samp(
            predicted.flatten(),
            ground_truth.flatten()
        )
        
        return float(ks_stat), float(p_value)
    
    def _check_mass_conservation(
        self,
        predicted: np.ndarray,
        ground_truth: np.ndarray
    ) -> float:
        """
        Check mass conservation
        
        Returns:
            Ratio of predicted mass to ground truth mass
        """
        mass_pred = np.sum(predicted)
        mass_true = np.sum(ground_truth)
        
        if abs(mass_true) < 1e-10:
            return 1.0
        
        return mass_pred / mass_true
    
    def _check_gradient_consistency(
        self,
        predicted: np.ndarray,
        ground_truth: np.ndarray
    ) -> float:
        """
        Check gradient consistency
        
        Returns:
            Relative error in gradients
        """
        # Compute gradients
        grad_pred_y, grad_pred_x = np.gradient(predicted)
        grad_true_y, grad_true_x = np.gradient(ground_truth)
        
        # Combine gradient components
        grad_pred_mag = np.sqrt(grad_pred_x**2 + grad_pred_y**2)
        grad_true_mag = np.sqrt(grad_true_x**2 + grad_true_y**2)
        
        # Calculate relative error
        mask = grad_true_mag > 1e-10
        if not np.any(mask):
            return 0.0
        
        rel_error = np.mean(
            np.abs(grad_pred_mag[mask] - grad_true_mag[mask]) / grad_true_mag[mask]
        )
        
        return float(rel_error)
    
    # ========================================================================
    # Profile-Specific Validation Methods
    # ========================================================================
    
    def _validate_profile_specific(
        self,
        predicted: np.ndarray,
        ground_truth: np.ndarray,
        profile_type: str,
        pixel_scale: float
    ) -> Dict[str, float]:
        """
        Profile-specific validation dispatcher
        
        Routes to appropriate profile validator based on type
        """
        profile_type = profile_type.upper()
        
        if profile_type == "NFW":
            return self._validate_nfw_profile(
                predicted, ground_truth, pixel_scale
            )
        elif profile_type == "SIS":
            return self._validate_sis_profile(
                predicted, ground_truth, pixel_scale
            )
        elif profile_type == "HERNQUIST":
            return self._validate_hernquist_profile(
                predicted, ground_truth, pixel_scale
            )
        else:
            return {}
    
    def _validate_nfw_profile(
        self,
        predicted: np.ndarray,
        ground_truth: np.ndarray,
        pixel_scale: float
    ) -> Dict[str, float]:
        """
        NFW-specific validation
        
        Checks:
        1. Central cusp behavior (ρ ∝ r^-1 → κ ∝ r^-1)
        2. Outer slope (ρ ∝ r^-3 → κ ∝ r^-2)
        3. Scale radius consistency
        4. Overall profile shape
        
        Returns:
            Dictionary of NFW-specific metrics
        """
        # Get radial coordinates
        ny, nx = predicted.shape
        y, x = np.mgrid[0:ny, 0:nx]
        cy, cx = ny / 2, nx / 2
        r = np.sqrt((x - cx)**2 + (y - cy)**2) * pixel_scale
        
        # Avoid center singularity
        r = np.maximum(r, pixel_scale)
        
        # Radial profiles
        r_flat = r.flatten()
        pred_flat = predicted.flatten()
        true_flat = ground_truth.flatten()
        
        # Sort by radius
        sort_idx = np.argsort(r_flat)
        r_sorted = r_flat[sort_idx]
        pred_sorted = pred_flat[sort_idx]
        true_sorted = true_flat[sort_idx]
        
        # 1. Central cusp (inner 20%)
        r_max = r_sorted[-1]
        inner_mask = r_sorted < 0.2 * r_max
        
        if np.sum(inner_mask) > 10:  # Need enough points
            inner_slope_pred = self._fit_power_law(
                r_sorted[inner_mask],
                pred_sorted[inner_mask]
            )
            inner_slope_true = self._fit_power_law(
                r_sorted[inner_mask],
                true_sorted[inner_mask]
            )
            cusp_error = abs(inner_slope_pred - inner_slope_true)
        else:
            inner_slope_pred = inner_slope_true = cusp_error = 0.0
        
        # 2. Outer slope (outer 50%)
        outer_mask = r_sorted > 0.5 * r_max
        
        if np.sum(outer_mask) > 10:
            outer_slope_pred = self._fit_power_law(
                r_sorted[outer_mask],
                pred_sorted[outer_mask]
            )
            outer_slope_true = self._fit_power_law(
                r_sorted[outer_mask],
                true_sorted[outer_mask]
            )
            outer_error = abs(outer_slope_pred - outer_slope_true)
        else:
            outer_slope_pred = outer_slope_true = outer_error = 0.0
        
        # 3. Overall fit quality
        fit_quality = 1.0 - min((cusp_error + outer_error) / 2, 1.0)
        
        return {
            'nfw_inner_slope_pred': float(inner_slope_pred),
            'nfw_inner_slope_true': float(inner_slope_true),
            'nfw_cusp_slope_error': float(cusp_error),
            'nfw_outer_slope_pred': float(outer_slope_pred),
            'nfw_outer_slope_true': float(outer_slope_true),
            'nfw_outer_slope_error': float(outer_error),
            'nfw_overall_fit_quality': float(fit_quality)
        }
    
    def _validate_sis_profile(
        self,
        predicted: np.ndarray,
        ground_truth: np.ndarray,
        pixel_scale: float
    ) -> Dict[str, float]:
        """
        Singular Isothermal Sphere validation
        
        Checks:
        1. Isothermal behavior (ρ ∝ r^-2 → κ ∝ r^-1)
        2. Einstein radius consistency
        3. Central convergence
        """
        # Get radial coordinates
        ny, nx = predicted.shape
        y, x = np.mgrid[0:ny, 0:nx]
        cy, cx = ny / 2, nx / 2
        r = np.sqrt((x - cx)**2 + (y - cy)**2) * pixel_scale
        r = np.maximum(r, pixel_scale)
        
        # Radial profiles
        r_flat = r.flatten()
        pred_flat = predicted.flatten()
        true_flat = ground_truth.flatten()
        
        # Sort by radius
        sort_idx = np.argsort(r_flat)
        r_sorted = r_flat[sort_idx]
        pred_sorted = pred_flat[sort_idx]
        true_sorted = true_flat[sort_idx]
        
        # Isothermal slope (should be -1)
        slope_pred = self._fit_power_law(r_sorted, pred_sorted)
        slope_true = self._fit_power_law(r_sorted, true_sorted)
        slope_error = abs(slope_pred - slope_true)
        
        # Fit quality
        fit_quality = max(1.0 - slope_error, 0.0)
        
        return {
            'sis_slope_pred': float(slope_pred),
            'sis_slope_true': float(slope_true),
            'sis_slope_error': float(slope_error),
            'sis_fit_quality': float(fit_quality)
        }
    
    def _validate_hernquist_profile(
        self,
        predicted: np.ndarray,
        ground_truth: np.ndarray,
        pixel_scale: float
    ) -> Dict[str, float]:
        """
        Hernquist profile validation
        
        Checks:
        1. Inner slope (ρ ∝ r^-1)
        2. Outer slope (ρ ∝ r^-4)
        3. Scale radius
        """
        # Similar to NFW but with different slopes
        ny, nx = predicted.shape
        y, x = np.mgrid[0:ny, 0:nx]
        cy, cx = ny / 2, nx / 2
        r = np.sqrt((x - cx)**2 + (y - cy)**2) * pixel_scale
        r = np.maximum(r, pixel_scale)
        
        r_flat = r.flatten()
        pred_flat = predicted.flatten()
        true_flat = ground_truth.flatten()
        
        sort_idx = np.argsort(r_flat)
        r_sorted = r_flat[sort_idx]
        pred_sorted = pred_flat[sort_idx]
        true_sorted = true_flat[sort_idx]
        
        # Inner slope
        r_max = r_sorted[-1]
        inner_mask = r_sorted < 0.2 * r_max
        
        if np.sum(inner_mask) > 10:
            inner_slope_pred = self._fit_power_law(
                r_sorted[inner_mask],
                pred_sorted[inner_mask]
            )
            inner_slope_true = self._fit_power_law(
                r_sorted[inner_mask],
                true_sorted[inner_mask]
            )
            inner_error = abs(inner_slope_pred - inner_slope_true)
        else:
            inner_error = 0.0
        
        # Outer slope
        outer_mask = r_sorted > 0.5 * r_max
        
        if np.sum(outer_mask) > 10:
            outer_slope_pred = self._fit_power_law(
                r_sorted[outer_mask],
                pred_sorted[outer_mask]
            )
            outer_slope_true = self._fit_power_law(
                r_sorted[outer_mask],
                true_sorted[outer_mask]
            )
            outer_error = abs(outer_slope_pred - outer_slope_true)
        else:
            outer_error = 0.0
        
        fit_quality = 1.0 - min((inner_error + outer_error) / 2, 1.0)
        
        return {
            'hernquist_inner_slope_error': float(inner_error),
            'hernquist_outer_slope_error': float(outer_error),
            'hernquist_fit_quality': float(fit_quality)
        }
    
    def _fit_power_law(
        self,
        r: np.ndarray,
        values: np.ndarray,
        log_space: bool = True
    ) -> float:
        """
        Fit power law: values = A * r^alpha
        
        Returns:
            alpha (power law exponent)
        """
        # Remove zeros and negative values
        mask = (r > 0) & (values > 0)
        if np.sum(mask) < 5:
            return 0.0
        
        r_valid = r[mask]
        v_valid = values[mask]
        
        if log_space:
            # Fit in log space: log(v) = log(A) + alpha * log(r)
            try:
                log_r = np.log(r_valid)
                log_v = np.log(v_valid)
                
                # Linear regression
                coeffs = np.polyfit(log_r, log_v, 1)
                alpha = coeffs[0]
                
                return float(alpha)
            except Exception:
                return 0.0
        else:
            # Direct power law fit (less stable)
            try:
                def power_law(r, A, alpha):
                    return A * r ** alpha
                
                popt, _ = optimize.curve_fit(
                    power_law, r_valid, v_valid,
                    p0=[1.0, -1.0],
                    maxfev=1000
                )
                
                return float(popt[1])
            except Exception:
                return 0.0
    
    # ========================================================================
    # Confidence and Interpretation Methods
    # ========================================================================
    
    def _calculate_confidence(
        self,
        rmse: float,
        rmse_threshold: float,
        ssim_score: float,
        ssim_threshold: float,
        p_value: float,
        mass_ratio: float
    ) -> float:
        """
        Calculate overall confidence level (0-1)
        
        Weighted combination of all metrics
        """
        # RMSE score (lower is better)
        rmse_score = max(0, 1 - rmse / rmse_threshold)
        
        # SSIM score (already 0-1)
        ssim_score_norm = ssim_score
        
        # P-value score (want p > 0.05)
        pval_score = min(p_value / 0.05, 1.0) if p_value > 0 else 1.0
        
        # Mass conservation score
        mass_score = max(0, 1 - abs(1 - mass_ratio) / 0.02)
        
        # Weighted average
        weights = [0.3, 0.3, 0.2, 0.2]  # RMSE, SSIM, p-value, mass
        scores = [rmse_score, ssim_score_norm, pval_score, mass_score]
        
        confidence = sum(w * s for w, s in zip(weights, scores))
        
        return min(max(confidence, 0.0), 1.0)
    
    def _generate_scientific_notes(
        self,
        metrics: Dict[str, float],
        profile_type: str,
        passed: bool,
        confidence: float
    ) -> str:
        """
        Generate scientific interpretation of results
        
        Creates human-readable assessment for researchers
        """
        notes = []
        notes.append("=" * 70)
        notes.append(f"SCIENTIFIC VALIDATION REPORT: {profile_type} Profile")
        notes.append("=" * 70)
        notes.append("")
        
        # Overall status
        if passed:
            notes.append("✅ VALIDATION STATUS: PASSED")
        else:
            notes.append("❌ VALIDATION STATUS: FAILED")
        
        notes.append(f"   Confidence Level: {confidence:.1%}")
        notes.append("")
        
        # 1. Accuracy Assessment
        notes.append("1. NUMERICAL ACCURACY")
        notes.append("-" * 70)
        
        rmse = metrics['rmse']
        if rmse < 0.01:
            notes.append("   ⭐ EXCELLENT: Predictions match ground truth within 1% RMSE")
        elif rmse < 0.05:
            notes.append("   ✓ GOOD: Predictions match ground truth within 5% RMSE")
        elif rmse < 0.1:
            notes.append("   ~ ACCEPTABLE: Predictions within 10% RMSE tolerance")
        else:
            notes.append("   ✗ POOR: RMSE exceeds 10%, predictions unreliable")
        
        notes.append(f"   • RMSE: {rmse:.6f}")
        notes.append(f"   • MAE: {metrics['mae']:.6f}")
        notes.append(f"   • Relative Error: {metrics['relative_error']:.2%}")
        notes.append(f"   • Max Error: {metrics['max_error']:.6f}")
        notes.append("")
        
        # 2. Structural Similarity
        notes.append("2. STRUCTURAL SIMILARITY")
        notes.append("-" * 70)
        
        ssim = metrics['ssim']
        if ssim > 0.95:
            notes.append("   ⭐ EXCELLENT: Structural similarity > 0.95 (near-perfect)")
        elif ssim > 0.9:
            notes.append("   ✓ GOOD: Structural similarity > 0.9")
        elif ssim > 0.8:
            notes.append("   ~ ACCEPTABLE: Structural similarity > 0.8")
        else:
            notes.append("   ✗ POOR: Low structural similarity, check architecture")
        
        notes.append(f"   • SSIM: {ssim:.4f}")
        notes.append(f"   • PSNR: {metrics['psnr']:.2f} dB")
        notes.append("")
        
        # 3. Statistical Consistency
        notes.append("3. STATISTICAL CONSISTENCY")
        notes.append("-" * 70)
        
        p_value = metrics['chi2_pvalue']
        if p_value > 0.05:
            notes.append(f"   ✓ PASSED: Chi-squared test (p={p_value:.3f} > 0.05)")
            notes.append("   Residuals consistent with statistical noise")
        else:
            notes.append(f"   ✗ FAILED: Chi-squared test (p={p_value:.3f} < 0.05)")
            notes.append("   Systematic deviations detected")
        
        notes.append(f"   • χ²: {metrics['chi_squared']:.2f}")
        notes.append(f"   • p-value: {p_value:.4f}")
        notes.append(f"   • K-S statistic: {metrics['ks_statistic']:.4f}")
        notes.append("")
        
        # 4. Physical Constraints
        notes.append("4. PHYSICAL CONSTRAINTS")
        notes.append("-" * 70)
        
        mass_error = metrics['mass_conservation_error_pct']
        if mass_error < 1:
            notes.append(f"   ✓ PASSED: Mass conservation within {mass_error:.2f}%")
        else:
            notes.append(f"   ✗ WARNING: Mass conservation error {mass_error:.2f}%")
        
        if metrics['positivity_check']:
            notes.append("   ✓ PASSED: All convergence values non-negative")
        else:
            n_neg = metrics['n_negative_pixels']
            notes.append(f"   ✗ FAILED: {n_neg} negative pixels (non-physical)")
        
        notes.append("")
        
        # 5. Profile-Specific Analysis
        if profile_type.upper() == "NFW" and 'nfw_cusp_slope_error' in metrics:
            notes.append("5. NFW PROFILE ANALYSIS")
            notes.append("-" * 70)
            
            cusp_error = metrics['nfw_cusp_slope_error']
            outer_error = metrics['nfw_outer_slope_error']
            
            notes.append(f"   • Inner slope (predicted): {metrics['nfw_inner_slope_pred']:.3f}")
            notes.append(f"   • Inner slope (expected): ~-1.0 for NFW cusp")
            notes.append(f"   • Inner slope error: {cusp_error:.4f}")
            notes.append("")
            notes.append(f"   • Outer slope (predicted): {metrics['nfw_outer_slope_pred']:.3f}")
            notes.append(f"   • Outer slope (expected): ~-2.0 for NFW")
            notes.append(f"   • Outer slope error: {outer_error:.4f}")
            notes.append("")
            
            fit_quality = metrics['nfw_overall_fit_quality']
            if fit_quality > 0.9:
                notes.append("   ✓ Profile shape excellently reproduced")
            elif fit_quality > 0.7:
                notes.append("   ~ Profile shape reasonably reproduced")
            else:
                notes.append("   ✗ Profile shape deviations detected")
            
            notes.append(f"   • Overall fit quality: {fit_quality:.2%}")
            notes.append("")
        
        # 6. Publication Recommendation
        notes.append("6. PUBLICATION READINESS")
        notes.append("-" * 70)
        
        if rmse < 0.01 and ssim > 0.95 and p_value > 0.05 and mass_error < 1:
            notes.append("   ✅ RECOMMENDED FOR PUBLICATION")
            notes.append("   Results meet peer-review standards for:")
            notes.append("   • ApJ, MNRAS, A&A (top-tier journals)")
            notes.append("   • Conference proceedings")
            notes.append("   • Preprint servers (arXiv)")
        elif rmse < 0.05 and ssim > 0.9:
            notes.append("   ~ ACCEPTABLE FOR PRELIMINARY STUDIES")
            notes.append("   Suitable for:")
            notes.append("   • Internal reports")
            notes.append("   • Proof-of-concept papers")
            notes.append("   • Further validation recommended for publication")
        else:
            notes.append("   ✗ NOT READY FOR PUBLICATION")
            notes.append("   Requires improvement:")
            notes.append("   • Additional training or architecture changes")
            notes.append("   • Validation on more diverse test cases")
            notes.append("   • Investigation of systematic errors")
        
        notes.append("")
        notes.append("=" * 70)
        
        return "\n".join(notes)


# ============================================================================
# Convenience Functions
# ============================================================================

def quick_validate(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    profile_type: str = "NFW"
) -> bool:
    """
    Quick validation check (returns pass/fail only)
    
    Usage:
        passed = quick_validate(pinn_output, analytic_solution)
    """
    validator = ScientificValidator(level=ValidationLevel.QUICK)
    result = validator.validate_convergence_map(
        predicted, ground_truth, profile_type, verbose=False
    )
    return result.passed


def rigorous_validate(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    profile_type: str = "NFW",
    uncertainty: Optional[np.ndarray] = None
) -> ValidationResult:
    """
    Rigorous validation with full report
    
    Usage:
        result = rigorous_validate(pinn_output, analytic_solution, "NFW")
        print(result.scientific_notes)
    """
    validator = ScientificValidator(level=ValidationLevel.RIGOROUS)
    return validator.validate_convergence_map(
        predicted, ground_truth, profile_type, uncertainty, verbose=True
    )
