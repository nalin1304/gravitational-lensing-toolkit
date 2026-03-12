"""
Uncertainty Quantification Module for Gravitational Lensing

Provides robust statistical methods for quantifying uncertainties in lens parameter
estimation and convergence map predictions. Implements Monte Carlo error propagation,
bootstrap resampling, and analytical error propagation formulas.

Scientific References:
    - Press et al. (1992): Numerical Recipes, 2nd Ed., Cambridge Univ. Press
      (Monte Carlo methods, Chapter 7)
    - Efron & Tibshirani (1993): An Introduction to the Bootstrap, Chapman & Hall
      (Bootstrap methods)
    - Bevington & Robinson (2003): Data Reduction and Error Analysis, 3rd Ed.
      (Error propagation, Chapter 3)
    - Barlow (1989): Statistics: A Guide to the Use of Statistical Methods
      (Confidence intervals and covariance)

Author: Phase 15 Implementation
Date: March 2026
"""

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from scipy import stats
from scipy.linalg import cholesky, solve_triangular
import warnings


@dataclass
class UncertaintyResult:
    """
    Container for uncertainty quantification results

    Attributes:
        mean: Mean value(s) of the quantity being estimated
        std: Standard deviation (1-sigma uncertainty)
        lower: Lower confidence bound
        upper: Upper confidence bound
        confidence: Confidence level (e.g., 0.95)
        covariance: Covariance matrix (if applicable)
        samples: MC samples used for estimation (if saved)

    References:
        Barlow (1989): Statistics, Chapter 5
    """

    mean: np.ndarray
    std: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    confidence: float
    covariance: Optional[np.ndarray] = None
    samples: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        result = {
            "mean": self.mean,
            "std": self.std,
            "lower": self.lower,
            "upper": self.upper,
            "confidence": self.confidence,
        }
        if self.covariance is not None:
            result["covariance"] = self.covariance
        if self.samples is not None:
            result["samples"] = self.samples
        return result


@dataclass
class PropagatedErrors:
    """
    Container for propagated parameter errors

    Attributes:
        values: Propagated values
        errors: Propagated 1-sigma errors
        correlation_matrix: Correlation matrix between outputs
        covariance_matrix: Full covariance matrix
        partial_derivatives: Jacobian matrix (if computed)
    """

    values: np.ndarray
    errors: np.ndarray
    correlation_matrix: np.ndarray
    covariance_matrix: np.ndarray
    partial_derivatives: Optional[np.ndarray] = None


def monte_carlo_error_propagation(
    model: Callable,
    param_samples: np.ndarray,
    n_samples: int = 1000,
    return_samples: bool = False,
    random_state: Optional[int] = None,
) -> UncertaintyResult:
    """
    Monte Carlo error propagation for lens parameters

    Implements the Monte Carlo method for error propagation as described in
    Press et al. (1992), Chapter 7. This method samples from the parameter
    distribution and computes the output distribution empirically.

    Mathematical basis:
        Given model f(θ) and parameter distribution p(θ), we sample
        θ_i ~ p(θ) for i = 1,...,N and compute y_i = f(θ_i).
        The output statistics are computed from {y_i}.

    Advantages over analytical methods:
        - No need for derivatives (works with black-box models)
        - Captures non-linear effects and correlations
        - Provides full output distribution, not just moments

    Args:
        model: Function that takes parameter array and returns output
               Signature: model(params) -> np.ndarray
        param_samples: Array of parameter samples [n_samples, n_params]
                       Each row is a parameter vector sampled from p(θ)
        n_samples: Number of Monte Carlo samples to use
        return_samples: If True, include all MC samples in result
        random_state: Random seed for reproducibility

    Returns:
        UncertaintyResult with mean, std, confidence intervals

    Example:
        >>> def lens_model(params):
        ...     M, c, z = params
        ...     return compute_convergence(M, c, z)
        >>>
        >>> # Generate parameter samples (e.g., from MCMC posterior)
        >>> param_samples = np.random.multivariate_normal(
        ...     mean=[1e14, 4.0, 0.5],
        ...     cov=[[1e26, 0, 0], [0, 0.04, 0], [0, 0, 0.001]],
        ...     size=1000
        ... )
        >>>
        >>> result = monte_carlo_error_propagation(
        ...     model=lens_model,
        ...     param_samples=param_samples,
        ...     n_samples=1000,
        ...     confidence=0.95
        ... )
        >>> print(f"Mean: {result.mean:.4f} ± {result.std:.4f}")

    References:
        Press et al. (1992): Numerical Recipes, 2nd Ed., Section 7.6
        Efron & Tibshirani (1993): Bootstrap methods, Chapter 6
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_available = len(param_samples)

    if n_available < n_samples:
        warnings.warn(
            f"Requested {n_samples} samples but only {n_available} available. "
            f"Using all {n_available} samples.",
            UserWarning,
        )
        n_samples = n_available

    # Randomly sample from parameter distribution
    indices = np.random.choice(n_available, size=n_samples, replace=False)
    sampled_params = param_samples[indices]

    # Run model on all parameter samples
    outputs = []
    for params in sampled_params:
        try:
            output = model(params)
            outputs.append(np.atleast_1d(output))
        except Exception as e:
            warnings.warn(f"Model evaluation failed for params {params}: {e}")
            continue

    if len(outputs) == 0:
        raise RuntimeError("All model evaluations failed")

    outputs = np.array(outputs)

    # Compute statistics
    mean = np.mean(outputs, axis=0)
    std = np.std(outputs, axis=0, ddof=1)

    # Compute confidence intervals (percentile method)
    confidence = 0.95
    alpha = (1 - confidence) / 2
    lower = np.percentile(outputs, 100 * alpha, axis=0)
    upper = np.percentile(outputs, 100 * (1 - alpha), axis=0)

    # Compute covariance matrix if output is multi-dimensional
    covariance = None
    if outputs.ndim > 1 and outputs.shape[1] > 1:
        covariance = np.cov(outputs.T)

    return UncertaintyResult(
        mean=mean,
        std=std,
        lower=lower,
        upper=upper,
        confidence=confidence,
        covariance=covariance,
        samples=outputs if return_samples else None,
    )


def compute_confidence_intervals(
    predictions: np.ndarray,
    confidence: float = 0.95,
    method: str = "percentile",
    axis: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate confidence intervals for predictions

    Supports multiple methods for computing confidence intervals:
        - 'percentile': Direct percentile method (non-parametric)
        - 'normal': Gaussian approximation
        - 'bc': Bias-corrected percentile (more accurate for skewed distributions)

    The percentile method is recommended as it makes no assumptions about
    the distribution shape (Efron & Tibshirani, 1993).

    Args:
        predictions: Array of predictions [n_samples, ...] or [..., n_samples]
        confidence: Confidence level (0 < confidence < 1), default 0.95
        method: Method for computing intervals ('percentile', 'normal', 'bc')
        axis: Axis along which to compute intervals

    Returns:
        Tuple of (lower_bound, upper_bound, point_estimate)
        - point_estimate is typically the median for percentile method

    Example:
        >>> # Predictions from MC sampling
        >>> predictions = np.random.normal(loc=1.0, scale=0.1, size=10000)
        >>> lower, upper, median = compute_confidence_intervals(
        ...     predictions, confidence=0.95
        ... )
        >>> print(f"95% CI: [{lower:.4f}, {upper:.4f}]")

    References:
        Efron & Tibshirani (1993): An Introduction to the Bootstrap
        Barlow (1989): Statistics, Section 5.3
    """
    if not 0 < confidence < 1:
        raise ValueError(f"Confidence must be between 0 and 1, got {confidence}")

    alpha = (1 - confidence) / 2

    if method == "percentile":
        # Direct percentile method
        lower = np.percentile(predictions, 100 * alpha, axis=axis)
        upper = np.percentile(predictions, 100 * (1 - alpha), axis=axis)
        point_estimate = np.median(predictions, axis=axis)

    elif method == "normal":
        # Gaussian approximation
        mean = np.mean(predictions, axis=axis)
        std = np.std(predictions, axis=axis, ddof=1)
        z_score = stats.norm.ppf(1 - alpha)

        lower = mean - z_score * std
        upper = mean + z_score * std
        point_estimate = mean

    elif method == "bc":
        # Bias-corrected percentile method
        # More accurate for skewed distributions
        z0 = stats.norm.ppf(np.mean(predictions < np.median(predictions)))
        z_alpha = stats.norm.ppf(alpha)
        z_1_alpha = stats.norm.ppf(1 - alpha)

        # Adjusted percentiles
        p_lower = stats.norm.cdf(2 * z0 + z_alpha) * 100
        p_upper = stats.norm.cdf(2 * z0 + z_1_alpha) * 100

        lower = np.percentile(predictions, p_lower, axis=axis)
        upper = np.percentile(predictions, p_upper, axis=axis)
        point_estimate = np.median(predictions, axis=axis)

    else:
        raise ValueError(
            f"Unknown method: {method}. Use 'percentile', 'normal', or 'bc'"
        )

    return lower, upper, point_estimate


def bootstrap_errors(
    data: np.ndarray,
    statistic_func: Callable[[np.ndarray], np.ndarray],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    return_samples: bool = False,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Bootstrap resampling for error estimation

    Implements the non-parametric bootstrap method (Efron, 1979) for estimating
    the sampling distribution of a statistic. This is particularly useful when
    the analytical form of the sampling distribution is unknown or complex.

    Algorithm:
        1. Draw B bootstrap samples (sampling with replacement)
        2. Compute statistic on each bootstrap sample
        3. Use distribution of bootstrap statistics to estimate errors

    Key assumptions:
        - Data is representative of the population
        - Bootstrap samples are drawn independently with replacement

    Args:
        data: Original data array [n_observations, ...]
        statistic_func: Function to compute statistic from data
                       Signature: statistic_func(data) -> np.ndarray
        n_bootstrap: Number of bootstrap resamples (Efron recommends ≥1000)
        confidence: Confidence level for intervals
        return_samples: If True, return all bootstrap statistics
        random_state: Random seed for reproducibility

    Returns:
        Dictionary containing:
            - 'statistic': Original statistic on full data
            - 'std_error': Bootstrap standard error
            - 'bias': Bootstrap bias estimate
            - 'lower': Lower confidence bound
            - 'upper': Upper confidence bound
            - 'confidence': Confidence level used
            - 'n_bootstrap': Number of bootstrap samples
            - 'bootstrap_samples': All bootstrap statistics (if return_samples=True)

    Example:
        >>> # Estimate error on mean convergence
        >>> def mean_convergence(data):
        ...     return np.mean(data['kappa'])
        >>>
        >>> lens_data = {'kappa': np.random.normal(1.0, 0.2, 100)}
        >>> result = bootstrap_errors(
        ...     data=lens_data,
        ...     statistic_func=mean_convergence,
        ...     n_bootstrap=10000,
        ...     confidence=0.95
        ... )
        >>> print(f"Mean: {result['statistic']:.4f} ± {result['std_error']:.4f}")

    References:
        Efron (1979): "Bootstrap Methods: Another Look at the Jackknife"
        Efron & Tibshirani (1993): An Introduction to the Bootstrap
        Press et al. (1992): Numerical Recipes, Section 15.6
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_obs = (
        len(data)
        if isinstance(data, (list, np.ndarray))
        else len(next(iter(data.values())))
    )

    # Compute original statistic
    original_stat = statistic_func(data)

    # Bootstrap resampling
    bootstrap_stats = []

    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_obs, size=n_obs, replace=True)

        if isinstance(data, dict):
            # Handle dictionary data
            bootstrap_sample = {
                key: np.array(val)[indices] for key, val in data.items()
            }
        else:
            # Handle array data
            bootstrap_sample = data[indices]

        # Compute statistic on bootstrap sample
        try:
            stat = statistic_func(bootstrap_sample)
            bootstrap_stats.append(np.atleast_1d(stat))
        except Exception as e:
            warnings.warn(f"Statistic computation failed: {e}")
            continue

    if len(bootstrap_stats) == 0:
        raise RuntimeError("All bootstrap computations failed")

    bootstrap_stats = np.array(bootstrap_stats)

    # Compute bootstrap statistics
    mean_bootstrap = np.mean(bootstrap_stats, axis=0)
    std_error = np.std(bootstrap_stats, axis=0, ddof=1)
    bias = mean_bootstrap - original_stat

    # Confidence intervals using percentile method
    lower, upper, _ = compute_confidence_intervals(
        bootstrap_stats, confidence=confidence, method="percentile"
    )

    result = {
        "statistic": original_stat,
        "std_error": std_error,
        "bias": bias,
        "lower": lower,
        "upper": upper,
        "confidence": confidence,
        "n_bootstrap": n_bootstrap,
    }

    if return_samples:
        result["bootstrap_samples"] = bootstrap_stats

    return result


def propagate_parameter_errors(
    model: Callable,
    param_values: np.ndarray,
    param_errors: np.ndarray,
    param_correlations: Optional[np.ndarray] = None,
    method: str = "analytical",
    n_samples: int = 10000,
    delta: float = 1e-6,
) -> PropagatedErrors:
    """
    Propagate parameter errors through a model

    Supports multiple error propagation methods:
        - 'analytical': First-order Taylor expansion (linear approximation)
        - 'monte_carlo': Monte Carlo sampling (non-linear, more accurate)
        - 'numerical': Numerical differentiation for derivatives

    Analytical method (default):
        Uses the standard error propagation formula:
        σ_y² = Σ_ij (∂f/∂θ_i)(∂f/∂θ_j) Cov(θ_i, θ_j)

        For uncorrelated parameters:
        σ_y² = Σ_i (∂f/∂θ_i)² σ_i²

    Args:
        model: Model function f(θ) -> output
        param_values: Parameter values θ (best-fit or mean)
        param_errors: Parameter 1-sigma errors σ_i
        param_correlations: Correlation matrix (optional, identity if None)
        method: Propagation method ('analytical', 'monte_carlo', 'numerical')
        n_samples: Number of MC samples (for monte_carlo method)
        delta: Step size for numerical differentiation

    Returns:
        PropagatedErrors with propagated values, errors, and correlation matrix

    Example:
        >>> def einstein_radius(M, c, z):
        ...     # Simplified Einstein radius calculation
        ...     return 1.0 * (M / 1e14)**0.5 * (1 + z)**(-0.5)
        >>>
        >>> params = np.array([1e14, 4.0, 0.5])  # M, c, z
        >>> errors = np.array([0.1e14, 0.5, 0.1])
        >>>
        >>> result = propagate_parameter_errors(
        ...     model=einstein_radius,
        ...     param_values=params,
        ...     param_errors=errors,
        ...     method='analytical'
        ... )
        >>> print(f"θ_E = {result.values[0]:.4f} ± {result.errors[0]:.4f}")

    References:
        Bevington & Robinson (2003): Data Reduction and Error Analysis, Chapter 3
        Press et al. (1992): Numerical Recipes, Section 15.6
    """
    param_values = np.atleast_1d(param_values)
    param_errors = np.atleast_1d(param_errors)
    n_params = len(param_values)

    # Build covariance matrix
    if param_correlations is None:
        # Assume uncorrelated parameters
        covariance = np.diag(param_errors**2)
    else:
        # Build from correlation matrix
        correlation = np.atleast_2d(param_correlations)
        std_matrix = np.outer(param_errors, param_errors)
        covariance = correlation * std_matrix

    if method == "analytical":
        # Analytical error propagation with numerical derivatives
        jacobian = _compute_jacobian(model, param_values, delta)

        # Compute propagated covariance: J Σ J^T
        propagated_cov = jacobian @ covariance @ jacobian.T

        # Propagated errors
        propagated_errors = np.sqrt(np.diag(propagated_cov))

        # Output values
        output_values = np.atleast_1d(model(param_values))

        # Correlation matrix
        correlation_matrix = _covariance_to_correlation(propagated_cov)

        return PropagatedErrors(
            values=output_values,
            errors=propagated_errors,
            correlation_matrix=correlation_matrix,
            covariance_matrix=propagated_cov,
            partial_derivatives=jacobian,
        )

    elif method == "monte_carlo":
        # Monte Carlo error propagation
        # Generate correlated parameter samples
        param_samples = np.random.multivariate_normal(
            mean=param_values, cov=covariance, size=n_samples
        )

        # Run Monte Carlo propagation
        mc_result = monte_carlo_error_propagation(
            model=model,
            param_samples=param_samples,
            n_samples=n_samples,
            return_samples=True,
        )

        # Compute correlation from MC samples
        if mc_result.samples is not None and mc_result.samples.shape[1] > 1:
            correlation_matrix = np.corrcoef(mc_result.samples.T)
        else:
            correlation_matrix = np.array([[1.0]])

        return PropagatedErrors(
            values=mc_result.mean,
            errors=mc_result.std,
            correlation_matrix=correlation_matrix,
            covariance_matrix=mc_result.covariance
            if mc_result.covariance is not None
            else np.diag(mc_result.std**2),
            partial_derivatives=None,
        )

    elif method == "numerical":
        # Same as analytical but emphasizes numerical derivatives
        jacobian = _compute_jacobian(model, param_values, delta, method="central")

        propagated_cov = jacobian @ covariance @ jacobian.T
        propagated_errors = np.sqrt(np.diag(propagated_cov))
        output_values = np.atleast_1d(model(param_values))
        correlation_matrix = _covariance_to_correlation(propagated_cov)

        return PropagatedErrors(
            values=output_values,
            errors=propagated_errors,
            correlation_matrix=correlation_matrix,
            covariance_matrix=propagated_cov,
            partial_derivatives=jacobian,
        )

    else:
        raise ValueError(f"Unknown method: {method}")


def compute_covariance_matrix(
    samples: np.ndarray, rowvar: bool = False, bias: bool = False
) -> np.ndarray:
    """
    Compute covariance matrix from samples

    Wrapper around numpy.cov with better handling for edge cases
    and additional validation.

    Args:
        samples: Array of samples [n_samples, n_variables]
        rowvar: If True, each row is a variable, each column an observation
        bias: If False (default), normalize by N-1 (unbiased estimator)
              If True, normalize by N (maximum likelihood estimator)

    Returns:
        Covariance matrix [n_variables, n_variables]

    References:
        Press et al. (1992): Numerical Recipes, Section 14.5
    """
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)

    # Remove NaN values
    valid_mask = ~np.any(np.isnan(samples), axis=1)
    samples_clean = samples[valid_mask]

    if len(samples_clean) < 2:
        raise ValueError("Need at least 2 valid samples to compute covariance")

    ddof = 0 if bias else 1
    cov = np.cov(samples_clean, rowvar=rowvar, bias=bias, ddof=ddof)

    # Ensure 2D even for single variable
    if np.isscalar(cov):
        cov = np.array([[cov]])

    return cov


def correlation_from_covariance(covariance: np.ndarray) -> np.ndarray:
    """
    Convert covariance matrix to correlation matrix

    Args:
        covariance: Covariance matrix Σ

    Returns:
        Correlation matrix ρ where ρ_ij = Σ_ij / sqrt(Σ_ii Σ_jj)
    """
    return _covariance_to_correlation(covariance)


def gaussian_error_propagation(
    values: np.ndarray, errors: np.ndarray, operation: str = "sum"
) -> Tuple[float, float]:
    """
    Analytical error propagation for common operations

    Assumes Gaussian errors and uses standard formulas.

    Supported operations:
        - 'sum': z = Σ x_i, σ_z² = Σ σ_i²
        - 'product': z = Π x_i, (σ_z/z)² = Σ (σ_i/x_i)²
        - 'ratio': z = x/y, (σ_z/z)² = (σ_x/x)² + (σ_y/y)²
        - 'power': z = x^n, σ_z/z = |n| σ_x/x

    Args:
        values: Input values
        errors: Input errors (1-sigma)
        operation: Operation type

    Returns:
        Tuple of (result, propagated_error)

    References:
        Bevington & Robinson (2003): Chapter 3, Error Propagation Formulas
    """
    values = np.atleast_1d(values)
    errors = np.atleast_1d(errors)

    if operation == "sum":
        result = np.sum(values)
        error = np.sqrt(np.sum(errors**2))

    elif operation == "product":
        result = np.prod(values)
        rel_errors = errors / values
        error = np.abs(result) * np.sqrt(np.sum(rel_errors**2))

    elif operation == "ratio":
        if len(values) != 2 or len(errors) != 2:
            raise ValueError("Ratio requires exactly 2 values")
        x, y = values
        sx, sy = errors
        result = x / y
        rel_error = np.sqrt((sx / x) ** 2 + (sy / y) ** 2)
        error = np.abs(result) * rel_error

    elif operation == "power":
        if len(values) != 2:
            raise ValueError("Power requires [base, exponent]")
        x, n = values
        result = x**n
        rel_error = np.abs(n) * errors[0] / x
        error = np.abs(result) * rel_error

    else:
        raise ValueError(f"Unknown operation: {operation}")

    return result, error


def compute_prediction_uncertainty_map(
    predictions: np.ndarray, confidence: float = 0.95, method: str = "percentile"
) -> Dict[str, np.ndarray]:
    """
    Compute uncertainty map for 2D predictions (e.g., convergence maps)

    Args:
        predictions: Array of predictions [n_samples, height, width]
        confidence: Confidence level
        method: Method for computing intervals

    Returns:
        Dictionary with 'mean', 'std', 'lower', 'upper' maps
    """
    mean_map = np.mean(predictions, axis=0)
    std_map = np.std(predictions, axis=0, ddof=1)

    lower_map, upper_map, _ = compute_confidence_intervals(
        predictions, confidence=confidence, method=method, axis=0
    )

    return {
        "mean": mean_map,
        "std": std_map,
        "lower": lower_map,
        "upper": upper_map,
        "confidence": confidence,
    }


# ============================================================================
# Helper Functions
# ============================================================================


def _compute_jacobian(
    func: Callable, params: np.ndarray, delta: float = 1e-6, method: str = "forward"
) -> np.ndarray:
    """
    Compute Jacobian matrix using numerical differentiation

    Args:
        func: Function f(x) -> array
        params: Parameter values at which to compute Jacobian
        delta: Step size for numerical differentiation
        method: Differentiation method ('forward', 'backward', 'central')

    Returns:
        Jacobian matrix J_ij = ∂f_i/∂x_j
    """
    params = np.atleast_1d(params)
    n_params = len(params)

    # Evaluate at reference point
    f0 = np.atleast_1d(func(params))
    n_outputs = len(f0)

    jacobian = np.zeros((n_outputs, n_params))

    for j in range(n_params):
        params_plus = params.copy()
        params_minus = params.copy()

        if method == "central":
            # Central difference (more accurate, O(h²))
            params_plus[j] += delta
            params_minus[j] -= delta
            f_plus = np.atleast_1d(func(params_plus))
            f_minus = np.atleast_1d(func(params_minus))
            jacobian[:, j] = (f_plus - f_minus) / (2 * delta)
        else:
            # Forward difference (O(h))
            params_plus[j] += delta
            f_plus = np.atleast_1d(func(params_plus))
            jacobian[:, j] = (f_plus - f0) / delta

    return jacobian


def _covariance_to_correlation(covariance: np.ndarray) -> np.ndarray:
    """
    Convert covariance matrix to correlation matrix

    Args:
        covariance: Covariance matrix

    Returns:
        Correlation matrix
    """
    std = np.sqrt(np.diag(covariance))
    std_matrix = np.outer(std, std)
    correlation = covariance / (std_matrix + 1e-10)

    # Ensure diagonal is exactly 1
    np.fill_diagonal(correlation, 1.0)

    return correlation


def _generate_correlated_samples(
    mean: np.ndarray,
    covariance: np.ndarray,
    n_samples: int,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Generate correlated Gaussian samples using Cholesky decomposition

    Args:
        mean: Mean vector
        covariance: Covariance matrix
        n_samples: Number of samples to generate
        random_state: Random seed

    Returns:
        Array of samples [n_samples, n_dim]
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Cholesky decomposition: Σ = L L^T
    L = cholesky(covariance, lower=True)

    # Generate standard normal samples
    z = np.random.standard_normal((n_samples, len(mean)))

    # Transform: x = μ + L z
    samples = mean + z @ L.T

    return samples


# ============================================================================
# Advanced Methods
# ============================================================================


def hierarchical_bootstrap(
    data_groups: List[np.ndarray],
    statistic_func: Callable,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Hierarchical bootstrap for grouped/correlated data

    For data with hierarchical structure (e.g., multiple observations per lens),
    standard bootstrap can underestimate errors. This method resamples at the
    group level first, then within groups.

    Args:
        data_groups: List of data arrays, one per group
        statistic_func: Function to compute statistic
        n_bootstrap: Number of bootstrap iterations
        confidence: Confidence level
        random_state: Random seed

    Returns:
        Bootstrap results dictionary

    References:
        Efron & Tibshirani (1993): Chapter 8, Complex data structures
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_groups = len(data_groups)
    original_stat = statistic_func(np.concatenate(data_groups))

    bootstrap_stats = []

    for _ in range(n_bootstrap):
        # Step 1: Resample groups
        group_indices = np.random.choice(n_groups, size=n_groups, replace=True)

        # Step 2: Resample within each selected group
        resampled_data = []
        for idx in group_indices:
            group = data_groups[idx]
            within_indices = np.random.choice(len(group), size=len(group), replace=True)
            resampled_data.append(group[within_indices])

        combined = np.concatenate(resampled_data)

        try:
            stat = statistic_func(combined)
            bootstrap_stats.append(np.atleast_1d(stat))
        except Exception as e:
            warnings.warn(f"Statistic computation failed: {e}")
            continue

    bootstrap_stats = np.array(bootstrap_stats)

    return {
        "statistic": original_stat,
        "mean": np.mean(bootstrap_stats, axis=0),
        "std_error": np.std(bootstrap_stats, axis=0, ddof=1),
        "bias": np.mean(bootstrap_stats, axis=0) - original_stat,
        "lower": np.percentile(bootstrap_stats, 100 * (1 - confidence) / 2, axis=0),
        "upper": np.percentile(bootstrap_stats, 100 * (1 + confidence) / 2, axis=0),
        "confidence": confidence,
        "n_bootstrap": n_bootstrap,
    }


def jackknife_errors(
    data: np.ndarray, statistic_func: Callable, return_samples: bool = False
) -> Dict[str, Any]:
    """
    Jackknife resampling for error estimation

    The jackknife is a resampling method where each observation is left out
    once. It is particularly useful for bias estimation and when bootstrap
    is computationally expensive.

    Args:
        data: Data array
        statistic_func: Function to compute statistic
        return_samples: Return all jackknife samples

    Returns:
        Jackknife results dictionary

    References:
        Efron (1979): Bootstrap methods
        Efron & Tibshirani (1993): Chapter 11
    """
    n = len(data)
    original_stat = statistic_func(data)

    # Leave-one-out samples
    jackknife_stats = []

    for i in range(n):
        # Remove i-th observation
        reduced_data = np.delete(data, i, axis=0)
        stat = statistic_func(reduced_data)
        jackknife_stats.append(np.atleast_1d(stat))

    jackknife_stats = np.array(jackknife_stats)

    # Jackknife mean
    jack_mean = np.mean(jackknife_stats, axis=0)

    # Jackknife standard error
    factor = (n - 1) / n
    jack_var = factor * np.sum((jackknife_stats - jack_mean) ** 2, axis=0)
    jack_std = np.sqrt(jack_var)

    # Bias estimate: (n-1) * (jack_mean - original)
    bias = (n - 1) * (jack_mean - original_stat)

    result = {
        "statistic": original_stat,
        "jackknife_mean": jack_mean,
        "std_error": jack_std,
        "bias": bias,
        "n": n,
    }

    if return_samples:
        result["jackknife_samples"] = jackknife_stats

    return result


def weighted_bootstrap(
    data: np.ndarray,
    weights: np.ndarray,
    statistic_func: Callable,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Weighted bootstrap for data with non-uniform sampling probabilities

    Args:
        data: Data array
        weights: Sampling weights (probabilities)
        statistic_func: Function to compute statistic
        n_bootstrap: Number of bootstrap iterations
        confidence: Confidence level
        random_state: Random seed

    Returns:
        Bootstrap results dictionary
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Normalize weights
    weights = np.array(weights) / np.sum(weights)
    n = len(data)

    original_stat = statistic_func(data)
    bootstrap_stats = []

    for _ in range(n_bootstrap):
        # Resample with weights
        indices = np.random.choice(n, size=n, replace=True, p=weights)
        bootstrap_sample = data[indices]

        try:
            stat = statistic_func(bootstrap_sample)
            bootstrap_stats.append(np.atleast_1d(stat))
        except Exception as e:
            warnings.warn(f"Statistic computation failed: {e}")
            continue

    bootstrap_stats = np.array(bootstrap_stats)

    return {
        "statistic": original_stat,
        "mean": np.mean(bootstrap_stats, axis=0),
        "std_error": np.std(bootstrap_stats, axis=0, ddof=1),
        "lower": np.percentile(bootstrap_stats, 100 * (1 - confidence) / 2, axis=0),
        "upper": np.percentile(bootstrap_stats, 100 * (1 + confidence) / 2, axis=0),
        "confidence": confidence,
        "n_bootstrap": n_bootstrap,
    }


# ============================================================================
# Visualization and Reporting Functions
# ============================================================================


def format_uncertainty(value: float, error: float, significant_figures: int = 2) -> str:
    """
    Format a value with its uncertainty in scientific notation

    Example:
        >>> format_uncertainty(1.23456, 0.0234)
        '1.235 ± 0.023'
    """
    if error == 0:
        return f"{value:.{significant_figures}f}"

    # Determine decimal places from error magnitude
    decimal_places = -int(np.floor(np.log10(error))) + significant_figures - 1
    decimal_places = max(0, decimal_places)

    return f"{value:.{decimal_places}f} ± {error:.{decimal_places}f}"


def print_uncertainty_report(
    result: Union[UncertaintyResult, PropagatedErrors, Dict],
    title: str = "Uncertainty Analysis",
) -> None:
    """
    Print a formatted uncertainty report

    Args:
        result: Uncertainty result object or dictionary
        title: Report title
    """
    print("=" * 70)
    print(f" {title}")
    print("=" * 70)
    print()

    if isinstance(result, UncertaintyResult):
        print(f"Confidence Level: {result.confidence:.1%}")
        print()
        print("PREDICTION STATISTICS:")
        print(f"  Mean:  {np.mean(result.mean):.6f}")
        print(f"  Std:   {np.mean(result.std):.6f}")
        print()
        print("CONFIDENCE INTERVALS:")
        print(f"  Lower: {np.mean(result.lower):.6f}")
        print(f"  Upper: {np.mean(result.upper):.6f}")

        if result.covariance is not None:
            print()
            print("COVARIANCE MATRIX:")
            print(result.covariance)

    elif isinstance(result, PropagatedErrors):
        print("PROPAGATED VALUES:")
        for i, (val, err) in enumerate(zip(result.values, result.errors)):
            print(f"  Parameter {i + 1}: {format_uncertainty(val, err)}")

        print()
        print("CORRELATION MATRIX:")
        print(result.correlation_matrix)

        print()
        print("COVARIANCE MATRIX:")
        print(result.covariance_matrix)

    elif isinstance(result, dict):
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                if value.ndim == 0 or (value.ndim == 1 and len(value) == 1):
                    print(f"  {key}: {value.item():.6f}")
                else:
                    print(f"  {key}: {value}")
            else:
                print(f"  {key}: {value}")

    print()
    print("=" * 70)


# ============================================================================
# Convenience Functions for Common Use Cases
# ============================================================================


def lens_parameter_uncertainty(
    mass: float,
    mass_error: float,
    concentration: float,
    concentration_error: float,
    redshift: float,
    redshift_error: float,
    correlations: Optional[np.ndarray] = None,
    output_quantity: str = "einstein_radius",
) -> PropagatedErrors:
    """
    Compute uncertainties for common lens parameters

    Args:
        mass: Virial mass in solar masses
        mass_error: 1-sigma error on mass
        concentration: Concentration parameter
        concentration_error: 1-sigma error on concentration
        redshift: Lens redshift
        redshift_error: 1-sigma error on redshift
        correlations: 3x3 correlation matrix (optional)
        output_quantity: Quantity to compute ('einstein_radius', 'mass_density', etc.)

    Returns:
        PropagatedErrors object
    """
    params = np.array([mass, concentration, redshift])
    errors = np.array([mass_error, concentration_error, redshift_error])

    def lens_model(p):
        M, c, z = p
        if output_quantity == "einstein_radius":
            # Simplified Einstein radius (arcseconds)
            # θ_E ∝ sqrt(M * D_ls / (D_l * D_s))
            # Approximation: θ_E ∝ sqrt(M) * (1+z)^(-0.5)
            return np.array([1.0 * np.sqrt(M / 1e14) * (1 + z) ** (-0.5)])
        elif output_quantity == "mass_density":
            # Central mass density
            return np.array([M * c**3 / (4 * np.pi * (np.log(1 + c) - c / (1 + c)))])
        else:
            raise ValueError(f"Unknown output quantity: {output_quantity}")

    return propagate_parameter_errors(
        model=lens_model,
        param_values=params,
        param_errors=errors,
        param_correlations=correlations,
        method="analytical",
    )


def convergence_map_uncertainty(
    convergence_maps: np.ndarray, confidence: float = 0.95
) -> Dict[str, np.ndarray]:
    """
    Compute uncertainty maps from multiple convergence predictions

    Args:
        convergence_maps: Array of convergence maps [n_samples, height, width]
        confidence: Confidence level

    Returns:
        Dictionary with mean, std, lower, upper maps
    """
    return compute_prediction_uncertainty_map(
        predictions=convergence_maps, confidence=confidence, method="percentile"
    )


__all__ = [
    # Main functions
    "monte_carlo_error_propagation",
    "compute_confidence_intervals",
    "bootstrap_errors",
    "propagate_parameter_errors",
    "compute_covariance_matrix",
    "correlation_from_covariance",
    "gaussian_error_propagation",
    "compute_prediction_uncertainty_map",
    # Advanced methods
    "hierarchical_bootstrap",
    "jackknife_errors",
    "weighted_bootstrap",
    # Convenience functions
    "lens_parameter_uncertainty",
    "convergence_map_uncertainty",
    "format_uncertainty",
    "print_uncertainty_report",
    # Data classes
    "UncertaintyResult",
    "PropagatedErrors",
]
