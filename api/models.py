"""
Pydantic Models for FastAPI Request/Response Validation

This module defines all Pydantic models used for API request and response
validation, ensuring type safety and automatic OpenAPI documentation generation.

Author: Gravitational Lensing Toolkit
Date: March 2025
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from enum import Enum


# ============================================================================
# Base Models
# ============================================================================


class LensSystemParams(BaseModel):
    """Parameters for defining a lens system with cosmology."""

    z_lens: float = Field(..., ge=0.001, le=10.0, description="Lens redshift")
    z_source: float = Field(
        ..., ge=0.001, le=15.0, description="Source redshift (must be > z_lens)"
    )
    H0: float = Field(
        70.0, ge=50.0, le=100.0, description="Hubble constant in km/s/Mpc"
    )
    Omega_m: float = Field(0.3, ge=0.1, le=0.9, description="Matter density parameter")

    @field_validator("z_source")
    @classmethod
    def validate_redshift_order(cls, v: float, info) -> float:
        if "z_lens" in info.data and v <= info.data["z_lens"]:
            raise ValueError("Source redshift must be greater than lens redshift")
        return v


class GridParams(BaseModel):
    """Parameters for computational grids."""

    extent: float = Field(10.0, gt=0.0, description="Half-width of grid in arcseconds")
    resolution: int = Field(
        128, ge=32, le=1024, description="Number of grid points per dimension"
    )


class PositionList(BaseModel):
    """List of 2D positions."""

    x: List[float] = Field(..., description="X coordinates")
    y: List[float] = Field(..., description="Y coordinates")

    @field_validator("y")
    @classmethod
    def validate_same_length(cls, v: List[float], info) -> List[float]:
        if "x" in info.data and len(v) != len(info.data["x"]):
            raise ValueError("x and y must have the same length")
        return v


# ============================================================================
# Request Models - Lens Creation
# ============================================================================


class PointMassLensRequest(BaseModel):
    """Request to create a point mass lens."""

    mass: float = Field(..., gt=0.0, description="Mass in solar masses")
    lens_system: LensSystemParams = Field(..., description="Cosmological parameters")


class NFWLensRequest(BaseModel):
    """Request to create an NFW lens."""

    M_vir: float = Field(..., gt=0.0, description="Virial mass in solar masses")
    concentration: float = Field(
        ..., gt=0.0, description="Concentration parameter c = r_vir / r_s"
    )
    lens_system: LensSystemParams = Field(..., description="Cosmological parameters")
    ellipticity: float = Field(0.0, ge=0.0, lt=1.0, description="Ellipticity parameter")
    ellipticity_angle: float = Field(
        0.0, ge=0.0, lt=360.0, description="Position angle in degrees"
    )
    include_subhalos: bool = Field(
        False, description="Whether to include subhalo population"
    )
    subhalo_fraction: float = Field(
        0.05, ge=0.0, le=0.2, description="Fraction of mass in subhalos"
    )


class SersicLensRequest(BaseModel):
    """Request to create a Sérsic lens."""

    total_mass: float = Field(..., gt=0.0, description="Total mass in solar masses")
    n: float = Field(
        ...,
        gt=0.0,
        description="Sérsic index (n=1 for exponential, n=4 for de Vaucouleurs)",
    )
    effective_radius: float = Field(..., gt=0.0, description="Effective radius in kpc")
    lens_system: LensSystemParams = Field(..., description="Cosmological parameters")
    ellipticity: float = Field(0.0, ge=0.0, lt=1.0, description="Ellipticity parameter")
    position_angle: float = Field(
        0.0, ge=0.0, lt=360.0, description="Position angle in degrees"
    )
    center_x: float = Field(0.0, description="Center x-coordinate in arcseconds")
    center_y: float = Field(0.0, description="Center y-coordinate in arcseconds")


class CompositeComponent(BaseModel):
    """Component for composite lens model."""

    type: str = Field(
        ..., description="Component type: 'point_mass', 'nfw', or 'sersic'"
    )
    parameters: Dict[str, float] = Field(
        ..., description="Component-specific parameters"
    )
    center_x: float = Field(0.0, description="Center x-coordinate in arcseconds")
    center_y: float = Field(0.0, description="Center y-coordinate in arcseconds")


class CompositeLensRequest(BaseModel):
    """Request to create a composite lens."""

    components: List[CompositeComponent] = Field(
        ..., min_length=1, max_length=10, description="List of lens components"
    )
    lens_system: LensSystemParams = Field(..., description="Cosmological parameters")


# ============================================================================
# Response Models - Lens Creation
# ============================================================================


class LensModelResponse(BaseModel):
    """Response containing lens model information."""

    model_id: str = Field(..., description="Unique model identifier")
    model_type: str = Field(..., description="Type of lens model")
    parameters: Dict[str, Any] = Field(..., description="Model parameters")
    metadata: Dict[str, Any] = Field(..., description="Additional metadata")
    created_at: str = Field(..., description="Creation timestamp")


# ============================================================================
# Request Models - Computations
# ============================================================================


class DeflectionRequest(BaseModel):
    """Request to compute deflection angles."""

    model_id: str = Field(..., description="Lens model identifier")
    positions: PositionList = Field(..., description="Positions to evaluate")


class ConvergenceRequest(BaseModel):
    """Request to compute convergence map."""

    model_id: str = Field(..., description="Lens model identifier")
    grid: GridParams = Field(
        default_factory=lambda: GridParams(), description="Grid parameters"
    )


class PotentialRequest(BaseModel):
    """Request to compute lensing potential."""

    model_id: str = Field(..., description="Lens model identifier")
    grid: GridParams = Field(
        default_factory=lambda: GridParams(), description="Grid parameters"
    )


class ImagesRequest(BaseModel):
    """Request to find lensed images."""

    model_id: str = Field(..., description="Lens model identifier")
    source_position: List[float] = Field(
        ...,
        min_length=2,
        max_length=2,
        description="Source position (x, y) in arcseconds",
    )
    grid_extent: float = Field(
        5.0, gt=0.0, description="Half-width of search grid in arcseconds"
    )
    grid_resolution: int = Field(
        300, ge=100, le=1000, description="Grid resolution for ray tracing"
    )
    threshold: float = Field(
        0.05, gt=0.0, description="Distance threshold for image identification"
    )


class TimeDelayRequest(BaseModel):
    """Request to compute time delays."""

    model_id: str = Field(..., description="Lens model identifier")
    image_positions: List[List[float]] = Field(
        ...,
        min_length=1,
        description="List of image positions [[x1, y1], [x2, y2], ...]",
    )
    source_position: List[float] = Field(
        ...,
        min_length=2,
        max_length=2,
        description="Source position (x, y) in arcseconds",
    )


# ============================================================================
# Response Models - Computations
# ============================================================================


class DeflectionResponse(BaseModel):
    """Response containing deflection angles."""

    job_id: str = Field(..., description="Unique job identifier")
    deflection_x: List[float] = Field(
        ..., description="X-component of deflection angle in arcseconds"
    )
    deflection_y: List[float] = Field(
        ..., description="Y-component of deflection angle in arcseconds"
    )
    magnitudes: List[float] = Field(..., description="Deflection angle magnitudes")
    timestamp: str = Field(..., description="Completion timestamp")


class ConvergenceResponse(BaseModel):
    """Response containing convergence map."""

    job_id: str = Field(..., description="Unique job identifier")
    convergence_map: List[List[float]] = Field(
        ..., description="2D convergence map (dimensionless)"
    )
    grid_x: List[float] = Field(..., description="X-coordinates of grid")
    grid_y: List[float] = Field(..., description="Y-coordinates of grid")
    min_value: float = Field(..., description="Minimum convergence value")
    max_value: float = Field(..., description="Maximum convergence value")
    mean_value: float = Field(..., description="Mean convergence value")
    timestamp: str = Field(..., description="Completion timestamp")


class PotentialResponse(BaseModel):
    """Response containing lensing potential."""

    job_id: str = Field(..., description="Unique job identifier")
    potential_map: List[List[float]] = Field(
        ..., description="2D lensing potential map in arcsec²"
    )
    grid_x: List[float] = Field(..., description="X-coordinates of grid")
    grid_y: List[float] = Field(..., description="Y-coordinates of grid")
    min_value: float = Field(..., description="Minimum potential value")
    max_value: float = Field(..., description="Maximum potential value")
    mean_value: float = Field(..., description="Mean potential value")
    timestamp: str = Field(..., description="Completion timestamp")


class ImagesResponse(BaseModel):
    """Response containing lensed image positions."""

    job_id: str = Field(..., description="Unique job identifier")
    image_positions: List[List[float]] = Field(
        ..., description="Image positions [[x1, y1], ...] in arcseconds"
    )
    magnifications: List[float] = Field(..., description="Magnification of each image")
    n_images: int = Field(..., description="Number of images found")
    source_position: List[float] = Field(
        ..., description="Source position (x, y) in arcseconds"
    )
    timestamp: str = Field(..., description="Completion timestamp")


class TimeDelayResponse(BaseModel):
    """Response containing time delays."""

    job_id: str = Field(..., description="Unique job identifier")
    time_delays: List[float] = Field(..., description="Time delays in days")
    relative_delays: List[float] = Field(
        ..., description="Time delays relative to first image"
    )
    image_positions: List[List[float]] = Field(..., description="Image positions")
    source_position: List[float] = Field(..., description="Source position")
    unit: str = Field("days", description="Unit of time delay")
    timestamp: str = Field(..., description="Completion timestamp")


# ============================================================================
# Request/Response Models - Wave Optics
# ============================================================================


class WaveAmplificationRequest(BaseModel):
    """Request to compute wave optical amplification."""

    model_id: str = Field(..., description="Lens model identifier")
    source_position: List[float] = Field(
        default=[0.5, 0.0],
        min_length=2,
        max_length=2,
        description="Source position (x, y) in arcseconds",
    )
    wavelength: float = Field(500.0, gt=0.0, description="Wavelength in nanometers")
    grid_size: int = Field(
        512, ge=128, le=2048, description="Grid size (power of 2 recommended)"
    )
    grid_extent: float = Field(3.0, gt=0.0, description="Grid extent in arcseconds")


class FringeInfo(BaseModel):
    """Information about detected interference fringes."""

    fringe_spacing: float = Field(
        ..., description="Average fringe spacing in arcseconds"
    )
    n_fringes: int = Field(..., description="Number of distinct fringes detected")
    fringe_contrast: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fringe contrast (I_max - I_min) / (I_max + I_min)",
    )


class WaveAmplificationResponse(BaseModel):
    """Response containing wave optical amplification."""

    job_id: str = Field(..., description="Unique job identifier")
    amplitude_map: List[List[float]] = Field(..., description="Intensity map |F|²")
    phase_map: List[List[float]] = Field(..., description="Phase map in radians")
    fermat_potential: List[List[float]] = Field(
        ..., description="Fermat potential Φ(θ) in arcsec²"
    )
    grid_x: List[float] = Field(..., description="X-coordinates")
    grid_y: List[float] = Field(..., description="Y-coordinates")
    wavelength: float = Field(..., description="Wavelength in nm")
    fringe_info: FringeInfo = Field(..., description="Detected fringe information")
    timestamp: str = Field(..., description="Completion timestamp")


class WaveComparisonRequest(BaseModel):
    """Request to compare wave vs geometric optics."""

    model_id: str = Field(..., description="Lens model identifier")
    source_position: List[float] = Field(
        default=[0.5, 0.0],
        min_length=2,
        max_length=2,
        description="Source position (x, y) in arcseconds",
    )
    wavelength: float = Field(500.0, gt=0.0, description="Wavelength in nanometers")
    grid_size: int = Field(512, ge=128, le=2048, description="Grid size")
    grid_extent: float = Field(3.0, gt=0.0, description="Grid extent in arcseconds")


class WaveComparisonResponse(BaseModel):
    """Response comparing wave vs geometric optics."""

    job_id: str = Field(..., description="Unique job identifier")
    wave_amplitude: List[List[float]] = Field(
        ..., description="Wave optics amplitude map"
    )
    geometric_convergence: List[List[float]] = Field(
        ..., description="Geometric optics convergence map"
    )
    fractional_difference: List[List[float]] = Field(
        ..., description="Fractional difference map"
    )
    max_difference: float = Field(..., description="Maximum fractional difference")
    mean_difference: float = Field(..., description="Mean fractional difference")
    significant_pixels: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of pixels with significant difference",
    )
    grid_x: List[float] = Field(..., description="X-coordinates")
    grid_y: List[float] = Field(..., description="Y-coordinates")
    wavelength: float = Field(..., description="Wavelength in nm")
    timestamp: str = Field(..., description="Completion timestamp")


# ============================================================================
# Request/Response Models - PINN
# ============================================================================


class PINNPredictRequest(BaseModel):
    """Request for PINN prediction."""

    convergence_map: List[List[float]] = Field(
        ..., description="2D convergence map for inference"
    )
    target_size: int = Field(
        64, ge=32, le=256, description="Target size for model input"
    )
    mc_samples: int = Field(
        1, ge=1, le=1000, description="Number of MC Dropout samples for uncertainty"
    )


class PINNPredictResponse(BaseModel):
    """Response from PINN prediction."""

    job_id: str = Field(..., description="Unique job identifier")
    predictions: Dict[str, float] = Field(
        ...,
        description="Predicted parameters (M_vir, r_s, ellipticity, beta_x, beta_y, H0)",
    )
    uncertainties: Optional[Dict[str, float]] = Field(
        None, description="Parameter uncertainties (if mc_samples > 1)"
    )
    classification: Dict[str, float] = Field(
        ..., description="Dark matter model classification probabilities"
    )
    entropy: float = Field(..., ge=0.0, description="Predictive entropy")
    timestamp: str = Field(..., description="Completion timestamp")


class PINNTrainRequest(BaseModel):
    """Request to start PINN training."""

    epochs: int = Field(100, ge=1, le=10000, description="Number of training epochs")
    learning_rate: float = Field(0.001, gt=0.0, description="Learning rate")
    batch_size: int = Field(32, ge=1, le=256, description="Batch size")
    input_size: int = Field(64, ge=32, le=256, description="Input image size")
    dropout_rate: float = Field(0.2, ge=0.0, le=0.5, description="Dropout rate")
    lambda_physics: float = Field(
        0.1, ge=0.0, le=1.0, description="Weight for physics loss term"
    )
    use_gpu: bool = Field(True, description="Whether to use GPU if available")


class PINNTrainResponse(BaseModel):
    """Response from PINN training submission."""

    job_id: str = Field(..., description="Unique training job identifier")
    status: str = Field(..., description="Job status (pending)")
    message: str = Field(..., description="Status message")
    status_url: str = Field(..., description="URL to check training status")
    submitted_at: str = Field(..., description="Submission timestamp")


class PINNStatusResponse(BaseModel):
    """Response containing PINN training status."""

    job_id: str = Field(..., description="Training job identifier")
    status: str = Field(
        ..., description="Current status (pending/running/completed/failed)"
    )
    progress: float = Field(
        ..., ge=0.0, le=100.0, description="Training progress percentage"
    )
    epoch: int = Field(..., ge=0, description="Current epoch")
    total_epochs: int = Field(..., ge=1, description="Total number of epochs")
    losses: Dict[str, float] = Field(
        default_factory=dict, description="Current loss values"
    )
    start_time: Optional[str] = Field(None, description="Training start time")
    end_time: Optional[str] = Field(None, description="Training end time")
    model_path: Optional[str] = Field(
        None, description="Path to saved model (if completed)"
    )
    error: Optional[str] = Field(None, description="Error message (if failed)")


# ============================================================================
# Request/Response Models - Validation
# ============================================================================


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="System status (healthy/degraded)")
    timestamp: str = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version")
    gpu_available: bool = Field(..., description="Whether GPU is available")
    database_connected: Optional[bool] = Field(
        None, description="Whether database is connected"
    )


class TestResult(BaseModel):
    """Individual test result."""

    test_name: str = Field(..., description="Name of the test")
    status: str = Field(..., description="Test status (passed/failed/skipped)")
    duration: float = Field(..., ge=0.0, description="Test duration in seconds")
    error: Optional[str] = Field(None, description="Error message if failed")


class TestSuiteResponse(BaseModel):
    """Test suite execution response."""

    job_id: str = Field(..., description="Unique job identifier")
    passed: int = Field(..., ge=0, description="Number of passed tests")
    failed: int = Field(..., ge=0, description="Number of failed tests")
    skipped: int = Field(..., ge=0, description="Number of skipped tests")
    total: int = Field(..., ge=0, description="Total number of tests")
    duration: float = Field(..., ge=0.0, description="Total duration in seconds")
    results: List[TestResult] = Field(..., description="Individual test results")
    timestamp: str = Field(..., description="Completion timestamp")


class BenchmarkResult(BaseModel):
    """Individual benchmark result."""

    name: str = Field(..., description="Benchmark name")
    mean_time: float = Field(..., ge=0.0, description="Mean execution time")
    std_time: float = Field(
        ..., ge=0.0, description="Standard deviation of execution time"
    )
    unit: str = Field(..., description="Time unit (seconds)")
    throughput: float = Field(
        ..., ge=0.0, description="Throughput (operations per second)"
    )


class SystemInfo(BaseModel):
    """System information."""

    cpu: str = Field(..., description="CPU information")
    gpu: str = Field(..., description="GPU information")
    ram_gb: int = Field(..., ge=1, description="RAM in GB")
    python_version: str = Field(..., description="Python version")


class BenchmarkResponse(BaseModel):
    """Benchmark results response."""

    job_id: str = Field(..., description="Unique job identifier")
    benchmarks: List[BenchmarkResult] = Field(..., description="Benchmark results")
    system_info: SystemInfo = Field(..., description="System information")
    timestamp: str = Field(..., description="Completion timestamp")


# ============================================================================
# Request/Response Models - Synthetic Generation
# ============================================================================


class SyntheticRequest(BaseModel):
    """Request for synthetic convergence map generation."""

    profile_type: str = Field(
        ..., description="Profile type: 'NFW' or 'Elliptical NFW'"
    )
    mass: float = Field(
        ..., ge=1e11, le=1e14, description="Virial mass in solar masses"
    )
    scale_radius: float = Field(
        200.0, ge=50.0, le=500.0, description="Scale radius in kpc"
    )
    ellipticity: float = Field(0.0, ge=0.0, le=0.5, description="Ellipticity parameter")
    grid_size: int = Field(64, description="Grid size (32, 64, or 128)")

    @field_validator("profile_type")
    @classmethod
    def validate_profile_type(cls, v: str) -> str:
        if v not in ["NFW", "Elliptical NFW"]:
            raise ValueError("profile_type must be 'NFW' or 'Elliptical NFW'")
        return v

    @field_validator("grid_size")
    @classmethod
    def validate_grid_size(cls, v: int) -> int:
        if v not in [32, 64, 128]:
            raise ValueError("grid_size must be 32, 64, or 128")
        return v


class CoordinateGrids(BaseModel):
    """Coordinate grids for synthetic map."""

    X: List[List[float]] = Field(..., description="X-coordinate grid")
    Y: List[List[float]] = Field(..., description="Y-coordinate grid")


class SyntheticResponse(BaseModel):
    """Response for synthetic convergence map generation."""

    job_id: str = Field(..., description="Unique job identifier")
    convergence_map: List[List[float]] = Field(..., description="2D convergence map")
    coordinates: CoordinateGrids = Field(..., description="Coordinate grids")
    metadata: Dict[str, Any] = Field(..., description="Generation metadata")
    timestamp: str = Field(..., description="Completion timestamp")
