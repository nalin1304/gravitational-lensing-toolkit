"""
FastAPI REST API for Gravitational Lensing Analysis

This package provides a comprehensive REST API for gravitational lensing
computations, including lens model creation, ray tracing, wave optics,
and Physics-Informed Neural Network (PINN) inference.

Modules
-------
main : FastAPI application with all endpoints
models : Pydantic models for request/response validation
auth : Authentication utilities (JWT, password hashing)

Example
-------
Run the API server:

    $ python -m api.main

Or using uvicorn directly:

    $ uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

The API documentation will be available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
"""

__version__ = "2.1.0"
__author__ = "Gravitational Lensing Toolkit"

from api.models import (
    # Base models
    LensSystemParams,
    GridParams,
    PositionList,
    # Lens request models
    PointMassLensRequest,
    NFWLensRequest,
    SersicLensRequest,
    CompositeLensRequest,
    CompositeComponent,
    # Lens response models
    LensModelResponse,
    # Computation request models
    DeflectionRequest,
    ConvergenceRequest,
    PotentialRequest,
    ImagesRequest,
    TimeDelayRequest,
    # Computation response models
    DeflectionResponse,
    ConvergenceResponse,
    PotentialResponse,
    ImagesResponse,
    TimeDelayResponse,
    # Wave optics models
    WaveAmplificationRequest,
    WaveAmplificationResponse,
    FringeInfo,
    WaveComparisonRequest,
    WaveComparisonResponse,
    # PINN models
    PINNPredictRequest,
    PINNPredictResponse,
    PINNTrainRequest,
    PINNTrainResponse,
    PINNStatusResponse,
    # Validation models
    HealthResponse,
    TestResult,
    TestSuiteResponse,
    BenchmarkResult,
    SystemInfo,
    BenchmarkResponse,
    # Synthetic models
    SyntheticRequest,
    CoordinateGrids,
    SyntheticResponse,
)

from api.auth import (
    verify_password,
    get_password_hash,
    create_access_token,
    create_refresh_token,
    decode_token,
    verify_token,
    get_current_user,
    get_current_active_user,
    get_current_admin_user,
    get_optional_user,
    generate_api_key,
    hash_api_key,
    verify_api_key,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Base models
    "LensSystemParams",
    "GridParams",
    "PositionList",
    # Lens models
    "PointMassLensRequest",
    "NFWLensRequest",
    "SersicLensRequest",
    "CompositeLensRequest",
    "CompositeComponent",
    "LensModelResponse",
    # Computation models
    "DeflectionRequest",
    "ConvergenceRequest",
    "PotentialRequest",
    "ImagesRequest",
    "TimeDelayRequest",
    "DeflectionResponse",
    "ConvergenceResponse",
    "PotentialResponse",
    "ImagesResponse",
    "TimeDelayResponse",
    # Wave optics models
    "WaveAmplificationRequest",
    "WaveAmplificationResponse",
    "FringeInfo",
    "WaveComparisonRequest",
    "WaveComparisonResponse",
    # PINN models
    "PINNPredictRequest",
    "PINNPredictResponse",
    "PINNTrainRequest",
    "PINNTrainResponse",
    "PINNStatusResponse",
    # Validation models
    "HealthResponse",
    "TestResult",
    "TestSuiteResponse",
    "BenchmarkResult",
    "SystemInfo",
    "BenchmarkResponse",
    # Synthetic models
    "SyntheticRequest",
    "CoordinateGrids",
    "SyntheticResponse",
    # Auth functions
    "verify_password",
    "get_password_hash",
    "create_access_token",
    "create_refresh_token",
    "decode_token",
    "verify_token",
    "get_current_user",
    "get_current_active_user",
    "get_current_admin_user",
    "get_optional_user",
    "generate_api_key",
    "hash_api_key",
    "verify_api_key",
]
