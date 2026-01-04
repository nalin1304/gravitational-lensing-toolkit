"""
Machine Learning Module for Gravitational Lensing

This module provides physics-informed neural networks for:
- Lens parameter inference
- Dark matter model classification
- Training and evaluation utilities

Phase 7 Updates:
- GPU acceleration and performance optimization
- Vectorized convergence map generation

Phase 9 Updates:
- Transfer learning from synthetic to real data
- Domain adaptation (DANN, MMD, CORAL)
- Bayesian uncertainty quantification
- Fine-tuning strategies
"""

from .pinn import PhysicsInformedNN, physics_informed_loss
from .generate_dataset import (
    generate_training_data,
    generate_convergence_map_vectorized,
    generate_convergence_map
)
from .evaluate import evaluate_model, compute_metrics
from .augmentation import (
    RandomRotation, RandomFlip, RandomBrightness, RandomNoise,
    Compose, ToTensor, Normalize, get_training_transforms
)
try:
    from .tensorboard_logger import PINNLogger
except Exception:
    # Optional dependency (TensorBoard/TensorFlow)
    PINNLogger = None  # type: ignore
from .performance import (
    get_backend, set_backend, GPU_AVAILABLE,
    PerformanceMonitor, timer,
    benchmark_convergence_map, compare_cpu_gpu_performance,
    cached_convergence, clear_cache
)
from .transfer_learning import (
    TransferConfig,
    DomainAdaptationNetwork,
    MMDLoss,
    CORALLoss,
    BayesianUncertaintyEstimator,
    TransferLearningTrainer,
    create_synthetic_to_real_pipeline,
    compute_domain_discrepancy
)

__all__ = [
    'PhysicsInformedNN',
    'physics_informed_loss',
    'generate_training_data',
    'generate_convergence_map_vectorized',
    'generate_convergence_map',
    'evaluate_model',
    'compute_metrics',
    # Augmentation
    'RandomRotation',
    'RandomFlip',
    'RandomBrightness',
    'RandomNoise',
    'Compose',
    'ToTensor',
    'Normalize',
    'get_training_transforms',
    # Logging
    'PINNLogger',
    # Performance (Phase 7)
    'get_backend',
    'set_backend',
    'GPU_AVAILABLE',
    'PerformanceMonitor',
    'timer',
    'benchmark_convergence_map',
    'compare_cpu_gpu_performance',
    'cached_convergence',
    'clear_cache',
    # Transfer Learning (Phase 9)
    'TransferConfig',
    'DomainAdaptationNetwork',
    'MMDLoss',
    'CORALLoss',
    'BayesianUncertaintyEstimator',
    'TransferLearningTrainer',
    'create_synthetic_to_real_pipeline',
    'compute_domain_discrepancy',
]
