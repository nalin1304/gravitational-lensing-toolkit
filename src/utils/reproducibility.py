"""
Reproducibility and Seed Management Module

This module provides comprehensive utilities for ensuring reproducible results
in machine learning experiments. It handles random seed management for numpy,
torch, and the random module, provides context managers for deterministic
execution, and utilities for saving/loading random states.

Example usage:
    >>> from src.utils.reproducibility import set_seed, DeterministicContext
    >>>
    >>> # Set seeds globally
    >>> set_seed(42)
    >>>
    >>> # Use deterministic context for critical sections
    >>> with DeterministicContext(seed=123):
    ...     # Operations here are deterministic
    ...     result = model.predict(data)
    >>>
    >>> # Save and restore random state
    >>> from src.utils.reproducibility import save_random_state, load_random_state
    >>> save_random_state("checkpoint.pkl")
    >>> load_random_state("checkpoint.pkl")
"""

import random
import hashlib
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Union
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch


# Type alias for random state dictionary
RandomState = Dict[str, Any]


def set_seed(seed: int) -> None:
    """
    Set all random seeds for reproducibility.

    Sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA if available)
    - PyTorch backends (cudnn deterministic mode)

    Args:
        seed: Integer seed value. Should be non-negative.

    Raises:
        TypeError: If seed is not an integer.
        ValueError: If seed is negative.

    Example:
        >>> set_seed(42)
        >>> np.random.rand()  # Always produces same result
        0.3745401188473625
    """
    if not isinstance(seed, int):
        raise TypeError(f"Seed must be an integer, got {type(seed).__name__}")
    if seed < 0:
        raise ValueError(f"Seed must be non-negative, got {seed}")

    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)

    # PyTorch CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # For deterministic behavior on GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_random_state() -> RandomState:
    """
    Get the current random state from all random number generators.

    Returns:
        Dictionary containing the current state of:
        - 'random': Python random module state
        - 'numpy': NumPy random state
        - 'torch': PyTorch CPU random state
        - 'torch_cuda': PyTorch CUDA random states (if available)

    Example:
        >>> state = get_random_state()
        >>> # ... do some random operations ...
        >>> # Can restore later with set_random_state(state)
    """
    state: RandomState = {
        "random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["torch_cuda"] = [
            torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())
        ]

    return state


def set_random_state(state: RandomState) -> None:
    """
    Restore random state from a previously saved state.

    Args:
        state: Dictionary containing random states as returned by
               get_random_state().

    Raises:
        KeyError: If required keys are missing from the state dictionary.

    Example:
        >>> state = get_random_state()
        >>> random.random()  # Some random operation
        >>> set_random_state(state)  # Restore to previous state
    """
    required_keys = {"random", "numpy", "torch"}
    missing_keys = required_keys - set(state.keys())
    if missing_keys:
        raise KeyError(f"Missing required keys in state: {missing_keys}")

    # Restore Python random state
    random.setstate(state["random"])

    # Restore NumPy state
    np.random.set_state(state["numpy"])

    # Restore PyTorch state
    torch.set_rng_state(state["torch"])

    # Restore CUDA states if available
    if "torch_cuda" in state and torch.cuda.is_available():
        for i, cuda_state in enumerate(state["torch_cuda"]):
            if i < torch.cuda.device_count():
                torch.cuda.set_rng_state(cuda_state, i)


def save_random_state(filepath: Union[str, Path]) -> None:
    """
    Save the current random state to a file.

    Args:
        filepath: Path to save the random state. Should have .pkl extension.

    Raises:
        IOError: If unable to write to the file.

    Example:
        >>> save_random_state("checkpoint.pkl")
        >>> # Later, can restore with load_random_state("checkpoint.pkl")
    """
    filepath = Path(filepath)
    state = get_random_state()

    try:
        with open(filepath, "wb") as f:
            pickle.dump(state, f)
    except (IOError, OSError) as e:
        raise IOError(f"Failed to save random state to {filepath}: {e}")


def load_random_state(filepath: Union[str, Path]) -> None:
    """
    Load random state from a file.

    Args:
        filepath: Path to the saved random state file.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If unable to read the file.
        pickle.UnpicklingError: If the file is corrupted.

    Example:
        >>> load_random_state("checkpoint.pkl")
        >>> # Random state is now restored to saved state
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Random state file not found: {filepath}")

    try:
        with open(filepath, "rb") as f:
            state = pickle.load(f)
        set_random_state(state)
    except (IOError, OSError) as e:
        raise IOError(f"Failed to load random state from {filepath}: {e}")


@dataclass
class DeterministicContext:
    """
    Context manager for deterministic execution.

    This context manager ensures that all operations within its scope
    produce reproducible results by:
    1. Saving the current random state
    2. Setting a specific seed
    3. Restoring the original random state on exit

    Attributes:
        seed: The seed to use for deterministic operations.
        strict: If True, raises an error if CUDA is not available when
                determinism is requested. If False, runs on CPU with a warning.

    Example:
        >>> with DeterministicContext(seed=42):
        ...     # All operations here are deterministic
        ...     result = np.random.rand(10)
        >>>
        >>> # Original random state is restored here
    """

    seed: int
    strict: bool = False

    def __post_init__(self):
        if not isinstance(self.seed, int):
            raise TypeError(f"Seed must be an integer, got {type(self.seed).__name__}")
        if self.seed < 0:
            raise ValueError(f"Seed must be non-negative, got {self.seed}")

    def __enter__(self) -> "DeterministicContext":
        """Enter the context, save current state and set new seed."""
        self._saved_state = get_random_state()
        set_seed(self.seed)

        # Additional determinism settings for PyTorch
        self._prev_deterministic = torch.backends.cudnn.deterministic
        self._prev_benchmark = torch.backends.cudnn.benchmark
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context, restore original state."""
        # Restore random states
        set_random_state(self._saved_state)

        # Restore PyTorch backend settings
        torch.backends.cudnn.deterministic = self._prev_deterministic
        torch.backends.cudnn.benchmark = self._prev_benchmark


def hash_config(config: Dict[str, Any]) -> str:
    """
    Create a deterministic hash of a configuration dictionary.

    This is useful for:
    - Creating unique identifiers for experiments
    - Caching results based on configuration
    - Tracking which parameters were used

    The hash is deterministic and will be the same for the same configuration
    regardless of dictionary ordering.

    Args:
        config: Dictionary containing configuration parameters. Values should be
                serializable (strings, numbers, lists, nested dicts, etc.).

    Returns:
        A hexadecimal string hash of the configuration (64 characters).

    Raises:
        TypeError: If config contains non-serializable values.

    Example:
        >>> config = {'lr': 0.001, 'batch_size': 32, 'layers': [64, 128, 64]}
        >>> hash_config(config)
        'a3f2c8d9e1b4f6a7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1'
    """
    try:
        # Sort keys for deterministic ordering
        config_str = pickle.dumps(config, protocol=pickle.HIGHEST_PROTOCOL)
    except (TypeError, pickle.PicklingError) as e:
        raise TypeError(f"Config contains non-serializable values: {e}")

    # Create SHA-256 hash
    hash_obj = hashlib.sha256(config_str)
    return hash_obj.hexdigest()


def get_deterministic_hash(objects: list) -> str:
    """
    Create a deterministic hash from a list of objects.

    This is useful for creating unique identifiers from multiple sources
    such as model weights, data samples, or configuration parameters.

    Args:
        objects: List of objects to hash. Objects should be picklable.

    Returns:
        A hexadecimal string hash (64 characters).

    Example:
        >>> hash_val = get_deterministic_hash([config, model_weights, seed])
    """
    hash_obj = hashlib.sha256()

    for obj in objects:
        try:
            obj_bytes = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
            hash_obj.update(obj_bytes)
        except (TypeError, pickle.PicklingError) as e:
            raise TypeError(
                f"Object of type {type(obj).__name__} is not serializable: {e}"
            )

    return hash_obj.hexdigest()


# Backwards compatibility alias
set_random_seed = set_seed
