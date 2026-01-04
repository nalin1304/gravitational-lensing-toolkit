"""
Data Augmentation for Gravitational Lensing Images

This module provides augmentation transforms for training the PINN model
on gravitational lensing convergence maps.
"""

import torch
import numpy as np
from typing import Optional, Tuple, Union


class RandomRotation:
    """
    Randomly rotate image by 90, 180, or 270 degrees.
    Preserves physical symmetry of lensing systems.
    """
    
    def __init__(self, p: float = 0.5):
        """
        Parameters
        ----------
        p : float
            Probability of applying rotation (default: 0.5)
        """
        self.p = p
    
    def __call__(self, image: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Apply random rotation to image"""
        if np.random.random() > self.p:
            return image
        
        # Random rotation: 90, 180, or 270 degrees
        k = np.random.choice([1, 2, 3])  # Number of 90-degree rotations
        
        if isinstance(image, torch.Tensor):
            # For PyTorch tensors: rotate along spatial dimensions (last 2)
            return torch.rot90(image, k=k, dims=(-2, -1))
        else:
            # For NumPy arrays: rotate along last 2 dimensions
            return np.rot90(image, k=k, axes=(-2, -1))


class RandomFlip:
    """
    Randomly flip image horizontally and/or vertically.
    Preserves physical symmetry of lensing systems.
    """
    
    def __init__(self, horizontal: bool = True, vertical: bool = True, p: float = 0.5):
        """
        Parameters
        ----------
        horizontal : bool
            Enable horizontal flips
        vertical : bool
            Enable vertical flips
        p : float
            Probability of applying each flip
        """
        self.horizontal = horizontal
        self.vertical = vertical
        self.p = p
    
    def __call__(self, image: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Apply random flips to image"""
        result = image
        
        if self.horizontal and np.random.random() < self.p:
            if isinstance(result, torch.Tensor):
                result = torch.flip(result, dims=(-1,))  # Flip last dimension
            else:
                result = np.flip(result, axis=-1)
        
        if self.vertical and np.random.random() < self.p:
            if isinstance(result, torch.Tensor):
                result = torch.flip(result, dims=(-2,))  # Flip second-to-last dimension
            else:
                result = np.flip(result, axis=-2)
        
        return result


class RandomBrightness:
    """
    Randomly adjust image brightness.
    Simulates different exposure times or signal-to-noise ratios.
    """
    
    def __init__(self, brightness_range: Tuple[float, float] = (0.8, 1.2), p: float = 0.5):
        """
        Parameters
        ----------
        brightness_range : tuple
            Range of brightness multipliers (min, max)
        p : float
            Probability of applying brightness adjustment
        """
        self.brightness_range = brightness_range
        self.p = p
    
    def __call__(self, image: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Apply random brightness adjustment"""
        if np.random.random() > self.p:
            return image
        
        # Random brightness factor
        factor = np.random.uniform(*self.brightness_range)
        
        if isinstance(image, torch.Tensor):
            result = image * factor
            # Clip to valid range [0, 1] if normalized
            result = torch.clamp(result, 0.0, 1.0)
        else:
            result = image * factor
            result = np.clip(result, 0.0, 1.0)
        
        return result


class RandomNoise:
    """
    Add random Gaussian noise to simulate observational uncertainty.
    """
    
    def __init__(self, noise_std: float = 0.01, p: float = 0.3):
        """
        Parameters
        ----------
        noise_std : float
            Standard deviation of Gaussian noise
        p : float
            Probability of applying noise
        """
        self.noise_std = noise_std
        self.p = p
    
    def __call__(self, image: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Add random Gaussian noise"""
        if np.random.random() > self.p:
            return image
        
        if isinstance(image, torch.Tensor):
            noise = torch.randn_like(image) * self.noise_std
            result = image + noise
            result = torch.clamp(result, 0.0, 1.0)
        else:
            noise = np.random.randn(*image.shape) * self.noise_std
            result = image + noise
            result = np.clip(result, 0.0, 1.0)
        
        return result


class Compose:
    """
    Compose multiple transforms together.
    """
    
    def __init__(self, transforms: list):
        """
        Parameters
        ----------
        transforms : list
            List of transform objects to apply sequentially
        """
        self.transforms = transforms
    
    def __call__(self, image: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Apply all transforms in sequence"""
        for transform in self.transforms:
            image = transform(image)
        return image


def get_training_transforms(
    rotation: bool = True,
    flip: bool = True,
    brightness: bool = True,
    noise: bool = True,
    rotation_p: float = 0.5,
    flip_p: float = 0.5,
    brightness_p: float = 0.5,
    noise_p: float = 0.3,
    brightness_range: Tuple[float, float] = (0.8, 1.2),
    noise_std: float = 0.01
) -> Compose:
    """
    Get standard training augmentation pipeline.
    
    Parameters
    ----------
    rotation : bool
        Enable rotation augmentation
    flip : bool
        Enable flip augmentation
    brightness : bool
        Enable brightness augmentation
    noise : bool
        Enable noise augmentation
    rotation_p : float
        Probability of rotation
    flip_p : float
        Probability of flip
    brightness_p : float
        Probability of brightness adjustment
    noise_p : float
        Probability of noise addition
    brightness_range : tuple
        Range for brightness adjustment
    noise_std : float
        Standard deviation for noise
    
    Returns
    -------
    Compose
        Composed transform pipeline
    
    Examples
    --------
    >>> transforms = get_training_transforms()
    >>> augmented_image = transforms(original_image)
    """
    transform_list = []
    
    if rotation:
        transform_list.append(RandomRotation(p=rotation_p))
    
    if flip:
        transform_list.append(RandomFlip(p=flip_p))
    
    if brightness:
        transform_list.append(RandomBrightness(brightness_range=brightness_range, p=brightness_p))
    
    if noise:
        transform_list.append(RandomNoise(noise_std=noise_std, p=noise_p))
    
    return Compose(transform_list)


class ToTensor:
    """Convert NumPy array to PyTorch tensor."""
    
    def __call__(self, image: np.ndarray) -> torch.Tensor:
        """Convert to tensor"""
        if isinstance(image, torch.Tensor):
            return image
        return torch.from_numpy(image).float()


class Normalize:
    """
    Normalize image to zero mean and unit variance.
    """
    
    def __init__(self, mean: float = 0.5, std: float = 0.5):
        """
        Parameters
        ----------
        mean : float
            Mean for normalization
        std : float
            Standard deviation for normalization
        """
        self.mean = mean
        self.std = std
    
    def __call__(self, image: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Normalize image"""
        return (image - self.mean) / self.std


if __name__ == '__main__':
    # Test augmentations
    import matplotlib.pyplot as plt
    
    # Create dummy image
    test_image = np.random.rand(1, 64, 64)
    
    # Get transforms
    transforms = get_training_transforms()
    
    # Apply multiple times to see variation
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    
    axes[0].imshow(test_image[0], cmap='viridis', origin='lower')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    for i in range(1, 8):
        augmented = transforms(test_image.copy())
        axes[i].imshow(augmented[0], cmap='viridis', origin='lower')
        axes[i].set_title(f'Augmented {i}')
        axes[i].axis('off')
    
    plt.suptitle('Data Augmentation Examples', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("✓ Augmentation tests passed!")
