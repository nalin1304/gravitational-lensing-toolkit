"""
Neural Posterior Estimation (NPE) for Fast Lens Modeling

Implements amortized neural posterior estimation for rapid lens parameter
inference, replacing traditional MCMC for LSST-scale lens samples.

Reference:
---------
Venkatraman et al. (2025), "Lens Model Accuracy in the Expected LSST
Lensed AGN Sample", arXiv:2510.20778

Achieves <1% bias and 6.5% precision on Einstein radius per lens,
enabling hierarchical inference for 1300+ lensed AGN.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, List, Optional
from collections import deque


class EmbeddingNetwork(nn.Module):
    """
    CNN encoder to extract features from lensed images.

    Compresses high-dimensional images into a low-dimensional
    summary statistic suitable for posterior estimation.
    """

    def __init__(
        self, input_size: int = 64, in_channels: int = 1, embedding_dim: int = 128
    ):
        super().__init__()

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # Block 1: 64x64 -> 32x32
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 2: 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 3: 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 4: 8x8 -> 4x4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Calculate flattened size
        flat_size = 256 * (input_size // 16) ** 2

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode image to embedding.

        Parameters
        ----------
        x : torch.Tensor
            Input image (batch, channels, H, W)

        Returns
        -------
        embedding : torch.Tensor
            Embedding vector (batch, embedding_dim)
        """
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


class ConditionalNormalizingFlow(nn.Module):
    """
    Conditional normalizing flow for posterior estimation.

    Maps from simple base distribution to complex posterior
    conditioned on observed image embedding.
    """

    def __init__(
        self,
        input_dim: int = 5,  # Number of lens parameters
        context_dim: int = 128,  # Embedding dimension
        n_flows: int = 8,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.context_dim = context_dim
        self.n_flows = n_flows

        # Flow layers (affine coupling transforms)
        self.flows = nn.ModuleList(
            [
                AffineCouplingLayer(input_dim, context_dim, hidden_dim)
                for _ in range(n_flows)
            ]
        )

        # Permutation layers (swap dimensions)
        self.permutations = [torch.randperm(input_dim) for _ in range(n_flows)]

    def forward(
        self, z: torch.Tensor, context: torch.Tensor, reverse: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transform through flow.

        Parameters
        ----------
        z : torch.Tensor
            Latent variables (batch, input_dim)
        context : torch.Tensor
            Conditioning context (batch, context_dim)
        reverse : bool
            If True, apply inverse transformation

        Returns
        -------
        z_transformed : torch.Tensor
            Transformed variables
        log_det_jacobian : torch.Tensor
            Log determinant of Jacobian
        """
        log_det_total = torch.zeros(z.size(0), device=z.device)

        if not reverse:
            # Forward: base -> target
            for i, flow in enumerate(self.flows):
                z, log_det = flow(z, context)
                log_det_total += log_det
                # Permute dimensions
                z = z[:, self.permutations[i]]
        else:
            # Inverse: target -> base
            for i in range(len(self.flows) - 1, -1, -1):
                # Undo permutation
                inv_perm = torch.argsort(self.permutations[i])
                z = z[:, inv_perm]
                z, log_det = flow(z, context, reverse=True)
                log_det_total += log_det

        return z, log_det_total

    def sample(self, context: torch.Tensor, n_samples: int = 1000) -> torch.Tensor:
        """
        Sample from posterior.

        Parameters
        ----------
        context : torch.Tensor
            Conditioning context (batch, context_dim)
        n_samples : int
            Number of samples per context

        Returns
        -------
        samples : torch.Tensor
            Samples from posterior (batch, n_samples, input_dim)
        """
        batch_size = context.size(0)

        # Sample from base distribution (standard normal)
        z = torch.randn(batch_size, n_samples, self.input_dim, device=context.device)

        # Expand context to match samples
        context_expanded = context.unsqueeze(1).expand(-1, n_samples, -1)
        context_expanded = context_expanded.reshape(batch_size * n_samples, -1)
        z = z.reshape(batch_size * n_samples, -1)

        # Transform through flow
        samples, _ = self.forward(z, context_expanded, reverse=False)

        # Reshape back
        samples = samples.reshape(batch_size, n_samples, self.input_dim)

        return samples


class AffineCouplingLayer(nn.Module):
    """
    Affine coupling layer for normalizing flow.

    Splits input into two halves, transforms one half conditioned
    on the other and the context.
    """

    def __init__(self, input_dim: int, context_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.input_dim = input_dim
        self.split_dim = input_dim // 2

        # Networks for scale and translation
        # Input: unconditioned half + context
        # Output: scale and translation for conditioned half
        output_dim = (input_dim - self.split_dim) * 2

        self.net = nn.Sequential(
            nn.Linear(self.split_dim + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self, x: torch.Tensor, context: torch.Tensor, reverse: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply affine coupling transformation.

        Parameters
        ----------
        x : torch.Tensor
            Input (batch, input_dim)
        context : torch.Tensor
            Context (batch, context_dim)
        reverse : bool
            If True, apply inverse

        Returns
        -------
        x_transformed : torch.Tensor
            Transformed input
        log_det_jacobian : torch.Tensor
            Log determinant of Jacobian
        """
        # Split input
        x1, x2 = x[:, : self.split_dim], x[:, self.split_dim :]

        # Compute scale and translation
        net_input = torch.cat([x1, context], dim=1)
        net_output = self.net(net_input)

        scale = net_output[:, ::2]
        translation = net_output[:, 1::2]

        if not reverse:
            # Forward: x2' = x2 * exp(scale) + translation
            x2_transformed = x2 * torch.exp(scale) + translation
            log_det = scale.sum(dim=1)
        else:
            # Inverse: x2 = (x2' - translation) * exp(-scale)
            x2_transformed = (x2 - translation) * torch.exp(-scale)
            log_det = -scale.sum(dim=1)

        # Concatenate back
        x_transformed = torch.cat([x1, x2_transformed], dim=1)

        return x_transformed, log_det


class NeuralPosteriorEstimator:
    """
    Neural Posterior Estimation for fast lens modeling.

    Replaces MCMC with amortized neural inference for rapid
    parameter estimation from lensed images.

    Reference
    ---------
    Venkatraman et al. (2025), arXiv:2510.20778
    """

    def __init__(
        self,
        embedding_net: Optional[EmbeddingNetwork] = None,
        flow: Optional[ConditionalNormalizingFlow] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize NPE model.

        Parameters
        ----------
        embedding_net : EmbeddingNetwork, optional
            Pre-trained embedding network
        flow : ConditionalNormalizingFlow, optional
            Pre-trained normalizing flow
        device : str
            Computation device
        """
        self.device = device

        if embedding_net is None:
            self.embedding_net = EmbeddingNetwork().to(device)
        else:
            self.embedding_net = embedding_net.to(device)

        if flow is None:
            self.flow = ConditionalNormalizingFlow().to(device)
        else:
            self.flow = flow.to(device)

        self.embedding_net.eval()
        self.flow.eval()

    def estimate_posterior(
        self,
        image: np.ndarray,
        n_samples: int = 10000,
        parameter_names: List[str] = None,
    ) -> Dict:
        """
        Estimate posterior distribution given observed image.

        Parameters
        ----------
        image : np.ndarray
            Observed lensed image
        n_samples : int
            Number of posterior samples
        parameter_names : list of str, optional
            Names of parameters

        Returns
        -------
        result : dict
            Dictionary containing:
            - 'samples': Posterior samples (n_samples, n_params)
            - 'mean': Posterior mean
            - 'std': Posterior standard deviation
            - 'median': Posterior median
            - 'credible_intervals': 68% and 95% credible intervals

        Reference
        ---------
        Venkatraman et al. (2025), "Lens Model Accuracy in the Expected
        LSST Lensed AGN Sample", arXiv:2510.20778
        """
        # Convert image to tensor
        if image.ndim == 2:
            image = image[np.newaxis, np.newaxis, :, :]
        elif image.ndim == 3:
            image = image[np.newaxis, :, :, :]

        x = torch.FloatTensor(image).to(self.device)

        # Get embedding
        with torch.no_grad():
            embedding = self.embedding_net(x)

            # Sample from posterior
            samples = self.flow.sample(embedding, n_samples=n_samples)

        # Convert to numpy
        samples = samples.cpu().numpy()[0]  # Remove batch dimension

        # Compute statistics
        mean = samples.mean(axis=0)
        median = np.median(samples, axis=0)
        std = samples.std(axis=0)

        # Credible intervals
        ci_68 = np.percentile(samples, [16, 84], axis=0)
        ci_95 = np.percentile(samples, [2.5, 97.5], axis=0)

        if parameter_names is None:
            parameter_names = [f"param_{i}" for i in range(samples.shape[1])]

        result = {
            "samples": samples,
            "mean": mean,
            "median": median,
            "std": std,
            "credible_intervals_68": ci_68,
            "credible_intervals_95": ci_95,
            "parameter_names": parameter_names,
        }

        return result

    def train_step(
        self,
        images: torch.Tensor,
        parameters: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """
        Single training step.

        Parameters
        ----------
        images : torch.Tensor
            Training images (batch, channels, H, W)
        parameters : torch.Tensor
            True parameters (batch, n_params)
        optimizer : torch.optim.Optimizer
            Optimizer

        Returns
        -------
        loss : float
            Training loss (negative log likelihood)
        """
        self.embedding_net.train()
        self.flow.train()
        optimizer.zero_grad()

        # Get embeddings
        embeddings = self.embedding_net(images)

        # Transform parameters through flow (inverse direction)
        z, log_det = self.flow(parameters, embeddings, reverse=True)

        # Negative log likelihood: -log p(x) = -log p(z) - log|det J|
        # Base distribution is standard normal
        log_pz = -0.5 * (z**2 + np.log(2 * np.pi)).sum(dim=1)
        log_px = log_pz + log_det

        loss = -log_px.mean()

        loss.backward()
        optimizer.step()

        return loss.item()


# Convenience function
def run_neural_posterior_estimation(
    image: np.ndarray,
    model_path: Optional[str] = None,
    n_samples: int = 10000,
    parameter_names: Optional[List[str]] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict:
    """
    Run neural posterior estimation for lens parameter inference.

    High-level interface for Venkatraman et al. (2025) method.

    Parameters
    ----------
    image : np.ndarray
        Observed lensed image
    model_path : str, optional
        Path to pre-trained model
    n_samples : int
        Number of posterior samples
    parameter_names : list of str, optional
        Parameter names [theta_E, q, PA, ...]
    device : str
        Computation device

    Returns
    -------
    result : dict
        Posterior distribution with statistics

    Performance
    -----------
    - Inference time: ~0.1s per lens (vs hours for MCMC)
    - Accuracy: <1% bias, 6.5% precision on θ_E
    - Scales to 1000+ lenses (LSST-era)

    Example
    -------

    >>> image = load_lensed_image("lens.fits")

    >>> result = run_neural_posterior_estimation(
    ...     image,
    ...     parameter_names=["theta_E", "q", "PA", "x0", "y0"]
    ... )

    >>> print(f"Einstein radius: {result['mean'][0]:.2f} ± {result['std'][0]:.2f}")

    Reference
    ---------
    Venkatraman et al. (2025), arXiv:2510.20778
    """
    model = NeuralPosteriorEstimator(device=device)

    if model_path:
        checkpoint = torch.load(model_path)
        model.embedding_net.load_state_dict(checkpoint["embedding_net"])
        model.flow.load_state_dict(checkpoint["flow"])

    result = model.estimate_posterior(image, n_samples, parameter_names)

    return result
