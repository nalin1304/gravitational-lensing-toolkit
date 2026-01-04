"""
TensorBoard Logging Utilities for PINN Training

This module provides utilities for logging training metrics, visualizations,
and model information to TensorBoard.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional, List
import io
from PIL import Image


class PINNLogger:
    """
    Logger for Physics-Informed Neural Network training.
    Handles metrics, images, and model graph logging to TensorBoard.
    """
    
    def __init__(self, log_dir: str = './runs', experiment_name: str = 'pinn_training'):
        """
        Parameters
        ----------
        log_dir : str
            Base directory for TensorBoard logs
        experiment_name : str
            Name of the experiment
        """
        self.writer = SummaryWriter(log_dir=f'{log_dir}/{experiment_name}')
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        
        print(f"TensorBoard logging to: {log_dir}/{experiment_name}")
        print(f"To view: tensorboard --logdir={log_dir}")
    
    def log_scalars(self, scalars: Dict[str, float], step: int, prefix: str = ''):
        """
        Log scalar values (losses, metrics).
        
        Parameters
        ----------
        scalars : dict
            Dictionary of scalar values to log
        step : int
            Current step/epoch number
        prefix : str
            Prefix for scalar names (e.g., 'train/', 'val/')
        """
        for name, value in scalars.items():
            self.writer.add_scalar(f'{prefix}{name}', value, step)
    
    def log_training_metrics(
        self,
        epoch: int,
        train_losses: Dict[str, float],
        val_losses: Dict[str, float],
        learning_rate: float
    ):
        """
        Log training and validation metrics for an epoch.
        
        Parameters
        ----------
        epoch : int
            Current epoch number
        train_losses : dict
            Training losses
        val_losses : dict
            Validation losses
        learning_rate : float
            Current learning rate
        """
        # Log training losses
        self.log_scalars(train_losses, epoch, prefix='train/')
        
        # Log validation losses
        self.log_scalars(val_losses, epoch, prefix='val/')
        
        # Log learning rate
        self.writer.add_scalar('learning_rate', learning_rate, epoch)
    
    def log_images(
        self,
        images: torch.Tensor,
        predictions: torch.Tensor,
        ground_truth: torch.Tensor,
        step: int,
        num_images: int = 4
    ):
        """
        Log sample images with predictions.
        
        Parameters
        ----------
        images : torch.Tensor
            Input images (B, C, H, W)
        predictions : torch.Tensor
            Model predictions
        ground_truth : torch.Tensor
            Ground truth values
        step : int
            Current step
        num_images : int
            Number of images to log
        """
        num_images = min(num_images, images.shape[0])
        
        # Log first few images
        for i in range(num_images):
            self.writer.add_image(
                f'images/sample_{i}',
                images[i],
                step,
                dataformats='CHW'
            )
    
    def log_predictions_comparison(
        self,
        model,
        dataloader,
        device,
        step: int,
        num_samples: int = 6
    ):
        """
        Log a comparison of predictions vs ground truth.
        
        Parameters
        ----------
        model : nn.Module
            Trained model
        dataloader : DataLoader
            Data loader
        device : torch.device
            Device to run on
        step : int
            Current step
        num_samples : int
            Number of samples to visualize
        """
        model.eval()
        
        with torch.no_grad():
            # Get one batch
            images, params, labels = next(iter(dataloader))
            images = images.float().to(device)
            
            # Make predictions
            pred_params, pred_class_logits = model(images)
            pred_classes = torch.argmax(pred_class_logits, dim=1)
            
            # Move to CPU
            images_cpu = images.cpu().numpy()
            params_cpu = params.numpy()
            labels_cpu = labels.numpy()
            pred_params_cpu = pred_params.cpu().numpy()
            pred_classes_cpu = pred_classes.cpu().numpy()
            
            # Create comparison figure
            fig = self._create_prediction_figure(
                images_cpu[:num_samples],
                params_cpu[:num_samples],
                labels_cpu[:num_samples],
                pred_params_cpu[:num_samples],
                pred_classes_cpu[:num_samples]
            )
            
            # Convert figure to image and log
            self.writer.add_figure('predictions/comparison', fig, step)
            plt.close(fig)
    
    def _create_prediction_figure(
        self,
        images: np.ndarray,
        true_params: np.ndarray,
        true_labels: np.ndarray,
        pred_params: np.ndarray,
        pred_labels: np.ndarray
    ):
        """Create a figure showing predictions vs ground truth"""
        n = len(images)
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.ravel()
        
        class_names = ['CDM', 'WDM', 'SIDM']
        
        for i in range(min(n, 6)):
            ax = axes[i]
            ax.imshow(images[i, 0], cmap='viridis', origin='lower')
            
            true_class = class_names[true_labels[i]]
            pred_class = class_names[pred_labels[i]]
            correct = "✓" if true_labels[i] == pred_labels[i] else "✗"
            color = 'green' if correct == "✓" else 'red'
            
            # Show key parameters
            title = f"{correct} True: {true_class} | Pred: {pred_class}\n"
            title += f"M_vir: {true_params[i,0]:.2e} → {pred_params[i,0]:.2e}"
            
            ax.set_title(title, fontsize=9, color=color, fontweight='bold')
            ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def log_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        class_names: List[str],
        step: int
    ):
        """
        Log confusion matrix as an image.
        
        Parameters
        ----------
        confusion_matrix : np.ndarray
            Confusion matrix (n_classes, n_classes)
        class_names : list
            List of class names
        step : int
            Current step
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Normalize
        cm_norm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        
        # Plot
        im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        
        # Labels
        ax.set(
            xticks=np.arange(len(class_names)),
            yticks=np.arange(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
            ylabel='True label',
            xlabel='Predicted label'
        )
        
        # Annotate
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                text = ax.text(
                    j, i, f'{confusion_matrix[i, j]}\n({cm_norm[i, j]:.2%})',
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black",
                    fontsize=10
                )
        
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        self.writer.add_figure('metrics/confusion_matrix', fig, step)
        plt.close(fig)
    
    def log_parameter_errors(
        self,
        pred_params: np.ndarray,
        true_params: np.ndarray,
        param_names: List[str],
        step: int
    ):
        """
        Log parameter prediction errors.
        
        Parameters
        ----------
        pred_params : np.ndarray
            Predicted parameters
        true_params : np.ndarray
            True parameters
        param_names : list
            Parameter names
        step : int
            Current step
        """
        errors = np.abs(pred_params - true_params)
        mae_per_param = np.mean(errors, axis=0)
        
        # Log MAE for each parameter
        for i, name in enumerate(param_names):
            self.writer.add_scalar(f'parameters/MAE_{name}', mae_per_param[i], step)
    
    def log_model_graph(self, model, input_size=(1, 1, 64, 64)):
        """
        Log model architecture graph.
        
        Parameters
        ----------
        model : nn.Module
            Model to log
        input_size : tuple
            Input tensor size
        """
        try:
            dummy_input = torch.randn(input_size)
            self.writer.add_graph(model, dummy_input)
            print("✓ Model graph logged to TensorBoard")
        except Exception as e:
            print(f"Warning: Could not log model graph: {e}")
    
    def log_histograms(self, model, step: int):
        """
        Log model parameter and gradient histograms.
        
        Parameters
        ----------
        model : nn.Module
            Model
        step : int
            Current step
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.writer.add_histogram(f'parameters/{name}', param, step)
                if param.grad is not None:
                    self.writer.add_histogram(f'gradients/{name}', param.grad, step)
    
    def log_text(self, tag: str, text: str, step: int = 0):
        """
        Log text information.
        
        Parameters
        ----------
        tag : str
            Tag for the text
        text : str
            Text to log
        step : int
            Step number
        """
        self.writer.add_text(tag, text, step)
    
    def log_hyperparameters(self, hparams: Dict, metrics: Dict):
        """
        Log hyperparameters and final metrics.
        
        Parameters
        ----------
        hparams : dict
            Hyperparameters
        metrics : dict
            Final metrics
        """
        self.writer.add_hparams(hparams, metrics)
    
    def close(self):
        """Close the TensorBoard writer."""
        self.writer.close()
        print("✓ TensorBoard logger closed")


def plot_to_tensorboard_image(fig) -> torch.Tensor:
    """
    Convert matplotlib figure to tensor for TensorBoard.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to convert
    
    Returns
    -------
    torch.Tensor
        Image tensor (C, H, W)
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    
    img = Image.open(buf)
    img_array = np.array(img)
    
    # Convert to (C, H, W) format
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
    
    return img_tensor


if __name__ == '__main__':
    # Test logger
    logger = PINNLogger(log_dir='./test_runs', experiment_name='test')
    
    # Log some dummy data
    for epoch in range(10):
        train_losses = {
            'total': np.random.rand(),
            'mse_params': np.random.rand(),
            'ce_class': np.random.rand(),
            'physics_residual': np.random.rand()
        }
        val_losses = {
            'total': np.random.rand(),
            'mse_params': np.random.rand(),
            'ce_class': np.random.rand(),
            'physics_residual': np.random.rand()
        }
        
        logger.log_training_metrics(epoch, train_losses, val_losses, 1e-3)
    
    logger.close()
    print("✓ Logger test passed!")
